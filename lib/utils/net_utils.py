# Main
import os
import torch
import random
import numpy as np
import torch.nn.functional as F

from torch import nn
from itertools import accumulate
from functools import lru_cache, reduce
from torch.distributed.optim import ZeroRedundancyOptimizer
from smplx.lbs import batch_rodrigues, batch_rigid_transform

# Typing
from typing import List, Callable, Tuple

# Utils
from lib.utils.log_utils import log
from lib.utils.base_utils import dotdict


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    if x.ndim == xp.ndim - 1:
        x = x[None]

    m = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1] + torch.finfo(xp.dtype).eps)  # slope
    b = fp[..., :-1] - (m * xp[..., :-1])

    indices = torch.sum(torch.ge(x[..., :, None], xp[..., None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    return m.gather(dim=-1, index=indices) * x + b.gather(dim=-1, index=indices)


def integrate_weights(w: torch.Tensor):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.
    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.
    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.cumsum(w[..., :-1], dim=-1).clip(max=1.0)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = torch.cat([cw.new_zeros(shape), cw, cw.new_ones(shape)], dim=-1)
    return cw0


def weighted_percentile(t: torch.Tensor, w: torch.Tensor, ps: list):
    """Compute the weighted percentiles of a step function. w's must sum to 1."""
    t, w = matchup_channels(t, w)
    cw = integrate_weights(w)
    # We want to interpolate into the integrated weights according to `ps`.
    # Vmap fn to an arbitrary number of leading dimensions.
    cw_mat = cw.reshape([-1, cw.shape[-1]])
    t_mat = t.reshape([-1, t.shape[-1]])
    wprctile_mat = interpolate(torch.from_numpy(np.array(ps)).to(t, non_blocking=True),
                               cw_mat,
                               t_mat)
    wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
    return wprctile


def ray_transfer(s: torch.Tensor,
                 tn: torch.Tensor,
                 tf: torch.Tensor,
                 g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ig: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ):
    # transfer ray depth from s space to t space (with inverse of g)
    return ig(s * g(tf) + (1 - s) * g(tn))


def inv_transfer(t: torch.Tensor,
                 tn: torch.Tensor,
                 tf: torch.Tensor,
                 g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ):
    # transfer ray depth from t space back to s space (with function g)
    return (g(t) - g(tn)) / (g(tf) - g(tn))

# implement the inverse distance sampling stragety of mipnerf360


def linear_sampling(device='cuda',
                    n_samples: int = 128,
                    perturb=False,
                    ):
    # calculate the steps for each ray
    s_vals = torch.linspace(0., 1. - 1 / n_samples, steps=n_samples, device=device)  # S,
    if perturb:
        s_vals = s_vals + torch.rand_like(s_vals) / n_samples  # S,
    return s_vals

# Hierarchical sampling (section 5.2)


def searchsorted(a: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find indices where v should be inserted into a to maintain order.
    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.
    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.
    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = torch.arange(a.shape[-1], device=a.device)  # 128
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.max(torch.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)[0]  # 128
    idx_hi = torch.min(torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)[0]
    return idx_lo, idx_hi


def invert_cdf(u, t, w):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = interpolate(u, cw, t)
    return t_new


def importance_sampling(t: torch.Tensor,
                        w: torch.Tensor,
                        num_samples: int,
                        perturb=True,
                        single_jitter=False,
                        ):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        rng: random number generator (or None for `linspace` sampling).
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
        w_logits: [..., num_bins], logits corresponding to bin weights
        num_samples: int, the number of samples.
        single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
        deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
        use_gpu_resampling: bool, If True this resamples the rays based on a
        "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
        this resamples the rays based on brute-force searches, which is fast on
        TPUs, but slow on GPUs.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """

    # preparing for size change
    sh = *t.shape[:-1], num_samples  # B, P, I
    t = t.reshape(-1, t.shape[-1])
    w = w.reshape(-1, w.shape[-1])

    # assuming sampling in s space
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)

    eps = torch.finfo(torch.float32).eps

    # Draw uniform samples.

    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps if perturb else 0
    d = 1 if single_jitter else num_samples
    u = (
        torch.linspace(0, 1 - u_max, num_samples, device=t.device) +
        torch.rand(t.shape[:-1] + (d,), device=t.device) * max_jitter
    )

    u = invert_cdf(u, t, w)

    # preparing for size change
    u = u.reshape(sh)
    return u


def matchup_channels(t: torch.Tensor, w: torch.Tensor):
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)  # 65
    return t, w


def weight_to_pdf(t: torch.Tensor, w: torch.Tensor, eps=torch.finfo(torch.float32).eps**2):
    t, w = matchup_channels(t, w)
    """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
    return w / (t[..., 1:] - t[..., :-1]).clip(eps)


def pdf_to_weight(t: torch.Tensor, p: torch.Tensor):
    t, p = matchup_channels(t, p)
    """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
    return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-torch.inf, torch.inf)):
    t, w = matchup_channels(t, w)
    """Dilate (via max-pooling) a non-negative step function."""
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate = torch.sort(torch.cat([t, t0, t1], dim=-1), dim=-1)[0]
    t_dilate = t_dilate.clip(*domain)
    w_dilate = torch.max(
        torch.where(
            (t0[..., None, :] <= t_dilate[..., None])
            & (t1[..., None, :] > t_dilate[..., None]),
            w[..., None, :],
            0,
        ),
        dim=-1)[0][..., :-1]
    return t_dilate, w_dilate


def max_dilate_weights(t,
                       w,
                       dilation,
                       domain=(-torch.inf, torch.inf),
                       renormalize=False,
                       eps=torch.finfo(torch.float32).eps**2):
    """Dilate (via max-pooling) a set of weights."""
    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= torch.sum(w_dilate, dim=-1, keepdim=True).clip(eps)
    return t_dilate, w_dilate


def query(tq, t, y, outside_value=0):
    """Look up the values of the step function (t, y) at locations tq."""
    idx_lo, idx_hi = searchsorted(t, tq)
    yq = torch.where(idx_lo == idx_hi, outside_value,
                     torch.take_along_dim(torch.cat([y, torch.full_like(y[..., :1], outside_value)], dim=-1), idx_lo, dim=-1))  # ?
    return yq


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=False)


def raw2alpha(raw, dists=0.005, bias=0.0, act_fn=F.relu):
    return 1. - torch.exp(-act_fn(raw + bias) * dists)


def alpha2raw(alpha, dists=0.005, bias=0.0, act_fn=F.relu):
    return act_fn(-torch.log(1 - alpha) / dists) - bias


def volume_rendering(rgb, alpha, eps=1e-8, bg_brightness=0.0, bg_image=None):
    # NOTE: here alpha's last dim is not 1, but n_samples
    # rgb: n_batch, n_rays, n_samples, 3
    # alpha: n_batch, n_rays, n_samples
    # bg_image: n_batch, n_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: n_batch, n_rays, n_samples
    # rgb_map: n_batch, n_rays, 3
    # acc_map: n_batch, n_rays

    def render_weights(alpha: torch.Tensor, eps=1e-8):
        # alpha: n_batch, n_rays, n_samples
        expanded_alpha = torch.cat([alpha.new_ones(*alpha.shape[:2], 1), 1. - alpha + eps], dim=-1)
        weights = alpha * torch.cumprod(expanded_alpha, dim=-1)[..., :-1]  # (n_batch, n_rays, n_samples)
        return weights

    if bg_image is not None:
        rgb[:, :, -1] = bg_image

    weights = render_weights(alpha, eps)  # (n_batch, n_rays, n_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (n_batch, n_rays, 3)
    acc_map = torch.sum(weights, -1)  # (n_batch, n_rays)
    if bg_brightness < 0:  # smaller than zeros means we want to use noise as background
        bg_brightness = torch.rand_like(rgb_map)
    rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_brightness

    return weights, rgb_map, acc_map


def take_jacobian(func: Callable, input: torch.Tensor, create_graph=False, vectorize=True, strategy='reverse-mode'):
    return torch.autograd.functional.jacobian(func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class GradModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradModule, self).__init__()

    def take_gradient(self, output: torch.Tensor, input: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return take_gradient(output, input, d_out, self.training or create_graph, self.training or retain_graph)

    def jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.take_gradient(o, input, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-1)
        return jac


class MLP(GradModule):
    def __init__(self, input_ch=32, W=256, D=8, out_ch=257, skips=[4], actvn=nn.ReLU(), out_actvn=nn.Identity(), init=nn.Identity()):
        super(MLP, self).__init__()
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W
            if i == 0:
                I = input_ch
            if i in skips:
                I = input_ch + W
            if i == D:
                O = out_ch
            self.linears.append(nn.Linear(I, O))
        self.linears = nn.ModuleList(self.linears)
        self.actvn = actvn
        self.out_actvn = out_actvn

        for l in self.linears:
            init(l.weight)

    def forward(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x


def project(xyz, K, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].T + RT[:, 3:].T
    xyz = xyz @ K.T
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy


def transform(xyz, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].T + RT[:, 3:].T
    return xyz


def fix_random(fix=True):
    if fix:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)


def load_model(net: nn.Module,
               optims: List[nn.Module],
               scheduler: nn.Module,
               recorder: nn.Module,
               model_dir,
               resume=True,
               strict=False,
               skips=[],
               only=[],
               allow_mismatch=[],
               epoch=-1,
               load_others=True):
    if not resume:
        log(f"removing trained weights: {model_dir}", 'red')
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    log('loading model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))), 'blue')
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')

    pretrained_net = pretrained_model['net']
    if skips:
        keys = list(pretrained_net.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_net[k]

    if only:
        keys = list(pretrained_net.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_net[k]
    for key in allow_mismatch:
        if key in net.state_dict() and key in pretrained_net:
            net_parent = net
            pre_parent = pretrained_net
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                net_parent = getattr(net_parent, k)
                pre_parent = pre_parent[k]
            last_name = chain[-1]
            setattr(net_parent, last_name, nn.Parameter(pre_parent[last_name], requires_grad=getattr(net_parent, last_name).requires_grad))  # just replace without copying

    net.load_state_dict(pretrained_model['net'], strict=strict)
    log(f'loaded model at epoch: {pretrained_model["epoch"]}', 'blue')
    if load_others:
        for i, optim in enumerate(optims):
            optim.load_state_dict(pretrained_model['optims'][i])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0


def save_model(net, optims, scheduler, recorder, model_dir, epoch, latest=False, rank=0, distributed=False):
    os.system('mkdir -p {}'.format(model_dir))
    if distributed:
        # all other ranks should consolidate the state dicts of the optimizer to the default rank: 0
        for opt in optims:
            if isinstance(opt, ZeroRedundancyOptimizer):
                opt.consolidate_state_dict()
                torch.cuda.synchronize()  # sync across devices to make sure that the state dict saved is full
        if rank != 0:
            return  # other processes don't need to save the model
    model = {
        'net': net.state_dict(),
        'optims': [optim.state_dict() for optim in optims],
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if latest:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def root_of_any(k, l):
    for s in l:
        a = accumulate(k.split('.'), lambda x, y: x + '.' + y)
        for r in a:
            if s == r:
                return True
    return False


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def load_network(
    net: nn.Module,
    model_dir,
    resume=True,
    epoch=-1,
    strict=False,
    skips=[],
    only=[],
    allow_mismatch=[],
):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        log(f'pretrained model: {model_dir} does not exist', 'red')
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    log('loading network: {}'.format(model_path), 'blue')
    # ordered dict cannot be mutated while iterating
    # vanilla dict cannot change size while iterating
    pretrained_model = torch.load(model_path)
    pretrained_net = pretrained_model['net']

    if skips:
        keys = list(pretrained_net.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_net[k]

    if only:
        keys = list(pretrained_net.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_net[k]

    for key in allow_mismatch:
        if key in net.state_dict() and key in pretrained_net and not strict:
            net_parent = net
            pre_parent = pretrained_net
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                net_parent = getattr(net_parent, k)
                pre_parent = pre_parent[k]
            last_name = chain[-1]
            setattr(net_parent, last_name, nn.Parameter(pre_parent[last_name], requires_grad=getattr(net_parent, last_name).requires_grad))  # just replace without copying

    net.load_state_dict(pretrained_net, strict=strict)
    log(f'loaded network at epoch: {pretrained_model["epoch"]}', 'blue')
    return pretrained_model['epoch'] + 1


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 20


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (x.norm(dim=-1, keepdim=True) + eps)

import torch
from typing import Callable, Tuple, List

from . import base_renderer
from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import volume_rendering, importance_sampling, linear_sampling, ray_transfer, inv_transfer, weighted_percentile, searchsorted, matchup_channels, max_dilate_weights

from lib.networks.environ import mipnerf360_network

# from mipnerf360


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)  # 129
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)  # 128
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]  # 127
    y0_inner = torch.where(idx_hi[..., :-1] <= idx_lo[..., 1:], cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
    return y0_inner, y0_outer

# from mipnerf360


def lossfun_outer(t: torch.Tensor, w: torch.Tensor, t_env: torch.Tensor, w_env: torch.Tensor, eps=torch.finfo(torch.float32).eps):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)
    """The proposal weight should be an upper envelope on the nerf weight."""
    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return (w - w_outer).clip(0.).pow(2) / (w + eps)


def lossfun_distortion(t: torch.Tensor, w: torch.Tensor):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2  # 64
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])  # 64
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
    """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
    # Distortion when the intervals do not overlap.
    d_disjoint = torch.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

    # Distortion when the intervals overlap.
    d_overlap = (2 *
                 (torch.minimum(t0_hi, t1_hi)**3 - torch.maximum(t0_lo, t1_lo)**3) +
                 3 * (t1_hi * t0_hi * torch.abs(t1_hi - t0_hi) +
                      t1_lo * t0_lo * torch.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                      (t0_lo - t1_hi) + t1_lo * t0_hi *
                      (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

    # Are the two intervals not overlapping?
    are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

    return torch.where(are_disjoint, d_disjoint, d_overlap)


def anneal_weights(t: torch.Tensor, w: torch.Tensor, train_frac: float, anneal_slope: float = 10.0):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)

    # Optionally anneal the weights as a function of training iteration.
    if anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        def bias(x, s): return (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, anneal_slope)
    else:
        anneal = 1.

    # A slightly more stable way to compute weights**anneal. If the distance
    # between adjacent intervals is zero then its weight is fixed to 0.
    logits_resample = torch.where(
        t[..., 1:] > t[..., :-1],
        anneal * torch.log(w), -torch.inf)

    w = torch.softmax(logits_resample, dim=-1)
    return w


class Renderer(base_renderer.Renderer):
    def get_pixel_value(self,
                        ray_o: torch.Tensor,
                        ray_d: torch.Tensor,
                        near: torch.Tensor,
                        far: torch.Tensor,
                        batch: dotdict,
                        ) -> dotdict:

        # preparing shapes
        B, P, C = ray_o.shape

        # type annotation
        self.net: mipnerf360_network.Network

        # decoder for density
        def coarse_decoder(x, v, d): return self.net.forward_coarse(x, v, d, batch)
        def fine_decoder(x, v, d): return self.net.forward_fine(x, v, d, batch)

        # an abstracted sample and inference (forward) function
        def sample_and_forward(S: int, S_product: int = None, s_vals: torch.Tensor = None, weights: torch.Tensor = None, decoder=coarse_decoder):  # lots of arguments captured
            if s_vals is None and weights is None:
                # linearly sample the first level
                s_vals = linear_sampling(ray_o.device, S, cfg.perturb > 0 and self.net.training)[None, None, :].expand(B, P, S)
            else:
                # MARK: GRAD
                # make sampling optimizable will typically render the optimization non-linear
                s_vals = s_vals.detach()
                weights = weights.detach()

                # perform dilation around sampled weights of previous level # * this is important for the stability of prop_loss and overall convergence
                dilation = cfg.dilation_bias + cfg.dilation_multiplier / S_product
                s_vals, weights = max_dilate_weights(s_vals, weights, dilation, domain=(0., 1.), renormalize=True)
                s_vals = s_vals[..., 1:-1]
                weights = weights[..., 1:-1]

                # perform importance sampling # * this is important for the overall quality of the rendered results (make it behave like a large network)
                train_frac = batch.meta.iter_step / (cfg.ep_iter * cfg.train.epoch) if self.net.training else 1.0  # TODO: training fraction, use 1.0 for visualization?
                weights = anneal_weights(s_vals, weights, train_frac)  # anneal the weights as a function of iter
                s_vals = importance_sampling(s_vals, weights, S, cfg.perturb > 0 and self.net.training)

            # transform depth in disparity space to euclidian space
            z_vals = ray_transfer(s_vals, near[..., None], far[..., None])
            wpts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]  # B, P, S, 3

            # compute the color and density
            out = self.get_density_color(wpts, ray_d, z_vals, decoder)  # will copy last distance

            # reshape to [num_rays, num_samples along ray, 4]
            rgb, occ = out.raw[..., :-1].reshape(B, P, S, -1), out.raw[..., -1:].reshape(B, P, S)  # B, P, S, 3; # B, P, S
            weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_brightness=cfg.bg_brightness)  # B, P, S; B, P, 3; B, P

            # preparing return values
            ret = dotdict()  # save some memory when rendering
            ret.rgb_map = rgb_map
            ret.acc_map = acc_map
            ret.s_vals = s_vals
            ret.z_vals = z_vals
            ret.weights = weights

            # save some memory or computation based on current mode
            if self.net.training:
                ret.update(out)  # update with extra network output
            if cfg.vis_depth_map:
                ret.depth_map = torch.sum(weights * z_vals, dim=-1).view(B, P)
            if cfg.vis_median_depth:
                ret.median_map = weighted_percentile(torch.cat([z_vals, far[..., None]], dim=-1),
                                                     torch.cat([weights, 1 - acc_map[..., None]], dim=-1),
                                                     [0.5],
                                                     ).view(B, P).clip(cfg.clip_near, cfg.clip_far)
            return ret

        # * coarse level sampling: round 1
        out = sample_and_forward(cfg.n_samples)
        s_vals_p0, weights_p0 = out.s_vals, out.weights

        # * coarse level sampling: round 2
        out = sample_and_forward(cfg.n_samples, cfg.n_samples ** 2, out.s_vals, out.weights, coarse_decoder)
        s_vals_p1, weights_p1 = out.s_vals, out.weights

        # * importance level sampling
        out = sample_and_forward(cfg.n_importance, cfg.n_samples ** 2 * cfg.n_importance, out.s_vals, out.weights, fine_decoder)
        s_vals, weights = out.s_vals, out.weights

        # ? move loss computation logics to trainer instead of in renderer? will prevent from visualizing these losses
        ret = out  # the output container, discard results from previous samples for now
        ret.distortion = lossfun_distortion(s_vals, weights)
        ret.interlevel = \
            lossfun_outer(s_vals.detach(),
                          weights.detach(),
                          s_vals_p0,
                          weights_p0) + \
            lossfun_outer(s_vals.detach(),
                          weights.detach(),
                          s_vals_p1,
                          weights_p1)  # MARK: GRAD

        return ret

import torch

from torch import nn
from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.log_utils import print_colorful_stacktrace
from lib.utils.net_utils import volume_rendering
from lib.networks.environ import mipnerf360_network


class Renderer(nn.Module):  # TODO: renderer should not be a module
    def __init__(self, net: mipnerf360_network.Network):
        super(Renderer, self).__init__()
        self.net = net

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.n_samples, device=near.device, dtype=near.dtype)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=upper.device, dtype=upper.dtype)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder) -> dotdict:
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        B, P, S = wpts.shape[:3]
        wpts = wpts.view(B, P * S, -1)
        viewdir = viewdir[:, :, None].expand(-1, -1, S, -1)
        viewdir = viewdir.reshape(B, P * S, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(B, P * S)

        ret = raw_decoder(wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch) -> dotdict:
        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        B, P, S = wpts.shape[:3]

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        def raw_decoder(wpts_val, viewdir_val, dists_val): return self.net(wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        raw: torch.Tensor = ret.raw
        rgb = raw[..., :-1].view(B, P, S, raw.shape[-1] - 1)  # B, P, S, 3
        occ = raw[..., -1:].view(B, P, S)  # B, P, S, 1

        # volume rendering of rgb values
        weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_brightness=cfg.bg_brightness)

        depth_map = torch.sum(weights * z_vals, dim=-1)

        rgb_map = rgb_map.view(B, P, -1)
        acc_map = acc_map.view(B, P)
        depth_map = depth_map.view(B, P)

        # prepare for regulariaztion on distortion loss
        ret.weights = weights
        ret.z_vals = z_vals

        if not self.net.training:
            ret = dotdict()  # save some memory

        ret.rgb_map = rgb_map
        ret.acc_map = acc_map
        ret.depth_map = depth_map

        # prepare for semantics and normal critics
        rgb = ret.rgb_map
        if rgb.shape[-1] >= 3:
            ret.rgb_map = rgb[..., :3]
            rgb = rgb[..., 3:]
        if rgb.shape[-1] >= 3:
            ret.norm_map = rgb[..., :3]
            rgb = rgb[..., 3:]

        return ret

    def render(self, batch):
        ray_o = batch.ray_o
        ray_d = batch.ray_d
        near = batch.near
        far = batch.far
        near = near.clip(min=cfg.clip_near)  # do not go back the camera
        far = far.clip(max=cfg.clip_far)  # do not go back the camera

        # volume rendering for each pixel
        B, P = ray_o.shape[:2]
        chunk = cfg.train_chunk_size if self.net.training else cfg.render_chunk_size
        ret_list = []
        for i in range(0, P, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk, batch)
            ret_list.append(pixel_value)
        try:
            keys = ret_list[0].keys()
            ret = {k: torch.cat([r[k] for r in ret_list if k in r], dim=1) for k in keys}
        except Exception as e:
            print_colorful_stacktrace()
            __import__('ipdb').set_trace()
        return dotdict(ret)

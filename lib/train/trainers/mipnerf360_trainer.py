import torch
import numpy as np

from torch import nn

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.networks.renderer import base_renderer, make_renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.renderer: base_renderer.Renderer = make_renderer(cfg, net)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = dotdict()
        loss = 0

        # mipnerf360 regularzation
        if 'interlevel' in ret:
            prop_loss = ret.interlevel.mean()
            scalar_stats.update({'prop_loss': prop_loss})
            loss += cfg.prop_loss_weight * prop_loss

        if 'distortion' in ret:
            dist_loss = ret.distortion.mean()
            scalar_stats.update({'dist_loss': dist_loss})
            loss += cfg.dist_loss_weight * dist_loss

        if 'rgb_map' in ret:
            rgb_pred = ret.rgb_map
            rgb_gt = batch.rgb
            msk_gt = batch.msk[..., None].expand(rgb_gt.shape)  # add last dimension
            denom = msk_gt.sum()

            resd_sq = (rgb_pred - rgb_gt) ** 2
            mse = (resd_sq * msk_gt).sum() / denom
            psnr = (1 / mse).log() * 10 / np.log(10)
            charb = (torch.sqrt(resd_sq + 0.001 ** 2) * msk_gt).sum() / denom

            img_loss = charb
            scalar_stats.psnr = psnr
            scalar_stats.img_loss = img_loss
            loss += cfg.img_loss_weight * img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats

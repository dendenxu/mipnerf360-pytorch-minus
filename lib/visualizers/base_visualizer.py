import os
import torch
import kornia

from lib.config import cfg
from lib.utils.log_utils import log, run
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import normalize
from lib.utils.color_utils import colormap
from lib.utils.data_utils import save_image


class Visualizer:
    def __init__(self):
        self.prepare_result_paths()

    def prepare_result_paths(self):
        result_dir = os.path.join(cfg.result_dir, 'comparison')
        result_dir = self.prepare_result_subfolder(result_dir)

        img_path = f'{result_dir}/frame{{0:04d}}_view{{1:04d}}.png'
        img_gt_path = os.path.splitext(img_path)[0] + '_gt' + os.path.splitext(img_path)[1]
        img_loss_path = os.path.splitext(img_path)[0] + '_loss' + os.path.splitext(img_path)[1]
        self.img_path = img_path
        self.img_gt_path = img_gt_path
        self.img_loss_path = img_loss_path

    @staticmethod
    def prepare_result_subfolder(result_dir: str):
        if result_dir.endswith('comparison'):
            result_dir = result_dir[:-len('comparison') - 1]
            default_appendix = 'comparison'
        else:
            default_appendix = ''
        if cfg.vis_acc_map:
            result_dir = f"{result_dir}/acc"
        elif cfg.vis_depth_map:
            result_dir = f"{result_dir}/depth"
        elif cfg.vis_distortion:
            result_dir = f"{result_dir}/distortion"
        else:
            result_dir = f"{result_dir}/{default_appendix}"
        return result_dir

    def update_result_dir(self):
        os.makedirs(self.result_dir, exist_ok=True)
        log(f'the results are saved at {self.result_dir}', 'yellow')

        if cfg.overwrite:
            log(f'will delete visualization in: {self.result_dir}', 'red')
            run(f'rm -rf {self.result_dir}')

    @staticmethod
    def generate_image(output: dotdict, batch: dotdict):
        H, W = batch['H'].item(), batch['W'].item()

        if cfg.vis_acc_map:  # visualize depth map (ray occupancy accumulation) or just shadow coefficients (sphere tracing)
            rgb_map = output.acc_map[0, ..., None]  # 0 - 1
            if 'msk' in batch:
                rgb_gt = batch.msk[0, ..., None]  # 0 - 1

        elif cfg.vis_depth_map:  # visualize depth map
            def depth_curve_fn(x): return torch.log(x + torch.finfo(torch.float32).eps)
            if cfg.vis_median_depth:
                depth_map = output.median_map[0]
            else:
                depth_map = output.depth_map[0]
            depth_map = depth_curve_fn(depth_map)
            percentile = 0.005
            percentile_number = int(percentile * depth_map.numel())
            depth_min = depth_map.ravel().topk(percentile_number, largest=False)[0].max() # a simple version of percentile
            depth_max = depth_map.ravel().topk(percentile_number, largest=True)[0].min() # a simple version of percentile
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            rgb_map = colormap(depth_map)

        elif cfg.vis_distortion:
            rgb_map = colormap(output.distortion[0])

        else:
            # visualize rgb map (default) (valid when relighting)
            rgb_map = output.rgb_map[0]
            if 'rgb' in batch:
                rgb_gt = batch.rgb[0]

        if rgb_map.ndim == 2:
            mask_at_box = batch.mask_at_box[0]
            mask_at_box = mask_at_box.reshape(H, W)
            mask_at_box = mask_at_box.nonzero(as_tuple=True)
            img_pred = rgb_map.new_ones(H, W, 3) * cfg.bg_brightness
            img_pred[mask_at_box] = rgb_map
        else:
            img_pred = rgb_map

        if 'rgb_gt' in locals():
            if rgb_gt.ndim == 2:
                img_gt = rgb_gt.new_ones(H, W, 3) * cfg.bg_brightness
                img_gt[mask_at_box] = rgb_gt
            else:
                img_gt = rgb_gt

        if 'orig_H' in batch and 'orig_W' in batch:
            img_pred = Visualizer.fill_image(img_pred, batch.orig_H.item(), batch.orig_W.item(), batch.crop_bbox[0])

            if 'img_gt' in locals():
                img_gt = Visualizer.fill_image(img_gt, batch.orig_H.item(), batch.orig_W.item(), batch.crop_bbox[0])

        if 'img_gt' in locals():
            img_loss = (img_pred - img_gt).pow(2).sum(dim=-1).clip(0, 1)[..., None].expand(img_pred.shape)
            img_loss = img_loss.detach().cpu().numpy()
            img_pred = img_pred.detach().cpu().numpy()
            img_gt = img_gt.detach().cpu().numpy()
            return img_pred, img_gt, img_loss
        else:
            img_pred = img_pred.detach().cpu().numpy()
            return img_pred

    @staticmethod
    def fill_image(img, orig_H, orig_W, bbox):
        full_img = img.new_ones(orig_H, orig_W, 3) * cfg.bg_brightness
        height = bbox[1, 1] - bbox[0, 1]
        width = bbox[1, 0] - bbox[0, 0]
        full_img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
        return full_img

    def visualize(self, output: dotdict, batch: dotdict):
        vis_ret = Visualizer.generate_image(output, batch)

        view_index = batch.meta.view_index.item()
        frame_index = batch.meta.frame_index.item()

        img_path = self.img_path.format(frame_index, view_index)
        img_gt_path = self.img_gt_path.format(frame_index, view_index)
        img_loss_path = self.img_loss_path.format(frame_index, view_index)

        if not hasattr(self, 'result_dir'):
            self.result_dir = os.path.dirname(img_path)
            self.update_result_dir()

        if isinstance(vis_ret, list) or isinstance(vis_ret, tuple):
            img_pred, img_gt, img_error = vis_ret
            save_image(img_path, img_pred)
            save_image(img_gt_path, img_gt)
            save_image(img_loss_path, img_error)
        else:
            save_image(img_path, vis_ret)

    def summarize(self):
        input = '"' + self.result_dir + '/*' + os.path.splitext(self.img_path)[1] + '"'
        output = self.result_dir + '.mp4'
        cmd = [
            'ffmpeg',
            '-framerate', cfg.fps,
            '-f', 'image2',
            '-pattern_type', 'glob',
            '-y',
            '-r', cfg.fps,
            '-i', input,
            '-c:v', 'libx264',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            output,
        ]
        run(cmd)
        log(f'video generated: {output}', 'yellow')

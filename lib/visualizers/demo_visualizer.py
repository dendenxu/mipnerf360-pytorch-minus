# This file is reused when we're performing textured rendering or just plain old rendering
import os

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.log_utils import log
from lib.utils.data_utils import save_image
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def prepare_result_paths(self):
        data_dir = f'data/novel_view/{cfg.exp_name}'
        data_dir = f'data/novel_view/{cfg.exp_name}'

        if cfg.perform:
            img_path = f'{data_dir}/perform/view{{1:04d}}_frame{{1:04d}}.png'
        elif 'sfm' in cfg.test_dataset_module or 'mipnerf360' in cfg.test_dataset_module:  # special treatment for sfm datasets
            img_path = f'{data_dir}/frame_{{0:04d}}_view_{{1:04d}}.png' # TODO: this is evil
        else:
            img_path = f'{data_dir}/frame_{{0:04d}}/view_{{1:04d}}.png'

        result_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        result_dir = self.prepare_result_subfolder(result_dir)

        img_path = os.path.join(result_dir, img_name)
        self.img_path = img_path

    def visualize(self, output: dotdict, batch: dotdict):
        img_pred = Visualizer.generate_image(output, batch)
        if isinstance(img_pred, tuple):
            img_pred = img_pred[0]

        frame_index = batch['frame_index'].item()
        view_index = batch['view_index'].item()

        img_path = self.img_path.format(frame_index, view_index)

        if not hasattr(self, 'result_dir'):
            self.result_dir = os.path.dirname(img_path)
            self.update_result_dir()

        save_image(img_path, img_pred)

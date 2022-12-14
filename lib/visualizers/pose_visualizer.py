import os
from lib.config import cfg
from lib.utils.data_utils import save_image
from lib.utils.base_utils import dotdict
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def prepare_result_paths(self):
        result_dir = f'data/pose_sequence/{cfg.exp_name}/view_{{1:04d}}'
        result_dir = self.prepare_result_subfolder(result_dir)

        img_path = f'{result_dir}/frame_{{0:04d}}.png'
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

# monocular structure from motion dataset
# a handheld camera, with moving people

import os
import cv2
import torch
import imageio
import numpy as np
from os.path import join

from lib.config import cfg
from lib.utils.log_utils import log
from lib.utils.parallel_utils import parallel_execution

from . import base_dataset


class Dataset(base_dataset.Dataset):
    def load_view(self):
        if len(cfg.training_view) != 1 and cfg.training_view[0] != 0:
            log(f'selected: {cfg.training_view} but handheld dataset only support view == [0], will use view = [0]', 'red')
        if 'train' in self.split:
            self.view = [0]
        else:
            self.view = cfg.test_view

    def load_ims_data(self):
        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame if 'train' in self.split else cfg.num_eval_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[[0]]
            for ims_data in self.annots['ims'][i:i + ni * i_intv][::i_intv]
        ])
        self.cam_inds = np.array([
            0 for i in range(len(self.ims))
        ])  # all is 0
        self.num_cams = 1

    def load_gt(self):
        if self.split != 'train' or cfg.no_data_cache:
            return super(Dataset, self).load_gt()
        action = super(Dataset, self).get_image_and_mask
        indices = list(range(len(self)))
        log(f'prefetching images and maskes...', 'blue')
        self.images_and_maskes = parallel_execution(indices, action=action, print_progress=True)

    def get_image_and_mask(self, index):
        if self.split != 'train' or cfg.no_data_cache:
            return super(Dataset, self).get_image_and_mask(index)
        return self.images_and_maskes[index]

    def get_indices(self, index):
        latent, frame, view, cam = super(Dataset, self).get_indices(index)
        return latent, frame, view, frame

import numpy as np
from lib.config import cfg
from lib.utils.log_utils import log

from . import pose_dataset


class Dataset(pose_dataset.Dataset):
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
        self.num_cams = 1

import numpy as np
from lib.config import cfg
from lib.utils.base_utils import dotdict
from . import sfm_dataset



class Dataset(sfm_dataset.Dataset):
    def get_blend(self, i):
        ret = dotdict()
        ret.meta = dotdict()
        ret.wbounds = np.array([[-cfg.box_far, -cfg.box_far, -cfg.box_far],
                                [cfg.box_far, cfg.box_far, cfg.box_far], ],
                               dtype=np.float32,
                               )
        return ret

    def load_bigpose(self):
        pass

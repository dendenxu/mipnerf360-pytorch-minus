from . import sfm_demo_dataset, mipnerf360_dataset
from lib.utils.base_utils import dotdict


class Dataset(sfm_demo_dataset.Dataset):
    def get_blend(self, i):
        return mipnerf360_dataset.Dataset.get_blend(self, i)  # evil inheritance

    def load_bigpose(self):
        return mipnerf360_dataset.Dataset.load_bigpose(self)

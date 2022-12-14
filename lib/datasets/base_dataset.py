import os
import cv2
import torch
import imageio
import numpy as np
import torch.utils.data as data

from os.path import join
from os import system, listdir

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.data_utils import sample_ray, load_image, load_unchanged, read_mask_by_img_path, get_bounds, to_tensor, to_numpy

from pytorch3d.structures import Meshes

# Naming convenstion for dataset functions:
# 1. load_stuff is to load some shared info from the disk (like the pose of bigpose or smpl face / weights like meta-data)
# 2. get_stuff is to prepare for per-index batch input data for the training of this index (implemented in a nested structure for now to make it DRY)


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        # do not change the order of data loading (has dependent structures)
        self.load_meta(data_root, human, ann_file, split)  # load shared metadata across dataset
        self.load_view()  # determine view to use for the current mode
        self.load_ims_data()  # only load the data, no strange processing
        self.load_ims_ravel()  # perform strange processing on data
        self.load_gt()  # for prefetching, for now, just skeleton

    def load_meta(self, data_root, human, ann_file, split):
        self.data_root = data_root
        self.human = human
        self.split = split

        self.annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = self.annots['cams']
        self.lbs_root = join(self.data_root, cfg.lbs)
        self.bkgd_root = join(self.data_root, cfg.bkgd)
        self.nrays = cfg.n_rays

        if os.path.exists(self.bkgd_root):
            self.BGs = np.stack([load_image(join(self.bkgd_root, f'{cam:02d}.jpg')) for cam in range(len(self.cams['K']))])

    def load_view(self):
        num_cams = len(self.cams['K'])
        training_view = cfg.training_view if len(cfg.training_view) else list(range(num_cams))
        if len(cfg.test_view) == 0:
            test_view = [i for i in range(num_cams) if i not in training_view]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = training_view if 'train' in self.split else test_view
        self.view = view

    def load_ims_data(self):
        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame if 'train' in self.split else cfg.num_eval_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[self.view]
            for idx, ims_data in enumerate(self.annots['ims'][i:i + ni * i_intv][::i_intv]) if idx * i_intv + i not in cfg.skip
        ])
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.view]
            for idx, ims_data in enumerate(self.annots['ims'][i:i + ni * i_intv][::i_intv]) if idx * i_intv + i not in cfg.skip
        ])
        self.num_cams = len(self.view)

    def load_ims_ravel(self):
        self.ims = self.ims.ravel()
        self.cam_inds = self.cam_inds.ravel()

    def load_gt(self):
        pass

    def get_mask(self, index):
        msk = read_mask_by_img_path(self.data_root, self.ims[index], cfg.erode_dilate_mask, cfg.mask_dir)
        H, W = int(msk.shape[0] * cfg.ratio), int(msk.shape[1] * cfg.ratio)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # load camera parameters
        view_index = self.cam_inds[index]
        K = np.array(self.cams['K'][view_index], dtype=np.float32)
        D = np.array(self.cams['D'][view_index], dtype=np.float32)

        # update camera parameters based on scaling
        K[:2] = K[:2] * cfg.ratio

        # undistort image & mask -> might overload CPU
        msk = cv2.undistort(msk, K, D)
        return msk

    def get_image_and_mask(self, index):
        img_path = join(self.data_root, self.ims[index])
        img = load_image(img_path)  # why do we need a separated get_mask function?
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        # load camera parameters
        view_index = self.cam_inds[index]
        K = np.array(self.cams['K'][view_index], dtype=np.float32)
        D = np.array(self.cams['D'][view_index], dtype=np.float32)

        # update camera parameters based on scaling
        K[:2] = K[:2] * cfg.ratio

        # undistort image & mask -> might overload CPU
        img = cv2.undistort(img, K, D)

        # masking images
        msk = self.get_mask(index)

        if cfg.mask_bkgd:
            msk = torch.from_numpy(msk)
            img = torch.from_numpy(img)
            img[msk == 0] = 0
            img = img.numpy()
            msk = msk.numpy()

        return img, msk

    def get_lbs_params(self, i):
        pass

    def get_blend(self, i):
        pass

    def get_indices(self, index):
        # store index data
        latent_index = index // len(self.view)

        # find frame index
        i = int(os.path.basename(self.ims[index])[:-4])
        frame_index = i

        # load camera parameters
        view_index = self.cam_inds[index]  # should always be zero

        # store camera sub
        cam_index = view_index

        return latent_index, frame_index, view_index, cam_index

    def get_gt(self, index):
        # read images & masks
        img, msk = self.get_image_and_mask(index)

        # get meta indices
        latent_index, frame_index, view_index, cam_index = self.get_indices(index)

        # load camera parameters
        K = np.array(self.cams['K'][cam_index], dtype=np.float32)
        D = np.array(self.cams['D'][cam_index], dtype=np.float32)
        R = np.array(self.cams['R'][cam_index], dtype=np.float32)
        T = np.array(self.cams['T'][cam_index], dtype=np.float32) / 1000.

        # update camera parameters based on scaling
        H, W = img.shape[:2]
        K[:2] = K[:2] * cfg.ratio

        # load SMPL & pose & human related parameters
        ret = self.get_blend(frame_index)

        # store image parameter
        meta = {
            'img': img,
            'msk': msk,
        }
        ret.update(meta)

        # store camera parameters
        meta = {
            'cam_K': K,
            'cam_R': R,
            'cam_T': T,
            'cam_RT': np.concatenate([R, T], axis=1),
            'H': H,
            'W': W,
        }
        ret.update(meta)
        ret.meta.update(meta)  # keep a copy on the cpu

        # store camera background images
        if hasattr(self, 'BGs'):
            BG = np.array(self.BGs[view_index], dtype=np.float32)
            BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_AREA)
            BG = cv2.undistort(BG, K, D)
            meta = {
                "cam_BG": BG,
            }
            ret.update(meta)
            ret.meta.update(meta)

        meta = {
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index,
        }
        ret.update(meta)
        ret.meta.update(meta)
        return ret

    def __getitem__(self, index):
        ret = self.get_gt(index)
        img, msk, H, W, K, R, T, RT = ret.img, ret.msk, ret.H, ret.W, ret.cam_K, ret.cam_R, ret.cam_T, ret.cam_RT
        wbounds = ret.wbounds  # need for near far and mask at box

        # sample rays
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = sample_ray(
            img, msk, K, R, T, wbounds, cfg.n_rays, self.split, cfg.subpixel_sample,
            cfg.body_sample_ratio, cfg.face_sample_ratio)

        # compute occupancy (mask value), whether sampled point is in mask
        msk = msk[coord[:, 0], coord[:, 1]].astype(np.float32)

        # store ray data
        meta = {
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'coord': coord,
            'msk': msk,
            'mask_at_box': mask_at_box,
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)

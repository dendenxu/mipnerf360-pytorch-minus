import os
import cv2
import imageio
import numpy as np
from lib.config import cfg
from lib.utils.render_utils import load_cam
from lib.utils.data_utils import read_mask_by_img_path, get_bounds, get_rays_within_bounds, load_image

from . import base_dataset


class Dataset(base_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__(data_root, human, ann_file, split)
        self.load_camera()

    def load_ims_ravel(self):  # just overload
        pass

    def load_camera(self):
        self.Ks = np.array(self.cams['K'])[self.view].astype(np.float32)
        self.Rs = np.array(self.cams['R'])[self.view].astype(np.float32)
        self.Ts = np.array(self.cams['T'])[self.view].astype(np.float32) / 1000.0
        self.Ds = np.array(self.cams['D'])[self.view].astype(np.float32)

        self.Ks[:, :2] = self.Ks[:, :2] * cfg.ratio  # prepare for rendering at different scale
        lower_row = np.array([[[0., 0., 0., 1.]]], dtype=np.float32).repeat(len(self.Ks), axis=0)  # 1, 1, 4 -> N, 1, 4
        self.RT = np.concatenate([self.Rs, self.Ts], axis=-1)  # N, 3, 3 + N, 1, 3
        self.RT = np.concatenate([self.RT, lower_row], axis=-2)  # N, 3, 4 + N, 1, 4

        if hasattr(self, 'BGs'):
            self.BGs = self.BGs[self.view].astype(np.float32)
            BGs = []
            for v in range(len(self.view)):
                D = self.Ds[v]
                K = self.Ks[v]
                BG = self.BGs[v]
                H, W = BG.shape[:2]
                H, W = int(H * cfg.ratio), int(W * cfg.ratio)
                BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_AREA)
                BG = cv2.undistort(BG, K, D)
                BGs.append(BG)
            self.BGs = np.stack(BGs)

    def get_mask(self, i):
        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]  # im is like 1/000000.png
            msk = read_mask_by_img_path(self.data_root, im)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        # ? BUG
        view_index = index % len(self.view)
        latent_index = index

        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)

        msks = self.get_mask(index)

        # reduce the image resolution by ratio
        img_path = os.path.join(self.data_root, self.ims[index][view_index])
        i = int(os.path.basename(img_path)[:-4])
        frame_index = i

        # These are SMPL bw, bounds, vertices
        wpts, ppts, A, joints, Rh, Th, poses, shapes = self.get_lbs_params(i)

        wbounds = get_bounds(wpts)

        img = imageio.imread(img_path)
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        K = self.Ks[view_index]
        RT = self.RT[view_index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = get_rays_within_bounds(H, W, K, R, T, wbounds)

        # load SMPL & pose & human related parameters
        ret = self.get_blend(frame_index)

        # store camera parameters
        meta = {
            "cam_K": K,
            "cam_R": R,
            "cam_T": T,
            "cam_RT": np.concatenate([R, T], axis=1),
            'H': H,
            'W': W,
            'RT': self.RT,
            'Ks': self.Ks,
            'msks': msks,
        }
        ret.update(meta)
        ret.meta.update(meta)

        # store camera background images
        if hasattr(self, 'BGs'):
            BG = self.BGs[view_index]
            meta = {
                "cam_BG": BG,
            }
            ret.update(meta)
            ret.meta.update(meta)

        # store ray data
        meta = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
        }
        ret.update(meta)
        latent_index = index // len(self.view)
        meta = {
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': self.view[view_index],
        }
        ret.update(meta)
        ret.meta.update(meta)
        return ret

    def __len__(self):
        return self.ims.size  # number of elements, regardless of dimensions

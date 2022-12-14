# monocular structure from motion dataset
# a handheld camera, with moving people

import os
import cv2
import h5py
import torch
import imageio
import numpy as np
from tqdm import tqdm
from os.path import join
from functools import partial

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import normalize
from lib.utils.parallel_utils import parallel_execution
from lib.utils.log_utils import log, print_colorful_stacktrace
from lib.utils.data_utils import export_h5, load_h5

from . import base_dataset


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    ray_o = -(R.mT @ T[..., 0])  # 3, 3 @ 3 -> 3,
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(H, dtype=torch.float),
                          torch.arange(W, dtype=torch.float),
                          indexing='ij')  # H, W
    # 0->H, 0->W
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # H, W, 3
    pixel_camera = xy1 @ torch.inverse(K).T  # H, W, 3
    pixel_world = (pixel_camera - T[..., 0]) @ R  # (H, W, 3 - 3) @ 3, 3
    # calculate the ray direction
    ray_d = pixel_world - ray_o  # H, W, 3 - 3
    ray_d = normalize(ray_d)  # H, W, 3
    # ray_o = ray_o[None, None].expand(ray_d.shape)  # H, W, 3
    ray_o = ray_o[None, None]  # 1, 1, 3
    return ray_o, ray_d  # float32


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    # assuming ray_d already normalized
    ray_d[(ray_d < 1e-5) & (ray_d > -1e-10)] = 1e-5
    ray_d[(ray_d > -1e-5) & (ray_d < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[0, 0]) / ray_d  # (1, 3 - 3) / H, W, 3 -> H, W, 3
    tmax = (bounds[1:2] - ray_o[0, 0]) / ray_d
    t1 = torch.minimum(tmin, tmax)
    t2 = torch.maximum(tmin, tmax)
    near = torch.max(t1, dim=-1)[0]
    far = torch.min(t2, dim=-1)[0]
    return near, far  # float32


def randperm_pass(n_runs: int, n_inds: int, n_extra: int, device='cuda', storage='cpu'):
    def impl():
        # one pass is not enough
        coords = []
        for i in tqdm(range(n_runs - 1)):
            inds = torch.randperm(n_inds, device=device)  # generate samples on gpus seems faster?
            inds *= n_runs - 1  # expand indices
            coords.append(inds.to(storage))
        coords.append(torch.randperm(n_extra, device=device).to(storage))
        coords = torch.cat(coords, dim=0)
        return coords
    try:
        return impl()
    except RuntimeError as e:  # OOM
        print_colorful_stacktrace()
        log(f'cuda out of memory, will try to use cpu implementation of torch.randperm, will be slower', 'red')
        log(f'this ususally means someone else is using your designated card, your training will fail but preloaded data will be saved to disk', 'red')
        device = 'cpu'
        return impl()


class Dataset(base_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        # do not change the order of data loading (has dependent structures)
        self.load_meta(data_root, human, ann_file, split)  # load shared metadata across dataset
        self.load_view()  # determine view to use for the current mode
        self.load_ims_data()  # only load the data, no strange processing
        self.load_ims_ravel()  # perform strange processing on data
        self.load_cameras_and_params()  # light

        ray_cache = join(self.data_root, cfg.ray_cache)
        use_data_cache = not cfg.no_data_cache
        cache_file_exists = os.path.exists(ray_cache)

        if not cache_file_exists or not use_data_cache:
            if cfg.distributed:
                log(f'using ddp with prefetch dataset would be extremely memory heavy', 'yellow')
                log(f'advise to first save the prepared data to disk and perform sharded loading', 'yellow')
            self.load_gt()  # will load maskes
            self.load_rays()  # will apply maskes
            self.load_coords()  # will delete maskes

        if not cache_file_exists and use_data_cache:
            log(f'saving generated data to disk: {ray_cache}, this could take forever', 'blue')
            export_h5(dotdict(images=self.images,
                              ray_o=self.ray_o,
                              ray_d=self.ray_d,
                              ), ray_cache)

        if cache_file_exists and use_data_cache:
            if cfg.distributed:
                log(f'will try to shard the data into multiple datasets for different processes, check your memory usage for progress', 'blue')
                log(f'local rank: {cfg.local_rank}, world size: {cfg.world_size}')
                with h5py.File(ray_cache, 'r') as ray_cache:  # copilot!
                    # cleaning up code, this should not exist
                    if 'None' in ray_cache:
                        ray_cache = ray_cache['None']
                    n = ray_cache['images'].shape[0]
                    n_per_rank = n // cfg.world_size
                    n_extra = n % cfg.world_size
                    start = cfg.local_rank * n_per_rank
                    end = start + n_per_rank
                    if cfg.local_rank == cfg.world_size - 1:
                        end += n_extra
                    log(f'[rank #{cfg.local_rank}/{cfg.world_size}] loading images from {start} to {end}')
                    self.images = torch.from_numpy(ray_cache['images'][start:end])
                    log(f'[rank #{cfg.local_rank}/{cfg.world_size}] loading ray_o from {start} to {end}')
                    self.ray_o = torch.from_numpy(ray_cache['ray_o'][start:end])
                    log(f'[rank #{cfg.local_rank}/{cfg.world_size}] loading ray_d from {start} to {end}')
                    self.ray_d = torch.from_numpy(ray_cache['ray_d'][start:end])
            else:
                log(f'loading pre-processed data from disk: {ray_cache}, this could take forever, check your memory usage for progress', 'blue')
                ray_cache = load_h5(ray_cache)
                # cleaning up code, this should not exist
                if 'None' in ray_cache:
                    ray_cache = ray_cache['None']
                self.images = torch.from_numpy(ray_cache.images)
                self.ray_o = torch.from_numpy(ray_cache.ray_o)
                self.ray_d = torch.from_numpy(ray_cache.ray_d)

        if cfg.pin_memory:
            self.images.pin_memory()
            self.ray_o.pin_memory()
            self.ray_d.pin_memory()

        if self.images.shape[1] != cfg.n_rays:
            log(f'loaded ray shape {self.images.shape} does not match user specification, will rearrange')
            self.images = self.images.reshape(-1, cfg.n_rays, *self.images.shape[2:])
            self.ray_o = self.ray_o.reshape(-1, cfg.n_rays, *self.ray_o.shape[2:])
            self.ray_d = self.ray_d.reshape(-1, cfg.n_rays, *self.ray_d.shape[2:])

        log(f'ray shape: {self.images.shape}')

        # now we have the following structures:
        self.images  # S, R, 3 * 1 -> 3
        self.ray_o  # S, R, 3 * 4 -> 12
        self.ray_d  # S, R, 3 * 4 -> 12
        # self.H  # scalar (maybe not exist)
        # self.W  # scalar (maybe not exist)
        self.Ks  # N, 3, 3
        self.Rs  # N, 3, 3
        self.Ts  # N, 3, 1
        self.Ds  # N, 4
        self.RT  # N, 3, 4
        # FIXME: on 1152 images of resolution 1920 * 1080, this will take up 128GBs of memory
        # not practical when the dataset is large enough (or contain large 4k images)
        # especially try when trying to pin memory to non-pageable section

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
        log(f'prefetching images and maskes...', 'blue')
        bundled = parallel_execution(list(range(len(self))), action=self.get_image_and_mask, print_progress=True)

        # split images and maskes
        # https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
        self.images, self.maskes = zip(*bundled)
        del bundled  # save some memory

        log(f'stacking individual maskes...')
        self.maskes = np.stack(self.maskes)  # N, H, W
        self.maskes = torch.from_numpy(self.maskes)  # N, H, W (at least one byte per element)
        # maskes will be reused when loading but will eventually be deleted (only valid pixels are retained)

        # only supports same resolution
        log(f'stacking indivisual images...')
        self.images = np.stack(self.images)  # N, H, W, 3
        self.images = (self.images.clip(0., 1.) * 255).astype(np.uint8)  # 1/4 memory
        self.images = torch.from_numpy(self.images)  # N, H, W, 3
        self.H = self.images.shape[1]
        self.W = self.images.shape[2]

        log(f'sampling valid pixels...')
        self.images = self.images[self.maskes]  # perform sampling with maskes

    def load_cameras_and_params(self):
        log(f'preloading camera parameters...', 'blue')
        # load camera parameters
        self.Ks = np.array(self.cams['K']).astype(np.float32)
        self.Rs = np.array(self.cams['R']).astype(np.float32)
        self.Ts = np.array(self.cams['T']).astype(np.float32) / 1000.0
        self.Ds = np.array(self.cams['D']).astype(np.float32)
        self.Ks[:, :2] = self.Ks[:, :2] * cfg.ratio  # prepare for rendering at different scale
        self.RT = np.concatenate([self.Rs, self.Ts], axis=-1)  # N, 3, 3 + N, 1, 3

        self.Ks = torch.from_numpy(self.Ks)
        self.Rs = torch.from_numpy(self.Rs)
        self.Ts = torch.from_numpy(self.Ts)
        self.Ds = torch.from_numpy(self.Ds)
        self.RT = torch.from_numpy(self.RT)

    def get_rays(self, cam_ind):
        H, W, K, R, T = self.H, self.W, self.Ks[cam_ind], self.Rs[cam_ind], self.Ts[cam_ind]
        ray_o, ray_d = get_rays(H, W, K, R, T)
        return ray_o, ray_d

    def load_rays(self):
        # note that parallel execution is only valid when input is list
        log(f'precomputing rays for sampling...', 'blue')
        bundled = parallel_execution(list(range(len(self))), action=self.get_rays, print_progress=True)
        self.ray_o, self.ray_d = zip(*bundled)
        del bundled
        del self.H, self.W  # avoid errors when accessing this later

        # let's just ignore mab during sampling
        log(f'stacking individual rays...')
        self.ray_o = torch.stack(self.ray_o)  # N, 1, 1, 3
        self.ray_d = torch.stack(self.ray_d)  # N, H, W, 3

        log(f'sampling valid pixels...')
        self.ray_o = self.ray_o.expand(self.ray_d.shape)[self.maskes]  # MARK: MEM
        self.ray_d = self.ray_d[self.maskes]  # MARK: MEM

    def load_coords(self):
        del self.maskes  # save some memory

        n_rays = cfg.n_rays  # we should not change this during training
        n_pixels = self.images.shape[0]
        n_samples = int(np.ceil(n_pixels / n_rays))
        extra_samples = n_samples * n_rays - n_pixels
        log(f'preparing random sampling indices for {n_pixels} pixels...', 'blue')
        log(f'this could take forever, be patient', 'blue')

        quota = 1  # GB
        quota = quota * 2**30  # B
        n_inds = quota // 8  # 64 bit int
        n_runs = int(np.ceil(n_pixels / n_inds))  # every run should fit on a gpu with quota memory
        n_extra = n_inds - (n_runs * n_inds - n_pixels)

        # first pass of random permutation
        log(f'generating permutated data...')
        if cfg.randperm_pass < 2:
            log(f'needs at least two random permutation passes, your value: {cfg.randperm_pass}', 'red')
            cfg.randperm_pass = 2
        passes = [randperm_pass(n_runs, n_inds, n_extra) for i in tqdm(range(cfg.randperm_pass))]
        log(f'generating random samples...')
        coords = passes[0]
        for i in tqdm(range(1, len(passes))):
            coords = coords[passes[i].flip(0)]  # two pass random permutation
        coords = torch.cat([coords, coords[:extra_samples]], dim=0)  # MP + EP, 3
        coords = coords.view(n_samples, n_rays)  # S, R
        log(f'sampling images...')
        self.images = self.images[coords]  # S, R
        log(f'sampling ray origins...')
        self.ray_o = self.ray_o[coords]  # S, R
        log(f'sampling ray directions...')
        self.ray_d = self.ray_d[coords]  # S, R
        del coords  # not to be reused

    def __getitem__(self, index):
        # well, indexing really doesn't matter

        # perform sampling for all rays
        # also just construct batch dimension
        ret = dotdict()
        ret.meta = dotdict()
        # should we design an epoch based sampling strategy?
        sampled = torch.randint(len(self.images), size=())  # sampled coordinates: scalar
        ret.ray_o = self.ray_o[sampled]
        ret.ray_d = self.ray_d[sampled]
        ret.rgb = self.images[sampled].float() / 255.0
        ret.msk = torch.ones_like(ret.ray_o[..., 0], dtype=torch.bool)
        ret.near = torch.ones_like(ret.ray_o[..., 0]) * cfg.clip_near
        ret.far = torch.ones_like(ret.ray_o[..., 0]) * cfg.clip_far
        ret.mask_at_box = torch.ones_like(ret.msk)

        return ret

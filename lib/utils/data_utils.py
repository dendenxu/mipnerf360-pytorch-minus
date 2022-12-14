import os
import cv2
import h5py
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from typing import Union

from lib.utils.base_utils import dotdict
from lib.utils.log_utils import log


def h5_to_dotdict(h5: h5py.File) -> dotdict:
    d = {key: h5_to_dotdict(h5[key]) if isinstance(h5[key], h5py.Group) else h5[key][:] for key in h5.keys()}  # loaded as numpy array
    d = dotdict(d)
    return d


def to_h5py(value, grp: h5py.File, key: str = None, compression: str = 'gzip'):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        grp.create_dataset(str(key), data=value, compression=compression)
    elif isinstance(value, list):
        if key is not None:
            grp = grp.create_group(str(key))
        [to_h5py(k, v, grp) for k, v in enumerate(value)]
    elif isinstance(value, dict):
        if key is not None:
            grp = grp.create_group(str(key))
        [to_h5py(k, v, grp) for k, v in value.items()]
    else:
        raise NotImplementedError('unsupport h5 file type')


def export_h5(batch: dotdict, filename):
    with h5py.File(filename, 'w') as f:
        to_h5py(batch, f)


def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        return h5_to_dotdict(f)


def to_cuda(batch, device="cuda") -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cuda(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cuda(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:  # numpy and others
        batch = torch.tensor(batch, device=device)
    return batch


def to_tensor(batch) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        batch = [to_tensor(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_tensor(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.tensor(batch)
    return batch


def to_cpu(batch, non_blocking=True) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cpu(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cpu(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking)
    else:  # numpy and others
        batch = torch.tensor(batch, device="cpu")
    return batch


def to_numpy(batch, non_blocking=True) -> np.ndarray:
    if isinstance(batch, (tuple, list)):
        batch = [to_numpy(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_numpy(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.array(batch)
    return batch


def remove_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [remove_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (remove_batch(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[0]
    else:
        batch = np.array(batch)[0]
    return batch


def add_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [add_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (add_batch(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[None]
    else:
        batch = np.array(batch)[None]
    return batch


def add_iter_step(batch, iter_step) -> Union[torch.Tensor, np.ndarray]:
    return add_scalar(batch, iter_step, name="iter_step")


def add_scalar(batch, value, name) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        for b in batch:
            add_scalar(b, value, name)

    if isinstance(batch, dict):
        batch[name] = torch.tensor(value)
        batch['meta'][name] = torch.tensor(value)
    return batch


def load_image(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im).astype(np.float32) / 255
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim >= 3 and image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        elif image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255  # BGR to RGB
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        return image


def load_unchanged(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)
        return image


def load_mask(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('L', (int(im.width * ratio), int(im.height * ratio)))
        return (np.asarray(im)[..., None] > 128).astype(np.uint8)
    else:
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[..., None] > 128  # BGR to RGB
        height, width = mask.shape[:2]
        if ratio != 1.0:
            mask = cv2.resize(mask.astype(np.uint8), (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)[..., None]
            # WTF: https://stackoverflow.com/questions/68502581/image-channel-missing-after-resizing-image-with-opencv
        return mask


def save_unchanged(img_path: str, img: np.ndarray, quality=100):
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_image(img_path: str, img: np.ndarray, quality=100):
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_mask(msk_path: str, msk: np.ndarray, quality=100):
    os.makedirs(os.path.dirname(msk_path), exist_ok=True)
    if msk.ndim == 2:
        msk = msk[..., None]
    return cv2.imwrite(msk_path, msk[..., 0] * 255, [cv2.IMWRITE_JPEG_QUALITY, quality])


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def read_mask_by_img_path(data_root: str, img_path: str, erode_dilate_edge: bool = False, mask_dir: str = '') -> np.ndarray:
    def read_mask_file(path):
        msk = load_mask(path).astype(np.uint8)
        if len(msk.shape) == 3:
            msk = msk[..., 0]
        return msk

    if mask_dir:
        msk_path = os.path.join(data_root, img_path.replace('images', mask_dir))
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask_dir)) + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask_dir))[:-4] + '.png'
        if not os.path.exists(msk_path):
            log(f'warning: defined mask path {msk_path} does not exist', 'yellow')

    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask_cihp', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'merged_mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.jpg'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):  # background matte v2
        msk_path = os.path.join(data_root, img_path.replace('images', 'bgmt'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.jpg'

    msk = read_mask_file(msk_path)
    # erode edge inconsistence when evaluating and training
    if erode_dilate_edge:  # eroding edge on matte might erode the actual human
        msk = fill_mask_edge_with(msk)

    return msk


def fill_mask_edge_with(msk, border=5, value=100):
    msk = msk.copy()
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = value
    return msk


def get_rays(H, W, K, R, T, subpixel=False):
    # calculate the camera origin
    ray_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(H, dtype=np.float32),
                       np.arange(W, dtype=np.float32),
                       indexing='ij')
    # 0->H, 0->W
    xy1 = np.stack([j, i, np.ones_like(i)], axis=2)
    if subpixel:
        rand = np.random.rand(H, W, 2) - 0.5
        xy1[:, :, :2] += rand
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    ray_d = pixel_world - ray_o[None, None]
    ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)
    ray_o = np.broadcast_to(ray_o, ray_d.shape)
    return ray_o, ray_d


def get_near_far(bounds, ray_o, ray_d):
    """
    calculate intersections with 3d bounding box
    return: near, far (indexed by mask_at_box (bounding box mask))
    """
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


def get_full_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near / norm_d[..., 0]
    far = far / norm_d[..., 0]
    return near, far, mask_at_box


def full_sample_ray(img, msk, K, R, T, bounds, split='train', subpixel=False):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T, subpixel)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    msk = msk * mask_at_box
    coord = np.argwhere(np.ones_like(mask_at_box))  # every pixel
    ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
    ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)
    near = near[coord[:, 0], coord[:, 1]].astype(np.float32)
    far = far[coord[:, 0], coord[:, 1]].astype(np.float32)
    rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray(img, msk, K, R, T, bounds, nrays, split='train', subpixel=False, body_ratio=0.5, face_ratio=0.0):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T, subpixel)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    msk = msk * mask_at_box
    if "train" in split:
        n_body = int(nrays * body_ratio)
        n_face = int(nrays * face_ratio)
        n_rays = nrays - n_body - n_face
        coord_body = np.argwhere(msk == 1)
        coord_face = np.argwhere(msk == 13)
        coord_rand = np.argwhere(mask_at_box == 1)
        coord_body = coord_body[np.random.randint(len(coord_body), size=[n_body, ])]
        coord_face = coord_face[np.random.randint(len(coord_face), size=[n_face, ])]
        coord_rand = coord_rand[np.random.randint(len(coord_rand), size=[n_rays, ])]
        coord = np.concatenate([coord_body, coord_face, coord_rand], axis=0)
        mask_at_box = mask_at_box[coord[:, 0], coord[:, 1]]  # always True when training
    else:
        coord = np.argwhere(mask_at_box == 1)
        # will not modify mask at box
    ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
    ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)
    near = near[coord[:, 0], coord[:, 1]].astype(np.float32)
    far = far[coord[:, 0], coord[:, 1]].astype(np.float32)
    rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_rays_within_bounds(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_bounds(xyz, box_padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= box_padding
    max_xyz += box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds

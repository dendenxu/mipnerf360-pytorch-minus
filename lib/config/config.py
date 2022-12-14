import os
import sys
import argparse
import warnings
import numpy as np
from . import yacs
from .yacs import CN
from rich import pretty
from rich import traceback

from lib.utils.log_utils import log
pretty.install()
traceback.install()
warnings.filterwarnings("ignore")


cfg = CN()

# a good configuration should like: no extra configuration for unused module
# multi models use different option sets

# cachable dataset will use this option
cfg.fps = 30
cfg.dilation_bias = 0.0025
cfg.dilation_multiplier = 0.5
cfg.randperm_pass = 2
cfg.clip_grad_norm = 40.0
cfg.no_data_cache = False

cfg.clip_near = 0.02
cfg.clip_far = 10.0
cfg.box_far = 5.0
cfg.find_unused_parameters = False

cfg.xyz_res = 10
cfg.view_res = 4
cfg.surf_reg_th = 0.02
cfg.temporal_dim = 256
cfg.overwrite = False
cfg.interpolate_path = False

cfg.print_network = True
cfg.table_row_limit = 5

cfg.profiling = CN()
cfg.profiling.enabled = False
cfg.profiling.clear_previous = True
cfg.profiling.skip_first = 10
cfg.profiling.wait = 5
cfg.profiling.warmup = 5
cfg.profiling.active = 10
cfg.profiling.repeat = 5
cfg.profiling.record_dir = ""
cfg.profiling.record_shapes = True
cfg.profiling.profile_memory = True
cfg.profiling.with_stack = True
cfg.profiling.with_flops = True
cfg.profiling.with_modules = True

cfg.detect_anomaly = False
cfg.sample_vert_cnt = 10
cfg.fixed_lbs_pose = -1

cfg.img_loss_weight = 1.0
cfg.eval_whole_img = True
cfg.dry_run = False
cfg.train_chunk_size = 2048
cfg.render_chunk_size = 2048
cfg.bg_brightness = 0.0

cfg.latent_dim = 128
cfg.collate = True
cfg.load_others = True
cfg.bw_sample_blend_K = 16

cfg.lbs = 'lbs'
cfg.smpl = 'smpl'
cfg.bkgd = 'bkgd/images'
cfg.params = 'params'
cfg.vertices = 'vertices'
cfg.smpl_meta = 'data/smpl-meta'
cfg.mask_dir = 'mask'  # empty mask dir, will iteration through predefined ones

cfg.pin_memory = True
cfg.prefetch_factor = 10
cfg.subpixel_sample = False
cfg.n_bones = 24
cfg.fixed_latent = -1  # whether to fix passed in latent code for feature
cfg.smoothing_term = 10.0  # how much smoothing to use on camera path, 1.0 is a lot
cfg.perform = False  # performing in visualing novel view?
cfg.crop_min_size = 180
cfg.crop_max_size = 200

cfg.perturb = 1.
cfg.n_samples = 64
cfg.n_importance = 128

cfg.mesh_simp_face = -1  # target number of faces to retain if doing mesh simplification
cfg.parent_cfg = 'configs/anisdf_xuzhen36.yaml'

# experiment name
cfg.exp_name = 'anisdf_xuzhen36'

# network
cfg.distributed = False

# data
cfg.skip = []
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.test_view = [0, 1, 2, 3]
cfg.begin_ith_latent = 0
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.num_render_frame = -1  # number of frames to render
cfg.frame_interval = 1
cfg.mask_bkgd = True
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.

# mesh
cfg.mesh_th = 0.5  # threshold of alpha

# task
cfg.task = 'deform'

# gpus
cfg.gpus = list(range(8))
cfg.resume = True  # if load the pretrained network

# epoch
cfg.ep_iter = -1
cfg.save_ep = 200
cfg.eval_ep = 100
cfg.save_latest_ep = 1

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.eps = 1e-8
cfg.train.weight_decay = 0.

cfg.train.lr_table = CN()  # will query the parameter, if found match, use lr in table
cfg.train.eps_table = CN()
cfg.train.weight_decap_table = CN()

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 1
cfg.record_interval = 5

# result
cfg.result_dir = 'data/result'

# training
cfg.tpose_geometry = 'bigpose'
cfg.erode_dilate_edge = True

# evaluation
cfg.replace_light = ''
cfg.test_light = []
cfg.rotate_ratio = 8  # will upscale then roll then downscale
cfg.fix_random = False
cfg.skip_eval = False
cfg.test_novel_pose = False

# visualization
cfg.novel_view_center = []
cfg.novel_view_z_off = -1

cfg.vis_median_depth = False
cfg.vis_distortion = False
cfg.vis_acc_map = False
cfg.vis_depth_map = False
cfg.vis_pose_sequence = False
cfg.vis_novel_view = False


def default_cfg():
    return cfg


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # NOTE: assign the gpus, this will ignore cfg.gpus if you've assigned CUDA_VISIBLE_DEVICES already
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    # Get rid of ugly TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)

    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']

    if cfg.profiling.enabled:
        cfg.train.epoch = 1
        cfg.ep_iter = cfg.profiling.skip_first + cfg.profiling.repeat * (cfg.profiling.wait + cfg.profiling.warmup + cfg.profiling.active)
        cfg.profiling.record_dir = cfg.record_dir


def update_cfg(cfg: CN, args):
    cfg_file = yacs.load_cfg(open(args.cfg_file, 'r'))
    cfg.merge_strain(cfg_file)
    cfg.merge_from_list(args.opts)  # load commandline config before merging

    if cfg.vis_pose_sequence:
        cfg.merge_from_other_cfg(cfg.pose_seq_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    cfg.merge_from_list(args.opts)  # load commandline config after merging
    parse_cfg(cfg, args)  # load some environment variables and defaults
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument('-c', "--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('-t', "--type", type=str, default="")
parser.add_argument('-r', '--local_rank', type=int, default=0)
parser.add_argument('-l', '--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
parser.add_argument('--test', action='store_true', dest='test', default=False)

args = None

if sys.argv[0].endswith('run.py') or sys.argv[0].endswith('train.py'):
    args = parser.parse_args()
    cfg = default_cfg()
    if len(args.type) > 0:
        cfg.task = "run"

    cfg = update_cfg(cfg, args)
    log(cfg.exp_name, 'magenta')

"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py -c configs/phdeform/monosdf_my_313.yaml distributed True exp_name monosdf_ddp load_normal False load_semantics False

torchrun --nproc_per_node=2 train.py -c configs/phdeform/phdeform_xuzhen36.yaml distributed True
"""

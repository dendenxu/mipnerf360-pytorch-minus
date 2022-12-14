# MipNeRF360-PyTorch (Minus the MipMap (Anti-Aliasing) Part)

This repo contains a **PyTorch implementation of [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://arxiv.org/abs/2111.12077) minus the mip-map part**.

The author was frequently amazed by the awesome results of the _mipnerf360_ paper (and code repository), but was lacking in motivation to learn a whole new framework (jax) when trying to use some of _mipnerf360_'s amazing techniques. So the author decided to reproduce the code / paper in PyTorch.

The main focus of this repo is to reproduce the fact that mipnerf360 is able to capture extremely detailed appearance and geometry of a large unbounded (or just large) scene with an extremely large 1024 x 8 NeRF network, while maintaining a somewhat reasonable training (2 days on two 3090s) / inference (0.05 fps on one 3090 for 512x512 images) speed. So most of the effort goes into making sure that the proposal network works fine (by making a 1024 x 8 network behave like an 256 x 8 one (16 times more parameters)). Then we ensure that the distortion loss and disparity parameteration (plus the contraction algorithm) works too since they further distribute the network representational power.

The reason why the mip-map part is omitted from the implementation is that it does not contribute too much to the final results (or the **stability of the training process**, which turned out to be the main prick when reproducing) as long as we do not try to render images with extremely low resolution.

This is extracted from one of the author's on-going research project. As a result, you may notice quite a lot of leftover code or configurations that have nothing to do with _mipnerf360_ or _mipnerf360-pytorch-minus_. The code may not well-organized and may contain (potentially quite a bit of) bugs. Please feel free to open issues if you have any questions (as long as there're people trying to use this project :).

This repo exists and should only exist for research purposes. The right of the algorithms reproduced in this repo belongs to Google.

## Installation

We advise using [`miniconda`](https://conda.io/miniconda.html) to manage your python environment, especially so when giant frameworks like PyTorch is involved.

Optionally, we recommend using [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) as a fast extension of conda for fast downloading of packages.

If you believe your environment has the correct PyTorch (and related) packages setup, please just run the example provided below and install any packages that are reported missing.

Alternatively, you can:

```shell
conda install mamba -n base -c conda-forge -y
mamba env create -n mipnerf360 python>=3.10 # mamba env create with -f is weird
mamba env update # inside the project directory

pip install -r requirements.txt
```

Note that after installing the packages, you may still encounter import errors due to the author's poor management of the project (I just extracted these from a much larger code base). Please install them manually.

## Data Example

We provide an example dataset capturing Room 412 of State Lab of CAD&CG, Zhejiang University.

I captured the data using my iPhone, so I guess I could distribute it freely?

## Usage Example

**Visualize novel views of the trained model**

Switch on vis_novel_view will switch datasets and visualizer used in the module.
This takes around 1.5 hours on a 3090, use ratio 0.25 for fast previewing.

```shell
python run.py -t visualize -c configs/mipnerf360_iphone_412.yaml vis_novel_view True ratio 0.5
```

**Visualize novel views depth of the trained model**

Switch on vis_novel_view will switch datasets and visualizer used in the module.
This takes around 1.5 hours on a 3090, use ratio 0.25 for fast previewing.
Median depth seems to be sharper than mean depth (default).

```shell
python run.py -t visualize -c configs/mipnerf360_iphone_412.yaml vis_novel_view True vis_depth_map True vis_median_depth True ratio 0.5
```

**Perform training on the iphone_412 dataset**

Note that the default training batch size for one gpu is 8192 rays per iteration instead of the 16384 mentioned in the paper.
However we expect the user to use 2x ddp (mentioned in the next example command) since only 8192 rays (power of 2) fits on a 3090.

Also note that the first time the training script is run, it will create a large (~45 GB) h5 file to store the preprocessed data for fast training.
In the provided example, we have over 1000 images in FHD (compared to the typical 50 or 100 images used in NeRF related projects).
When creating the large h5 cache (called ray_cache in the code), the program takes up ~128 GB of memory.
If your machine does not have enough memory, you can try to reduce the size of images used in the cache (by setting the ratio parameter to a smaller value)
Or try to reduce the number of images used in the cache (by setting the frame_interval parameter to a larger value).

Alternatively you can also download the preprocessed ray_cache file prepared by the author to avoid the hustle.
(As long as you have a LARGE bandwidth internet... which I don't so I would recommend lowering resolution or image count first)

```shell
python train.py -c configs/mipnerf360_iphone_412.yaml
```

**Perform training on 2 gpus**

This will train the network with a total effective batch size of 16384 rays per iteration.

Typical speed for training on 3090 is 0.75s/iter.
It's strongly recommend you create the ray_cache file first by running the previous command.
(When the training starts, you can interrupt the single-gpu training and call this command).
(I would print out various messages to let you know where you are during training/rendering).

```shell
torchrun --nproc_per_node=2 train.py -c configs/mipnerf360_iphone_412.yaml distributed True
```

## Implementation Detail

To be completed.

## Code Structure

The code structure of this repo is borrowed from [Neural Body](https://github.com/zju3dv/neuralbody).

Datasets exist for loading data from disk or memory and process them on the fly during traing or rendering.

Renderers store logic for volume rendering and ray marching.

Networks define and capture all optimizable network parameters along with the core forward functions intrinsically attached to the network.

Visualizers handle data visualization (mainly writing images to disk).

Trainers handle loss functions and training loops. The outer most trainer handles the training loop and the inner most trainer handles the loss function for a single iteration. The outer most script [`train.py`](train.py) handles epoch loops and invokation of any other modules.

Note that the program entry point is actually [`lib/config/config.py`](lib/config/config.py). This is why we need to `from lib.config import cfg` in any files that use the global configuration first.

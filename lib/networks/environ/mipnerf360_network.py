import torch
from torch import nn
from torch.nn import functional as F

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import MLP, GradModule, raw2alpha

from ..embedder import PositionalEncoding
# here we only implment the constraction and regularization of the mipnerf360 papers
# ignoring the original paper's antialiasing efforts since it may not really matter


def contract(x: torch.Tensor, radius: float = 1.0):
    x_norm = x.norm(dim=-1, keepdim=True)
    msk = x_norm <= radius
    return x * msk + ~msk * (1 + radius - radius / x_norm) * x / x_norm


class Network(GradModule):
    def __init__(self):
        super(Network, self).__init__()

        # shared componenets
        self.embedder_xyz = PositionalEncoding(multires=12)  # 10 * 2 * 3 + 3 = 63
        self.embedder_dir = PositionalEncoding(multires=4)  # 4 * 2 * 3 + 3 = 27
        self.geometry_actvn = nn.Softplus()
        self.appearance_actvn = nn.Sigmoid()

        # fine network components
        self.fine_geometry = MLP(self.embedder_xyz.get_dim(3), 1024, 8, 256 + 1, actvn=nn.ReLU(), init=nn.Identity())  # 63 -> 256, 8 -> 257
        self.fine_appearance = MLP(self.embedder_dir.get_dim(3) + 256, 128, 1, 3, actvn=nn.ReLU(), init=nn.Identity())  # 256 + 27 -> 256, 8 -> 3
        self.coarse_geometry = MLP(self.embedder_xyz.get_dim(3), 256, 4, 1, actvn=nn.ReLU(), init=nn.Identity())  # 63 -> 128, 8 -> 1
        self.forward = self.forward_fine

    def forward_fine(self, x: torch.Tensor, v: torch.Tensor, d: torch.Tensor, batch: dotdict):
        x = contract(x)
        ebd_xyz = self.embedder_xyz(x)
        ebd_dir = self.embedder_dir(v)
        feat = self.fine_geometry(ebd_xyz)
        feat, density = feat[..., :-1], feat[..., -1:]
        occ = raw2alpha(density, d[..., None], act_fn=self.geometry_actvn)
        rgb = self.fine_appearance(torch.cat([feat, ebd_dir], dim=-1))
        rgb = self.appearance_actvn(rgb)
        return dotdict(raw=torch.cat([rgb, occ], dim=-1))

    def forward_coarse(self, x: torch.Tensor, v: torch.Tensor, d: torch.Tensor, batch: dotdict):
        x = contract(x)
        ebd_xyz = self.embedder_xyz(x)
        density = self.coarse_geometry(ebd_xyz)
        occ = raw2alpha(density, d[..., None], act_fn=self.geometry_actvn)
        return dotdict(raw=occ)

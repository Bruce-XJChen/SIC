import contextlib
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from thop import profile
from PIL import Image

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
    "freeze_module",
    "adjust_status",
]


def get_model_info(model: nn.Module, tsize: Sequence[int], preprocess) -> str:
    # img = torch.zeros((1, 3, tsize[0], tsize[1]), device=next(model.parameters()).device)
    img = preprocess(Image.open("figures/clip.png")).unsqueeze(0)
    img = img.cuda()
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info
import torch
import torch.nn as nn
from collections import namedtuple

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

def quantization_range(bitwidth):
    q_max = (1 << (bitwidth - 1)) - 1
    q_min = -(1 << (bitwidth - 1))
    return q_min, q_max

def linear_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):


class LinearQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.S, self.Z = LinearQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        for name, param in model.named_parameters():
            # print(name, param)
            # if param.dim() > 1:
            codebook[name] = linear_quantize(param, bitwidth=bitwidth)
        return codebook
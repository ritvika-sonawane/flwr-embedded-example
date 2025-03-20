import torch
import torch.nn as nn
from collections import namedtuple

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

from fast_pytorch_kmeans import KMeans

def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    if codebook is None:
        # n_clusters = 1 << bitwidth
        n_clusters = 2**bitwidth
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    quantized_tensor = codebook.centroids[codebook.labels].view_as(fp32_tensor)
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        for name, param in model.named_parameters():
            # print(name, param)
            # if param.dim() > 1:
            codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook
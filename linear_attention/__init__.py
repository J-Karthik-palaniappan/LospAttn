import torch.nn as nn
import torch.nn.functional as F

from .symmetricNystrom import symmetric_nystrom, symmetric_nystrom_torch
from .nystromformer import nystromformer, nystromformer_torch
from .wildcat import WildCat
from .performer import FastAttention
from .triton_codes.fused_error import approx_attention_error_triton


class StatelessAttention(nn.Module):
    def __init__(self, fn, **args):
        super().__init__()
        self.fn = fn
        self.args = args

    def forward(self, q, k, v):
        return self.fn(q, k, v, **self.args)


def original(q, k, v, **args):
    return F.scaled_dot_product_attention(q, k, v, **args)


ATTNS = {
    "original": lambda **args: StatelessAttention(original, **args),
    "nystromformer": lambda **args: StatelessAttention(nystromformer, **args),
    "nystromformer_torch": lambda **args: StatelessAttention(nystromformer_torch, **args),
    "symmetric_nystrom": lambda **args: StatelessAttention(symmetric_nystrom, **args),
    "symmetric_nystrom_torch": lambda **args: StatelessAttention(symmetric_nystrom_torch, **args),
    "performer": lambda **args: FastAttention(**args),
    "wildcat": lambda **args: WildCat(**args),
}

__all__ = ["ATTNS", "StatelessAttention", "approx_attention_error_triton"]

import math
from collections import OrderedDict
from functools import partial
from types import FunctionType
from typing import Callable, Optional, List, Union, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

from EResNet import ENNLinear, ENNConv2d                                        


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

class ENNMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.dropout_att = nn.Dropout(dropout)

        self.q_proj   = ENNLinear(embed_dim, embed_dim, bias=True)
        self.k_proj   = ENNLinear(embed_dim, embed_dim, bias=True)
        self.v_proj   = ENNLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = ENNLinear(embed_dim, embed_dim, bias=True)

    def _split(self, x: Tensor) -> Tensor:     
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:     
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout_att(torch.softmax(attn, dim=-1))

        out = attn @ v
        out = self._merge(out)
        return self.out_proj(out)


class ENNMLP(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = ENNLinear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.fc2 = ENNLinear(mlp_dim, in_dim)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.do1(self.act(self.fc1(x)))
        x = self.do2(self.fc2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.ln1  = norm_layer(hidden_dim)
        self.attn = ENNMultiheadAttention(hidden_dim, num_heads, attention_dropout)
        self.do1  = nn.Dropout(dropout)

        self.ln2 = norm_layer(hidden_dim)
        self.mlp = ENNMLP(hidden_dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.do1(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x



class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.drop      = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
            for _ in range(num_layers)
        ])
        self.ln_final  = norm_layer(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(x + self.pos_embed)
        for blk in self.layers:
            x = blk(x)
        return self.ln_final(x)



class EViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        orth_weight: float = 5e-4,            
    ):
        super().__init__()
        _log_api_usage_once(self)

        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.hidden_dim  = hidden_dim
        self.orth_weight = orth_weight


        self.conv_proj = ENNConv2d(in_channels=3,
                                   out_channels=hidden_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   bias=True)

        seq_len = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_len += 1

        self.encoder = Encoder(seq_len, num_layers, num_heads,
                               hidden_dim, mlp_dim,
                               dropout, attention_dropout, norm_layer)

        if representation_size is None:
            self.pre_logits = None
            head_in = hidden_dim
        else:
            self.pre_logits = ENNLinear(hidden_dim, representation_size)
            self.act        = nn.Tanh()
            head_in = representation_size
        self.head = ENNLinear(head_in, num_classes)

    def _patchify(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)                       
        x = x.flatten(2).transpose(1, 2)            
        return x


    def forward(self, imgs: Tensor) -> Tensor:
        B, C, H, W = imgs.shape
        assert H == self.image_size and W == self.image_size, \
            f"Expected {self.image_size}×{self.image_size} input"

        tokens = self._patchify(imgs)
        cls    = self.cls_token.expand(B, -1, -1)
        x      = torch.cat((cls, tokens), dim=1)

        x = self.encoder(x)
        x = x[:, 0]                                 # CLS token

        if self.pre_logits is not None:
            x = self.act(self.pre_logits(x))
        return self.head(x)


    def orth_loss(self) -> Tensor:
        loss = torch.zeros(1, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, (ENNLinear, ENNConv2d)):
                if hasattr(m, "orthonormalize"):
                    loss = loss + m.orthonormalize()
                elif hasattr(m, "orth_loss"):
                    loss = loss + m.orth_loss()
        return loss


def test():
    model = EViT(image_size=224, patch_size=16,
                 num_layers=12, num_heads=12,
                 hidden_dim=768, mlp_dim=3072,
                 num_classes=1000)
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print("logits:", logits.shape)
    print("orth loss:", model.orth_loss())

# test()

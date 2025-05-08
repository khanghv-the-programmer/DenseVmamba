import torch.nn as nn
from typing import Callable, Any
import torch
from functools import partial
from models.components.mlp import Mlp
from models.components.ss2d import SS2D
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
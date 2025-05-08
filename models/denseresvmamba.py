from functools import partial
import torch.nn as nn
import torch



from models.components.dense_stage import DenseStage
from models.components.permute import Permute
from models.components.vss_block import VSSBlock

class DenseResMamba(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        growths=[32, 32, 64, 64],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_cls=VSSBlock,
        drop_path_rate=0.1,
        **block_kwargs,
    ):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Permute(0, 2, 3, 1),
            norm_layer(dims[0]),
            Permute(0, 3, 1, 2),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_channels = dims[0]
        idx = 0
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = DenseStage(
                in_channels=cur_channels,
                growth_rate=growths[i],
                num_blocks=depths[i],
                drop_path_rates=dpr[idx:idx + depths[i]],
                block_cls=block_cls,
                norm_layer=norm_layer,
                **block_kwargs,
            )
            idx += depths[i]
            self.stages.append(stage)
            cur_channels = stage.out_channels

        self.norm = norm_layer(cur_channels)
        self.head = nn.Linear(cur_channels, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        x = nn.AdaptiveAvgPool2d(1)(x).flatten(1)
        return self.head(x)

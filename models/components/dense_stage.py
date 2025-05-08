import torch.nn as nn
import torch

class DenseStage(nn.Module):
    def __init__(self, in_channels, growth_rate, num_blocks, drop_path_rates, block_cls, norm_layer, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.aligns = nn.ModuleList()
        cur_channels = in_channels
        for i in range(num_blocks):
            block = block_cls(hidden_dim=cur_channels, drop_path=drop_path_rates[i], norm_layer=norm_layer, **block_kwargs)
            self.blocks.append(block)
            self.aligns.append(nn.Conv2d(cur_channels, growth_rate, kernel_size=1, bias=False))
            cur_channels += growth_rate
        self.out_channels = cur_channels

    def forward(self, x):
        features = [x]
        for block, align in zip(self.blocks, self.aligns):
            out = block(torch.cat(features, dim=1).permute(0, 2, 3, 1))  # B,C,H,W -> B,H,W,C
            out = out.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
            out = align(out)
            features.append(out)
        return torch.cat(features, dim=1)
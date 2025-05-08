

from components.vssm import VSSM
import torch.nn as nn
import torch


class Backbone_VSSM(VSSM):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], 
                 d_state=16, ssm_ratio=2.0, attn_drop_rate=0., 
                 drop_rate=0., drop_path_rate=0.1, mlp_ratio=4.0,
                 patch_norm=True, norm_layer=nn.LayerNorm,
                 downsample_version: str = "v2",
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3), pretrained=None, 
                 **kwargs,
        ):
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, 
                         depths=depths, dims=dims, 
                         d_state=d_state, ssm_ratio=ssm_ratio, attn_drop_rate=attn_drop_rate, 
                         drop_rate=drop_rate, drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio,
                         patch_norm=patch_norm, norm_layer=norm_layer,
                         downsample_version=downsample_version,
                         use_checkpoint=use_checkpoint,
                         **kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x) if not isinstance(l.downsample, nn.Identity) else x
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs
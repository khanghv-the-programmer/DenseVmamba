
import torch
from timm.models.layers import DropPath

import selective_scan_cuda_core as selective_scan_cuda # type: ignore


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)
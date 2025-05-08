import torch
import selective_scan_cuda_core as selective_scan_cuda # type: ignore


    
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
        
        
        
        

class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    






class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None






def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops
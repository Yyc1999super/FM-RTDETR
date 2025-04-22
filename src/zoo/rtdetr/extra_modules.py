from xml.sax.xmlreader import InputSource
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from functools import partial
from einops import rearrange, repeat
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .conv import *
from .conv import autopad
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

from mamba_ssm import Mamba2

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):

        b, c , h , w = feats.size()
        ar = w / h

        return torch.cat([
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)

class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x
    

class FusionCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.cut_c = Cut(in_channels=dim, out_channels=dim)
        
        self.q = Conv(dim, dim, k=1, s=1)
        self.kv = Conv(dim, dim * 2, k=3, s=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv(dim, dim, k=1, s=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B, C, H1, W1 = x.shape
        N1 = H1*W1
        q = self.q(x).reshape(B, C, -1).permute(0, 2, 1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(self.cut_c(y)).reshape(B, 2*C, -1).permute(0, 2, 1)
        kv1 = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv1[0], kv1[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C).permute(0, 2, 1).reshape(B, C, H1, W1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1).to(x.dtype)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = input.permute((0, 2, 3, 1))
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x.permute((0, 3, 1, 2))
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        return x

class FusionMiMBlock(nn.Module):
    """ MiM-ISTD Block
    """
    def __init__(self, inner_dim, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=2, drop=0.):
        super().__init__()
        # Inner
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = VSSAttention(inner_dim)

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj = Conv(inner_dim, inner_dim, k=1, s=1)
        self.localconv = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.tanh_spatial = nn.Tanh()

    def forward(self, inputs):
        x, y = inputs
        xx = torch.cat([x, y], dim=1)
        localattn = self.localconv(xx)
        yy = self.drop_path(self.inner_attn(self.inner_norm1(localattn)))
        yyy = yy + yy * self.tanh_spatial(self.proj(self.proj_norm1(yy))) # B, N, C
        outer_tokens = localattn + yyy

        return outer_tokens
  

class SEAM(nn.Module):
    def __init__(self, c1, c2, n, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
class ParallelAtrousConv(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)
    
    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))
    

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(DSConv(mid_C * i, mid_C, 3))

        self.fuse = DSConv(in_C + mid_C, out_C, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K= DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)
        self.INF_K= DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)       
        self.Second_reduce = DSConv(in_C, out_C, 3)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        Q = self.Q(torch.cat([x,y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width*height)
        RGB_K = RGB_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width*height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height,width)
        RGB_refine = self.gamma1*RGB_refine+y
        
        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width*height)
        INF_K = INF_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width*height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height,width) 
        INF_refine = self.gamma2 * INF_refine + x
        
        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))        
        return out 

class PSFM(nn.Module):
    def __init__(self, Channel):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)           
        self.obj_fuse = GEFM(Channel * 2, Channel)
        
    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)        
        Inf_sum = self.Infobj(depth)        
        out = self.obj_fuse(rgb_sum,Inf_sum)
        return out
    
class ParallelAtrousConv1(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)
    
    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class ParallelAtrousConv2(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        hidden_dim = inc//2
        self.branch1 = ParallelAtrousConv1(hidden_dim)
        self.branch2 = nn.Sequential(Conv(hidden_dim, hidden_dim, k=3, g=hidden_dim), Conv(hidden_dim, hidden_dim, k=1))
        self.conv1 = Conv(inc, inc, k=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x11 = self.branch1(x1)
        x22 = x2 + self.sigmoid(self.branch2(x2)) * x2
        result = self.conv1(torch.cat([x11, x22], dim=1))
        return result
    
class SDFM(nn.Module):
    '''
    superficial detail fusion module
    '''

    def __init__(self, channels=64, r=4):
        super(SDFM, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(2 * channels, 2 * inter_channels),
            Conv(2 * inter_channels, 2 * channels, act=nn.Sigmoid()),
        )

        self.channel_agg = Conv(2 * channels, channels)

        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1, act=False),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重 
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo

class SDFMAtrousFusion(nn.Module):

    def __init__(self, inc):
        super(SDFMAtrousFusion, self).__init__()
        self.branch1 = SDFM(inc//2)
        self.branch2 = iAFF(inc//2)
        self.parallelAtrous = ParallelAtrousConv1(inc)
        self.conv1 = Conv(inc, inc//2, k=1)
        self.conv2 = Conv(inc, inc//2, k=1)
        self.conv3 = Conv(inc, inc, k=1)
        self.conv4 = Conv(2*inc, inc, k=1)

        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = torch.chunk(y, 2, dim=1)
        xxyy1 = self.branch1((x1, y1))
        xxyy2 = self.branch1((x2, y2))
        xxxyyy = torch.cat([xxyy1, xxyy2], dim=1)
        xxxyyy1 = self.conv1(xxxyyy)
        xxxyyy2 = self.conv2(xxxyyy)
        ww1 = self.sigmoid1(xxxyyy1)
        ww2 = self.sigmoid2(xxxyyy2)
        res1 = ww1*x1 + (1-ww1)*y1
        res2 = ww2*x2 + (1-ww2)*y2
        res3 = self.conv3(torch.cat([res1, res2], dim=1))

        xyinit = torch.cat([x, y], dim=1)
        xymid = self.parallelAtrous(self.conv4(xyinit))

        ww3 = self.sigmoid3(res3)
        result = ww3*xymid + (1-ww3)*res3
        return result

class ParallelAtrousConv3(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 3, inc, k=1)
        self.conv5 = Conv(inc, inc, k=1)
    
    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv5(x)], dim=1))

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        return x + self.m(x)

# class ParallelAtrousConv4(nn.Module):
#     def __init__(self, inc, ratio=[1, 3, 5]) -> None:
#         super().__init__()
#         cc = inc//2
#         self.rlength = len(ratio)
#         self.paraAtrousModulesList = nn.ModuleList()
#         self.conv1x1List = nn.ModuleList()
#         self.batchNormList = nn.ModuleList()
#         self.ratioConvList = nn.ModuleList()
#         for idx in range(self.rlength):
#             self.ratioConvList.add_module('{0}'.format(idx+1), nn.Conv2d(cc, cc, 3, 1, autopad(3, None, ratio[idx]), ratio[idx], cc))

#         for idx in range(self.rlength):
#             self.conv1x1List.append(Conv(cc, cc, k=1, s=1))
#             self.batchNormList.append(nn.BatchNorm2d(cc))
        
#         self.conv1 = Conv(inc, cc, k=1)
#         self.convfuse = Conv(self.rlength*cc, inc, k=1)

#     def forward(self, x):
#         xx = self.conv1(x)
#         # for idx in range(self.rlength):
#         inp = xx
#         ratioResultList = []
#         for index, module in enumerate(self.ratioConvList):  # 按执行顺序遍历网络各层操作
#             inp = module(inp)
#             ratioResultList.append(self.batchNormList[index](xx + self.conv1x1List[index](inp)))

#         result = self.convfuse(torch.cat(ratioResultList, dim=1))
#         return result

class ParallelDWConvBlock(nn.Module):
    def __init__(self, inc, dilation_rate=2, ss=[1, 3, 5]) -> None:
        super().__init__()
        cc= int(inc/2)
        self.dwconv1 = nn.Sequential(Conv(inc, inc, k=ss[0], g=inc), Conv(inc, inc, k=1))
        self.dwconv2 = nn.Sequential(Conv(inc, inc, k=ss[0], g=inc), Conv(inc, inc, k=1))
        self.dwconv3 = nn.Sequential(Conv(inc, inc, k=ss[0], g=inc), Conv(inc, inc, k=1))
        self.conv1 = Conv(3*inc, inc, k=1)

        self.paralellAtrousConv = ParallelAtrousConv4(inc)
        # self.F_sur = nn.Conv2d(cc, cc, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=cc)
        # self.fglo = FGlo(inc, reduction=16)
        self.convfuse1 = nn.Conv2d(inc*2, inc, 1, 1, autopad(1, None, 1), 1)
        self.convfuse2 = nn.Conv2d(inc*2, inc, 1, 1, autopad(1, None, 1), 1)
        # self.sigmoid1 = nn.Sigmoid()
        # self.sigmoid2 = nn.Sigmoid()
        self.inc = inc
        self.w1 = nn.Parameter(torch.ones(2*self.inc, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, inputs):
        x, y = inputs
        N, C, H, W = x.shape
        xy = self.convfuse1(torch.cat([x, y], dim=1))
        xy1 = self.conv1(torch.cat([self.dwconv1(xy), self.dwconv2(xy), self.dwconv3(xy)], dim=1))
        xy2 = self.paralellAtrousConv(xy)

        w1 = self.w1[:(2*self.inc)] # 加了这一行可以确保能够剪枝
        weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        xxx = (weight[:C] * xy1.view(N, H, W, C)).view(N, C, H, W)
        yyy = (weight[C:] * xy2.view(N, H, W, C)).view(N, C, H, W)
        xxx = xxx + xy1
        yyy = yyy + xy2

        out = self.convfuse2(torch.cat([xxx, yyy], dim=1))
        return out

    # def forward(self, x):
    #     x1, x2 = torch.chunk(x, 2, dim=1)
    #     ww1 = self.fglo(self.conv1(torch.cat([self.dwconv1(x1), self.dwconv2(x1), self.dwconv3(x1)], dim=1)))
    #     xx = ww1 * x + x
        
    #     yy = self.paralellAtrousConv(x2)
    #     out = self.convfuse(torch.cat([xx, yy], dim=1))
    #     return out

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True, is_enhance=False):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=n) # surrounding context
        
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.is_enhance = is_enhance
        self.F_glo= FGlo(nOut, reduction)

        self.paraAtrous = ParallelAtrousConv4(n)
        self.conv1x1_2 = Conv(3*n, nOut, 1, 1)

        self.conv3x3_1 = Conv(nOut, nOut, 1, 1, g=nOut)
        self.conv1x1_3 = Conv(nOut, nOut, 1, 1)

    def forward(self, input):
        output = self.conv1x1(input)
        output_pAtrous = self.paraAtrous(output)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = self.conv1x1_2(torch.cat([loc, sur, output_pAtrous], 1))

        joi_feat = self.bn_act(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature

        if self.is_enhance:
            input = self.conv3x3_1(self.conv1x1_3(input))

        # if residual version
        if self.add:
            output  = input + output
        
        return output

class MMParallelBlock(nn.Module):
    """ MiM-ISTD Block
    """
    def __init__(self, inner_dim, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=2, drop=0., residual=False):
        super().__init__()
        self.residual = residual
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = VSSAttention(inner_dim)

        self.paralellAtrousConv = ParallelAtrousConv3(inner_dim)

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj_norm2 = norm_layer(inner_dim)
        self.proj = Conv(inner_dim, inner_dim, k=1, s=1)

        self.conv1 = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.F_glo1 = FGlo(2*inner_dim, reduction=8)
        # self.F_glo2= FGlo(inner_dim, reduction=8)
        # self.conv2 = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.sigmoid = nn.Sigmoid()
        # self.tanh_spatial = nn.Tanh()

    def forward(self, inputs):
        x, y = inputs
        input1 = torch.cat([x, y], dim=1)
        ww1 = self.F_glo1(input1)
        intput2 = input1 + ww1 * input1
        xx, yy = torch.chunk(intput2, 2, dim =1)
        xx1 = self.conv1(intput2)
        yy = self.drop_path(self.inner_attn(self.inner_norm1(xx1)))
        xxx = self.proj(self.proj_norm1(yy))
        yyy = self.paralellAtrousConv(self.proj_norm2(xx1))
        
        ww2 = self.sigmoid(xxx * yyy) 
        result = xx * ww2 + yy * (1 - ww2)

        if self.residual:
            result = result + xx1
        return result

class ParallelAtrousConv4(nn.Module):
    def __init__(self, inc, ratio=[1, 3, 5, 7]) -> None:
        super().__init__()
        cc = inc//2
        self.rlength = len(ratio)
        self.paraAtrousModulesList = nn.ModuleList()
        self.conv1x1List = nn.ModuleList()
        self.batchNormList = nn.ModuleList()
        self.ratioConvList = nn.ModuleList()
        for idx in range(self.rlength):
            self.ratioConvList.add_module('{0}'.format(idx+1), nn.Conv2d(cc, cc, 3, 1, autopad(3, None, ratio[idx]), ratio[idx], 1))

        for idx in range(self.rlength):
            self.conv1x1List.append(Conv(cc, cc, k=1, s=1))
            self.batchNormList.append(nn.BatchNorm2d(cc))
        
        self.conv1 = Conv(inc, cc, k=1)
        self.convfuse = Conv(self.rlength*cc, inc, k=1)

    def forward(self, x):
        xx = self.conv1(x)
        # for idx in range(self.rlength):
        inp = xx
        ratioResultList = []
        for index, module in enumerate(self.ratioConvList):  # 按执行顺序遍历网络各层操作
            inp = module(inp)
            ratioResultList.append(self.batchNormList[index](xx + self.conv1x1List[index](inp)))

        result = self.convfuse(torch.cat(ratioResultList, dim=1))
        return result

class ParallelAtrousConv5(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv0 = Conv(inc, inc//2, k=1)
        self.conv1 = Conv(inc//2, inc//2, k=3, d=ratio[0])
        self.conv2 = Conv(inc//2, inc//2, k=3, d=ratio[1])
        self.conv3 = Conv(inc//2, inc//2, k=3, d=ratio[2])
        self.conv4 = Conv(inc//2 * 3, inc, k=1)
        self.conv5 = Conv(inc*2, inc, k=1)
    
    def forward(self, x):
        xx = self.conv0(x)
        return self.conv5(torch.cat([self.conv4(torch.cat([self.conv1(xx), self.conv2(xx), self.conv3(xx)], dim=1)), x]))

class ParallelAtrousConv6(nn.Module):
    def __init__(self, inc, ratio=[1, 3, 5, 7]) -> None:
        super().__init__()
        cc = inc//2
        self.rlength = len(ratio)
        self.conv1x1List = nn.ModuleList()
        self.concatFeatureFusionList = nn.ModuleList()

        for idx in range(self.rlength):
            self.conv1x1List.append(Conv(cc, cc, k=1, s=1))
            self.concatFeatureFusionList.append(ConcatFeatureFusion(cc, cc, branch_1=nn.Sequential(Conv(cc, cc, 1)), dilate=ratio[idx]))
        
        self.conv1 = Conv(inc, cc, k=1)
        self.convfuse = Conv(self.rlength*cc, inc, k=1)

    def forward(self, x):
        xx = self.conv1(x)
        ratioResultList = []
        for index in range(self.rlength):  # 按执行顺序遍历网络各层操作
            ratioResultList.append(self.conv1x1List[index](self.concatFeatureFusionList[index](xx)))

        result = self.convfuse(torch.cat(ratioResultList, dim=1))
        return result

class FusionMiMBlock1(nn.Module):

    def __init__(self, inner_dim, drop_path=0., ratiolen=3, act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=2, drop=0., residual=False):
        super().__init__()
        self.ratioList = [1, 3, 5, 7]
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = VSSAttention(inner_dim)
        self.parallelAtrous = ParallelAtrousConv6(inner_dim, ratio=self.ratioList[:ratiolen])

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj = Conv(inner_dim, inner_dim, k=1, s=1)
        self.localconv = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.tanh_spatial = nn.Tanh()

    def forward(self, inputs):
        x0, y0 = inputs
        y = self.parallelAtrous(y0)
        xx = torch.cat([x0, y], dim=1)
        localattn = self.localconv(xx)
        yy = self.drop_path(self.inner_attn(self.inner_norm1(localattn)))
        yyy = yy + yy * self.tanh_spatial(self.proj(self.proj_norm1(yy)))
        outer_token = localattn + yyy
        return outer_token

class FusionMiMBlock2(nn.Module):
    """ MiM-ISTD Block
    """
    def __init__(self, inner_dim, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=2, drop=0.):
        super().__init__()
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = VSSAttention(inner_dim)

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj = Conv(inner_dim, inner_dim, k=1, s=1)
        self.localconv = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.tanh_spatial = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, y = inputs
        xx = torch.cat([x, y], dim=1)
        localattn = self.localconv(xx)
        yy = self.drop_path(self.inner_attn(self.inner_norm1(localattn)))
        yyy = yy + yy * self.tanh_spatial(self.proj(self.proj_norm1(yy)))
        outer_token = localattn + yyy
        return outer_token
    

class FusionMiMBlock3(nn.Module):

    def __init__(self, inner_dim, drop_path=0., ratiolen=3, act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=2, drop=0., residual=False):
        super().__init__()
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = VSSAttention(inner_dim)
        self.parallelAtrous = ParallelAtrousConv5(inner_dim)

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj = Conv(inner_dim, inner_dim, k=1, s=1)
        self.localconv = Conv(2*inner_dim, inner_dim, k=1, s=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.tanh_spatial = nn.Tanh()

    def forward(self, inputs):
        x0, y0 = inputs
        y = self.parallelAtrous(y0)
        xx = torch.cat([x0, y], dim=1)
        localattn = self.localconv(xx)
        yy = self.drop_path(self.inner_attn(self.inner_norm1(localattn)))
        yyy = yy + yy * self.tanh_spatial(self.proj(self.proj_norm1(yy)))
        outer_token = localattn + yyy
        return outer_token

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

# class AFFAtrousLayer(nn.Module):
#     def __init__(self, inplanes, r=2):
#         super(AFFAtrousLayer, self).__init__()
#         mid_dim = inplanes//2
#         self.mid_dim = mid_dim
#         planes = inplanes // r
#         self.local_1 = nn.Sequential(
#             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inplanes),
#         )
#         self.local_2 = nn.Sequential(
#             nn.Conv2d(inplanes, planes, kernel_size=3, stride=3, padding=0),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inplanes),
#             nn.Conv2d(planes, planes, kernel_size=3, stride=3, padding=0),
#             nn.BatchNorm2d(inplanes),
#         )

#         # self.global_att = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm2d(planes),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm2d(inplanes),
#         # )
#         self.global_att = VSSAttention(inplanes)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         x, residual = inputs
#         xa = x + residual
#         xl = self.local_1(xa)
#         xg = self.global_att(xa)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)

#         xo = 2 * x * wei + 2 * residual * (1 - wei)
#         # affresult = self.conv0()

#         # xx = self.conv1(torch.cat(inputs, dim=1))
#         # xxxx1 = self.paralellAtrousConv(xx)
#         # xxxx2 = self.conv2(xx)

#         # xxyy = torch.cat([xxxx1, xxxx2], dim=1)
#         # xxyy = self.conv3(xxyy)
        
#         # result = self.conv5(self.conv4(torch.cat([affresult, xxyy], dim=1)))
#         return xo


class AFFAtrousLayer(nn.Module):
    def __init__(self, hidden_dim, r=2):
        super(AFFAtrousLayer, self).__init__()
        mid_dim = hidden_dim//2

        self.aff = AFF(hidden_dim)
        self.paralellAtrousConv = ParallelAtrousConv(mid_dim)
        self.conv0 = Conv(hidden_dim, mid_dim, k=1, s=1)
        self.conv1 = Conv(2*hidden_dim, mid_dim, k=1, s=1)
        self.conv2 = Conv(mid_dim, mid_dim, k=1, s=1)
        self.conv3 = Conv(hidden_dim, mid_dim, k=1, s=1)
        self.conv4 = Conv(hidden_dim, hidden_dim, k=3, s=1)
        self.conv5 = Conv(hidden_dim, hidden_dim, k=1, s=1)
        

    def forward(self, inputs):
        affresult = self.conv0(self.aff(inputs))
        xx = self.conv1(torch.cat(inputs, dim=1))
        xx1 = self.paralellAtrousConv(xx)
        xx2 = self.conv2(xx)
        xxyy = torch.cat([xx1, xx2], dim=1)
        xxyy = self.conv3(xxyy)

        result = self.conv5(self.conv4(torch.cat([affresult, xxyy], dim=1)))
        return result

class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(3, 1, 3, 1), e=0.5) -> None:
    # def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv4 = Conv(hidc * 3, inc[1], k=1)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)
        
        x = self.conv4(x + feature)
        return x

class AFF(nn.Module):
    def __init__(self, inplanes, r=4):
        super(AFF, self).__init__()
        planes = inplanes // r
        self.local_att = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inplanes),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inplanes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, residual = inputs
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, residual = inputs
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class ConcatFeatureFusion(nn.Module):
    def __init__(self, in_ch, out_ch, branch_1=nn.Identity(), dilate=1):
        super(ConcatFeatureFusion, self).__init__()
        self.branch_1 = branch_1

        self.dconv1 = Conv(in_ch, in_ch, 3, d=dilate) if dilate else nn.Identity()
        self.fuseconv = Conv(in_ch*2, out_ch, 1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.dconv1(x)
        result = self.fuseconv(torch.concat([x1, x2], dim=1))
        return result

class TwoBranchFeatureFusion(nn.Module):
    def __init__(self, inplanes, r=2):
        super(TwoBranchFeatureFusion, self).__init__()
        planes = inplanes // r
        self.branch_1 = nn.Sequential(
            Conv(inplanes, planes, 1),
            Conv(planes, planes, 3),
            Conv(planes, inplanes, 1, act=False)
        )

        self.branch_2 = ConcatFeatureFusion(inplanes, inplanes, branch_1=nn.Sequential(Conv(inplanes, planes, 1),
                                            ParallelAtrousConv6(planes, ratio=[1, 3, 5]),
                                            Conv(planes, inplanes, 1, act=False)), dilate=None)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x1, x2 = inputs
        xx = x1 + x2
        xl = self.branch_1(xx)
        xg = self.branch_2(xx)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x1 * wei + x2 * (1 - wei)
        return xo


class EnhanceP2FeatureBlock(nn.Module):
    def __init__(self, inc, outc):
        super(EnhanceP2FeatureBlock, self).__init__()
        cc = inc // 2
        self.branch_1 = ClipFeatureEnhance(inc, inc, branch_1=nn.Sequential(Conv(cc, cc, 1),
                          DWConv(cc, cc, 3),
                          Conv(cc, cc, 1)))
        self.branch_2 = ClipFeatureEnhance(inc, inc, branch_1=nn.Sequential(DWConv(cc, cc, 3),
                          Conv(cc, cc, 1),
                          DWConv(cc, cc, 3)))
        self.branch_3 = ConcatFeatureFusion(inc, inc, branch_1=nn.Sequential(DWConv(inc, inc, 3),
                          Conv(inc, inc, 1),
                          Conv(inc, inc, 3, d=3)), dilate=1),
        self.branch_4 = ConcatFeatureFusion(inc, inc, branch_1=nn.Sequential(DWConv(inc, inc, 3),
                          Conv(inc, inc, 1),
                          Conv(inc, inc, 3, d=5)), dilate=1)
        
        self.inc = inc  
        self.convfuse = Conv(4*inc, outc, k=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        
        result = self.convfuse(channel_shuffle(torch.cat([x1, x2, x3, x4], dim=1), 4*self.inc))
        return result
    
class EnchanceLocalFeatureBlock(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        cc = in_ch // 2
        self.branch_1 = ConcatFeatureFusion(in_ch, in_ch, branch_1=nn.Sequential(
            Conv(in_ch, in_ch, k=(1,3), p=(0,1)),
            Conv(in_ch, in_ch, k=(3,1), p=(1,0)),
            Conv(in_ch, in_ch, 3, d=3)), dilate=1)

        self.branch_2 = ConcatFeatureFusion(in_ch, in_ch, branch_1=nn.Sequential(
            Conv(in_ch, in_ch, k=(3,1), p=(1,0)),
            Conv(in_ch, in_ch, k=(1,3), p=(0,1)),
            Conv(in_ch, in_ch, 3, d=5)), dilate=1)

        self.branch_3 = ConcatFeatureFusion(in_ch, in_ch, branch_1=ClipFeatureEnhance(in_ch, in_ch, branch_1=nn.Sequential(
             DWConv(cc, cc, 3),
             Conv(cc, cc, 1),
             Conv(cc, cc, 3),
        )), dilate=1)

        self.convfuse = Conv(in_ch * 4, out_ch, 1, 1)
        
    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        xx = self.convfuse(torch.cat([x1, x2, x3, x], dim=1))
        return xx
    
class ClipFeatureEnhance(nn.Module):
    def __init__(self, in_ch, out_ch, branch_1=nn.Identity(), short=True):
        super(ClipFeatureEnhance, self).__init__()
        self.cc = in_ch // 2
        self.branch_1 = branch_1
        self.fuseconv = Conv(self.cc*2, out_ch, 1)
        self.short = short
        self.shortConv = Conv(self.cc, self.cc, 1) if self.short else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x11 = self.branch_1(x1)
        x22 = self.shortConv(x2)
        result = self.fuseconv(torch.concat([x11, x22], dim=1))
        return result
    
class EnhanceP2FeatureFusionBlock(nn.Module):
    def __init__(self, inc):
        super(EnhanceP2FeatureFusionBlock, self).__init__()
        cc = inc // 2
        self.branch_1 = FusionMiMBlock2(cc)
        
        self.cc = cc  
        self.inc = inc
        self.w1 = nn.Parameter(torch.ones(2*cc, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        self.convfuse = Conv(inc, inc, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        x1, x2 = xs
        x11, x12 = torch.chunk(x1, 2, dim=1)
        x21, x22 = torch.chunk(x2, 2, dim=1)
        x111 = self.branch_1((x11, x21))

        N, C, H, W = x12.shape
        w1 = self.w1[:(2*self.cc)] # 加了这一行可以确保能够剪枝
        weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        xxx = (weight[:C] * x12.view(N, H, W, C)).view(N, C, H, W)
        yyy = (weight[C:] * x22.view(N, H, W, C)).view(N, C, H, W)
        xyxy = xxx + yyy
        
        result = self.convfuse(channel_shuffle(torch.cat([x111, xyxy], dim=1), 2))
        # result = self.convfuse(torch.cat([x111, xyxy], dim=1))
        return result
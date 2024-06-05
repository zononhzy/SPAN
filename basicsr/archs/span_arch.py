# model: SPAN
# Lightweight Image Super-Resolution with Sliding Proxy Attention Network

import math
import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F
from itertools import repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialGate(nn.Module):
    """Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormProxy(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x):
        # Split
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x2 = self.conv(self.norm(x2))
        return x1 * x2

class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Conv2d(hidden_features // 2, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature with shape of (b, c, h, w).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def window_overlap_partition(x, window_size, overlap_size, padding_size=0, num_head=6):
    """
    Args:
        x: (b, c, h, w)
        window_size (int): window size
        overlap_size (int): overlap size

    Returns:
        windows: (b, num_window, num_head*3, window_size*window_size, c_head)
    """
    _, c, _, _ = x.shape
    x = F.unfold(x, kernel_size=(window_size, window_size), stride=window_size - overlap_size, padding=padding_size)
    x = x.reshape(x.shape[0], num_head*3, c // (num_head * 3), window_size * window_size, -1).permute(0, 4, 1, 3, 2).contiguous()
    return x


def window_overlap_reverse(windows, h, w, batch_img, window_size, overlap_size, padding_size=0):
    """
    Args:
        windows: (b, num_window, num_head, c, window_size*window_size) or (b, num_window, num_head, window_size*window_size)
        window_size (int): Window size
        overlap_size (int): overlap size

    Returns:
        x: (b, c, h, w)
    """
    windows = windows.permute(0, 2, 4, 3, 1).contiguous().reshape(batch_img, -1, windows.shape[1])
    windows = F.fold(
        input=windows,
        output_size=(h, w),
        kernel_size=(window_size, window_size),
        stride=window_size - overlap_size,
        padding=padding_size,
    )
    return windows


class ESA_Blcok(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA_Blcok, self).__init__()
        self.n_feats = n_feats
        self.esa_channels = esa_channels
        self.conv1 = conv(n_feats, esa_channels, kernel_size=1)
        self.conv_f = conv(esa_channels, esa_channels, kernel_size=1)
        self.conv2 = conv(esa_channels, esa_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(esa_channels, esa_channels, kernel_size=3, padding=1)
        self.conv4 = conv(esa_channels, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class HFTAttention(nn.Module):
    """High frequency texture attention based on convolution.

    Args:
        dim (int): Number of input channels.
        hidden_dim (int): Number of hidden channels.
        scale (int): Scale of downsample and upsample.
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.pool_scale = [2, 4, 8]
        self.conv_after_upsample = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): (b, c, h, w)
        Returns: high texture enhanced feature, (b, c1, h, w)
        """
        # high frequency texture attention
        b, c, h, w = x.shape
        # x_after = self.conv_before(x)
        attn = len(self.pool_scale) * x
        for i in range(len(self.pool_scale)):
            x_pool = F.adaptive_avg_pool2d(x, output_size=(h // self.pool_scale[i], w // self.pool_scale[i]))
            attn -= F.interpolate(x_pool, size=(h, w), mode="nearest")
        attn = self.conv_after_upsample(attn) * x

        return attn


class ELFEB(nn.Module):
    def __init__(self, in_channels, drop=0.0):
        super(ELFEB, self).__init__()
        self.in_channels = in_channels

        self.weight = nn.Parameter(torch.zeros(in_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = in_channels // self.n_div
        self.weight[0 * g : 1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g : 2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g : 3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g : 4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g :, 0, 1, 1] = 1.0  ## identity

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels),
            nn.GELU(),
            nn.Dropout(drop),
        )

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shortcut = x
        x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.in_channels)
        x = self.conv_block(x)
        x = (
            self.conv2(F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.in_channels))
            + shortcut
        )
        return x

class SPAB(nn.Module):
    r"""Sliding Proxy Attention Block

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        overlap_size (int): Overlap size for SPAB.
        padding_size (int): Padding size for SPAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        overlap_size=0,
        padding_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNormProxy,
    ):
        super(SPAB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.window_size = window_size
        self.overlap_size = overlap_size
        self.padding_size = padding_size
        self.mlp_ratio = mlp_ratio

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.overlap_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.overlap_size < self.window_size, "overlap_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(drop)
        self.hft_attention = HFTAttention(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.conv_ffn = SGFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        b_, c, h, w = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.qkv(x)
        # partition windows
        x_windows = window_overlap_partition(
            x, self.window_size, self.overlap_size, self.padding_size, self.num_heads
        )  # b, n, nH*3, hc, Wh*Ww

        q, k, v = torch.chunk(x_windows, 3, dim=2)  # b, n, nH, Wh*Ww, hc
        q = q * self.temperature[None, None]  # b, n, nH, Wh*Ww, hc
        attn = q @ k.transpose(-2, -1)  # b, n, nH, Wh*Ww, Wh*Ww

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size**2, self.window_size**2, -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias[None, None]  # b, n, nH, Wh*Ww, Wh*Ww

        max_attn = torch.max(attn, dim=-1, keepdim=True)[0]  # b, n, nH, Wh*Ww, 1 #for numerical stability
        exp_attn = torch.exp(attn - max_attn)  # b, n, nH, Wh*Ww, Wh*Ww
        sum_attn = torch.sum(exp_attn, dim=-1, keepdim=True)  # b, n, nH, Wh*Ww, 1
        sum_attn = window_overlap_reverse(
            sum_attn,
            h,
            w,
            batch_img=b_,
            window_size=self.window_size,
            overlap_size=self.overlap_size,
            padding_size=self.padding_size,
        )  # b_, nH, h, w

        exp_attn = exp_attn @ v  # b n nH Wh*Ww c

        exp_attn = window_overlap_reverse(
            exp_attn,
            h,
            w,
            batch_img=b_,
            window_size=self.window_size,
            overlap_size=self.overlap_size,
            padding_size=self.padding_size,
        ).reshape(b_, self.num_heads, self.head_dim, h, w)

        attn = exp_attn / sum_attn.unsqueeze(2)  # b_, nH, d_H, h, w
        attn = attn.reshape(b_, c, h, w)  # b_, c, h, w

        x = self.proj_drop(self.proj_out(attn))  # b_, c, h, w
        x = self.hft_attention(x)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.conv_ffn(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, overlap_size={self.overlap_size}, padding_size={self.padding_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """A basic layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        overlap_size (int): Local window overlap size.
        padding_size (int): Image padding size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        overlap_size,
        padding_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNormProxy,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [ELFEB(dim)]
            + [
                SPAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    padding_size=padding_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x) + x
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SPTB(nn.Module):
    """Sliding Proxy Transformer Block

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        overlap_size (int): Local window overlap size.
        padding_size (int): Image padding size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        overlap_size,
        padding_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNormProxy,
        downsample=None,
        use_checkpoint=False,
    ):
        super(SPTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            overlap_size=overlap_size,
            padding_size=padding_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        self.esa = ESA_Blcok(
            esa_channels=max(16, dim // 4),
            n_feats=dim,
        )

    def forward(self, x):
        return self.esa(self.residual_group(x))


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


@ARCH_REGISTRY.register()
class SPAN(nn.Module):
    r"""SPAN
        A PyTorch implement of : `Lightweight Image Super-Resolution with Sliding Proxy Attention Network`.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        overlap_size (int): Overlap size for OPSA. Default: 0
        padding_size (int): Padding size for OPSA. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        overlap_size=0,
        padding_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNormProxy,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        **kwargs,
    ):
        super(SPAN, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.padding_size = padding_size
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        input_resolution = to_2tuple(img_size)
        self.input_resolution = input_resolution

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Permuted Self Attention Group  (PSA_Group)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SPTB(
                dim=embed_dim,
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                overlap_size=overlap_size,
                padding_size=padding_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (input_resolution[0], input_resolution[1]))
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (
            self.window_size
            - self.overlap_size
            - (h + self.padding_size * 2 - self.overlap_size) % (self.window_size - self.overlap_size)
        ) % (self.window_size - self.overlap_size)
        mod_pad_w = (
            self.window_size
            - self.overlap_size
            - (w + self.padding_size * 2 - self.overlap_size) % (self.window_size - self.overlap_size)
        ) % (self.window_size - self.overlap_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]


if __name__ == "__main__":
    with torch.no_grad():
        upscale = 2
        overlap_size = 3
        window_size = 12
        padding_size = 0
        height = 1280 // upscale
        width = 720 // upscale
        mod_pad_h = (
            window_size - overlap_size - (height + padding_size * 2 - overlap_size) % (window_size - overlap_size)
        ) % (window_size - overlap_size)
        mod_pad_w = (
            window_size - overlap_size - (width + padding_size * 2 - overlap_size) % (window_size - overlap_size)
        ) % (window_size - overlap_size)

        height = height + mod_pad_h
        width = width + mod_pad_w
        model = SPAN(
            upscale=upscale,
            in_chans=3,
            img_size=(height, width),
            window_size=window_size,
            overlap_size=overlap_size,
            padding_size=padding_size,
            img_range=1.0,
            depths=[2, 2, 2, 2, 2, 2],
            embed_dim=60,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        ).to("cuda:1")
        model.eval()
        input = torch.randn(1, 3, height, width).to("cuda:1")
        flops = FlopCountAnalysis(model, (input,))
        print(flop_count_table(flops, show_param_shapes=False))

        torch.set_float32_matmul_precision('high')
        script_model = torch.compile(model)

        import numpy as np
        import tqdm

        repititions = 10

        print("warming up ...\n")
        with torch.no_grad():
            for i in range(repititions):
                script_model(input)

        torch.cuda.synchronize()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repititions, 1))

        print("Measuring ...")

        with torch.no_grad():
            for rep in tqdm.tqdm(range(repititions)):
                starter.record()
                _ = script_model(input)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)

        avg = np.sum(timings) / repititions
        print(f"Average time: {avg} ms")
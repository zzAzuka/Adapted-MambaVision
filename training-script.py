import math

import torch
import torch.nn as nn
from timm.models._builder import resolve_pretrained_cfg
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
import os
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from PIL import Image
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp  # , #PatchEmbed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Dataset Function
class SegmentationDataset(Dataset):
    """Dataset loader for IDD-20k segmentation"""

    def __init__(
        self, img_dir, label_dir, transform=None, img_size=512, num_classes=26
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes

        # Get all image files
        self.img_files = sorted(
            list(self.img_dir.glob("*.jpg"))
            + list(self.img_dir.glob("*.jpeg"))
            + list(self.img_dir.glob("*.png"))
        )

        print(f"Found {len(self.img_files)} images in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")

        # Load corresponding label
        label_name = img_path.stem + ".png"
        label_path = self.label_dir / label_name

        if not label_path.exists():
            label_path = self.label_dir / (img_path.stem + img_path.suffix)

        label = Image.open(label_path)

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        label = label.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        # Convert to tensors
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = torch.from_numpy(np.array(label, dtype=np.int64))

        # Ensure label values are in valid range
        label = torch.clamp(label, 0, self.num_classes - 1)

        return img, label


# Helper Fucntions
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


# LayerNorm2d
class LayerNorm2d(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


# DownSample
class Downsample(nn.Module):
    """
    Down-sampling block: This is added after every stage (from 1 to 4), where we try to create
    a spatial pyramid by using a 3x3 convolution with stride 2. [We half the resolution]
    Args:
        dim: feature size dimension.
        norm_layer: normalization layer.
        keep_dim: bool argument for maintaining the resolution.
    """

    def __init__(
        self,
        dim,
        keep_dim=False,
    ):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


# Stem(PatchEmbed)
class PatchEmbed(nn.Module):
    """
    Patch embedding block: This is disguised under two convolution layers [with kernel 3x3 and
    stride=2] which is used to make overlapping patches by reducing the resolution by 2.
    Args:
        in_chans: number of input channels.
        in_dim: input dimension.
        dim: feature size dimension.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


# Conv Block
class ConvBlock(nn.Module):
    """
    ConvBlock: This is the basic building block of the model. It is a 2-layer convolution block with
    a skip connection. The dimensions are preserved and it is used for local feature extraction.
    Args:
        dim: feature size dimension.
        drop_path: drop path rate. This is used to control the dropout rate of the skip connection.
        layer_scale: layer scaling coefficient. This is used to control the scaling of the output of the block.
        kernel_size: kernel size. This is used to control the size of the convolution kernel.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):

        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


# MambaVisionMixer
class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


# Attention Module
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Block (LayerNorm + Mixer + Drop Path + LayerNorm + MLP)
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        layer_scale=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )
        self.gamma_2 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# MambaVisionLayer (Alternate between Stages 1,2 and Stages 3,4)
class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        conv=False,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        transformer_blocks=[],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=dim,
                        counter=i,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        layer_scale=layer_scale,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


# MambaVision (Full BackBone Pipeline)
class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(
        self,
        dim=128,
        in_dim=64,
        depths=(3, 3, 10, 5),
        window_size=(8, 8, 14, 7),
        mlp_ratio=4.0,
        num_heads=(2, 4, 8, 16),
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        **kwargs,
    ):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i]))
                if depths[i] % 2 != 0
                else list(range(depths[i] // 2, depths[i])),
            )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = (
            nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Decoder
class FPNDecoder(nn.Module):
    """
    Simple Feature Pyramid Network decoder.
    Takes 4 skip feature maps from the backbone stages,
    progressively upsamples and fuses them, then produces
    a per-pixel class map.

    Channel sizes coming out of MambaVision (default dim=128):
        stage 0 skip: 128   (H/4)
        stage 1 skip: 256   (H/8)
        stage 2 skip: 512   (H/16)
        stage 3 skip: 1024  (H/32)
    """

    def __init__(self, stage_channels, num_classes, decoder_dim=256):
        super().__init__()

        # Lateral 1×1 convs: project each stage to decoder_dim
        self.laterals = nn.ModuleList(
            [nn.Conv2d(ch, decoder_dim, kernel_size=1) for ch in stage_channels]
        )

        # Output 3×3 convs to smooth after upsampling
        self.outputs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in stage_channels
            ]
        )

        # Final segmentation head (upsamples from H/4 → H)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, features):
        """
        features: list of 4 tensors, coarsest (stage3) first
                  i.e. [skip0, skip1, skip2, skip3]
        We process top-down: start from skip3 (coarsest),
        upsample and add skip2, ..., down to skip0.
        """
        # Lateral projections
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # Top-down pathway: start at coarsest (index 3)
        x = laterals[3]
        x = self.outputs[3](x)

        for i in [2, 1, 0]:
            x = F.interpolate(
                x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + laterals[i]
            x = self.outputs[i](x)

        # x is now at H/4 resolution; upsample 4× to full resolution
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.seg_head(x)
        return x  # (B, num_classes, H, W)


# Final Combination of Backbone + Decoder
class MambaVisionSeg(nn.Module):
    """
    Full segmentation model:
      MambaVision backbone (feature extractor) + FPNDecoder (segmentation head)
    """

    def __init__(self, backbone: MambaVision, num_classes: int):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.levels = backbone.levels
        # Drop the classifier pieces — we don't need them
        # backbone.norm / avgpool / head are unused

        dim = backbone.patch_embed.conv_down[-2].num_features  # = dim arg (128)
        # Channel count doubles every stage: dim, 2*dim, 4*dim, 8*dim
        stage_channels = [dim * (2**i) for i in range(4)]  # [128,256,512,1024]

        self.decoder = FPNDecoder(stage_channels, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, dim, H/4, W/4)

        skips = []
        for level in self.levels:
            x, skip = level(x)  # level returns (downsampled, skip)
            skips.append(skip)
        # skips = [s0(H/4), s1(H/8), s2(H/16), s3(H/32)]

        logits = self.decoder(skips)  # (B, num_classes, H, W)
        return logits


class SegmentationLoss(nn.Module):
    """
    Cross-Entropy + Dice loss combo — standard for segmentation.
    ignore_index: mask pixel value to skip (e.g. 255 = unlabelled in PASCAL VOC)
    """

    def __init__(self, num_classes, ignore_index=255, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def dice_loss(self, preds, targets):
        # preds: (B, C, H, W) raw logits
        probs = F.softmax(preds, dim=1)
        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        targets_one_hot = F.one_hot(targets_clean, self.num_classes)  # (B,H,W,C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()

        intersection = (probs * targets_one_hot * valid).sum(dim=(2, 3))
        union = ((probs + targets_one_hot) * valid).sum(dim=(2, 3))
        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean()

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice_loss(preds, targets)
        return ce_loss + self.dice_weight * dice_loss


def compute_iou(preds, targets, num_classes, ignore_index=255):
    """Returns mean IoU over valid classes."""
    preds = preds.argmax(dim=1)  # (B, H, W)
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        valid_mask = targets != ignore_index

        intersection = (pred_mask & target_mask & valid_mask).sum().item()
        union = ((pred_mask | target_mask) & valid_mask).sum().item()

        if union == 0:
            continue  # class not present, skip
        ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    nan_count = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        autocast_ctx = (
            torch.cuda.amp.autocast()
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_ctx:
            logits = model(images)  # (B, C, H, W)
            loss = criterion(logits, masks)

        # NaN/Inf loss checking (Detects and skips batches with NaN or Inf loss)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping")
            nan_count += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}]  Loss: {loss.item():.4f}")

    if nan_count > 0:
        print(f"  WARNING: Skipped {nan_count} batches due to NaN/Inf loss")

    return (
        total_loss / (len(loader) - nan_count)
        if nan_count < len(loader)
        else float("inf")
    )


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    nan_count = 0

    autocast_ctx = (
        torch.cuda.amp.autocast() if use_amp else torch.autocast("cpu", enabled=False)
    )

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, masks)

        # NaN/Inf loss checking
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item()
        total_iou += compute_iou(logits, masks, num_classes)

    if nan_count > 0:
        print(f"  WARNING: Skipped {nan_count} validation batches due to NaN/Inf loss")

    valid_batches = len(loader) - nan_count
    return total_loss / valid_batches if valid_batches > 0 else float(
        "inf"
    ), total_iou / valid_batches if valid_batches > 0 else 0.0


def main():

    # ── Dataset Paths ───────────────────────
    DATA_ROOT = "/home/shch/mamba_3neurons/idd20k_final"

    TRAIN_IMG_DIR = f"{DATA_ROOT}/train/images"
    TRAIN_LABEL_DIR = f"{DATA_ROOT}/train/labels"

    VALID_IMG_DIR = f"{DATA_ROOT}/valid/images"
    VALID_LABEL_DIR = f"{DATA_ROOT}/valid/labels"

    OUTPUT_DIR = "/home/shch/mamba_3neurons/outputs"

    # ── Config ──────────────────────────────
    NUM_CLASSES = 26
    IMG_SIZE = 512
    BATCH_SIZE = 16  # Increased for A40 GPU (46GB VRAM)
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    IGNORE_INDEX = 255

    # Early Stopping Config
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    MIN_DELTA = 0.001  # Minimum change to qualify as improvement

    # LR Warmup Config
    WARMUP_EPOCHS = 5  # Number of warmup epochs

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRETRAINED = None
    USE_AMP = DEVICE.type == "cuda"  # Only use mixed precision on CUDA

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ── Create Output Directory ────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # ── Datasets ────────────────────────────
    train_dataset = SegmentationDataset(
        TRAIN_IMG_DIR, TRAIN_LABEL_DIR, img_size=IMG_SIZE
    )

    val_dataset = SegmentationDataset(VALID_IMG_DIR, VALID_LABEL_DIR, img_size=IMG_SIZE)

    # ── DataLoaders ─────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Increased for faster data loading
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Model ───────────────────────────────
    backbone = MambaVision(
        dim=128,
        in_dim=64,
        depths=(3, 3, 10, 5),
        window_size=(8, 8, 14, 7),
        num_heads=(2, 4, 8, 16),
        num_classes=0,
    )

    if PRETRAINED:
        backbone._load_state_dict(PRETRAINED)
        print(f"Loaded pretrained weights from {PRETRAINED}")

    model = MambaVisionSeg(backbone, num_classes=NUM_CLASSES).to(DEVICE)

    # ── Loss, Optimizer, Scheduler ──────────
    criterion = SegmentationLoss(NUM_CLASSES, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # LR Warmup + Cosine Annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # ── Training Loop ───────────────────────
    best_iou = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]  LR: {current_lr:.6f}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler, USE_AMP
        )

        val_loss, val_iou = validate(
            model, val_loader, criterion, DEVICE, NUM_CLASSES, USE_AMP
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | mIoU: {val_iou:.4f}")

        # Early stopping check
        if val_iou > best_iou + MIN_DELTA:
            best_iou = val_iou
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                },
                f"{OUTPUT_DIR}/best_model.pth",
            )

            print(f"✓ Saved new best model (mIoU={best_iou:.4f})")
        else:
            patience_counter += 1
            print(
                f"No improvement for {patience_counter} epoch(s) (best mIoU={best_iou:.4f})"
            )

            # Check early stopping condition
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best mIoU: {best_iou:.4f}")
                break


if __name__ == "__main__":
    main()

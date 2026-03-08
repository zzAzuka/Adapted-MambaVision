import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path

import os
from functools import partial
from typing import Callable

from torch.utils import checkpoint

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Downsample(nn.Module):
    """
    Down-sampling block: This is added after every stage (from 1 to 4), where we try to create
    a spatial pyramid by using a 3x3 convolution with stride 2. [We half the resolution]
    Args:
        dim: feature size dimension.
        norm_layer: normalization layer.
        keep_dim: bool argument for maintaining the resolution.
    """

    def __init__(self, dim, keep_dim=False,):
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
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

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
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
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
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
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


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
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
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
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
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
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
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
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


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
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
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
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
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        return {'rpb'}

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

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)

try:
    from mambavision_model import MambaVision          # put your model file alongside this script
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARN] mambavision_model not found – using a ResNet-50 stub for smoke-testing.")


# ─────────────────────────────────────────────
# 2.  SEGMENTATION HEAD  (UPerNet-style simple version)
# ─────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """Lightweight FPN-style decoder that sits on top of the MambaVision backbone."""

    def __init__(self, in_channels_list, num_classes, embed_dim=256):
        super().__init__()
        # Lateral 1×1 projections
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1) for c in in_channels_list
        ])
        # Output conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features, target_size):
        # features: list of (B, C, H, W) tensors, coarse→fine order
        x = self.laterals[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            x = nn.functional.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False)
            x = x + self.laterals[i](features[i])
        x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.output_conv(x)


# ─────────────────────────────────────────────
# 3.  FULL SEGMENTATION MODEL
# ─────────────────────────────────────────────
class MambaVisionSeg(nn.Module):
    def __init__(self, num_classes, backbone_cfg: dict):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.backbone = MambaVision(**backbone_cfg)
            # Remove classification head
            self.backbone.head = nn.Identity()
            self.backbone.avgpool = nn.Identity()
            # Feature channel sizes per stage (dim * 2^i)
            dim = backbone_cfg.get("dim", 128)
            depths = backbone_cfg.get("depths", (3, 3, 10, 5))
            in_ch = [int(dim * 2 ** i) for i in range(len(depths))]
        else:
            import torchvision.models as tvm
            base = tvm.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            in_ch = [256, 512, 1024, 2048]

        self.seg_head = SegmentationHead(in_ch, num_classes)
        self._in_ch = in_ch

    def forward(self, x):
        H, W = x.shape[-2:]
        if MAMBA_AVAILABLE:
            feats = []
            xi = self.backbone.patch_embed(x)
            for level in self.backbone.levels:
                xi, skip = level(xi)
                feats.append(skip)
        else:
            # Stub: just get one feature map and replicate
            feat = self.backbone(x)
            feats = [feat, feat, feat, feat]

        return self.seg_head(feats, (H, W))


# ─────────────────────────────────────────────
# 4.  DATASETS
# ─────────────────────────────────────────────
class SegmentationDataset(Dataset):
    """
    Expects:
        root/images/*.jpg (or .png)
        root/masks/*.png   (grayscale, values = class indices 0..N-1)
    """
    def __init__(self, root, img_size=512, augment=False):
        self.img_dir  = Path(root) / "images"
        self.mask_dir = Path(root) / "masks"
        self.img_size = img_size
        self.augment  = augment
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.imgs = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.imgs)

    def _sync_transform(self, img, mask):
        # Resize
        img  = TF.resize(img,  [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.img_size, self.img_size))
            img  = TF.crop(img,  i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        return img, mask

    def __getitem__(self, idx):
        img_path  = self.imgs[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)          # keep as-is (grayscale / palette)

        img, mask = self._sync_transform(img, mask)

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                                  std =[0.229, 0.224, 0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


def get_classification_loaders(data_dir, img_size=224, batch_size=32, workers=4):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256/224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


def get_segmentation_loaders(data_dir, img_size=512, batch_size=8, workers=4):
    train_ds = SegmentationDataset(os.path.join(data_dir, "train"), img_size, augment=True)
    val_ds   = SegmentationDataset(os.path.join(data_dir, "val"),   img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 5.  METRICS
# ─────────────────────────────────────────────
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_miou(preds, labels, num_classes, ignore_index=255):
    """Mean Intersection-over-Union for a batch."""
    ious = []
    preds  = preds.view(-1)
    labels = labels.view(-1)
    mask   = labels != ignore_index
    preds, labels = preds[mask], labels[mask]
    for cls in range(num_classes):
        pred_cls  = preds  == cls
        label_cls = labels == cls
        inter = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ─────────────────────────────────────────────
# 6.  TRAIN / VALIDATE LOOPS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, task):
    model.train()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)

        if task == "classification":
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            metric_meter.update(acc1, bs)
            metric_name = "Acc@1"
        else:
            preds = outputs.argmax(1)
            miou  = compute_miou(preds.cpu(), targets.cpu(), num_classes=outputs.shape[1])
            metric_meter.update(miou * 100, bs)
            metric_name = "mIoU"

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} [{i}/{len(loader)}]  "
                  f"Loss: {loss_meter.avg:.4f}  {metric_name}: {metric_meter.avg:.2f}  "
                  f"({elapsed:.1f}s)")

    return loss_meter.avg, metric_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, task, num_classes):
    model.eval()
    loss_meter   = AverageMeter()
    metric_meter = AverageMeter()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, targets)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)

        if task == "classification":
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            metric_meter.update(acc1, bs)
        else:
            preds = outputs.argmax(1)
            miou  = compute_miou(preds.cpu(), targets.cpu(), num_classes=num_classes)
            metric_meter.update(miou * 100, bs)

    return loss_meter.avg, metric_meter.avg


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("MambaVision Trainer")
    # Data
    p.add_argument("--data_dir",    required=True, help="Root of dataset folder")
    p.add_argument("--task",        default="segmentation", choices=["classification", "segmentation"])
    p.add_argument("--num_classes", type=int, default=None,
                   help="Number of classes (auto-detected for classification)")
    p.add_argument("--img_size",    type=int, default=512)
    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.05)
    p.add_argument("--workers",     type=int,   default=4)
    # Checkpointing
    p.add_argument("--output_dir",  default="./checkpoints")
    p.add_argument("--resume",      default="",  help="Path to checkpoint to resume from")
    p.add_argument("--pretrained",  default="",  help="Path to pretrained backbone weights")
    # Model (backbone)
    p.add_argument("--dim",         type=int,   default=128)
    p.add_argument("--in_dim",      type=int,   default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────
    if args.task == "classification":
        train_loader, val_loader, num_classes = get_classification_loaders(
            args.data_dir, args.img_size, args.batch_size, args.workers)
        if args.num_classes is not None:
            num_classes = args.num_classes
    else:
        train_loader, val_loader = get_segmentation_loaders(
            args.data_dir, args.img_size, args.batch_size, args.workers)
        assert args.num_classes, "Please pass --num_classes for segmentation"
        num_classes = args.num_classes

    print(f"Task: {args.task}  |  Classes: {num_classes}  |  "
          f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────
    backbone_cfg = dict(
        dim=args.dim,
        in_dim=args.in_dim,
        depths=(3, 3, 10, 5),
        window_size=(8, 8, 14, 7),
        num_heads=(2, 4, 8, 16),
        num_classes=num_classes,
    )

    if args.task == "segmentation":
        model = MambaVisionSeg(num_classes=num_classes, backbone_cfg=backbone_cfg)
    else:
        if MAMBA_AVAILABLE:
            model = MambaVision(**backbone_cfg)
        else:
            import torchvision.models as tvm
            model = tvm.resnet50(num_classes=num_classes)

    # Load pretrained backbone
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")

    model = model.to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Loss ──────────────────────────────────
    if args.task == "segmentation":
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer + Scheduler ─────────────────
    # Separate weight-decay groups (no decay on biases / norms)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW([
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ────────────────────────────────
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_metric  = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}
    metric_name = "Acc@1" if args.task == "classification" else "mIoU"

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}   LR: {scheduler.get_last_lr()[0]:.2e}")
        print("="*60)

        train_loss, train_metric = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1, args.task)

        val_loss, val_metric = validate(
            model, val_loader, criterion, device, args.task, num_classes)

        scheduler.step()

        print(f"\n  ✔ Train  Loss: {train_loss:.4f}  {metric_name}: {train_metric:.2f}")
        print(f"  ✔ Val    Loss: {val_loss:.4f}  {metric_name}: {val_metric:.2f}")

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        # Save checkpoint
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_metric": best_metric,
            "args":        vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"  ★ New best {metric_name}: {best_metric:.2f} — checkpoint saved")

    # Save history JSON
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best {metric_name}: {best_metric:.2f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
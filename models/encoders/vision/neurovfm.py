# Copyright (c) 2022, Tri Dao.
# Adapted: removed flash_attn dependency, pure PyTorch implementation.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math
import json
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from torch.utils.checkpoint import checkpoint

from positional_encodings.torch_encodings import PositionalEncoding3D


MLP_CHECKPOINT_LVL = 2  # 保留常量，兼容旧引用


# ============================================================
# 纯 PyTorch 等价实现（替代 flash_attn 组件）
# ============================================================
FusedDense = nn.Linear
RMSNorm = None  # 不用 RMSNorm，统一 LayerNorm


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    eps=1e-5,
    dropout_p=0.0,
    rowscale=None,
    prenorm=True,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """
    纯 PyTorch 等价实现，签名与 flash_attn.ops.triton.layer_norm.layer_norm_fn 兼容。

    prenorm=True  -> 返回 (normed_x, residual)，residual 为归一化前的 x（含旧 residual）
    prenorm=False -> 返回最终 hidden state（已 add residual + norm），单个 tensor
    """
    if residual is not None:
        x = x + residual

    residual_out = x  # 新的 residual

    x_f = x.float() if residual_in_fp32 else x

    if is_rms_norm:
        x_n = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            x_n = x_n * weight
        if bias is not None:
            x_n = x_n + bias
    else:
        x_n = F.layer_norm(x_f, (x_f.shape[-1],), weight, bias, eps)

    if rowscale is not None:
        x_n = x_n * rowscale.unsqueeze(-1)

    if dropout_p > 0.0:
        x_n = F.dropout(x_n, p=dropout_p, training=True)

    x_n = x_n.to(x.dtype)

    if prenorm:
        return x_n, residual_out
    else:
        return x_n


class FusedMLP(nn.Module):
    """nn.Sequential 包装，签名与 flash_attn.modules.mlp.FusedMLP 兼容。"""

    def __init__(self, in_features, hidden_features, checkpoint_lvl=0,
                 return_residual=False, activation=F.gelu):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.return_residual = return_residual

    def forward(self, x):
        y = self.fc2(self.act(self.fc1(x)))
        if self.return_residual:
            return y, x
        return y


# ============================================================
# Packed <-> Padded 辅助函数
# ============================================================
def pad_packed(data, cu_seqlens, max_seqlen, batch_first=True, padding_value=0.0):
    """把 packed [total, D] 按 cu_seqlens 还原成 padded [B, max_seqlen, D] + mask [B, max_seqlen]。"""
    sequences = [data[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first,
                                             padding_value=padding_value)
    B, N, D = padded.shape
    if N < max_seqlen:
        padding_size = max_seqlen - N
        pad_shape = (B, padding_size, D) if batch_first else (padding_size, B, D)
        paddings = torch.full(pad_shape, padding_value, device=data.device, dtype=data.dtype)
        dim_to_pad = 1 if batch_first else 0
        padded = torch.cat([padded, paddings], dim=dim_to_pad)
    elif N > max_seqlen:
        padded = padded[:, :max_seqlen, :] if batch_first else padded[:max_seqlen, :, :]

    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.arange(max_seqlen, device=data.device)
    if batch_first:
        mask = indices.unsqueeze(0) < seq_lengths.unsqueeze(1)  # [B, max_seqlen]
    else:
        raise NotImplementedError("pad_packed currently requires batch_first=True")

    return padded, mask


def unpad_packed(padded_data, mask):
    """[B, max_seqlen, D] + mask [B, max_seqlen] -> [total, D]"""
    return padded_data[mask]


# ============================================================
# SelfAttention / CrossAttention（纯 PyTorch + SDPA）
# ============================================================
class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = qk_scale or dim_head ** -0.5
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, dim, bias=False)
        self.attn_drop = attn_drop

    def forward(self, x, cu_seqlens=None, max_seqlen=None,
                use_flash_attn=False, return_attn_weights=False):
        """
        x: [B, N, D] 或 packed [total, D] + cu_seqlens/max_seqlen
        返回: (out, attn_weights or None)
        """
        is_packed = cu_seqlens is not None
        if is_packed:
            if max_seqlen is None:
                raise ValueError("max_seqlen required when cu_seqlens provided")
            x_padded, attn_mask = pad_packed(x, cu_seqlens, max_seqlen, batch_first=True)
        else:
            x_padded = x
            B_, N_ = x_padded.shape[0], x_padded.shape[1]
            attn_mask = torch.ones(B_, N_, dtype=torch.bool, device=x_padded.device)

        B, N, D = x_padded.shape
        C = self.num_heads * self.dim_head

        qkv = self.qkv(x_padded)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, H, N, Dh]

        # SDPA mask: [B, 1, 1, N]，True 表示保留
        sdpa_mask = None
        if not attn_mask.all():
            sdpa_mask = attn_mask[:, None, None, :]

        if return_attn_weights:
            # 手动算，返回权重（N 大时不建议）
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if sdpa_mask is not None:
                attn = attn.masked_fill(~sdpa_mask, torch.finfo(attn.dtype).min)
            attn = attn.softmax(dim=-1)
            attn_weights = attn
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            y = attn @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None

        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)

        if is_packed:
            y = unpad_packed(y, attn_mask)
        return y, attn_weights


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_head, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = qk_scale or dim_head ** -0.5
        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, dim, bias=False)
        self.attn_drop = attn_drop

    def forward(self, x, cu_seqlens=None, max_seqlen=None,
                k=None, cu_seqlens_k=None, max_seqlen_k=None):
        # 简化：暂不支持 varlen cross-attn；当前 encoder/predictor 未使用
        raise NotImplementedError("CrossAttention not implemented in pure-PyTorch version")


# ============================================================
# Block
# ============================================================
class Block(nn.Module):
    def __init__(self, dim, dim_head, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, attn_drop=0.0, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout_layer=nn.Dropout,
                 resid_dropout1=0.0, resid_dropout2=0.0, drop_path1=0.0, drop_path2=0.0,
                 use_cross_attn=False):
        super().__init__()
        if use_cross_attn:
            self.mixer = CrossAttention(dim, dim_head, num_heads, qkv_bias, qk_scale, attn_drop)
        else:
            self.mixer = SelfAttention(dim, dim_head, num_heads, qkv_bias, qk_scale, attn_drop)
        self.use_cross_attn = use_cross_attn

        self.drop1 = dropout_layer(resid_dropout1)
        self.drop2 = dropout_layer(resid_dropout2)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.drop_path2 = StochasticDepth(drop_path2, mode="row")
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim,
                            checkpoint_lvl=MLP_CHECKPOINT_LVL, return_residual=False)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, residual=None,
                use_flash_attn=False, return_attn_weights=False):
        if self.drop_path1.p == 0 or not self.training:
            rowscale1 = None
        else:
            rowscale1 = self.drop_path1(
                torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype))

        x, residual = layer_norm_fn(
            x, self.norm1.weight, self.norm1.bias,
            residual=residual, eps=self.norm1.eps,
            dropout_p=self.drop1.p if self.training else 0.0,
            rowscale=rowscale1, prenorm=True, residual_in_fp32=True,
            is_rms_norm=False,
        )

        x, attn_weights = self.mixer(
            x, cu_seqlens, max_seqlen,
            use_flash_attn=False, return_attn_weights=return_attn_weights)

        if self.use_cross_attn and cu_seqlens is not None:
            residual = residual[cu_seqlens[:-1]]

        if self.drop_path2.p == 0 or not self.training:
            rowscale2 = None
        else:
            rowscale2 = self.drop_path2(
                torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype))

        x, residual = layer_norm_fn(
            x, self.norm2.weight, self.norm2.bias,
            residual=residual, eps=self.norm2.eps,
            dropout_p=self.drop2.p if self.training else 0.0,
            rowscale=rowscale2, prenorm=True, residual_in_fp32=True,
            is_rms_norm=False,
        )
        x = self.mlp(x)
        return x, residual, attn_weights


# ============================================================
# TransformerEncoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, prefix_len, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 init_std=0.02):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_prefix_tokens = prefix_len
        if self.num_prefix_tokens > 0:
            raise NotImplementedError("Prefix tokens not supported yet")
        self.cls_token = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        assert embed_dim % num_heads == 0
        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim // num_heads, num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None,
                  attn_drop=attn_drop_rate,
                  drop_path1=dpr[i - 1] if i > 0 else 0.0,
                  drop_path2=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer,
                  use_cross_attn=False)
            for i in range(depth)
        ])

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.mixer.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cu_seqlens=None, max_seqlen=None,
                use_flash_attn=False, return_attn_weights=False):
        hidden_states = x
        if self.cls_token is not None:
            raise NotImplementedError("Prefix tokens not supported yet")

        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for blk in self.blocks:
            block_output = blk(hidden_states, cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen, residual=residual,
                               use_flash_attn=False,
                               return_attn_weights=return_attn_weights)
            hidden_states = block_output[0]
            residual = block_output[1]
            block_attn_weights = block_output[2]
            if return_attn_weights and block_attn_weights is not None:
                all_attn_weights.append(block_attn_weights)

        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(hidden_states.shape[:-1],
                           device=hidden_states.device, dtype=hidden_states.dtype))

        if self.norm is not None:
            hidden_states = layer_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias,
                residual=residual, eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale, prenorm=False, is_rms_norm=False,
            )

        if return_attn_weights:
            return hidden_states, all_attn_weights
        return hidden_states


# ============================================================
# PositionalEncoding3DWrapper（不变）
# ============================================================
class PositionalEncoding3DWrapper(nn.Module):
    def __init__(self, in_dim=256, d=30, d_size=128, hw_size=192, pe_factor=1, concat=True):
        super().__init__()
        self.concat = concat
        if self.concat:
            self.num_out = in_dim + d
        else:
            self.num_out = in_dim
            d = in_dim
        assert d % 3 == 0
        p_enc = PositionalEncoding3D(d)
        self.d_size = d_size
        self.hw_size = hw_size

        pe_table = p_enc(torch.zeros(1, d_size, hw_size, hw_size, d))[0].view(
            d_size * (hw_size ** 2), d)
        self.register_buffer('p_enc', pe_table, persistent=False)

        self.pe_factor = pe_factor
        self.d = d

    def forward(self, x, coords):
        batch_size, num_tokens, _ = coords.shape
        tocatid = coords[:, :, 0] * (self.hw_size ** 2) + coords[:, :, 1] * self.hw_size + coords[:, :, 2]
        tocat = self.p_enc[tocatid]
        tocat = (tocat * self.pe_factor).to(dtype=x.dtype)
        if self.concat:
            return torch.cat((x, tocat), dim=-1)
        return tocat


# ============================================================
# PatchEmbed（本地版，纯 nn.Linear）
# ============================================================
from torch.nn.modules.utils import _pair


class PatchEmbed(nn.Module):
    def __init__(self, patch_hw_size: int = 16, patch_d_size: int = 4, in_chans: int = 1,
                 embed_dim: int = 768, norm_layer: nn.Module = None, bias: bool = True,
                 fused_bias_fc: bool = False):
        super().__init__()
        patch_hw_size = _pair(patch_hw_size)
        self.patch_hw_size = patch_hw_size
        self.patch_d_size = patch_d_size

        self.proj = nn.Linear(
            in_chans * patch_hw_size[0] * patch_hw_size[1] * patch_d_size,
            embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


# ============================================================
# VisionPredictor（保留，CrossAttention 未启用）
# ============================================================
PerceiverBlock = None


class VisionPredictor(nn.Module):
    def __init__(self, vision_encoder_dim, dim, depth, dim_head, num_heads, prefix_len,
                 text_encoder_dim=None, use_perceiver_block=False, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, use_flash_attn=False, init_std=0.02,
                 pos_emb_cf=None, use_mask_tokens=True, num_mask_tokens=1,
                 zero_init_mask_tokens=True):
        super().__init__()
        self.num_features = self.dim = dim
        self.num_prefix_tokens = prefix_len

        if pos_emb_cf is not None and pos_emb_cf["params"]["concat"]:
            self.from_vision_encoder = nn.Linear(
                vision_encoder_dim + pos_emb_cf["params"]["d"], dim, bias=True)
        else:
            self.from_vision_encoder = nn.Linear(vision_encoder_dim, dim, bias=True)

        self.to_vision_encoder = nn.Linear(dim, vision_encoder_dim, bias=True)
        if text_encoder_dim is not None:
            self.from_text_encoder = nn.Linear(text_encoder_dim, dim, bias=True)
        else:
            self.from_text_encoder = None

        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, vision_encoder_dim))
                for _ in range(num_mask_tokens)
            ])

        assert depth > 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        assert self.from_text_encoder is not None or not use_perceiver_block

        self.blocks = nn.ModuleList([
            Block(dim, dim_head, num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate,
                  drop_path1=dpr[i - 1] if i > 0 else 0.0, drop_path2=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, use_cross_attn=False)
            for i in range(depth)
        ])

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(dim)

        if not zero_init_mask_tokens and self.mask_tokens is not None:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

        if pos_emb_cf is not None:
            if pos_emb_cf["which"] == "pe3d":
                self.pos_embed = PositionalEncoding3DWrapper(**pos_emb_cf["params"])
            else:
                raise ValueError(f"Positional embedding {pos_emb_cf.get('which')} not supported")
        else:
            self.pos_embed = None

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for block_id, block in enumerate(self.blocks):
            rescale(block.mixer.proj.weight.data, block_id + 1)
            rescale(block.mlp.fc2.weight.data, block_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, vision_encoder_out, coords, info_cls_ctxt, info_tgt,
                info_cls_ctxt_tgt=None, text_encoder_out=None, info_media=None,
                mask_index=0, use_flash_attn=False, return_attn_weights=False,
                return_latents=False):
        masks_cls_ctxt, masks_tgt = info_cls_ctxt[0], info_tgt[0]
        assert masks_cls_ctxt is not None and masks_tgt is not None

        if info_cls_ctxt_tgt is not None and info_cls_ctxt_tgt[0] is not None:
            coords = coords[info_cls_ctxt_tgt[0]]

        if self.pos_embed is not None:
            if self.pos_embed.concat:
                vision_encoder_out = self.pos_embed(
                    vision_encoder_out.unsqueeze(0),
                    coords[masks_cls_ctxt].unsqueeze(0)).squeeze(0)
            else:
                vision_encoder_out = vision_encoder_out + self.pos_embed(
                    vision_encoder_out.unsqueeze(0),
                    coords[masks_cls_ctxt].unsqueeze(0)).squeeze(0)

        latents = self.from_vision_encoder(vision_encoder_out)
        N_cls_ctxt, D = latents.shape

        if self.mask_tokens is None:
            raise NotImplementedError()
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index].repeat(masks_tgt.size(0), 1)

        if self.pos_embed is not None:
            if self.pos_embed.concat:
                pred_tokens = self.pos_embed(
                    pred_tokens.unsqueeze(0),
                    coords[masks_tgt].unsqueeze(0)).squeeze(0)
            else:
                pred_tokens = pred_tokens + self.pos_embed(
                    pred_tokens.unsqueeze(0),
                    coords[masks_tgt].unsqueeze(0)).squeeze(0)
        pred_tokens = self.from_vision_encoder(pred_tokens)

        cls_ctxt_sizes = info_cls_ctxt[1][1:] - info_cls_ctxt[1][:-1]
        tgt_sizes = info_tgt[1][1:] - info_tgt[1][:-1]
        batch_sizes = cls_ctxt_sizes + tgt_sizes
        cu_seqlens_q = torch.cat([torch.tensor([0], device=batch_sizes.device),
                                  batch_sizes.cumsum(0)]).to(dtype=torch.int32)
        max_seqlen_q = batch_sizes.max().item()

        latents = torch.cat([
            torch.cat([l_slice, p_slice])
            for l_slice, p_slice in zip(
                torch.split(latents, cls_ctxt_sizes.tolist()),
                torch.split(pred_tokens, tgt_sizes.tolist()))
        ])
        toselect = []
        for cs, ts in zip(cls_ctxt_sizes, tgt_sizes):
            toselect.extend([0] * cs + [1] * ts)
        toselect = torch.tensor(toselect, dtype=torch.bool).to(latents.device)

        residual = None
        all_attn_weights = [] if return_attn_weights else None
        for block in self.blocks:
            block_output = block(latents, cu_seqlens=cu_seqlens_q,
                                 max_seqlen=max_seqlen_q, residual=residual,
                                 use_flash_attn=False,
                                 return_attn_weights=return_attn_weights)
            latents = block_output[0]
            residual = block_output[1]
            aw = block_output[2]
            if return_attn_weights and aw is not None:
                all_attn_weights.append(aw)

        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(latents.shape[:-1], device=latents.device, dtype=latents.dtype))

        if self.norm is not None:
            latents = layer_norm_fn(
                latents, self.norm.weight, self.norm.bias,
                residual=residual, eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale, prenorm=False, is_rms_norm=False)

        if return_latents:
            return latents, toselect
        if return_attn_weights:
            return self.to_vision_encoder(latents), toselect, all_attn_weights
        return self.to_vision_encoder(latents), toselect


# ============================================================
# VisionTransformer
# ============================================================
class VisionTransformer(TransformerEncoder):
    def __init__(self, embed_dim, depth, num_heads, prefix_len, token_dim=259,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, embed_layer_cf=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 init_std=0.02, pos_emb_cf=None):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         prefix_len=prefix_len, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer, act_layer=act_layer)

        if embed_layer_cf["which"] == "voxel":
            self.token_embed = PatchEmbed(**embed_layer_cf["params"])
        else:
            if pos_emb_cf is not None and pos_emb_cf["params"]["concat"]:
                self.token_embed = nn.Linear(
                    token_dim + pos_emb_cf["params"]["d"], embed_dim, bias=True)
            else:
                self.token_embed = nn.Linear(token_dim, embed_dim, bias=True)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if pos_emb_cf is not None:
            if pos_emb_cf["which"] == "pe3d":
                self.pos_embed = PositionalEncoding3DWrapper(**pos_emb_cf["params"])
            else:
                raise ValueError(f"Positional embedding {pos_emb_cf.get('which')} not supported")
        else:
            self.pos_embed = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, coords, masks=None, masks_enc=None,
                         cu_seqlens=None, max_seqlen=None,
                         use_flash_attn=False, return_attn_weights=False):
        if isinstance(self.token_embed, PatchEmbed):
            if masks is not None:
                x = x[masks]; coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]; coords = coords[masks_enc]

            x = self.dropout(x)

            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.token_embed(x).reshape(
                        -1, self.embed_dim - self.pos_embed.d)
                else:
                    hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)
            else:
                hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)

            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.pos_embed(
                        hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    hidden_states = hidden_states + self.pos_embed(
                        hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        else:
            if masks is not None:
                x = x[masks]; coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]; coords = coords[masks_enc]

            x = self.dropout(x)
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    x = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    x = x + self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)

            hidden_states = self.token_embed(x)

        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for blk in self.blocks:
            if self.training:
                block_output = checkpoint(
                    blk, hidden_states, cu_seqlens, max_seqlen, residual,
                    False, return_attn_weights, use_reentrant=False)
            else:
                block_output = blk(hidden_states, cu_seqlens=cu_seqlens,
                                   max_seqlen=max_seqlen, residual=residual,
                                   use_flash_attn=False,
                                   return_attn_weights=return_attn_weights)

            hidden_states = block_output[0]
            residual = block_output[1]
            aw = block_output[2]
            if return_attn_weights and aw is not None:
                all_attn_weights.append(aw)

        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(hidden_states.shape[:-1],
                           device=hidden_states.device, dtype=hidden_states.dtype))

        if self.norm is not None:
            hidden_states = layer_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias,
                residual=residual, eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale, prenorm=False, is_rms_norm=False)

        if return_attn_weights:
            return hidden_states, all_attn_weights
        return hidden_states

    def forward_penultimate_features(self, x, coords, masks=None, masks_enc=None,
                                     cu_seqlens=None, max_seqlen=None,
                                     use_flash_attn=False, return_attn_weights=False):
        # 复用 forward_features 主体，跳过最后 block 与最终 norm
        if isinstance(self.token_embed, PatchEmbed):
            if masks is not None:
                x = x[masks]; coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]; coords = coords[masks_enc]
            x = self.dropout(x)
            if self.pos_embed is not None and self.pos_embed.concat:
                hidden_states = self.token_embed(x).reshape(
                    -1, self.embed_dim - self.pos_embed.d)
            else:
                hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.pos_embed(
                        hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    hidden_states = hidden_states + self.pos_embed(
                        hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        else:
            if masks is not None:
                x = x[masks]; coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]; coords = coords[masks_enc]
            x = self.dropout(x)
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    x = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    x = x + self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
            hidden_states = self.token_embed(x)

        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                continue
            if self.training:
                block_output = checkpoint(
                    blk, hidden_states, cu_seqlens, max_seqlen, residual,
                    False, return_attn_weights, use_reentrant=False)
            else:
                block_output = blk(hidden_states, cu_seqlens=cu_seqlens,
                                   max_seqlen=max_seqlen, residual=residual,
                                   use_flash_attn=False,
                                   return_attn_weights=return_attn_weights)
            hidden_states = block_output[0]
            residual = block_output[1]
            aw = block_output[2]
            if return_attn_weights and aw is not None:
                all_attn_weights.append(aw)

        hidden_states = hidden_states + residual if residual is not None else hidden_states
        if return_attn_weights:
            return hidden_states, all_attn_weights
        return hidden_states

    def forward(self, x, coords, masks=None, masks_enc=None, cu_seqlens=None,
                max_seqlen=None, use_flash_attn=False, return_attn_weights=False):
        return self.forward_features(x, coords, masks, masks_enc, cu_seqlens,
                                     max_seqlen, use_flash_attn, return_attn_weights)


# ============================================================
# get_vit_backbone / load_pretrained_model
# ============================================================
def get_vit_backbone(which: str = "vit_base", params: Dict = {}) -> VisionTransformer:
    if which == "vit_tiny":
        return VisionTransformer(embed_dim=192, depth=12, num_heads=3, prefix_len=0,
                                 mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **params)
    elif which == "vit_small":
        return VisionTransformer(embed_dim=384, depth=12, num_heads=6, prefix_len=0,
                                 mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **params)
    elif which == "vit_base":
        return VisionTransformer(embed_dim=768, depth=12, num_heads=12, prefix_len=0,
                                 mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **params)
    elif which == "vit_large":
        return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, prefix_len=0,
                                 mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **params)
    elif which == "vit_huge":
        return VisionTransformer(embed_dim=1280, depth=32, num_heads=16, prefix_len=0,
                                 mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **params)
    elif which == "vit":
        return VisionTransformer(**params)
    else:
        raise ValueError("ViT backbone name must be in [vit_tiny, vit_small, vit_base, "
                         "vit_large, vit_huge, vit]")


def load_pretrained_model(model_config_path, model_weight_path):
    with open(model_config_path) as f:
        config = json.load(f)
    model = get_vit_backbone(**config)
    state_dict = torch.load(model_weight_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 兼容：原 FusedDense 权重 key 可能带 .weight/.bias，nn.Linear 也带，通常一致。
    # 若个别 key 不匹配（比如 FusedMLP 的 fc1/fc2），这里只做 strict 加载；
    # 如遇 key mismatch，用 strict=False 并打印缺失 key。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_pretrained_model] missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[load_pretrained_model] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    return model, config


# ============================================================
# NeurovfmVisionEncoder（纯 PyTorch，裸跑 fp32）
# ============================================================
from .base import VisionEncoder, VisionEncoderConfig


class NeurovfmVisionEncoder(VisionEncoder):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__(cfg)
        init_cfg = self.cfg.args
        model, config = load_pretrained_model(**init_cfg)
        self.model: VisionTransformer = model
        self.out_dim = config["params"]["embed_dim"]
        self.patch_size = (4, 16, 16)

    def tokenize_volume(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        assert D % pd == 0 and H % ph == 0 and W % pw == 0
        n_patches_d, n_patches_h, n_patches_w = D // pd, H // ph, W // pw

        tokens_torch = rearrange(
            x,
            "b c (d pd) (h ph) (w pw) -> b (d h w) (c pd ph pw)",
            d=n_patches_d, h=n_patches_h, w=n_patches_w,
            pd=pd, ph=ph, pw=pw)

        coords_d, coords_h, coords_w = torch.meshgrid(
            torch.arange(n_patches_d), torch.arange(n_patches_h),
            torch.arange(n_patches_w), indexing='ij')
        coords_torch = torch.stack(
            [coords_d.flatten(), coords_h.flatten(), coords_w.flatten()],
            dim=1).long().to(x.device)
        return tokens_torch, coords_torch

    def forward(self, images):
        tokens_torch, coords_torch = self.tokenize_volume(images)
        B, N, _ = tokens_torch.shape

        mean, std = 0.3141, 0.2623
        tokens_torch = (tokens_torch - mean) / std

        tokens_flat = tokens_torch.reshape(B * N, -1)
        coords_flat = coords_torch.repeat(B, 1)

        cu_seqlens = torch.arange(0, (B + 1) * N, step=N,
                                  dtype=torch.int32, device=images.device)
        max_seqlen = N

        embs = self.model(
            tokens_flat, coords_flat,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            use_flash_attn=False,
        )
        return embs.reshape(B, N, -1)


# ============================================================
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     cfg = VisionEncoderConfig(
#         name="neurovfm",
#         args={
#             "model_config_path": "/hpc/jtia475/model_weights/neurovfm/config.json",
#             "model_weight_path": "/hpc/jtia475/model_weights/neurovfm/pytorch_model.bin",
#         },
#     )
#     model = NeurovfmVisionEncoder(cfg).to(device).eval()
#     x = torch.randn((2, 1, 128, 352, 352)).float().to(device)
#     with torch.no_grad():
#         emb = model(x)
#     print("output:", emb.shape, emb.dtype)
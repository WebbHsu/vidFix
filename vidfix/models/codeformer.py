"""CodeFormer network (Apache-style architecture from sczhou/CodeFormer)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .vqgan import ResBlock, VQAutoEncoder


def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    size = feat.size()
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        return dec_feat + residual


class CodeFormer(VQAutoEncoder):
    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=None,
        fix_modules=None,
    ):
        if connect_list is None:
            connect_list = ["32", "64", "128", "256"]
        super().__init__(512, 64, [1, 2, 2, 4, 4, 8], "nearest", 2, [16], codebook_size)
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)
        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
                for _ in range(self.n_layers)
            ]
        )
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False))
        self.channels = {
            "16": 512,
            "32": 256,
            "64": 256,
            "128": 128,
            "256": 128,
            "512": 64,
        }
        self.fuse_encoder_block = {"512": 2, "256": 5, "128": 8, "64": 11, "32": 14, "16": 18}
        self.fuse_generator_block = {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21}
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        logits = self.idx_pred_layer(query_emb)
        logits = logits.permute(1, 0, 2)
        if code_only:
            return logits, lq_feat

        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        if detach_16:
            quant_feat = quant_feat.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        return x, logits, lq_feat


def load_codeformer(weights_path, device) -> CodeFormer:
    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")
    except Exception:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        ckpt = ckpt["params_ema"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        ckpt = ckpt["params"]
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    net.to(device)
    return net

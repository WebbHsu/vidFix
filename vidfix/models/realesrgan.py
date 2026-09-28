"""Real-ESRGAN networks (BSD-3, xinntao/Real-ESRGAN) — self-contained, no basicsr."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..paths import REALESRGAN_DIR

# Official release URLs (xinntao/Real-ESRGAN).
WEIGHT_URLS = {
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    ),
    "realesr-general-wdn-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ),
}

# Default: compact general model — real-world video, denoise DNI, low VRAM vs RRDBNet.
DEFAULT_MODEL = "realesr-general-x4v3"

# Blend of enhanced vs original (RealScaler-style interpolation).
BLEND = {"mild": 0.45, "medium": 0.75, "strong": 1.0}
# DNI denoise strength for x4v3 (0=keep texture via wdn, 1=main model).
DN_STRENGTH = {"mild": 0.35, "medium": 0.55, "strong": 0.75}


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR net (realesr-general-x4v3 / animevideov3)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)]
        body.append(_act(act_type, num_feat))
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_act(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN RRDBNet (RealESRGAN_x2plus / x4plus)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        num_in = num_in_ch * (4 if scale == 2 else 1)
        self.conv_first = nn.Conv2d(num_in, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        else:
            feat = x
        feat = self.conv_first(feat)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def pixel_unshuffle(x: Tensor, scale: int) -> Tensor:
    b, c, h, w = x.shape
    out_c = c * (scale**2)
    assert h % scale == 0 and w % scale == 0
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, out_c, h // scale, w // scale)


def _act(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.PReLU(num_parameters=num_feat)


def weight_path(name: str) -> Path:
    return REALESRGAN_DIR / f"{name}.pth"


def weights_present(model: str = DEFAULT_MODEL) -> bool:
    """Main weights are required; the x4v3 wdn file is optional (without it DNI is skipped)."""
    return weight_path(model).is_file()


def missing_weights_message(model: str = DEFAULT_MODEL) -> str:
    needed = [weight_path(model)]
    if model == "realesr-general-x4v3":
        needed.append(weight_path("realesr-general-wdn-x4v3"))
    paths = "、".join(str(p) for p in needed)
    return (
        f"找不到 Real-ESRGAN 權重（需要：{paths}）。"
        f"請執行 python scripts/download_weights.py"
    )


def _load_state(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "params_ema" in obj:
            return obj["params_ema"]
        if "params" in obj:
            return obj["params"]
    return obj


def _dni_state(path_a: Path, path_b: Path, wa: float, wb: float) -> dict:
    a = _load_state(path_a)
    b = _load_state(path_b)
    return {k: wa * a[k] + wb * b[k] for k in a}


def build_net(model: str = DEFAULT_MODEL) -> tuple[nn.Module, int]:
    if model == "realesr-general-x4v3":
        return (
            SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
            4,
        )
    if model == "RealESRGAN_x2plus":
        return (
            RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32),
            2,
        )
    raise ValueError(f"未知 Real-ESRGAN 模型：{model}")


class FrameEnhancer:
    """Real-ESRGAN upscale (x4 or x2), INTER_AREA back to the input size, blend with input.

    Output always has the input H×W. CUDA runs fp16; CPU fp32. Tiles are cut on the
    input with ``tile_pad`` context on each side (no seams). VRAM notes: docs/architecture.md.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        strength: str = "medium",
        tile: int = 512,
        tile_pad: int = 10,
        denoise_strength: float | None = None,
    ):
        if not weight_path(model).is_file():
            raise FileNotFoundError(missing_weights_message(model))
        self.model_name = model
        self.strength = strength if strength in BLEND else "medium"
        self.blend = float(BLEND[self.strength])
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        net, self.scale = build_net(model)
        dn = denoise_strength
        if dn is None:
            dn = float(DN_STRENGTH.get(self.strength, 0.55))
        state = self._resolve_state(model, dn)
        net.load_state_dict(state, strict=True)
        net.eval()
        self.net = net.to(self.device)
        if self.half:
            self.net = self.net.half()

    def _resolve_state(self, model: str, dn: float) -> dict:
        main = weight_path(model)
        if model == "realesr-general-x4v3":
            wdn = weight_path("realesr-general-wdn-x4v3")
            dn = float(np.clip(dn, 0.0, 1.0))
            if wdn.is_file() and abs(dn - 1.0) > 1e-6:
                # dni_weight [dn, 1-dn]: more main model = stronger denoise
                return _dni_state(main, wdn, dn, 1.0 - dn)
            return _load_state(main)
        return _load_state(main)

    @torch.no_grad()
    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        """Return BGR uint8 at the same HxW as input."""
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        up = self._infer(rgb)
        # INTER_AREA: box average for the integer 4x/2x reduction (no ringing, no aliasing)
        down = cv2.resize(up, (w0, h0), interpolation=cv2.INTER_AREA)
        down_bgr = cv2.cvtColor(down, cv2.COLOR_RGB2BGR)
        if self.blend >= 0.999:
            return down_bgr
        a = self.blend
        out = down_bgr.astype(np.float32) * a + bgr.astype(np.float32) * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _infer(self, rgb01: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(rgb01.transpose(2, 0, 1)).float().unsqueeze(0)
        img = img.to(self.device)
        if self.half:
            img = img.half()
        # RRDBNet x2 pixel-unshuffles by 2, so H/W must be even
        mod = 2 if self.scale == 2 else 1
        _, _, h, w = img.shape
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        if self.tile > 0 and (img.shape[2] > self.tile or img.shape[3] > self.tile):
            out = self._tile_infer(img)
        else:
            out = self.net(img)
        if pad_h or pad_w:
            out = out[:, :, : h * self.scale, : w * self.scale]
        arr = out.squeeze(0).clamp_(0, 1).cpu().float().numpy().transpose(1, 2, 0)
        return (arr * 255.0).round().astype(np.uint8)

    def _tile_infer(self, img: Tensor) -> Tensor:
        _, _, height, width = img.shape
        out = img.new_zeros((1, 3, height * self.scale, width * self.scale))
        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                ofs_x = tx * self.tile
                ofs_y = ty * self.tile
                i0x, i1x = ofs_x, min(ofs_x + self.tile, width)
                i0y, i1y = ofs_y, min(ofs_y + self.tile, height)
                p0x = max(i0x - self.tile_pad, 0)
                p1x = min(i1x + self.tile_pad, width)
                p0y = max(i0y - self.tile_pad, 0)
                p1y = min(i1y + self.tile_pad, height)
                tile = img[:, :, p0y:p1y, p0x:p1x]
                pred = self.net(tile)
                o0x, o1x = i0x * self.scale, i1x * self.scale
                o0y, o1y = i0y * self.scale, i1y * self.scale
                t0x = (i0x - p0x) * self.scale
                t1x = t0x + (i1x - i0x) * self.scale
                t0y = (i0y - p0y) * self.scale
                t1y = t0y + (i1y - i0y) * self.scale
                out[:, :, o0y:o1y, o0x:o1x] = pred[:, :, t0y:t1y, t0x:t1x]
        return out

    def close(self) -> None:
        self.net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

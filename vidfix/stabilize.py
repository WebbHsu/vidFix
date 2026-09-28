"""Offline face-track smoothing.

Per-frame detectors jitter by 1–3 px. That jitter becomes rotation/scale shake
when the restored 512 crop is warped back. For a 20–30 s clip we can look at
the whole track and smooth the rigid pose (scale, rotation, translation) with
a Savitzky–Golay filter — the usual offline-VFX approach, stronger than causal EMA.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .models.restorer import FACE_TEMPLATE


def affine_from_kps(kps: np.ndarray) -> np.ndarray | None:
    M, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps[:5], dtype=np.float32),
        FACE_TEMPLATE,
        method=cv2.LMEDS,
    )
    return None if M is None else M.astype(np.float32)


def _M_to_params(M: np.ndarray) -> tuple[float, float, float, float]:
    a, _b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, _d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    scale = math.hypot(a, c)
    rot = math.atan2(c, a)
    return scale, rot, tx, ty


def _params_to_M(scale: float, rot: float, tx: float, ty: float) -> np.ndarray:
    ca = math.cos(rot) * scale
    sa = math.sin(rot) * scale
    return np.array([[ca, -sa, tx], [sa, ca, ty]], dtype=np.float32)


def smooth_affine_track(
    kps_list: list[np.ndarray | None],
    fps: float,
    window_sec: float = 0.45,
) -> list[np.ndarray | None]:
    """Return one 2x3 affine (kps → 512 template) per frame, or None if no face."""
    n = len(kps_list)
    if n == 0:
        return []

    params = np.full((n, 4), np.nan, dtype=np.float64)
    for i, kps in enumerate(kps_list):
        if kps is None:
            continue
        M = affine_from_kps(kps)
        if M is None:
            continue
        params[i] = _M_to_params(M)

    valid = ~np.isnan(params[:, 0])
    if int(valid.sum()) < 3:
        return [affine_from_kps(k) if k is not None else None for k in kps_list]

    good = np.flatnonzero(valid)
    params[good, 1] = np.unwrap(params[good, 1])
    idx = np.arange(n, dtype=np.float64)
    for c in range(4):
        params[:, c] = np.interp(idx, good.astype(np.float64), params[good, c])

    win = int(round(float(fps) * window_sec)) | 1
    win = max(5, min(win, n if n % 2 == 1 else n - 1))
    poly = 2 if win > 4 else 1
    if win >= 5 and n >= win:
        try:
            from scipy.signal import savgol_filter

            for c in range(4):
                params[:, c] = savgol_filter(params[:, c], win, poly, mode="interp")
        except Exception:
            pass

    max_gap = max(4, int(round(float(fps) * 0.25)))
    out: list[np.ndarray | None] = []
    for i in range(n):
        nearest = int(np.min(np.abs(good - i)))
        if nearest > max_gap:
            out.append(None)
            continue
        out.append(_params_to_M(*params[i]))
    return out

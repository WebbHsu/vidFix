from __future__ import annotations

import cv2
import numpy as np
import torch

from ..paths import CODEFORMER_WEIGHTS, INSIGHTFACE_ROOT
from .codeformer import load_codeformer

# FFHQ 512 5-point template used by GFPGAN / CodeFormer FaceRestoreHelper
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


class FaceRestorer:
    def __init__(
        self,
        fidelity: float = 0.40,
        visibility: float = 0.60,
        tiny_face_px: int = 75,
        kps_smooth: float = 0.42,
        face_smooth: float = 0.70,
        hold_miss: int = 2,
    ):
        if not CODEFORMER_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"找不到 CodeFormer 權重：{CODEFORMER_WEIGHTS}\n請執行 python scripts/download_weights.py"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_codeformer(CODEFORMER_WEIGHTS, self.device)
        self.fidelity = float(fidelity)
        self.visibility = float(visibility)
        self.tiny_face_px = int(tiny_face_px)
        # EMA weight for the *new* frame: lower = smoother, more lag.
        self.kps_smooth = float(np.clip(kps_smooth, 0.05, 1.0))
        self.face_smooth = float(np.clip(face_smooth, 0.05, 1.0))
        self.hold_miss = int(max(0, hold_miss))
        self.detector = _build_detector()
        self.reset()

    def reset(self) -> None:
        """Call at the start of each clip so smoothing does not leak across cuts."""
        self._smooth_kps: np.ndarray | None = None
        self._prev_restored: np.ndarray | None = None
        self._miss = 0
        self._face_w = 160.0

    def detect_kps(self, bgr: np.ndarray) -> np.ndarray | None:
        faces = self.detector.get(bgr)
        if not faces:
            return None
        face = max(faces, key=_face_area)
        kps = np.asarray(face.kps, dtype=np.float32)
        if kps.shape[0] < 5:
            return None
        bbox = np.asarray(face.bbox, dtype=np.float32)
        self._face_w = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        return kps[:5]

    def restore_frame(self, bgr: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        vis = self.visibility
        w = self.fidelity
        if M is None:
            kps = self.detect_kps(bgr)
            if kps is None:
                self._miss += 1
                if self._smooth_kps is None or self._miss > self.hold_miss:
                    self._prev_restored = None
                    return bgr
                kps = self._smooth_kps
                vis = vis * max(0.0, 1.0 - self._miss / (self.hold_miss + 1))
            else:
                self._miss = 0
            kps = self._smooth_landmarks(kps)
            aligned, M = _align_face(bgr, kps)
        else:
            self._miss = 0
            scale = float(np.hypot(M[0, 0], M[1, 0]))
            if scale > 1e-6:
                self._face_w = 512.0 / scale
            aligned = cv2.warpAffine(
                bgr,
                M,
                (512, 512),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if aligned is None or M is None:
            return bgr
        if self._face_w < self.tiny_face_px:
            w = min(0.55, max(w, 0.50))
            vis = min(vis, 0.55)
        restored = self._codeformer(aligned, w)
        restored = self._smooth_restored(restored)
        return _paste_face(bgr, restored, M, vis)

    def _smooth_landmarks(self, kps: np.ndarray) -> np.ndarray:
        if self._smooth_kps is None:
            self._smooth_kps = kps.copy()
            return self._smooth_kps
        jump = float(np.linalg.norm(kps - self._smooth_kps, axis=1).mean())
        if jump > 80.0:
            self._smooth_kps = kps.copy()
            return self._smooth_kps
        # Small detector jitter is damped; real head motion is followed.
        a = 0.58 if jump > 10.0 else self.kps_smooth
        self._smooth_kps = ((1.0 - a) * self._smooth_kps + a * kps).astype(np.float32)
        return self._smooth_kps

    def _smooth_restored(self, restored: np.ndarray) -> np.ndarray:
        cur = restored.astype(np.float32)
        if self._prev_restored is None or self._prev_restored.shape != cur.shape:
            self._prev_restored = cur
            return restored
        a = self.face_smooth
        blended = (1.0 - a) * self._prev_restored + a * cur
        self._prev_restored = blended
        return np.clip(blended, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _codeformer(self, face_bgr: np.ndarray, w: float) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        t = (t - 0.5) / 0.5
        t = t.to(self.device)
        out = self.net(t, w=w, adain=True)[0]
        out = out.clamp(-1, 1)
        out = (out + 1) / 2
        arr = out.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.net = None
        self.detector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_detector():
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise RuntimeError("請安裝 insightface：pip install insightface") from e
    INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
    providers = []
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    except Exception:
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        allowed_modules=["detection"],
        providers=providers,
    )
    ctx_id = 0 if torch.cuda.is_available() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _face_area(face) -> float:
    b = face.bbox
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _align_face(img: np.ndarray, kps: np.ndarray):
    if kps is None or len(kps) < 5:
        return None, None
    M, _ = cv2.estimateAffinePartial2D(kps[:5], FACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None, None
    aligned = cv2.warpAffine(
        img,
        M,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, M


def _inscribed_ellipse_mask(size: int) -> np.ndarray:
    """Full strength on the face; zero at the 512-square corners (no visible box)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) * 0.5
    rx = size * 0.48
    ry = size * 0.50
    d = np.sqrt(((xx - c) / rx) ** 2 + ((yy - c) / ry) ** 2)
    inner, outer = 0.78, 1.0
    mask = np.ones_like(d, dtype=np.float32)
    band = (d > inner) & (d <= outer)
    t = (d[band] - inner) / (outer - inner)
    mask[band] = 0.5 * (1.0 + np.cos(np.pi * t))
    mask[d > outer] = 0.0
    k = max(9, (int(size * 0.04) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    peak = float(mask.max())
    if peak > 0:
        mask /= peak
    return mask.astype(np.float32)


_MASK_512 = None


def _paste_face(img: np.ndarray, restored: np.ndarray, M: np.ndarray, visibility: float) -> np.ndarray:
    global _MASK_512
    h, w = img.shape[:2]
    inv = cv2.invertAffineTransform(M)
    fh, fw = restored.shape[:2]
    if fh == 512 and fw == 512:
        if _MASK_512 is None:
            _MASK_512 = _inscribed_ellipse_mask(512)
        mask = _MASK_512
    else:
        mask = _inscribed_ellipse_mask(fw)
    warped = cv2.warpAffine(
        restored, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_w = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_w = np.clip(mask_w.astype(np.float32), 0.0, 1.0)
    vis = float(np.clip(visibility, 0.0, 1.0))
    composed = _seamless_or_alpha(img, warped, mask_w)
    m = (mask_w * vis)[..., None]
    out = composed.astype(np.float32) * m + img.astype(np.float32) * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _seamless_or_alpha(img: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Poisson-blend so the restored face matches surrounding lighting; no square seam."""
    h, w = img.shape[:2]
    mask_u8 = (mask * 255.0).astype(np.uint8)
    mask_u8[0, :] = 0
    mask_u8[-1, :] = 0
    mask_u8[:, 0] = 0
    mask_u8[:, -1] = 0
    ys, xs = np.where(mask_u8 > 16)
    if len(xs) < 30:
        return img
    x, y, bw, bh = cv2.boundingRect(mask_u8)
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img
    src = warped[y0:y1, x0:x1]
    m = mask_u8[y0:y1, x0:x1]
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0
    if int(m.max()) < 16:
        return img
    cx = x0 + (x1 - x0) // 2
    cy = y0 + (y1 - y0) // 2
    cx = int(np.clip(cx, 1, w - 2))
    cy = int(np.clip(cy, 1, h - 2))
    try:
        return cv2.seamlessClone(src, img, m, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        blend = mask[..., None]
        return np.clip(warped.astype(np.float32) * blend + img.astype(np.float32) * (1.0 - blend), 0, 255).astype(
            np.uint8
        )


def weights_status() -> dict:
    det_dir = INSIGHTFACE_ROOT / "models" / "buffalo_l"
    has_det = det_dir.is_dir() and any(det_dir.glob("*.onnx"))
    return {
        "codeformer": CODEFORMER_WEIGHTS.is_file(),
        "codeformer_path": str(CODEFORMER_WEIGHTS),
        "insightface": has_det,
        "insightface_path": str(det_dir),
        "cuda": bool(torch.cuda.is_available()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

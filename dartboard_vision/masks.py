"""Aligned temporal HSV differences, with independent per-camera shadows.

No target coordinates, camera mounting angles, dart colours or dart count are
used. Thresholds have explicit HSV units; spatial lengths scale with image size.
"""

from dataclasses import asdict, dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class MaskSettings:
    saturation_change: float = 0.12
    hue_chroma_change: float = 0.12
    value_change: float = 0.06
    local_value_change: float = 0.075
    color_value_change: float = 0.075
    illumination_sigma: float = 9.0


def differences(before, after, valid, settings=MaskSettings()):
    scale = before.shape[0] / 900.0
    old = cv.GaussianBlur(before.astype(np.float32) / 255.0, (3, 3), 0.6)
    new = cv.GaussianBlur(after.astype(np.float32) / 255.0, (3, 3), 0.6)
    # Robust global channel gains compensate exposure / white balance changes.
    # The board is mostly unchanged; no dart position or pixel labels are used.
    gains = []
    for k in range(3):
        keep = (
            valid
            & (old[..., k] > 0.12)
            & (old[..., k] < 0.90)
            & (new[..., k] > 0.05)
            & (new[..., k] < 0.98)
        )
        gains.append(
            float(np.median(new[..., k][keep] / old[..., k][keep]))
            if keep.any()
            else 1.0
        )
    old = np.clip(old * np.array(gains, dtype=np.float32), 0, 1)
    a = cv.cvtColor(old, cv.COLOR_BGR2HSV)
    b = cv.cvtColor(new, cv.COLOR_BGR2HSV)
    dh = np.abs(a[..., 0] - b[..., 0])
    dh = np.minimum(dh, 360 - dh)
    # Circular hue distance weighted by chroma, zero on neutral pixels.
    hc = 2 * np.minimum(a[..., 1], b[..., 1]) * np.sin(np.deg2rad(dh) * 0.5)
    ds = np.abs(a[..., 1] - b[..., 1])
    dv = b[..., 2] - a[..., 2]
    sigma = max(1.0, settings.illumination_sigma * scale)
    old_low = cv.GaussianBlur(a[..., 2], (0, 0), sigma)
    new_low = cv.GaussianBlur(b[..., 2], (0, 0), sigma)
    local_ratio = new_low / np.maximum(old_low, 0.04)
    local_value = np.abs(b[..., 2] - a[..., 2] * local_ratio)
    signed_s = b[..., 1] - a[..., 1]
    local_saturation = abs(signed_s - cv.GaussianBlur(signed_s, (0, 0), sigma))
    ha = np.deg2rad(a[..., 0])
    hb = np.deg2rad(b[..., 0])
    dc = np.stack(
        [
            b[..., 1] * np.cos(hb) - a[..., 1] * np.cos(ha),
            b[..., 1] * np.sin(hb) - a[..., 1] * np.sin(ha),
        ],
        axis=-1,
    )
    local_chroma = np.linalg.norm(dc - cv.GaussianBlur(dc, (0, 0), sigma), axis=2)
    rgb_change = np.max(abs(new - old), axis=2)
    return dict(
        hue_degrees=dh,
        hue_chroma=hc,
        saturation=ds,
        value=dv,
        rgb_change=rgb_change,
        local_value=local_value,
        local_saturation=local_saturation,
        local_chroma=local_chroma,
        before_hsv=a,
        after_hsv=b,
        channel_gains=gains,
    )


def dart_mask(before, after, valid, settings=MaskSettings()):
    f = differences(before, after, valid, settings)
    mask, info = mask_from_fields(f, valid, settings)
    return mask, dict(info, fields=f)


def evidence_score(f, settings=MaskSettings()):
    """Continuous threshold-normalized HSV signal: one is the mask threshold."""
    s = settings
    colour = np.maximum(
        f["local_saturation"] / s.saturation_change,
        f["local_chroma"] / s.hue_chroma_change,
    )
    colour = np.where(f["rgb_change"] > s.color_value_change, colour, 0.0)
    value = np.minimum(
        abs(f["value"]) / s.value_change, f["local_value"] / s.local_value_change
    )
    return np.maximum(colour, value).astype(np.float32)


def mask_from_fields(f, valid, settings=MaskSettings()):
    s = settings
    color = (
        (f["local_saturation"] > s.saturation_change)
        | (f["local_chroma"] > s.hue_chroma_change)
    ) & (f["rgb_change"] > s.color_value_change)
    value = (abs(f["value"]) > s.value_change) & (
        f["local_value"] > s.local_value_change
    )
    changed = valid & (color | value)
    # Diagnostic shading evidence: value fell but the locally corrected HSV
    # signals did not indicate an object. This is not a pixel ground-truth label.
    shadow = valid & (f["value"] < -s.value_change) & ~changed
    mask = np.uint8(changed) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (
        np.uint8(stats[1:, cv.CC_STAT_AREA] >= max(5, 5 * (mask.shape[0] / 900.0) ** 2))
        * 255
    )
    mask = keep[labels]
    mask[~valid] = 0
    return mask, dict(
        shadow=np.uint8(shadow) * 255,
        settings=asdict(s),
        foreground_pixels=int(np.count_nonzero(mask)),
    )

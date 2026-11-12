"""Cached spatial precision proxies from board texture and the source-map Jacobian.

This is a bounded information heuristic, not measured geometric uncertainty.
No observed lettering or numbered-ring pixels enter the sharpness calculation.
"""

import time

import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt

from .alignment import plane_features, source_mapping
from .geometry import BOARD, R, geometry_fields


def masked_average(value, mask, sigma):
    m = mask.astype(np.float64)
    return cv.GaussianBlur(value * m, (0, 0), sigma) / np.maximum(
        cv.GaussianBlur(m, (0, 0), sigma), 1e-12
    )


def focus_proxy(gray, roi, interior, sigma):
    f = plane_features(gray, None, roi, 1.0)[:, :, 0]
    stencil = cv.erode(np.uint8(roi), np.ones((3, 3), np.uint8)) > 0
    energies = []
    for blur in (0.6, 1.8):
        smooth = masked_average(f, roi, blur)
        gx = cv.Scharr(smooth, cv.CV_64F, 1, 0, scale=1 / 32)
        gy = cv.Scharr(smooth, cv.CV_64F, 0, 1, scale=1 / 32)
        energies.append(masked_average(gx * gx + gy * gy, interior & stencil, sigma))
    # A flat patch gets neutral sharpness; this prevents division by zero.
    e0, e1 = energies
    eps = 1e-12
    focus = (e0 + eps) / (e0 + e1 + 2 * eps)
    # Outside the scoring ROI, use nearest interior evidence, never lettering.
    nearest = distance_transform_edt(~roi, return_distances=False, return_indices=True)
    return focus[tuple(nearest)]


def sampling_precision(xy):
    """Smallest squared singular value: source pixels per output pixel."""
    dx = np.gradient(xy, axis=1)
    dy = np.gradient(xy, axis=0)
    a = np.sum(dx * dx, axis=-1)
    b = np.sum(dx * dy, axis=-1)
    c = np.sum(dy * dy, axis=-1)
    return np.maximum(0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b * b)), 0)


def normalize_confidence(scores):
    scores = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Invalid confidence scores")
    total = scores.sum(axis=0, keepdims=True)
    return np.divide(3 * scores, total, out=np.ones_like(scores), where=total > 0)


def build_confidence(frames, models, board=BOARD, window_sigma_mm=12.0):
    start = time.perf_counter()
    size = board.size
    yy, xx = np.mgrid[:size, :size]
    q = np.c_[
        (xx.ravel() - board.center[0]) / R,
        (yy.ravel() - board.center[1]) / R,
        np.ones(size * size),
    ]
    _, distance, roi, _ = geometry_fields(size, board)
    interior = roi & (distance > 2.5)
    scores = []
    focuses = []
    densities = []
    planes = []
    visible = []
    for frame, model in zip(frames, models):
        p = np.asarray(model["inverse_homography_normalized"]).ravel()[:8]
        xy = (
            source_mapping(q, p, model, scale=1.0, derivatives=False)
            .reshape(size, size, 2)
            .astype(np.float32)
        )
        valid = (
            (xy[:, :, 0] >= 1)
            & (xy[:, :, 0] < frame.shape[1] - 2)
            & (xy[:, :, 1] >= 1)
            & (xy[:, :, 1] < frame.shape[0] - 2)
        )
        plane = cv.remap(frame, xy[:, :, 0], xy[:, :, 1], cv.INTER_LINEAR)
        focus = focus_proxy(
            cv.cvtColor(plane, cv.COLOR_BGR2GRAY).astype(float),
            roi,
            interior,
            window_sigma_mm * board.ppm,
        )
        density = sampling_precision(xy.astype(float))
        score = focus * density / (1 + density) * valid
        scores.append(score)
        focuses.append(focus)
        densities.append(density)
        planes.append(plane)
        visible.append(valid)
    c = normalize_confidence(scores)
    summary = dict(
        build_ms=(time.perf_counter() - start) * 1000,
        window_sigma_mm=window_sigma_mm,
        source="Empty-board scoring ROI only; nearest interior focus extrapolation outside ROI",
        formula="score = fine_energy/(fine_energy+coarse_energy) * min_singular_value_squared/(1+min_singular_value_squared); normalize sum=3",
        focus_scales_output_px=[0.6, 1.8],
        heuristic_not_calibrated_uncertainty=True,
    )
    return c, dict(
        summary=summary,
        focus=np.array(focuses),
        density=np.array(densities),
        planes=planes,
        visible=visible,
        roi=roi,
    )


def fuse(planes, visible, confidence=None):
    valid = np.array(visible, dtype=float)
    weights = valid if confidence is None else valid * np.asarray(confidence)
    total = weights.sum(axis=0)
    return np.uint8(
        np.clip(
            np.sum(np.array(planes, dtype=float) * weights[:, :, :, None], axis=0)
            / np.maximum(total[:, :, None], 1e-12),
            0,
            255,
        )
    )


def overlay_masked_views(background, planes, masks, visible=None):
    """Replace dart-mask pixels with the mean of their contributing cameras.

    All inputs are already in the same board coordinates. Masks are binary:
    a view either contributes its current RGB pixel or contributes nothing.
    Pixels outside every valid mask retain the background exactly. The empty
    reference is not blended, and this operation does not affect detection.
    """
    images = np.asarray(planes)
    support = np.asarray(masks) > 0
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Planes must have shape (cameras, height, width, 3)")
    if images.dtype != np.uint8 or background.dtype != np.uint8:
        raise ValueError("Images must be uint8 BGR")
    if support.shape != images.shape[:3] or background.shape != images.shape[1:]:
        raise ValueError("Background, planes and masks must share board coordinates")
    if visible is not None:
        valid = np.asarray(visible, dtype=bool)
        if valid.shape != support.shape:
            raise ValueError("Visibility must match the camera masks")
        support &= valid
    count = support.sum(axis=0)
    selected_sum = np.sum(images.astype(np.float32) * support[..., None], axis=0)
    selected_mean = selected_sum / np.maximum(count[..., None], 1)
    result = background.copy()
    active = count > 0
    result[active] = np.uint8(np.clip(selected_mean[active], 0, 255))
    return result


class CachedFusion:
    """Cache normalized display weights and buffers; geometry is never updated."""

    def __init__(self, confidence, visible):
        self.confidence = np.asarray(confidence, dtype=np.float32).copy()
        if (
            self.confidence.ndim != 3
            or np.any(~np.isfinite(self.confidence))
            or np.any(self.confidence < 0)
        ):
            raise ValueError("Invalid confidence")
        self.shape = (*self.confidence.shape[1:], 3)
        self.acc = np.empty(self.shape, np.float32)
        self.work = np.empty_like(self.acc)
        self.set_visibility(visible)

    def set_visibility(self, visible):
        valid = np.asarray(visible, dtype=bool)
        if valid.shape != self.confidence.shape:
            raise ValueError("Visibility shape differs from weights")
        self.visible = valid.copy()
        w = self.confidence * valid
        total = w.sum(axis=0)
        self.shares = np.divide(
            w, total[None, :, :], out=np.zeros_like(w), where=total[None, :, :] > 0
        )
        self.rgb_weights = [np.repeat(s[:, :, None], 3, axis=2) for s in self.shares]

    def blend(self, planes, visible=None):
        if visible is not None and not np.array_equal(visible, self.visible):
            self.set_visibility(visible)
        if len(planes) != len(self.rgb_weights) or any(
            p.shape != self.shape for p in planes
        ):
            raise ValueError("Image shape differs from weights")
        self.acc.fill(0)
        for image, weight in zip(planes, self.rgb_weights):
            cv.multiply(image, weight, dst=self.work, dtype=cv.CV_32F)
            cv.add(self.acc, self.work, dst=self.acc)
        np.clip(self.acc, 0, 255, out=self.acc)
        return self.acc.astype(np.uint8)

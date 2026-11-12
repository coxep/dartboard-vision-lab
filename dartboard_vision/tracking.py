"""Pyramidal temporal ECC after joint calibration; all lens terms stay frozen.

Tracking uses a small rectified grayscale image and compiled OpenCV gradients.
Output RGB is sampled once from the raw camera with the updated combined map.
Temporal references are per-camera empty-board images in the shared board plane;
there is no per-frame joint calibration or new distortion estimation.
"""

import time

import cv2 as cv
import numpy as np

from .geometry import BOARD, R
from .photometry import contrast_normalized


class PyramidTracker:
    def __init__(self, frame, model, size=300, iterations=5):
        self.model = dict(model)
        self.H = np.asarray(model["inverse_homography_normalized"], dtype=float).copy()
        self.size = size
        self.iterations = iterations
        self.coordinates = {}
        self.maps = {}
        self.reference, self.reference_valid = self.rectify(frame, size, gray=True)
        self.reference = contrast_normalized(self.reference, size / 900).astype(
            np.float32
        )
        yy, xx = np.mgrid[:size, :size]
        radius = np.hypot(xx * 900 / size - 450, yy * 900 / size - 450) / BOARD.ppm
        self.roi = radius < 230.0
        self.N = np.array(
            [[900 / size / R, 0, -450 / R], [0, 900 / size / R, -450 / R], [0, 0, 1.0]]
        )
        self.probes = np.float32(
            [[150, 150], [750, 150], [750, 750], [150, 750], [450, 450]]
        )

    def mapping(self, size, H=None):
        H = self.H if H is None else H
        if size in self.maps and np.array_equal(self.maps[size][0], H):
            return self.maps[size][1:]
        if size not in self.coordinates:
            y, x = np.mgrid[:size, :size]
            self.coordinates[size] = np.stack(
                [(x * 900 / size - 450) / R, (y * 900 / size - 450) / R], axis=-1
            ).astype(np.float32)
        uv = cv.perspectiveTransform(self.coordinates[size], H)
        x, y = cv.split(uv)
        r2 = x * x + y * y
        lam = self.model["division_lambda"]
        root = np.sqrt(np.maximum(1 - 4 * lam * r2, 1e-8))
        factor = 2 / (1 + root)
        x = x * factor
        y = y * factor
        rd2 = x * x + y * y
        beta = self.model.get("residual_radial4", 0.0)
        p1, p2 = self.model.get("tangential", [0.0, 0.0])
        extra = 1 + beta * rd2 * rd2
        cross = x * y
        u = (x * extra + 2 * p1 * cross + p2 * (rd2 + 2 * x * x)) * self.model[
            "normalization_radius_px"
        ] + self.model["distortion_centre_px"][0]
        v = (y * extra + p1 * (rd2 + 2 * y * y) + 2 * p2 * cross) * self.model[
            "normalization_radius_px"
        ] + self.model["distortion_centre_px"][1]
        self.maps[size] = (H.copy(), u.astype(np.float32), v.astype(np.float32))
        return self.maps[size][1:]

    def rectify(self, frame, size=900, gray=False, H=None):
        mx, my = self.mapping(size, H)
        source = (
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if gray and frame.ndim == 3 else frame
        )
        image = cv.remap(source, mx, my, cv.INTER_LINEAR)
        valid = (
            (mx >= 1)
            & (mx < frame.shape[1] - 2)
            & (my >= 1)
            & (my < frame.shape[0] - 2)
        )
        return image, valid

    def update(self, frame):
        started = time.perf_counter()
        seed = self.H.copy()
        observed, valid = self.rectify(frame, self.size, gray=True)
        observed = contrast_normalized(observed, self.size / 900).astype(np.float32)
        mask = self.roi & valid & self.reference_valid

        def correlation(image, keep):
            a = self.reference[keep].astype(float)
            b = image[keep].astype(float)
            a -= a.mean()
            b -= b.mean()
            return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

        if mask.sum() < 0.5 * self.roi.sum():
            return dict(
                accepted=False,
                reason="insufficient visible board support",
                needs_reacquisition=True,
                elapsed_ms=1000 * (time.perf_counter() - started),
            )
        before = correlation(observed, mask)
        global_before = before
        support_ref = mask
        W = np.eye(3, dtype=np.float32)
        levels = []
        try:
            for ratio in (0.5, 1.0):
                S = np.diag([ratio, ratio, 1.0])
                shape = (round(self.size * ratio),) * 2
                local = (S @ W @ np.linalg.inv(S)).astype(np.float32)
                target = cv.resize(self.reference, shape, interpolation=cv.INTER_AREA)
                current = cv.resize(observed, shape, interpolation=cv.INTER_AREA)
                keep = (
                    cv.resize(
                        mask.astype(np.uint8), shape, interpolation=cv.INTER_NEAREST
                    )
                    * 255
                )
                # After a coarse alignment, large residuals can exclude darts/occlusions.
                if ratio == 1.0:
                    aligned = cv.warpPerspective(
                        observed,
                        W,
                        (self.size, self.size),
                        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                    )
                    residual = abs(self.reference - aligned)
                    v = residual[mask]
                    median = float(np.median(v))
                    mad = 1.4826 * float(np.median(abs(v - median)))
                    good = mask & (residual <= max(0.8, median + 4 * mad))
                    if good.sum() >= 0.5 * mask.sum():
                        support_ref = good
                        keep = cv.warpPerspective(
                            np.uint8(good) * 255,
                            W,
                            (self.size, self.size),
                            flags=cv.INTER_NEAREST,
                        )
                cc, local = cv.findTransformECC(
                    target,
                    current,
                    local,
                    cv.MOTION_HOMOGRAPHY,
                    (
                        cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS,
                        self.iterations,
                        1e-4,
                    ),
                    inputMask=keep,
                    gaussFiltSize=5,
                )
                W = np.linalg.inv(S) @ local @ S
                levels.append(dict(scale=ratio, ecc=float(cc)))
            aligned = cv.warpPerspective(
                observed,
                W,
                (self.size, self.size),
                flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
            )
            global_after = correlation(aligned, mask)
            before = correlation(observed, support_ref)
            after = correlation(aligned, support_ref)
            full = (
                np.diag([900 / self.size, 900 / self.size, 1.0])
                @ W
                @ np.diag([self.size / 900, self.size / 900, 1.0])
            )
            moved = cv.perspectiveTransform(
                self.probes.reshape(-1, 1, 2), full
            ).reshape(-1, 2)
            correction = float(np.linalg.norm(moved - self.probes, axis=1).max())
            candidate = seed @ self.N @ W @ np.linalg.inv(self.N)
            candidate /= candidate[2, 2]
            accepted = bool(
                np.all(np.isfinite(candidate))
                and after >= before - 1e-4
                and correction <= 8.0
            )
            if accepted:
                self.H = candidate
            reason = (
                "local appearance objective accepted"
                if accepted
                else "local correction requires recovery"
            )
        except cv.error:
            accepted = False
            after = before
            global_after = global_before
            correction = None
            reason = "ECC did not converge"
        return dict(
            accepted=accepted,
            reason=reason,
            needs_reacquisition=not accepted,
            correlation_before=before,
            correlation_after=after,
            global_correlation_before=global_before,
            global_correlation_after=global_after,
            retained_support_fraction=float(support_ref.sum() / mask.sum()),
            maximum_correction_px=correction,
            levels=levels,
            elapsed_ms=1000 * (time.perf_counter() - started),
            lens_fixed=True,
            per_frame_joint_fit=False,
        )

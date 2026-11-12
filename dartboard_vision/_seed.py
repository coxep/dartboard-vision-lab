"""Internal seed-map state for joint alignment, without capture or tracking."""

from functools import lru_cache

import cv2 as cv
import numpy as np

from .geometry import BOARD, R


@lru_cache(maxsize=3)
def points(size):
    yy, xx = np.mgrid[:size, :size]
    scale = BOARD.size / size
    return np.c_[
        (xx.ravel() * scale - BOARD.center[0]) / R,
        (yy.ravel() * scale - BOARD.center[1]) / R,
        np.ones(size * size),
    ]


class SeedRectifier:
    def __init__(self, frame, model, use_klt=False):
        if use_klt:
            raise ValueError(
                "Joint seed rectifiers do not perform optical-flow tracking"
            )
        self.model = model
        self.J = np.array(model["inverse_homography_normalized"])
        self.lam = model["division_lambda"]
        self.lam2 = model.get("division_lambda2", 0.0)
        self.c = np.array(model["distortion_centre_px"])
        self.s = model["normalization_radius_px"]
        maximum = (
            np.linalg.norm(np.maximum(self.c, np.array(frame.shape[1::-1]) - self.c))
            / self.s
            + 0.02
        )
        self.radial_d = np.linspace(0, maximum, 65537)
        r2 = self.radial_d**2
        self.radial_u = self.radial_d / (1 + self.lam * r2 + self.lam2 * r2 * r2)
        if not np.all(np.diff(self.radial_u) > 0):
            raise ValueError("Radial map folds within sensor")

    def rectify(self, frame, size=900, J=None):
        J = self.J if J is None else J
        q = points(size) @ J.T
        u = q[:, :2] / q[:, 2:]
        ru = np.linalg.norm(u, axis=1)
        rd = np.interp(ru, self.radial_u, self.radial_d, right=self.radial_d[-1] + 1)
        xy = (u * (rd / np.maximum(ru, 1e-12))[:, None] * self.s + self.c).reshape(
            size, size, 2
        )
        plane = cv.remap(
            frame,
            xy[:, :, 0].astype(np.float32),
            xy[:, :, 1].astype(np.float32),
            cv.INTER_LINEAR,
        )
        valid = (
            (xy[:, :, 0] >= 0)
            & (xy[:, :, 0] < frame.shape[1] - 1)
            & (xy[:, :, 1] >= 0)
            & (xy[:, :, 1] < frame.shape[0] - 1)
        )
        return plane, valid

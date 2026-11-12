"""Metric board geometry and canonical painted-region labels.

Coordinates use x right, y down; angles run clockwise from twelve o'clock.
The default profile is the strip board used by the current real-data pipeline.
"""

import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np

DEFAULT_RADII = (97.4, 107.4, 160.0, 170.0)
NUMBERS = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
ANCHOR_SPOKES = (0, 5, 10, 15)


@dataclass
class Board:
    size: int = 900
    radii: tuple = DEFAULT_RADII

    @property
    def center(self):
        return np.array([self.size / 2, self.size / 2], np.float32)

    @property
    def ppm(self):
        return 0.36 * self.size / self.radii[-1]

    def point(self, radius, angle):
        """Angle in degrees, clockwise from 12 o'clock."""
        a = np.deg2rad(angle)
        return self.center + radius * self.ppm * np.array([np.sin(a), -np.cos(a)])

    def landmarks(self):
        return np.float32(
            [self.point(r, -9 + 18 * s) for r in self.radii for s in range(20)]
        )

    def anchors(self):
        return self.landmarks()[60 + np.array(ANCHOR_SPOKES)]

    def mask(self, factor=1.0):
        mask = np.zeros((self.size, self.size), np.uint8)
        cv.circle(
            mask,
            tuple(self.center.astype(int)),
            round(self.radii[-1] * self.ppm * factor),
            255,
            -1,
        )
        return mask

    def grid(self, image):
        out = image.copy()
        center = tuple(self.center.astype(int))
        for r in (6.35, 15.9, *self.radii):
            cv.circle(out, center, round(r * self.ppm), (100, 220, 255), 1, cv.LINE_AA)
        for s in range(20):
            a = -9 + 18 * s
            cv.line(
                out,
                tuple(self.point(15.9, a).astype(int)),
                tuple(self.point(self.radii[-1], a).astype(int)),
                (100, 220, 255),
                1,
                cv.LINE_AA,
            )
            p = self.point(self.radii[-1] * 1.12, 18 * s).astype(int)
            text(out, str(NUMBERS[s]), (p[0] - 12, p[1] + 6), scale=0.55)
        return out

    def score(self, pixel, uncertainty_mm=2.0):
        x, y = (np.asarray(pixel) - self.center) / self.ppm
        radius = math.hypot(x, y)
        angle = math.degrees(math.atan2(x, -y)) % 360
        segment = int(NUMBERS[int((angle + 9) // 18) % 20])
        ti, to, di, do = self.radii
        if radius > do:
            label, value = "MISS", 0
        elif radius <= 6.35:
            label, value = "DB", 50
        elif radius <= 15.9:
            label, value = "SB", 25
        elif di <= radius <= do:
            label, value = f"D{segment}", 2 * segment
        elif ti <= radius <= to:
            label, value = f"T{segment}", 3 * segment
        else:
            label, value = f"S{segment}", segment
        # Close calls require review; this is a tolerance, not a statistical CI.
        near_ring = min(abs(radius - r) for r in (6.35, 15.9, *self.radii))
        boundary_angle = abs(((angle + 9) % 18) - 9)
        # Distance to closest boundary: center angles are 0,18,...; boundaries +/-9.
        near_spoke = radius * math.sin(math.radians(9 - boundary_angle))
        review = near_ring <= uncertainty_mm or (
            15.9 < radius <= do and near_spoke <= uncertainty_mm
        )
        return dict(
            label=label,
            score=value,
            x_mm=float(x),
            y_mm=float(y),
            radius_mm=radius,
            boundary_review=bool(review),
        )


def text(image, label, at=(12, 26), color=(255, 255, 255), scale=0.65):
    cv.putText(
        image,
        label,
        tuple(map(int, at)),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        label,
        tuple(map(int, at)),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv.LINE_AA,
    )


BOARD = Board()
R = BOARD.radii[-1] * BOARD.ppm


def geometry_fields(size, board=BOARD, wire_band_mm=None):
    # Exactly match JointTracker.rectify's x*900/size pixel sampling convention.
    yy, xx = np.mgrid[:size, :size]
    x = (xx * board.size / size - board.center[0]) / board.ppm
    y = (yy * board.size / size - board.center[1]) / board.ppm
    radius = np.hypot(x, y)
    angle = np.arctan2(x, -y)
    sector = np.floor(((angle + np.pi / 20) % (2 * np.pi)) / (np.pi / 10)).astype(int)
    labels = sector % 2
    ti, to, di, do = board.radii
    band = ((radius >= ti) & (radius <= to)) | ((radius >= di) & (radius <= do))
    labels[band] += 2
    labels[radius <= 15.9] = 3
    labels[radius <= 6.35] = 2
    labels[radius > do] = 0
    ring_distance = np.min(
        abs(radius[..., None] - np.array([6.35, 15.9, ti, to, di, do])), axis=-1
    )
    da = (angle + np.pi / 20) % (np.pi / 10)
    spoke = radius * np.sin(np.minimum(da, np.pi / 10 - da))
    spoke[(radius < 15.9) | (radius > do)] = np.inf
    distance = np.minimum(ring_distance, spoke)
    roi = (radius > 25) & (radius < do + 4.0)
    if wire_band_mm is not None:
        roi &= distance > wire_band_mm
    # A single contrast code lets the compiled ECC solver compare all four classes.
    # It is an encoding, not a fit to the board's observed RGB colours.
    codes = np.array([-1.0, 1.0, -0.35, 0.35], np.float32)
    reference = cv.GaussianBlur(codes[labels], (0, 0), max(0.6, 1.2 * size / 300))
    return labels, distance, roi, reference

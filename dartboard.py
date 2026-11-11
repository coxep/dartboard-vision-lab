#!/usr/bin/env python3
"""Three-camera dartboard rectification and experimental shaft-intersection scoring.

Python 3.10+, numpy, opencv-python (with GUI support). See README.md.
No cameras are opened until this script is run with `calibrate` or `run`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment


NUMBERS = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                    3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
# Nominal radii in mm; measure your board if wire/colour boundaries differ.
DEFAULT_RADII = (99.0, 107.0, 162.0, 170.0)
ANCHOR_SPOKES = [0, 5, 10, 15]
ANCHOR_LABELS = ["5 / 20 (upper left)", "13 / 6 (upper right)",
                 "17 / 3 (lower right)", "8 / 11 (lower left)"]


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
        return np.float32([self.point(r, -9 + 18 * s)
                           for r in self.radii for s in range(20)])

    def anchors(self):
        return self.landmarks()[60 + np.array(ANCHOR_SPOKES)]

    def mask(self, factor=1.0):
        mask = np.zeros((self.size, self.size), np.uint8)
        cv.circle(mask, tuple(self.center.astype(int)),
                  round(self.radii[-1] * self.ppm * factor), 255, -1)
        return mask

    def grid(self, image):
        out = image.copy()
        center = tuple(self.center.astype(int))
        for r in (6.35, 15.9, *self.radii):
            cv.circle(out, center, round(r * self.ppm), (100, 220, 255), 1, cv.LINE_AA)
        for s in range(20):
            a = -9 + 18 * s
            cv.line(out, tuple(self.point(15.9, a).astype(int)),
                    tuple(self.point(self.radii[-1], a).astype(int)),
                    (100, 220, 255), 1, cv.LINE_AA)
            p = self.point(self.radii[-1] * 1.12, 18 * s).astype(int)
            text(out, str(NUMBERS[s]), (p[0] - 12, p[1] + 6), scale=.55)
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
            15.9 < radius <= do and near_spoke <= uncertainty_mm)
        return dict(label=label, score=value, x_mm=float(x), y_mm=float(y),
                    radius_mm=radius, boundary_review=bool(review))


def text(image, label, at=(12, 26), color=(255, 255, 255), scale=.65):
    cv.putText(image, label, tuple(map(int, at)), cv.FONT_HERSHEY_SIMPLEX,
               scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, label, tuple(map(int, at)), cv.FONT_HERSHEY_SIMPLEX,
               scale, color, 1, cv.LINE_AA)


def project(points, H):
    return cv.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)


def warp(frame, H, board):
    return cv.warpPerspective(frame, H, (board.size, board.size))


def seed_homography(clicks, board):
    points = np.float32(clicks)
    if (points.shape != (4, 2) or not cv.isContourConvex(points.reshape(-1, 1, 2))
            or abs(cv.contourArea(points)) < 100):
        raise ValueError("Choose four well-separated points in the illustrated order.")
    H = cv.getPerspectiveTransform(points, board.anchors())
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Degenerate calibration points.")
    return H


def adaptive_block_size(shape):
    """A wedge-scale neighborhood, rather than a board-texture neighborhood."""
    return max(3, round(min(shape[:2]) * .5) | 1)


def binarize(gray, block_size=None):
    """Preserve solid dark/light wedges under slowly changing illumination.

    At 1920x1080 the default block is 541; at 900x900 it is 451. A 31px
    neighborhood estimates the local wedge itself and enhances texture instead.
    The output comes only from observed pixels, never from model-filled sectors.
    """
    block_size = adaptive_block_size(gray.shape) if block_size is None else int(block_size)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("Adaptive threshold block_size must be odd and >= 3")
    smooth = cv.GaussianBlur(gray, (5, 5), 1.2)
    return cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, block_size, 5)



def board_color_masks(frame):
    """Observed red/green ring segments in HSV; keep colours separate for corners.

    Saturation gates prevent black/white wedges (whose hue is unreliable) from
    entering these masks. Red spans both ends of OpenCV's 0..179 hue range.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red = (cv.inRange(hsv, (0, 65, 30), (12, 255, 255))
           | cv.inRange(hsv, (170, 65, 30), (179, 255, 255)))
    green = cv.inRange(hsv, (35, 55, 30), (110, 255, 255))
    return dict(red=cv.medianBlur(red, 3), green=cv.medianBlur(green, 3))



def board_feature_edges(frame):
    """Observed wedge + colour-boundary edges for the full geometry check."""
    images = [binarize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)), *board_color_masks(frame).values()]
    edges = np.zeros(frame.shape[:2], np.uint8)
    for image in images:
        edges |= cv.Canny(image, 50, 150)
    return edges


def ellipse_normalizer(ellipse):
    """Source ellipse -> unit circle, preserving handedness."""
    center, axes, angle = ellipse
    angle = math.radians(angle)
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    A = np.diag(2 / np.asarray(axes)) @ R.T
    E = np.eye(3)
    E[:2, :2], E[:2, 2] = A, -A @ center
    return E


def nominal_homography(board, K, position, look_at=(0, 0, 0)):
    """Approximate camera geometry -> preliminary image-to-board projection.

    Coordinates: board z=0, x right, y up, z toward the player. Camera roll=0.
    This is a PRIOR, never the sampled true pose in a Monte Carlo trial.
    """
    center = np.asarray(position, np.float64)
    forward = np.asarray(look_at, np.float64) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.array([right, down, forward])
    camera = np.asarray(K) @ np.c_[R, -R @ center]
    plane = np.array([[1 / board.ppm, 0, -board.center[0] / board.ppm],
                      [0, -1 / board.ppm, board.center[1] / board.ppm],
                      [0, 0, 0], [0, 0, 1.]])
    return np.linalg.inv(camera @ plane)


def refine_edge_template(frame, seed, board):
    """Iteratively align the entire metric wire pattern to observed image edges.

    Optimize an 8-parameter correction to the PRIOR homography. Concentric
    circles and equally spaced straight spokes are fixed in the shared plane.
    The result still warps actual input pixels; no synthetic output replaces them.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    radius = board.radii[-1]
    samples = [board.point(r, a) for r in (6.35, 15.9, *board.radii)
               for a in np.linspace(0, 360, 180, endpoint=False)]
    samples += [board.point(r, -9 + 18 * s) for s in range(20)
                for r in np.linspace(20, radius, 60)]
    normalized = (np.asarray(samples) - board.center) / (radius * board.ppm)
    points = np.c_[normalized, np.ones(len(normalized))]
    S = np.array([[radius * board.ppm, 0, board.center[0]],
                  [0, radius * board.ppm, board.center[1]], [0, 0, 1.]])
    base = np.linalg.inv(seed) @ S

    def correction(p):
        return np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [p[6], p[7], 1.]])

    p = np.zeros(8)
    final_residual = None
    for scale in (.25, .5, 1.):
        small = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        smooth = cv.GaussianBlur(small, (5, 5), 1.1)
        edges = cv.Canny(smooth, 25, 70)
        distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 5)
        distance = cv.GaussianBlur(distance, (3, 3), .5).astype(np.float64)
        hp = points @ (base @ correction(p)).T
        xy = hp[:, :2] / hp[:, 2:]
        # Keep residual length fixed, exclude boundaries cropped at initialization.
        visible = ((xy[:, 0] * scale > 3) & (xy[:, 0] * scale < small.shape[1]-4)
                   & (xy[:, 1] * scale > 3) & (xy[:, 1] * scale < small.shape[0]-4))
        active = points[visible]
        if len(active) < len(points) * .65:
            raise ValueError("Too little predicted board is visible for template refinement")

        def residual(parameters):
            q = active @ (base @ correction(parameters)).T
            x, y = (q[:, :2] / q[:, 2:] * scale).T
            x = np.clip(x, 0, small.shape[1]-1.001)
            y = np.clip(y, 0, small.shape[0]-1.001)
            xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
            dx, dy = x-xi, y-yi
            # Float64 bilinear sampling keeps finite-difference gradients smooth.
            value = ((1-dx)*(1-dy)*distance[yi, xi] + dx*(1-dy)*distance[yi, xi+1]
                     + (1-dx)*dy*distance[yi+1, xi] + dx*dy*distance[yi+1, xi+1])
            return np.r_[value, .05 * parameters]

        fit = least_squares(residual, p, bounds=(-.4, .4), loss="soft_l1",
                            f_scale=1.5, max_nfev=60, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        p = fit.x
        final_residual = residual(p)[:-8] / scale
    refined = S @ np.linalg.inv(base @ correction(p))
    info = dict(method="prior plane projection + full wire-template optimization",
                template_median_source_px=float(np.median(final_residual)),
                template_p90_source_px=float(np.percentile(final_residual, 90)),
                correction=p.tolist())
    if info["template_median_source_px"] > 3 or info["template_p90_source_px"] > 8:
        raise ValueError("Plane template fit residual is too large")
    return refined, info


def ideal_board_template(board):
    """Geometry-only grayscale template; no renderer texture or camera truth."""
    yy, xx = np.mgrid[:board.size, :board.size]
    x, y = (xx-board.center[0])/board.ppm, (yy-board.center[1])/board.ppm
    radius = np.hypot(x, y)
    angle = np.degrees(np.arctan2(x, -y)) % 360
    sector = ((angle+9)//18).astype(int) % 20
    template = np.where(sector % 2 == 0, 41, 191).astype(np.uint8)
    ti, to, di, do = board.radii
    band = ((radius >= ti) & (radius <= to)) | ((radius >= di) & (radius <= do))
    template[band & (sector % 2 == 0)] = 97
    template[band & (sector % 2 == 1)] = 106
    template[radius <= 15.9] = 106
    template[radius <= 6.35] = 97
    template[radius > do] = 30
    for r in (6.35, 15.9, *board.radii):
        cv.circle(template, tuple(board.center.astype(int)), round(r*board.ppm), 160, 1, cv.LINE_AA)
    for s in range(20):
        cv.line(template, tuple(board.point(15.9, -9+18*s).astype(int)),
                tuple(board.point(do, -9+18*s).astype(int)), 160, 1, cv.LINE_AA)
    return template


def refine_plane_template(frame, seed, board):
    """Coarse-to-fine whole-pattern registration, followed by metric edge polish."""
    template = ideal_board_template(board)
    preliminary = warp(frame, seed, board)
    observed = cv.cvtColor(preliminary, cv.COLOR_BGR2GRAY)
    valid = warp(np.full(frame.shape[:2], 255, np.uint8), seed, board)
    mask = np.uint8((valid > 250) & (board.mask(.97) > 0))*255
    def register(initial, affine_first):
        W = initial.copy()  # ECC maps TEMPLATE pixels to OBSERVED pixels.
        correlation = None
        for scale in (.25, .5, 1.):
            shape = (round(board.size*scale), round(board.size*scale))
            T = np.diag([scale, scale, 1.]).astype(np.float32)
            small_W = T @ W @ np.linalg.inv(T)
            ref = cv.resize(template, shape, interpolation=cv.INTER_AREA)
            image = cv.resize(observed, shape, interpolation=cv.INTER_AREA)
            roi = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 1e-5)
            if scale == .25 and affine_first:
                try:
                    _, affine = cv.findTransformECC(ref, image, small_W[:2].copy(), cv.MOTION_AFFINE,
                                                    criteria, inputMask=roi, gaussFiltSize=5)
                    small_W = np.vstack([affine, [0, 0, 1]]).astype(np.float32)
                except cv.error:
                    pass
            correlation, small_W = cv.findTransformECC(ref, image, small_W, cv.MOTION_HOMOGRAPHY,
                                                       criteria, inputMask=roi, gaussFiltSize=5)
            W = np.linalg.inv(T) @ small_W @ T
        return W, correlation

    identity = np.eye(3, dtype=np.float32)
    starts = [(identity, True), (identity, False)]
    # A separate translation estimate supplies a bounded recovery when the coarse
    # affine iteration diverges. No true camera pose or hit location is consulted.
    window = cv.createHanningWindow((board.size, board.size), cv.CV_32F)
    shift, response = cv.phaseCorrelate(template.astype(np.float32), observed.astype(np.float32), window)
    if response > .05 and max(abs(shift[0]), abs(shift[1])) < board.size*.25:
        translated = identity.copy()
        translated[:2, 2] = shift
        starts.extend([(translated, True), (translated, False)])
    # The projected bull is an observed origin cue. It improves the translation
    # seed without assigning any of the repeated treble/double ring boundaries.
    hsv = cv.cvtColor(preliminary, cv.COLOR_BGR2HSV)
    red = cv.inRange(hsv, (0, 110, 45), (12, 255, 255)) | cv.inRange(hsv, (170, 110, 45), (179, 255, 255))
    yy, xx = np.mgrid[:board.size, :board.size]
    red[np.hypot(xx-board.center[0], yy-board.center[1]) > board.size*.18] = 0
    contours, _ = cv.findContours(red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    expected_area = math.pi*(6.35*board.ppm)**2
    bulls = []
    for c in contours:
        area = cv.contourArea(c)
        if len(c) < 5 or not expected_area*.2 < area < expected_area*3:
            continue
        ellipse = cv.fitEllipse(c)
        if min(ellipse[1])/max(ellipse[1]) < .25:
            continue
        fill = area/(math.pi*np.prod(ellipse[1])/4)
        if fill > .65:
            bulls.append((abs(math.log(area/expected_area)), np.array(ellipse[0])))
    if bulls:
        origin = min(bulls, key=lambda b:b[0])[1]
        translated = identity.copy()
        translated[:2, 2] = origin-board.center
        starts.extend([(translated, True), (translated, False)])
    failures = []
    for attempt, (initial, affine_first) in enumerate(starts):
        try:
            W, correlation = register(initial, affine_first)
            if correlation is None or correlation < .85:
                raise ValueError("Whole-board template correlation is too low")
            up = project([board.center, board.center + [0, -100]], W)
            vector = up[1]-up[0]
            rotation = math.degrees(math.atan2(vector[0], -vector[1]))
            if abs(rotation) > 12:
                raise ValueError("Template fit violates upright-camera orientation prior")
            candidate = np.linalg.inv(W) @ seed
            H, info = refine_edge_template(frame, candidate, board)
            info.update(method="nominal plane projection + multiscale ECC + metric edge polish",
                        ecc_correlation=float(correlation), initialization_attempt=attempt+1)
            return H, info
        except (cv.error, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    raise ValueError("Whole-board template registration failed all bounded starts: " + failures[-1])


def automatic_homography(frame, board, nominal_H=None, K=None):
    """Rectify actual pixels; optional known K enables pose/association hypotheses."""
    bootstrap = None
    if nominal_H is None:
        # The colour/ellipse fit is an INITIALIZER, not a final calibration.
        # Real boards expose the sparse-fit bias; refine their observed pixels too.
        nominal_H, bootstrap = ellipse_homography(frame, board)
    candidates, pose_search = [], None
    if K is not None:
        from pose_alignment import seeded_pose_hypotheses
        try:
            hypotheses, pose_search = seeded_pose_hypotheses(frame, board, K, nominal_H, ideal_board_template(board))
            candidates = [(H, info) for H, info in hypotheses if info["correlation"] > .75]
        except (cv.error, ValueError, np.linalg.LinAlgError) as exc:
            pose_search = dict(reason=str(exc), hypotheses=0)
    # Retain the existing full-pattern initializer as an explicit fallback.
    candidates.append((nominal_H, None))
    failures = []
    for initialization, pose_info in candidates:
        try:
            H, info = refine_plane_template(frame, initialization, board)
            # Verify orientation against the ORIGINAL belief, not just the latest
            # hypothesis, so a periodic sector shift cannot reset the prior.
            up = project([board.center, board.center+[0, -100]], nominal_H@np.linalg.inv(H))
            v = up[1]-up[0]
            if abs(math.degrees(math.atan2(v[0], -v[1]))) > 12:
                raise ValueError("Final alignment violates original upright pose belief")
            _, corners = harris_refine(warp(frame, H, board), np.eye(3), board, gate=6)
            if not corners["refined"]:
                raise ValueError("Plane-template fit did not pass the Harris geometry check: "+corners.get("reason", "unknown"))
            # Sparse wire-edge corners can bias a dense fit; use for validation.
            corners["harris_role"] = "validation only; dense plane-template H retained"
            corners["matched_source_px"] = project(corners["matched_source_px"], np.linalg.inv(H)).tolist()
            corners.update(info)
            corners["binary_template_support"] = binary_template_support(frame, H, board)
            if corners["binary_template_support"] < .6:
                raise ValueError("Final plane-template edge support is too low")
            corners["edge_inputs"] = ["adaptive wedges", "HSV red", "HSV green"]
            corners["threshold_block_size_in_plane"] = adaptive_block_size((board.size,board.size))
            corners["ellipse_bootstrap"] = bootstrap
            corners["pose_search"] = pose_search
            corners["pose_initialization"] = pose_info
            corners["pose_initializer_used"] = pose_info is not None
            corners["failed_initializations"] = failures
            return H, corners
        except (cv.error, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    raise ValueError("All pose/template hypotheses failed: "+failures[-1])


def binary_template_support(frame, H, board):
    edges = board_feature_edges(warp(frame, H, board))
    distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 3)
    samples = [board.point(r, a) for r in board.radii for a in np.linspace(0, 360, 360, endpoint=False)]
    samples += [board.point(r, -9 + 18 * s) for s in range(20) for r in np.linspace(20, board.radii[-1], 100)]
    samples = np.rint(samples).astype(int)
    return float(np.mean(distance[samples[:, 1], samples[:, 0]] <= 3 * board.size / 900))


def ellipse_homography(frame, board):
    """Coarse initializer, not final acceptance, for a red/green upright board.

    Outer double ellipse + projected bull center determine a metric plane
    rectification up to rotation. Camera-up supplies the remaining orientation.
    Colour only locates the rings; adaptive binary Harris corners refine the fit.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    colour = (cv.inRange(hsv, (0, 65, 30), (12, 255, 255))
              | cv.inRange(hsv, (170, 65, 30), (179, 255, 255))
              | cv.inRange(hsv, (35, 55, 30), (110, 255, 255)))
    closed = cv.morphologyEx(colour, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        area = abs(cv.contourArea(contour))
        if len(contour) < 20 or area < .025 * frame.shape[0] * frame.shape[1]:
            continue
        ellipse = cv.fitEllipse(contour)
        _, axes, _ = ellipse
        if min(axes) < 25 or min(axes) / max(axes) < .12:
            continue
        ellipse_area = math.pi * axes[0] * axes[1] / 4
        ratio = area / ellipse_area
        normalized = project(contour.reshape(-1, 2), ellipse_normalizer(ellipse))
        residual = np.abs(np.linalg.norm(normalized, axis=1) - 1)
        # A cropped annulus is an open arc, not a filled elliptical contour.
        # Its contour contains BOTH ring edges. Fit its outer half explicitly.
        if .03 < ratio < .5 and np.median(residual) < .08:
            p = contour.reshape(-1, 2).astype(np.float32)
            radii = np.linalg.norm(normalized, axis=1)
            outer = p[radii > np.percentile(radii, 55)]
            if len(outer) >= 20:
                ellipse = cv.fitEllipse(outer.reshape(-1, 1, 2))
                contour = outer.reshape(-1, 1, 2)
                ellipse_area = math.pi * ellipse[1][0] * ellipse[1][1] / 4
        if (.03 < ratio < 1.15 and np.median(residual) < .08
                and max(ellipse[1]) < 1.5 * max(frame.shape[:2])):
            candidates.append((ellipse_area, contour, ellipse))
    if not candidates:
        raise ValueError("Automatic board detection failed: no complete red/green ring ellipse")
    # The double ring is the largest closed coloured ring on the board.
    _, contour, ellipse = max(candidates, key=lambda c: c[0])
    points = contour.reshape(-1, 2).astype(np.float32)
    for _ in range(3):
        E = ellipse_normalizer(ellipse)
        residual = np.abs(np.linalg.norm(project(points, E), axis=1) - 1)
        keep = residual < max(.018, 2.5 * float(np.median(residual)))
        if keep.sum() >= 20:
            ellipse = cv.fitEllipse(points[keep].reshape(-1, 1, 2))
    E = ellipse_normalizer(ellipse)
    # Outer bull is a small connected coloured component inside the double ring.
    yy, xx = np.mgrid[:frame.shape[0], :frame.shape[1]]
    local = project(np.c_[xx.ravel(), yy.ravel()], E).reshape(*xx.shape, 2)
    central = np.uint8((np.linalg.norm(local, axis=2) < .48) & (colour > 0)) * 255
    bulls, _ = cv.findContours(central, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    outer_area = math.pi * ellipse[1][0] * ellipse[1][1] / 4
    bulls = [c for c in bulls if len(c) >= 5 and .001 * outer_area < cv.contourArea(c) < .035 * outer_area
             and max(cv.fitEllipse(c)[1]) < .22 * max(ellipse[1])
             and cv.contourArea(c) / (math.pi * np.prod(cv.fitEllipse(c)[1]) / 4) > .7]
    if not bulls:
        raise ValueError("Automatic board detection failed: bull not visible")
    bull = max(bulls, key=cv.contourArea)
    bull_pixel = np.array(cv.fitEllipse(bull)[0])
    b = project([bull_pixel], E)[0]
    if np.linalg.norm(b) >= .5:
        raise ValueError("Detected bull and outer ring are inconsistent")
    # For unit-circle conic diag(1,1,-1), its pole at b gives the vanishing
    # line [b.x,b.y,-1]. Send this line to infinity, then whiten the ellipse.
    P = np.array([[1., 0, -b[0]], [0, 1., -b[1]], [-b[0], -b[1], 1.]])
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.float32(np.c_[np.cos(a), np.sin(a)])
    rectified_ellipse = cv.fitEllipse(project(circle, P).reshape(-1, 1, 2))
    G = ellipse_normalizer(rectified_ellipse) @ P @ E
    projected_up = project([bull_pixel, bull_pixel + [0, -1]], G)
    up = projected_up[1] - projected_up[0]
    theta = -math.atan2(up[0], -up[1])
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta), math.cos(theta), 0], [0, 0, 1.]])
    radius = board.radii[-1] * board.ppm
    S = np.array([[radius, 0, board.center[0]], [0, radius, board.center[1]], [0, 0, 1.]])
    H = S @ R @ G
    H, report = harris_refine(frame, H, board, gate=10)
    if not report["refined"]:
        raise ValueError("Automatic ring fit found, but distributed binary Harris corners did not verify it")
    # Check binary edges against the known ring/spoke template in canonical space.
    edges = board_feature_edges(warp(frame, H, board))
    distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 3)
    samples = [board.point(r, a) for r in board.radii for a in np.linspace(0, 360, 360, endpoint=False)]
    samples += [board.point(r, -9 + 18 * s) for s in range(20) for r in np.linspace(20, board.radii[-1], 100)]
    samples = np.rint(samples).astype(int)
    support = float(np.mean(distance[samples[:, 1], samples[:, 0]] <= 3 * board.size / 900))
    # Return this coarse candidate to automatic_homography for mandatory dense
    # refinement and final geometry validation, even if initial support is low.
    report["binary_template_support"] = support
    report["bootstrap_only"] = True
    return H, report


def harris_refine(frame, seed, board, gate=6.0):
    """Match Harris corners from clean wedges and separate HSV ring masks.

    The union of candidate POINTS retains boundaries lost in black/white alone.

    goodFeaturesToTrack(useHarrisDetector=True) uses the Harris response and
    non-maximum suppression, NOT the default Shi-Tomasi detector.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    roi = cv.warpPerspective(board.mask(1.04), np.linalg.inv(seed),
                             (frame.shape[1], frame.shape[0]), flags=cv.INTER_NEAREST)
    feature_images = dict(wedges=binarize(gray), **board_color_masks(frame))
    detected = []
    counts = {}
    for name, feature_image in feature_images.items():
        candidates = cv.goodFeaturesToTrack(feature_image, maxCorners=3000, qualityLevel=.005,
                                            minDistance=3, mask=roi, blockSize=3,
                                            useHarrisDetector=True, k=.04)
        counts[name] = 0 if candidates is None else len(candidates)
        if candidates is not None:
            detected.append(candidates)
    corners = np.concatenate(detected) if detected else None
    if corners is None:
        return seed, dict(refined=False, reason="No Harris corners")
    corners = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                              (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, .01))
    corners = corners.reshape(-1, 2)
    target = board.landmarks()
    distance = np.linalg.norm(project(corners, seed)[:, None] - target[None], axis=2)
    # Gated global one-to-one association, with explicit unmatched model slots.
    # This avoids the order-dependent correspondences of greedy nearest matching.
    limit = gate * board.size / 900
    costs = np.where(distance.T < limit, distance.T/limit, 1e6)
    costs = np.c_[costs, np.full((len(target), len(target)), 100.)]
    model_ids, detections = linear_sum_assignment(costs)
    pairs = [(int(c), int(t)) for t,c in zip(model_ids,detections)
             if c < len(corners) and distance[c,t] < limit]
    if len(pairs) < 12:
        return seed, dict(refined=False, reason=f"Only {len(pairs)} matched corners")
    c, t = np.array(pairs).T
    H, inliers = cv.findHomography(corners[c], target[t], cv.RANSAC,
                                  3 * board.size / 900)
    if H is None or inliers is None:
        return seed, dict(refined=False, reason="Harris homography failed")
    good = inliers.ravel().astype(bool)
    ids = t[good]
    # Require support spread around the board and across at least three rings.
    enough = (good.sum() >= 12 and len(set(ids % 20)) >= 8
              and len(set(ids // 20)) >= 3 and len(set((ids % 20) // 5)) == 4)
    shift = np.linalg.norm(project(board.anchors(), H @ np.linalg.inv(seed))
                           - board.anchors(), axis=1).max()
    if not enough or not np.isfinite(H).all() or shift > 2 * gate * board.size / 900:
        return seed, dict(refined=False, reason="Harris support/geometry check failed")
    error = np.linalg.norm(project(corners[c[good]], H) - target[ids], axis=1)
    return H, dict(refined=True, inliers=int(good.sum()), corner_candidates=counts,
                   corner_inputs=list(feature_images),
                   fit_median_px=float(np.median(error)),
                   matched_source_px=corners[c[good]].tolist(), matched_target_px=target[ids].tolist())


class Input:
    """USB indices, URLs or video files; grab all before retrieve for rough sync."""
    def __init__(self, sources=None, images=None, width=None, height=None):
        self.captures = []
        self.images = None
        self.sources = list(images if images is not None else sources)
        if images is not None:
            self.images = [cv.imread(str(p)) for p in images]
            if any(f is None for f in self.images):
                raise ValueError("Could not read one of the input images")
            return
        try:
            for source in sources:
                cap = cv.VideoCapture(int(source) if str(source).isdigit() else str(source))
                self.captures.append(cap)
                if not cap.isOpened():
                    raise ValueError(f"Could not open camera/video source {len(self.captures) - 1}")
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this.
                if width:
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            self.close()
            raise

    def read(self):
        if self.images is not None:
            return [f.copy() for f in self.images]
        if not all([cap.grab() for cap in self.captures]):
            raise RuntimeError("Camera disconnected or video ended; scoring stopped")
        frames = []
        for i, cap in enumerate(self.captures):
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Could not retrieve frame from camera {i}")
            frames.append(frame)
        return frames

    def close(self):
        for cap in self.captures:
            cap.release()


class Lens:
    def __init__(self, shape, data=None):
        self.shape = tuple(shape[:2])
        self.data, self.maps = data, None
        if data is not None:
            K = np.asarray(data["K"], np.float64)
            D = np.asarray(data["D"], np.float64)
            if (K.shape != (3, 3) or D.size not in (4, 5, 8, 12, 14)
                    or not np.isfinite(K).all() or not np.isfinite(D).all()):
                raise ValueError("Invalid OpenCV pinhole lens calibration")
            if list(data["image_size"]) != [shape[1], shape[0]]:
                raise ValueError("Lens calibration resolution does not match the stream")
            self.maps = cv.initUndistortRectifyMap(K, D, None, K,
                                                  (shape[1], shape[0]), cv.CV_32FC1)

    def apply(self, frame):
        if frame.shape[:2] != self.shape:
            raise ValueError("Camera resolution changed; recalibrate at the new resolution")
        return frame if self.maps is None else cv.remap(frame, *self.maps, cv.INTER_LINEAR)


def pick_points(frame, board, title):
    scale = min(1., 1000 / frame.shape[1], 780 / frame.shape[0])
    image = cv.resize(frame, None, fx=scale, fy=scale)
    guide = board.grid(np.zeros((board.size, board.size, 3), np.uint8))
    for i, p in enumerate(board.anchors()):
        cv.circle(guide, tuple(p.astype(int)), 12, (0, 0, 255), 2)
        text(guide, str(i + 1), p + [12, -12], color=(0, 100, 255))
    cv.imshow("Click guide: OUTER edge of double ring", cv.resize(guide, (650, 650)))
    points = []

    def mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])

    cv.namedWindow(title)
    cv.setMouseCallback(title, mouse)
    while True:
        canvas = image.copy()
        for i, p in enumerate(points):
            xy = tuple((np.asarray(p) * scale).astype(int))
            cv.circle(canvas, xy, 5, (0, 0, 255), -1)
            text(canvas, str(i + 1), np.array(xy) + [8, -8])
        label = f"Click {len(points) + 1}: {ANCHOR_LABELS[len(points)]}" if len(points) < 4 else "ENTER: fit; U: undo"
        text(canvas, label)
        text(canvas, "Outer DOUBLE ring wire intersection; U: undo, ESC: cancel", (12, 52), scale=.5)
        cv.imshow(title, canvas)
        key = cv.waitKey(20) & 0xff
        if key == 27:
            raise KeyboardInterrupt
        if key in (ord('u'), 8) and points:
            points.pop()
        if key in (10, 13) and len(points) == 4:
            cv.destroyWindow(title)
            return np.float32(points)


def calibrate(args, inputs):
    board = Board(args.size, tuple(args.radii))
    folder = Path(args.calibration).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    frames = inputs.read()
    prior = json.loads(Path(args.geometry_prior).read_text()) if args.geometry_prior else None
    positions = prior.get("camera_positions_mm", prior.get("positions_mm")) if prior else None
    if positions is not None and len(positions) != len(frames):
        raise ValueError("Provide one nominal camera position per input in the geometry prior")
    if args.intrinsics and len(args.intrinsics) != len(frames):
        raise ValueError("Provide one intrinsics NPZ per camera")
    entries = []
    for i, raw in enumerate(frames):
        data = None
        if args.intrinsics:
            with np.load(args.intrinsics[i], allow_pickle=False) as z:
                data = {k: z[k].tolist() for k in ("K", "D", "image_size")}
        frame = Lens(raw.shape, data).apply(raw)
        if not args.manual:
            seed = None
            if positions is not None:
                focal = math.hypot(raw.shape[1], raw.shape[0]) / (2 * math.tan(math.radians(78 / 2)))
                K = np.asarray(data["K"]) if data else np.array([[focal, 0, raw.shape[1]/2], [0, focal, raw.shape[0]/2], [0, 0, 1.]])
                seed = nominal_homography(board, K, positions[i])
            if args.estimate_radial_distortion:
                from edge_calibration import calibrate_unknown_radial
                if positions is None:
                    focal = math.hypot(raw.shape[1], raw.shape[0]) / (2 * math.tan(math.radians(78 / 2)))
                    K = np.asarray(data["K"]) if data else np.array([[focal,0,raw.shape[1]/2],[0,focal,raw.shape[0]/2],[0,0,1.]])
                H, data, report, frame, circle_overlay = calibrate_unknown_radial(
                    raw,board,K,seed,np.asarray(data["D"]) if data else None)
                Path(args.output).mkdir(parents=True,exist_ok=True)
                cv.imwrite(str(Path(args.output)/f"camera_{i}_postfit_circles.png"),circle_overlay)
            else:
                H, report = automatic_homography(frame, board, seed, K if positions is not None else None)
            print(f"Camera {i}: automatic calibration; {report['inliers']} Harris inliers; template support {report['binary_template_support']:.3f}")
            Path(args.output).mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(Path(args.output) / f"camera_{i}_calibration.png"), board.grid(warp(frame, H, board)))
        else:
            H, report = manual_calibration(frame, board, i)
        # Immutable appearance reference for automatic recovery after camera movement.
        ref = Path(args.calibration).stem + f"_camera_{i}_reference.png"
        if not cv.imwrite(str(folder / ref), frame):
            raise IOError("Could not save calibration reference")
        entries.append(dict(H=H.tolist(), shape=list(raw.shape[:2]), lens=data,
                            reference=ref, harris=report,
                            pose_K=K.tolist() if (positions is not None or args.estimate_radial_distortion) and not args.manual else None))
    config = dict(version=1, size=board.size, radii=board.radii, cameras=entries)
    Path(args.calibration).write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {args.calibration}. Camera order must remain the same when running.")


def manual_calibration(frame, board, i):
    """Optional fallback, invoked ONLY with --manual."""
    while True:
            clicks = pick_points(frame, board, f"Camera {i}: four anchors")
            try:
                H, report = harris_refine(frame, seed_homography(clicks, board), board)
            except ValueError as exc:
                print(exc)
                continue
            print(f"Camera {i}: {report}")
            preview = board.grid(warp(frame, H, board))
            text(preview, "ENTER: accept; R: redo; ESC: cancel")
            if not report["refined"]:
                text(preview, "MANUAL FIT: Harris refinement unavailable", (12, 52), (0, 180, 255))
            cv.imshow("Rectified calibration - check ALL rings and spokes", preview)
            while True:
                key = cv.waitKey(20) & 0xff
                if key in (10, 13, ord('r'), 27):
                    break
            if key == 27:
                raise KeyboardInterrupt
            if key in (10, 13):
                return H, report


class Tracker:
    """Track fixed board features; known K enables seeded physical pose updates."""
    def __init__(self, reference, H, board, K=None):
        self.board, self.H = board, np.asarray(H, np.float64)
        self.reference_H = self.H.copy()
        self.K = None if K is None else np.asarray(K, np.float64)
        self.orb = cv.ORB_create(nfeatures=6000, fastThreshold=12)
        mask = cv.warpPerspective(board.mask(1.03), np.linalg.inv(self.H),
                                  (reference.shape[1], reference.shape[0]), flags=cv.INTER_NEAREST)
        self.kp, self.desc = self.orb.detectAndCompute(cv.cvtColor(reference, cv.COLOR_BGR2GRAY), mask)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        self.valid, self.reason = False, "awaiting registration"
        self.previous_gray, self.flow_points, self.flow_targets = None, None, None

    def physical_fit(self, points, targets):
        """Refine the CURRENT pose belief using tracked/reference correspondences."""
        from pose_alignment import (pose_from_homography, fit_seeded_pose,
                                    homography_from_pose, pose_is_plausible)
        xy = (targets-self.board.center)/self.board.ppm
        objects = np.c_[xy[:, 0], -xy[:, 1], np.zeros(len(xy))].astype(float)
        r, t = pose_from_homography(self.H, self.K, self.board)
        center = (-cv.Rodrigues(r)[0].T@t).ravel()
        fit = fit_seeded_pose(objects, points, self.K, r, t)
        if fit is None:
            raise ValueError("Tracked features did not support a physical pose")
        _, r, t, good, _ = fit
        if good.sum()<16 or good.mean()<.65 or not pose_is_plausible(r,t,center,objects):
            raise ValueError("Tracked pose lacks sufficient support or violates mount belief")
        return homography_from_pose(r,t,self.K,self.board)

    def track(self, frame):
        """Per-frame LK tracking against FIXED canonical targets, not accumulated Hs."""
        if not self.valid or self.flow_points is None:
            return False
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p0 = self.flow_points.reshape(-1, 1, 2)
        p1, status, error = cv.calcOpticalFlowPyrLK(self.previous_gray, gray, p0, None,
                                                  winSize=(21, 21), maxLevel=3,
                                                  criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 30, .01))
        if p1 is None:
            self.valid, self.reason = False, "optical flow lost"
            return False
        back, backward_status, _ = cv.calcOpticalFlowPyrLK(gray, self.previous_gray, p1, None,
                                                          winSize=(21, 21), maxLevel=3)
        if back is None:
            self.valid, self.reason = False, "reverse optical flow lost"
            return False
        good = (status.ravel().astype(bool) & backward_status.ravel().astype(bool)
                & (np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1) < 1.5)
                & (error.ravel() < 30))
        points, targets = p1.reshape(-1, 2)[good], self.flow_targets[good]
        if len(points) < 16:
            self.valid, self.reason = False, "too few tracked board points"
            return False
        H, inliers = cv.findHomography(points, targets, cv.RANSAC, 3 * self.board.size / 900)
        if H is None or inliers is None:
            self.valid, self.reason = False, "tracked homography failed"
            return False
        keep = inliers.ravel().astype(bool)
        hull = cv.convexHull(targets[keep])
        area = math.pi * (self.board.radii[-1] * self.board.ppm) ** 2
        if keep.sum() < 16 or keep.mean() < .65 or cv.contourArea(hull) < .25 * area:
            self.valid, self.reason = False, "tracked points lack reliable board coverage"
            return False
        if self.K is not None:
            try:
                H = self.physical_fit(points[keep], targets[keep])
            except (ValueError, cv.error) as exc:
                self.valid, self.reason = False, str(exc)
                return False
        drift = np.linalg.norm(project(self.board.landmarks(), H @ np.linalg.inv(self.H))
                               - self.board.landmarks(), axis=1).max()
        self.H = H
        self.previous_gray = gray
        self.flow_points, self.flow_targets = points[keep], targets[keep]
        self.reason = f"{int(keep.sum())} optical-flow inliers"
        return bool(drift > .75 * self.board.size / 900)

    def update(self, frame):
        self.valid = False
        kp, desc = self.orb.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
        if desc is None or self.desc is None:
            self.reason = "not enough image features"
            return False
        matches = self.matcher.knnMatch(desc, self.desc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .72 * m[1].distance]
        # One current corner per reference feature.
        good = list({m.trainIdx: m for m in sorted(good, key=lambda m: -m.distance)}.values())
        if len(good) < 20:
            self.reason = "too few reference matches"
            return False
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.kp[m.trainIdx].pt for m in good])
        relative, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)
        if relative is None or inliers is None:
            self.reason = "reference homography failed"
            return False
        keep = inliers.ravel().astype(bool)
        canonical = project(dst[keep], self.reference_H)
        hull_area = abs(cv.contourArea(cv.convexHull(canonical))) if len(canonical) >= 3 else 0
        board_area = math.pi * (self.board.radii[-1] * self.board.ppm) ** 2
        if keep.sum() < 20 or keep.mean() < .35 or hull_area < .25 * board_area:
            self.reason = "reference matches not sufficiently consistent/spread out"
            return False
        candidate = self.reference_H @ relative
        if self.K is not None:
            try:
                candidate = self.physical_fit(src[keep], canonical)
            except (ValueError, cv.error) as exc:
                self.reason = str(exc)
                return False
            # Preserve the physical pose; corners validate without a sparse refit.
            _, report = harris_refine(frame, candidate, self.board)
        else:
            candidate, report = harris_refine(frame, candidate, self.board)
        # Reject horizons crossing the board and implausibly small/offscreen boards.
        perimeter = np.float32([self.board.point(self.board.radii[-1], a)
                                 for a in np.linspace(0, 360, 80, endpoint=False)])
        inverse = np.linalg.inv(candidate)
        denom = np.c_[perimeter, np.ones(len(perimeter))] @ inverse[2]
        visible = project(perimeter, inverse)
        inside = ((visible[:, 0] >= 0) & (visible[:, 0] < frame.shape[1])
                  & (visible[:, 1] >= 0) & (visible[:, 1] < frame.shape[0]))
        if (not np.isfinite(candidate).all() or np.min(denom) * np.max(denom) <= 0
                or inside.mean() < .80 or abs(cv.contourArea(visible)) < 4000):
            self.reason = "too little board is visible or perspective is degenerate"
            return False
        drift = np.linalg.norm(project(self.board.landmarks(), candidate @ np.linalg.inv(self.H))
                               - self.board.landmarks(), axis=1).max()
        moved = drift > .75 * self.board.size / 900
        if moved:
            self.H = candidate
        # Renew tracked features from the immutable reference on every global check.
        self.previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.flow_points = src[keep].copy()
        self.flow_targets = project(self.flow_points, self.H)
        self.valid = True
        self.reason = f"{int(keep.sum())} reference inliers; Harris {'refined' if report['refined'] else 'seed retained'}"
        return bool(moved)


def gray_image(image):
    return cv.GaussianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), (3, 3), 0)


def change_mask(gray, background, mask, threshold):
    result = np.uint8((cv.absdiff(gray, background) >= threshold) & (mask > 0)) * 255
    return cv.morphologyEx(result, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def shaft_line(mask, board):
    """Fit the center axis of ONE new elongated silhouette. Reject clutter."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    minimum = max(12, round(25 * (board.size / 900) ** 2))
    parts = sorted(range(1, count), key=lambda i: stats[i, cv.CC_STAT_AREA], reverse=True)
    parts = [i for i in parts if stats[i, cv.CC_STAT_AREA] >= minimum]
    if not parts:
        return None
    first = parts[0]
    y, x = np.where(labels == first)
    points = np.float32(np.c_[x, y])
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    if eigenvalues[-1] < (8 * board.size / 900) ** 2 or eigenvalues[-1] / max(eigenvalues[0], 1) < 4:
        return None
    vx, vy, px, py = cv.fitLine(points, cv.DIST_HUBER, 0, .01, .01).ravel()
    # A shaft can disappear where its brightness matches a coloured ring. Join
    # disconnected, collinear fragments; reject a second substantial off-axis object.
    normal = np.array([-vy, vx])
    origin = np.array([px, py])
    direction = np.array([vx, vy])
    along = (points - origin) @ direction
    fragments = [points]
    for part in parts[1:]:
        fy, fx = np.where(labels == part)
        fragment = np.float32(np.c_[fx, fy])
        across = np.abs((fragment - origin) @ normal)
        interval = (fragment - origin) @ direction
        gap = max(float(interval.min() - along.max()), float(along.min() - interval.max()), 0.)
        aligned = (np.median(across) <= max(6 * board.size / 900, 2 * math.sqrt(eigenvalues[0]))
                   and gap <= 40 * board.size / 900)
        if aligned:
            fragments.append(fragment)
        elif stats[part, cv.CC_STAT_AREA] > .45 * stats[first, cv.CC_STAT_AREA]:
            return None
    points = np.vstack(fragments)
    vx, vy, px, py = cv.fitLine(points, cv.DIST_HUBER, 0, .01, .01).ravel()
    direction = np.array([vx, vy], np.float64)
    normal = np.array([-vy, vx], np.float64)
    origin = np.array([px, py], np.float64)
    return dict(normal=normal, constant=float(normal @ origin), origin=origin,
                direction=direction, points=points)


def intersect_lines(lines, tolerance=5.0, support_distance=16.0):
    """Robust 2D line intersection, using the third camera as a consistency test."""
    present = [(i, line) for i, line in enumerate(lines) if line is not None]
    if len(present) < 2:
        return None, [], "fewer than two usable shaft axes"
    candidates = []
    for (_, a), (_, b) in itertools.combinations(present, 2):
        A = np.array([a["normal"], b["normal"]])
        if abs(np.linalg.det(A)) < math.sin(math.radians(15)):
            continue
        p = np.linalg.solve(A, [a["constant"], b["constant"]])
        support = []
        residual = 0.
        for i, line in present:
            distance = abs(line["normal"] @ p - line["constant"])
            # Tip must lie on/near the observed silhouette, not far along an extension.
            close = np.linalg.norm(line["points"] - p, axis=1).min() <= support_distance
            if distance <= tolerance and close:
                support.append(i)
                residual += distance
        candidates.append((len(support), -residual, p, support))
    if not candidates:
        return None, [], "shaft axes nearly parallel"
    candidates.sort(key=lambda v: (v[0], v[1]), reverse=True)
    best = candidates[0]
    if best[0] < 2:
        return None, [], "shaft axes do not meet near the observed dart"
    if len(present) >= 3 and best[0] < 3:
        # Three inconsistent lines can produce three plausible pairs: never pick arbitrarily.
        return None, [], "third camera disagrees; manual review required"
    ids = best[3]
    A = np.array([lines[i]["normal"] for i in ids])
    b = np.array([lines[i]["constant"] for i in ids])
    if np.linalg.cond(A) > 10:
        return None, [], "poor intersection geometry"
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    if any(abs(lines[i]["normal"] @ p - lines[i]["constant"]) > tolerance
           or np.linalg.norm(lines[i]["points"] - p, axis=1).min() > support_distance for i in ids):
        return None, [], "intersection failed final consistency check"
    return p, ids, "three views" if len(ids) == 3 else "two views; reduced confidence"


def blend_images(images, masks):
    weights = np.stack(masks).astype(np.float32) / 255
    denominator = weights.sum(axis=0)
    numerator = (np.stack(images).astype(np.float32) * weights[..., None]).sum(axis=0)
    return np.uint8(np.clip(numerator / np.maximum(denominator[..., None], 1), 0, 255))


def save_views(folder, images, blend, board):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if not cv.imwrite(str(folder / f"camera_{i}_rectified.png"), image):
            raise IOError("Could not save rectified image")
    cv.imwrite(str(folder / "aligned_blend.png"), blend)
    cv.imwrite(str(folder / "alignment_grid.png"), board.grid(blend))


def run(args, inputs):
    config_path = Path(args.calibration).resolve()
    config = json.loads(config_path.read_text())
    if config.get("version") != 1:
        raise ValueError("Unsupported calibration version")
    board = Board(config["size"], tuple(config["radii"]))
    entries = config["cameras"]
    if len(entries) != len(inputs.sources):
        raise ValueError("Input count/order must match calibration")
    lenses, trackers = [], []
    for entry in entries:
        lenses.append(Lens(tuple(entry["shape"]) + (3,), entry["lens"]))
        ref = cv.imread(str(config_path.parent / entry["reference"]))
        if ref is None:
            raise ValueError("Missing calibration reference image")
        trackers.append(Tracker(ref, entry["H"], board, entry.get("pose_K")))
    board_mask = board.mask(1.08)
    background, previous, stable_since = None, None, None
    last_check, last_status = -math.inf, ""
    last_event = "Ready: throw ONE dart, then wait for the estimate"
    grid_enabled, reset, paused = True, False, False
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        frames = [lens.apply(f) for lens, f in zip(lenses, inputs.read())]
        now = time.monotonic()
        was_valid = all(t.valid for t in trackers)
        moved = any([t.track(f) for t, f in zip(trackers, frames)])
        if now - last_check >= args.rectify_interval:
            moved = any([t.update(f) for t, f in zip(trackers, frames)]) or moved
            last_check = time.monotonic()
        valid = all(t.valid for t in trackers)
        images = [warp(f, t.H, board) for f, t in zip(frames, trackers)]
        masks = [warp(np.full(f.shape[:2], 255, np.uint8), t.H, board)
                 for f, t in zip(frames, trackers)]
        blend = blend_images(images, masks)
        if args.once:
            save_views(args.output, images, blend, board)
            print(f"Saved rectified views to {args.output}; registration valid={valid}")
            return
        gray = [gray_image(image) for image in images]
        # Any recovered/moved camera invalidates temporal differences. Rebase together.
        if background is None or moved or reset or (valid and not was_valid):
            background = [g.copy() for g in gray]
            previous = gray
            stable_since, reset = None, False
            if moved:
                last_event = "Camera movement corrected; baseline refreshed; throw the NEXT dart"
        status = ("PAUSED: remove darts, then R" if paused else "TRACKING") if valid else "PAUSED: registration lost"
        if status != last_status:
            print(status, "; ".join(f"camera {i}: {t.reason}" for i, t in enumerate(trackers)))
            last_status = status
        deltas = [change_mask(g, b, board_mask, args.diff_threshold) for g, b in zip(gray, background)]
        motion = max(float(np.count_nonzero(change_mask(g, p, board_mask, args.diff_threshold)))
                     / np.count_nonzero(board_mask) for g, p in zip(gray, previous))
        fractions = [np.count_nonzero(d) / np.count_nonzero(board_mask) for d in deltas]
        previous = gray
        if not valid or paused or motion > .002 or max(fractions) > .12:
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        changed = sum(np.count_nonzero(d) >= 30 * (board.size / 900) ** 2 for d in deltas) >= 2
        if valid and not paused and changed and stable_since is not None and time.monotonic() - stable_since >= args.settle_seconds:
            # Check CURRENT alignment again immediately before considering a score.
            moved_at_event = any([t.update(f) for t, f in zip(trackers, frames)])
            last_check = time.monotonic()
            if moved_at_event or not all(t.valid for t in trackers):
                background, stable_since = None, None
                last_event = "Alignment changed: event skipped; throw next dart after TRACKING"
                print(last_event)
                continue
            lines = [shaft_line(d, board) for d in deltas]
            p, ids, reason = intersect_lines(lines, tolerance=5 * board.size / 900,
                                             support_distance=16 * board.size / 900)
            event = dict(timestamp=time.time(), cameras=ids, note=reason, estimated=True)
            if p is not None and np.linalg.norm(p - board.center) <= board.radii[-1] * board.ppm * 1.08:
                event.update(board.score(p, args.boundary_mm))
                event["pixel"] = p.tolist()
                last_event = f"ESTIMATE {event['label']} = {event['score']} points"
                if len(ids) < 3 or event["boundary_review"]:
                    last_event += " - REVIEW"
                cv.circle(blend, tuple(np.rint(p).astype(int)), 7, (0, 0, 255), 2)
            else:
                event["note"] = reason if p is None else "intersection outside observable board area"
                last_event = f"REVIEW: {event['note']}"
            print(last_event)
            with log_path.open("a") as log:
                log.write(json.dumps(event) + "\n")
            # Diagnostic record includes masks and fitted axes for reviewing mistakes.
            event_dir = Path(args.output) / f"event_{time.time_ns()}"
            save_views(event_dir, images, blend, board)
            for i, (d, line) in enumerate(zip(deltas, lines)):
                diagnostic = cv.cvtColor(d, cv.COLOR_GRAY2BGR)
                if line is not None:
                    start = line["origin"] - board.size * line["direction"]
                    end = line["origin"] + board.size * line["direction"]
                    cv.line(diagnostic, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 0), 1)
                cv.imwrite(str(event_dir / f"camera_{i}_shaft.png"), diagnostic)
            background = [g.copy() for g in gray]
            stable_since = None
        if not args.headless:
            tiles = []
            for i, image in enumerate(images):
                tile = board.grid(image) if grid_enabled else image.copy()
                text(tile, f"Camera {i}: {'OK' if trackers[i].valid else 'LOST'}")
                tiles.append(cv.resize(tile, (480, 480)))
            combined = board.grid(blend) if grid_enabled else blend.copy()
            text(combined, status, color=(0, 255, 0) if valid else (0, 0, 255))
            text(combined, last_event[:100], (12, 52), scale=.45)
            text(combined, "Q quit | P pause/remove | R rebase/resume | G grid | S save", (12, 78), scale=.45)
            tiles.append(cv.resize(combined, (480, 480)))
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            dashboard = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
            cv.imshow("Dartboard - alignment and experimental scores", dashboard)
            key = cv.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                return
            if key == ord('g'):
                grid_enabled = not grid_enabled
            if key == ord('r'):
                reset, paused = True, False
                last_event = "Baseline reset; ready for next dart"
            if key == ord('p'):
                paused = not paused
            if key == ord('s'):
                save_views(Path(args.output) / "snapshot", images, blend, board)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["calibrate", "run"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sources", nargs="+", default=["0", "1", "2"], help="Camera indices, stream URLs, or video paths, IN CALIBRATION ORDER")
    source.add_argument("--images", nargs="+", help="Saved images; useful for offline calibration or run --once")
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--size", type=int, default=900, help="Calibration: canonical output size")
    parser.add_argument("--radii", type=float, nargs=4, default=DEFAULT_RADII,
                        metavar=("TREBLE_IN", "TREBLE_OUT", "DOUBLE_IN", "DOUBLE_OUT"))
    parser.add_argument("--intrinsics", nargs="+", help="Calibration: one NPZ per camera, containing K, D, image_size")
    parser.add_argument("--estimate-radial-distortion", action="store_true", help="Experimental calibration: jointly fit H and k1 to raw edges; hold K and other distortion terms fixed; save post-fit Hough circles")
    parser.add_argument("--manual", action="store_true", help="Optional four-click calibration fallback; default is automatic upright-board detection")
    parser.add_argument("--geometry-prior", help="JSON with nominal camera_positions_mm or positions_mm, in source order")
    parser.add_argument("--rectify-interval", type=float, default=.5, help="Seconds between automatic registration checks")
    parser.add_argument("--settle-seconds", type=float, default=.8)
    parser.add_argument("--diff-threshold", type=int, default=28)
    parser.add_argument("--boundary-mm", type=float, default=2., help="Flag scores near a wire for review")
    parser.add_argument("--log", default="dart_events.jsonl")
    parser.add_argument("--output", default="dartboard_output")
    parser.add_argument("--once", action="store_true", help="Run: save one aligned set and exit without a window or scoring")
    parser.add_argument("--headless", action="store_true", help="Run without dashboard; Ctrl+C to stop")
    args = parser.parse_args()
    if (args.size < 400 or not 0 < args.rectify_interval or not 0 < args.settle_seconds
            or not 0 < args.diff_threshold < 256 or args.boundary_mm < 0
            or not 15.9 < args.radii[0] < args.radii[1] < args.radii[2] < args.radii[3]):
        parser.error("Invalid size, timing, threshold, uncertainty, or increasing ring radii")
    if args.manual and args.estimate_radial_distortion:
        parser.error("Choose automatic radial fitting or manual calibration")
    if args.command == "run" and args.images and not args.once:
        parser.error("Use --once with still images; live scoring requires streams/videos")
    inputs = None
    try:
        inputs = Input(args.sources, args.images, args.width, args.height)
        if args.command == "calibrate":
            calibrate(args, inputs)
        else:
            run(args, inputs)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, RuntimeError, OSError, KeyError, cv.error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        if inputs is not None:
            inputs.close()
        if not args.headless and not args.once:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()

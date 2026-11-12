"""Hough proposals from temporal HSV masks, fitted shaft centres and tip ends."""

import itertools
import math

import cv2 as cv
import numpy as np


def shaft_lines(mask, evidence=None):
    scale = mask.shape[0] / 900.0
    skeleton = cv.ximgproc.thinning(mask)
    distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
    # Use thin parts of the medial axis, not flight silhouettes or mask edges.
    thin = np.uint8((skeleton > 0) & (distance <= 8 * scale)) * 255
    lines = cv.HoughLinesP(
        thin, 1, np.pi / 720, 18, minLineLength=25 * scale, maxLineGap=12 * scale
    )
    if lines is None:
        return [], dict(skeleton=skeleton, thin=thin)
    y, x = np.where(thin)
    points = np.float32(np.c_[x, y])
    axes = []
    for raw in sorted(lines[:, 0], key=lambda a: -np.linalg.norm(a[2:] - a[:2])):
        start, end = raw[:2].astype(float), raw[2:].astype(float)
        direction = end - start
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        near = points[abs((points - start) @ normal) <= 3 * scale]
        if len(near) < 15:
            continue
        t = (near - start) @ direction
        order = np.argsort(t)
        near = near[order]
        t = t[order]
        runs = np.split(np.arange(len(t)), np.where(np.diff(t) > 15 * scale)[0] + 1)
        midpoint = 0.5 * np.linalg.norm(end - start)
        run = min(runs, key=lambda ids: abs(np.median(t[ids]) - midpoint))
        fit = near[run]
        if len(fit) < 15 or np.ptp((fit - start) @ direction) < 25 * scale:
            continue
        vx, vy, px, py = cv.fitLine(fit, cv.DIST_HUBER, 0, 0.001, 0.001).ravel()
        direction = np.array([vx, vy], float)
        origin = np.array([px, py], float)
        normal = np.array([-vy, vx])
        residual = abs((fit - origin) @ normal)
        fit = fit[residual < 2.5 * scale]
        if len(fit) < 15:
            continue
        vx, vy, px, py = cv.fitLine(fit, cv.DIST_HUBER, 0, 0.001, 0.001).ravel()
        direction = np.array([vx, vy], float)
        origin = np.array([px, py], float)
        normal = np.array([-vy, vx])
        residual = abs((fit - origin) @ normal)
        rms = float(np.sqrt(np.mean(residual**2)))
        t = (fit - origin) @ direction
        length = float(np.ptp(t))
        if rms > 2 * scale:
            continue
        ends = np.array([origin + t.min() * direction, origin + t.max() * direction])
        strong_fraction = 1.0
        if evidence is not None:
            pixel = np.rint(fit).astype(int)
            strength = evidence[pixel[:, 1], pixel[:, 0]]
            strong_fraction = float(np.mean(strength >= 2.0))
            # Weak mask bridges may connect the shaft, but a proposed line also
            # needs substantial high-response HSV support. Same rule every view.
            if strong_fraction < 0.35:
                continue
        candidate = dict(
            origin=origin,
            direction=direction,
            normal=normal,
            constant=float(normal @ origin),
            points=fit,
            ends=ends,
            length_px=length,
            rms_px=rms,
            quality=length * strong_fraction / (1 + rms),
            strong_fraction=strong_fraction,
            mask=mask,
        )
        duplicate = next(
            (
                a
                for a in axes
                if abs(a["direction"] @ direction) > math.cos(math.radians(3))
                and abs(a["normal"] @ origin - a["constant"]) < 5 * scale
                and min(np.linalg.norm(a["ends"] - e, axis=1).min() for e in ends)
                < 25 * scale
            ),
            None,
        )
        if duplicate is None:
            axes.append(candidate)
        elif candidate["quality"] > duplicate["quality"]:
            axes = [a for a in axes if a is not duplicate]
            axes.append(candidate)
    return sorted(axes, key=lambda a: -a["quality"]), dict(skeleton=skeleton, thin=thin)


def terminal_evidence(axis, point):
    """A contact should terminate observed shaft evidence in at least two views.

    Test the original HSV mask on both sides, not the fitted segment endpoints:
    a hole in the mask must not turn the middle of a barrel into a new tip.
    """
    if "mask" not in axis:
        return True, []
    mask = axis["mask"]
    scale = mask.shape[0] / 900.0
    t = np.arange(10.0, 50.0) * scale
    cross = np.arange(-3.0, 4.0) * scale
    coverage = []
    for sign in (-1, 1):
        q = (
            np.array(point)
            + sign * t[:, None, None] * axis["direction"]
            + cross[None, :, None] * axis["normal"]
        )
        sampled = cv.remap(
            mask,
            q[..., 0].astype(np.float32),
            q[..., 1].astype(np.float32),
            cv.INTER_NEAREST,
        )
        coverage.append(float(np.mean(np.any(sampled > 0, axis=1))))
    return min(coverage) <= 0.25 and max(coverage) >= 0.4, coverage


def intersections(
    axes, center, radius, tolerance=3.0, tip_distance=20.0, minimum_angle=15.0
):
    proposals = []
    for i, j in itertools.combinations(range(len(axes)), 2):
        for ai, a in enumerate(axes[i]):
            for bi, b in enumerate(axes[j]):
                A = np.array([a["normal"], b["normal"]])
                if abs(np.linalg.det(A)) < math.sin(math.radians(minimum_angle)):
                    continue
                point = np.linalg.solve(A, [a["constant"], b["constant"]])
                if np.linalg.norm(point - center) > radius:
                    continue
                if any(
                    np.linalg.norm(axis["ends"] - point, axis=1).min() > tip_distance
                    for axis in [a, b]
                ):
                    continue
                if not all(terminal_evidence(axis, point)[0] for axis in (a, b)):
                    continue
                selected = {i: ai, j: bi}
                for camera, options in enumerate(axes):
                    if camera in selected:
                        continue
                    eligible = [
                        (abs(axis["normal"] @ point - axis["constant"]), index)
                        for index, axis in enumerate(options)
                        if np.linalg.norm(axis["ends"] - point, axis=1).min()
                        <= tip_distance
                        and abs(axis["normal"] @ point - axis["constant"]) <= tolerance
                        and terminal_evidence(axis, point)[0]
                    ]
                    if eligible:
                        selected[camera] = min(eligible)[1]
                chosen = [axes[k][v] for k, v in selected.items()]
                A = np.array([a["normal"] for a in chosen])
                b = np.array([a["constant"] for a in chosen])
                point = np.linalg.lstsq(A, b, rcond=None)[0]
                residual = [
                    float(abs(a["normal"] @ point - a["constant"])) for a in chosen
                ]
                if max(residual) > tolerance:
                    continue
                if not all(terminal_evidence(axis, point)[0] for axis in chosen):
                    continue
                end_dist = [
                    float(np.linalg.norm(a["ends"] - point, axis=1).min())
                    for a in chosen
                ]
                proposals.append(
                    dict(
                        point_pixels=point.tolist(),
                        axes=selected,
                        support=len(selected),
                        residual_px=residual,
                        tip_end_distance_px=end_dist,
                        mask_side_coverage=[
                            terminal_evidence(axis, point)[1] for axis in chosen
                        ],
                        cost=sum(residual) + 0.2 * sum(end_dist),
                    )
                )
    proposals.sort(key=lambda p: (-p["support"], p["cost"]))
    selected = []
    used = set()
    for p in proposals:
        keys = set(p["axes"].items())
        if keys & used or any(
            np.linalg.norm(np.array(p["point_pixels"]) - q["point_pixels"])
            < tip_distance
            for q in selected
        ):
            continue
        selected.append(p)
        used |= keys
    return selected, proposals

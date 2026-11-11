"""Frozen pre-improvement detector from commit 39e2dfd, for paired comparisons."""
import itertools
import math
import cv2 as cv
import numpy as np

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


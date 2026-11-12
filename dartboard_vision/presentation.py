"""Board template, contact markers and restrained video presentation.

These functions draw existing estimates; they do not fit geometry or detect hits.
"""

import cv2 as cv
import numpy as np

from .geometry import BOARD, geometry_fields


def template_board(board=BOARD):
    labels, _, _, _ = geometry_fields(board.size, board)
    palette = np.array(
        [[28, 28, 28], [200, 220, 232], [35, 45, 185], [65, 140, 35]], np.uint8
    )
    result = palette[labels]
    y, x = np.mgrid[: board.size, : board.size]
    r = np.hypot(x - board.center[0], y - board.center[1]) / board.ppm
    result[r > board.radii[-1]] = [24, 28, 26]
    center = tuple(board.center.astype(int))
    for radius in (6.35, 15.9, *board.radii):
        cv.circle(
            result, center, round(radius * board.ppm), (195, 200, 195), 1, cv.LINE_AA
        )
    for k in range(20):
        angle = -9 + 18 * k
        a = tuple(np.rint(board.point(15.9, angle)).astype(int))
        b = tuple(np.rint(board.point(board.radii[-1], angle)).astype(int))
        cv.line(result, a, b, (195, 200, 195), 1, cv.LINE_AA)
    # No numeric scores: absolute sector identity is not yet independently checked.
    return result


def mark_hits(image, contacts, size=18):
    result = image.copy()
    for hit in contacts:
        point = tuple(np.rint(hit["point_pixels"]).astype(int))
        cv.drawMarker(
            result, point, (18, 20, 18), cv.MARKER_TILTED_CROSS, size, 5, cv.LINE_AA
        )
        cv.drawMarker(
            result, point, (190, 230, 250), cv.MARKER_TILTED_CROSS, size, 2, cv.LINE_AA
        )
    return result


def compose(plane, template, contacts, t_s, glow=None):
    result = plane.astype(np.float32)
    if glow is not None:
        core = glow.astype(np.float32) / 255.0
        inner = cv.GaussianBlur(core, (0, 0), 1.1)
        halo = cv.GaussianBlur(core, (0, 0), 5.5)
        alpha = (0.10 * inner + 0.12 * halo)[..., None]
        result = result * (1 - alpha) + np.array([185, 225, 255], np.float32) * alpha
    yy, xx = np.mgrid[:900, :900]
    vignette = 1 - 0.10 * np.clip((np.hypot(xx - 450, yy - 450) / 600.0) ** 4, 0, 1)
    result = np.uint8(np.clip(result * vignette[..., None], 0, 255))
    result = mark_hits(result, contacts)
    small = cv.resize(
        mark_hits(template, contacts, 32), (200, 200), interpolation=cv.INTER_AREA
    )
    # The little board occupies the corner outside the scoring disc.
    x, y = 688, 666
    card = np.full((226, 204, 3), (20, 25, 23), np.uint8)
    edge = np.zeros((226, 204), np.uint8)
    cv.rectangle(edge, (9, 0), (194, 225), 255, -1)
    cv.rectangle(edge, (0, 9), (203, 216), 255, -1)
    for p in [(9, 9), (194, 9), (9, 216), (194, 216)]:
        cv.circle(edge, p, 9, 255, -1, cv.LINE_AA)
    area = result[y : y + 226, x : x + 204]
    a = edge.astype(np.float32)[..., None] / 255.0 * 0.94
    area[:] = np.uint8(area * (1 - a) + card * a)
    cv.putText(
        result,
        "CONTACT MAP",
        (x + 12, y + 17),
        cv.FONT_HERSHEY_SIMPLEX,
        0.33,
        (200, 215, 210),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        result,
        f"{len(contacts):02d}",
        (x + 172, y + 18),
        cv.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 230, 250),
        1,
        cv.LINE_AA,
    )
    result[y + 24 : y + 224, x + 2 : x + 202] = small
    cv.putText(
        result,
        "BOARD VIEW",
        (20, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 235, 230),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        result,
        f"{t_s:05.2f} s",
        (20, 47),
        cv.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 205, 195),
        1,
        cv.LINE_AA,
    )
    return result

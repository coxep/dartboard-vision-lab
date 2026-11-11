#!/usr/bin/env python3
"""Render physically consistent pinhole views, rectification and shaft intersections.

Run: python render_synthetic.py
All views are generated from the SAME board plane and SAME 3D dart.
These are geometric illustrations, not photorealistic performance evidence.
"""
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from dartboard import Board, NUMBERS, Tracker, project, shaft_line, intersect_lines, text, warp

COLORS = [(65, 185, 255), (255, 205, 75), (185, 100, 255)]


def board_texture(board):
    s = board.size
    yy, xx = np.mgrid[:s, :s]
    x, y = (xx - s / 2) / board.ppm, (yy - s / 2) / board.ppm
    r = np.hypot(x, y)
    a = np.degrees(np.arctan2(x, -y)) % 360
    sector = ((a + 9) // 18).astype(int) % 20
    image = np.full((s, s, 3), (27, 31, 35), np.uint8)
    image[r < 225] = (30, 33, 35)
    even = sector % 2 == 0
    playing = r <= 170
    image[playing & even] = (37, 40, 43)
    image[playing & ~even] = (156, 189, 209)
    ti, to, di, do = board.radii
    band = ((r >= ti) & (r <= to)) | ((r >= di) & (r <= do))
    image[band & even] = (55, 55, 195)
    image[band & ~even] = (117, 143, 30)
    image[r <= 15.9] = (112, 148, 25)
    image[r <= 6.35] = (50, 50, 195)
    # Fixed, unique texture makes reference registration testable despite sector repetition.
    rng = np.random.default_rng(37)
    noise = rng.normal(0, 9, (s, s, 1))
    image = np.uint8(np.clip(image.astype(float) + noise * (r < 225)[..., None], 0, 255))
    for _ in range(4500):
        px, py = rng.integers(0, s, 2)
        if r[py, px] < 220:
            color = tuple(int(c) for c in np.clip(image[py, px].astype(int) + rng.integers(-35, 36), 0, 255))
            cv.line(image, (int(px), int(py)), (int(px + rng.integers(-2, 3)), int(py + rng.integers(2, 6))), color, 1)
    for rad in (6.35, 15.9, *board.radii, 223):
        cv.circle(image, tuple(board.center.astype(int)), round(rad * board.ppm), (155, 160, 165), 1, cv.LINE_AA)
    for k in range(20):
        angle = -9 + 18 * k
        cv.line(image, tuple(board.point(15.9, angle).astype(int)),
                tuple(board.point(170, angle).astype(int)), (155, 160, 165), 1, cv.LINE_AA)
        p = board.point(202, 18 * k)
        size, _ = cv.getTextSize(str(NUMBERS[k]), cv.FONT_HERSHEY_SIMPLEX, .8, 2)
        text(image, str(NUMBERS[k]), p + [-size[0] / 2, size[1] / 2], scale=.8)
    return image


def camera_matrix(center, width=960, height=720, focal=650, look_at=None, roll=0):
    center = np.asarray(center, np.float64)
    target = np.zeros(3) if look_at is None else np.asarray(look_at, np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.array([right, down, forward])
    a = math.radians(roll)
    R = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1.]]) @ R
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
    return K @ np.c_[R, -R @ center]


def project3(points, P):
    p = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))] @ P.T
    return p[:, :2] / p[:, 2:]


def render_camera(texture, board, position, dart=True, tip=None, width=960, height=720, focal=650, look_at=None, roll=0, dart_config=None, existing_darts=None):
    P = camera_matrix(position, width, height, focal, look_at, roll)
    plane = np.array([[1 / board.ppm, 0, -board.center[0] / board.ppm],
                      [0, -1 / board.ppm, board.center[1] / board.ppm],
                      [0, 0, 0], [0, 0, 1.]])
    board_to_camera = P @ plane
    image = cv.warpPerspective(texture, board_to_camera, (width, height), borderValue=(46, 43, 40))
    # Existing darts appear in both reference and changed frames. Painter-order
    # rendering is an approximation; this is not a depth-tested mesh renderer.
    configurations=list(existing_darts or [])
    if dart:configurations.append(dict(dart_config or {},tip=[0.,103.,0.] if tip is None else tip))
    pixels=None
    for config in configurations:
        tip3=np.asarray(config.get('tip',[0.,103.,0.]),float)
        direction=np.asarray(config.get('direction',[.12,-.18,1.]),float)
        length=float(config.get('length',73.))
        points3=np.array([tip3,tip3+13/73*length*direction,tip3+40/73*length*direction,tip3+length*direction])
        pixels=project3(points3,P)
        def line(a,b,color,thickness):
            cv.line(image,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,thickness,cv.LINE_AA)
        line(pixels[0],pixels[1],tuple(config.get('tip_color',(225,230,235))),2)
        line(pixels[1],pixels[2],tuple(config.get('barrel_color',(100,110,120))),6)
        line(pixels[2],pixels[3],tuple(config.get('shaft_color',(215,220,225))),3)
        base=tip3+59/73*length*direction
        width=float(config.get('flight_width',5.))
        flight=project3([base+[-width,0,0],points3[-1],base+[width,0,0],tip3+54/73*length*direction],P)
        cv.fillConvexPoly(image,np.rint(flight).astype(np.int32),tuple(config.get('flight_color',(60,160,235))),cv.LINE_AA)
    if pixels is None:
        tip3=np.array([0.,103.,0.]) if tip is None else np.asarray(tip,float)
        pixels=project3([tip3,tip3+73*np.array([.12,-.18,1.])],P)

    return image, np.linalg.inv(board_to_camera), pixels[[0, -1]]


def panel(image, title, subtitle="", color=(225, 230, 235), width=620, height=510):
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    canvas = np.full((height, width, 3), (24, 22, 20), np.uint8)
    scale = min((width - 16) / image.shape[1], (height - 80) / image.shape[0])
    resized = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    x = (width - resized.shape[1]) // 2
    canvas[72:72 + resized.shape[0], x:x + resized.shape[1]] = resized
    text(canvas, title, (18, 30), color=color, scale=.7)
    text(canvas, subtitle, (18, 55), color=(185, 190, 195), scale=.43)
    return canvas


def main():
    root = Path(__file__).resolve().parent
    out = root / "synthetic"
    out.mkdir(exist_ok=True)
    board = Board()
    texture = board_texture(board)
    positions = [[-340, 240, 270], [340, 240, 270], [0, -390, 270]]
    rows = [[], []]
    entries, rectified = [], []
    common = texture.copy()
    true_tip = board.point(103, 0)
    for i, (position, color) in enumerate(zip(positions, COLORS)):
        empty, H, _ = render_camera(texture, board, position, dart=False)
        raw, _, axis = render_camera(texture, board, position)
        cv.imwrite(str(out / f"camera_{i}_reference.png"), empty)
        cv.imwrite(str(out / f"camera_{i}_dart.png"), raw)
        entries.append(dict(H=H.tolist(), shape=list(raw.shape[:2]), lens=None,
                            reference=f"camera_{i}_reference.png", harris=dict(refined=False, reason="exact synthetic geometry")))
        rect = warp(raw, H, board)
        rectified.append(rect)
        points = project(axis, H)
        v = points[1] - points[0]
        v /= np.linalg.norm(v)
        a, b = points[0] - 1700 * v, points[0] + 1700 * v
        cv.line(common, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv.LINE_AA)
        for target in (raw, rect):
            xy = axis if target is raw else points
            delta = xy[1] - xy[0]
            delta /= np.linalg.norm(delta)
            start, end = xy[0] - 1000 * delta, xy[0] + 1000 * delta
            cv.line(target, tuple(start.astype(int)), tuple(end.astype(int)), color, 2, cv.LINE_AA)
            cv.circle(target, tuple(np.rint(xy[0]).astype(int)), 6, color, 2)
        rows[0].append(panel(raw, f"CAMERA {i + 1}: OBLIQUE VIEW", "Same board and same 3D dart in all three views", color))
        rows[1].append(panel(rect, f"CAMERA {i + 1}: RECTIFIED", "Board surface aligned; projected shaft direction differs", color))
    montage = np.vstack([np.hstack(row) for row in rows])
    cv.imwrite(str(out / "three_views_and_rectification.png"), montage)
    cv.circle(common, tuple(np.rint(true_tip).astype(int)), 10, (255, 255, 255), 2, cv.LINE_AA)
    common = panel(common, "THREE SHAFT AXES, ONE BOARD ENTRY POINT", "Known target: treble 20, 60 points. Lines use exact camera geometry.", width=1000, height=1060)
    for i, color in enumerate(COLORS):
        text(common, f"Camera {i + 1} axis", (26 + i * 315, 1035), color, .65)
    cv.imwrite(str(out / "shaft_intersection.png"), common)
    config = dict(version=1, size=board.size, radii=board.radii, cameras=entries)
    (out / "calibration.json").write_text(json.dumps(config, indent=2) + "\n")

    # An actually estimated recovery, not an illustration using a known inverse.
    old, H, _ = render_camera(texture, board, positions[0], dart=False)
    moved, truth, _ = render_camera(texture, board, [-285, 300, 300], dart=False)
    tracker = Tracker(old, H, board)
    did_move = tracker.update(moved)
    if not tracker.valid or not did_move:
        raise RuntimeError("Synthetic camera-movement recovery failed: " + tracker.reason)
    stale = board.grid(warp(moved, H, board))
    recovered = board.grid(warp(moved, tracker.H, board))
    errors = np.linalg.norm(project(board.landmarks(), tracker.H @ np.linalg.inv(truth)) - board.landmarks(), axis=1)
    movement = np.hstack([
        panel(board.grid(warp(old, H, board)), "1. ORIGINAL CAMERA", "Reference alignment: grid follows board boundaries", width=650, height=690),
        panel(stale, "2. CAMERA MOVED: OLD MAPPING", "The old homography no longer matches the board", width=650, height=690),
        panel(recovered, "3. AUTOMATIC RECOVERY", f"ORB + Harris: median landmark error {np.median(errors):.2f} px", width=650, height=690),
    ])
    cv.imwrite(str(out / "camera_movement_recovery.png"), movement)
    # Verify the silhouette-based estimator separately from the exact-axis illustration.
    from dartboard import change_mask, gray_image
    lines = []
    for position in positions:
        empty, H, _ = render_camera(texture, board, position, dart=False)
        dart, _, _ = render_camera(texture, board, position)
        mask = change_mask(gray_image(warp(dart, H, board)), gray_image(warp(empty, H, board)), board.mask(1.08), 28)
        lines.append(shaft_line(mask, board))
    tip, ids, reason = intersect_lines(lines, tolerance=5, support_distance=16)
    measurement = dict(camera_recovery_median_px=float(np.median(errors)),
                       camera_recovery_max_px=float(np.max(errors)),
                       detected_tip=None if tip is None else tip.tolist(), cameras=ids, note=reason)
    if tip is not None:
        measurement.update(board.score(tip))
        measurement["tip_error_mm"] = float(np.linalg.norm(tip - true_tip) / board.ppm)
    (out / "measurements.json").write_text(json.dumps(measurement, indent=2) + "\n")
    print(json.dumps(measurement, indent=2))
    print(f"Rendered images and offline inputs in {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible C920-like synthetic test of the actual dartboard.py functions.

No hand-labelled corners, no ground-truth homographies passed to the detector.
Known synthetic K/D are used to undo the injected lens distortion. Camera poses,
distortion, noise and lighting are illustrative, NOT measured from a real C920.
Run: python benchmark_c920.py
"""
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2 as cv
import numpy as np

from dartboard import (Board, Lens, automatic_homography, binarize, change_mask,
                       gray_image, intersect_lines, project, shaft_line, text, warp,
                       nominal_homography, adaptive_block_size, board_color_masks)
from render_synthetic import board_texture, panel, render_camera, COLORS
from pose_alignment import homography_from_pose
from dart_detection import detect_dart, diagnostic_image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "c920_benchmark"
WIDTH, HEIGHT, DFOV = 1920, 1080, 78.
FOCAL = math.hypot(WIDTH, HEIGHT) / (2 * math.tan(math.radians(DFOV / 2)))
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.]])
# Brown-Conrady: k1, k2, p1, p2, k3. Illustrative mild barrel distortion.
D = np.array([-.045, .012, .0007, -.0005, 0.])
MOUNT_RADIUS_MM = 14.75 * 25.4
STANDOFF_MM = 8.5 * 25.4
_top = np.array([-340., 290.]) / math.hypot(340, 290) * MOUNT_RADIUS_MM
POSITIONS = [[float(_top[0]), float(_top[1]), STANDOFF_MM],
             [float(-_top[0]), float(_top[1]), STANDOFF_MM],
             [0., -MOUNT_RADIUS_MM, STANDOFF_MM]]
CASES = [("treble20", 103., 0., "T20", 60),
         ("double16", 166., 234., "D16", 32),
         ("single6", 135., 90., "S6", 6),
         ("doublebull", 0., 0., "DB", 50),
         ("singlebull", 11., 0., "SB", 25),
         ("treble19", 103., 198., "T19", 57),
         ("double3", 166., 180., "D3", 6),
         ("miss", 178., 54., "MISS", 0)]


def distortion_map(K, D, width, height):
    y, x = np.mgrid[:height, :width]
    distorted = np.float32(np.c_[x.ravel(), y.ravel()]).reshape(-1, 1, 2)
    undistorted = cv.undistortPoints(distorted, K, D, P=K).reshape(height, width, 2)
    return undistorted[:, :, 0], undistorted[:, :, 1]


def observed(ideal, maps, seed):
    image = cv.remap(ideal, *maps, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    vignette = 1 - .09 * (((x - WIDTH / 2) / WIDTH) ** 2 + ((y - HEIGHT / 2) / HEIGHT) ** 2)
    image = image.astype(float) * vignette[..., None] + rng.normal(0, .8, image.shape)
    image = cv.GaussianBlur(np.uint8(np.clip(image, 0, 255)), (3, 3), .45)
    # Detector receives exactly the JPEG bytes that are published as inputs.
    ok, jpg = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, 94])
    assert ok
    return cv.imdecode(jpg, cv.IMREAD_COLOR), jpg


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv.imwrite(str(path), image, [cv.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise IOError(path)


def draw_axis(image, line, color):
    result = image.copy()
    if line is not None:
        start = line["origin"] - 1500 * line["direction"]
        end = line["origin"] + 1500 * line["direction"]
        cv.line(result, tuple(start.astype(int)), tuple(end.astype(int)), color, 2, cv.LINE_AA)
    return result


def evaluate_case(name, radius, angle, label, score, board, texture, cameras, maps, seed):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    truth_pixel = board.point(radius, angle)
    xy = (truth_pixel - board.center) / board.ppm
    tip3 = [float(xy[0]), float(-xy[1]), 0.]
    lines, rectified, camera_rows, deltas = [], [], [], []
    diagnostics = []
    for i, camera in enumerate(cameras):
        ideal, true_H, _ = render_camera(texture, board, camera["position_mm"], tip=tip3,
                                        width=WIDTH, height=HEIGHT, focal=FOCAL)
        distorted, jpg = observed(ideal, maps, seed + i)
        (folder / f"camera_{i}_input.jpg").write_bytes(jpg.tobytes())
        corrected = camera["lens"].apply(distorted)
        H = camera["H"]
        rect = warp(corrected, H, board)
        delta = change_mask(gray_image(rect), camera["background"], board.mask(1.08), 28)
        deltas.append(delta)
        line = None
        rectified.append(rect)
        write_image(folder / f"camera_{i}_mask.png", delta)
        axis_image = draw_axis(rect, line, COLORS[i])
        write_image(folder / f"camera_{i}_detected.jpg", axis_image)
        diagnostics.append(dict(camera=i, shaft_found=line is not None,
                                raw=f"{name}/camera_{i}_input.jpg",
                                mask=f"{name}/camera_{i}_mask.png",
                                detected=f"{name}/camera_{i}_detected.jpg"))
        camera_rows.append(panel(distorted, f"CAMERA {i + 1}: DISTORTED INPUT", "1920 x 1080 / 78 deg diagonal / mild barrel + tangential distortion", COLORS[i], 640, 440))
    point, ids, reason, lines, info = detect_dart(deltas, board)
    for i,(rect,delta,line) in enumerate(zip(rectified,deltas,lines)):
        write_image(folder / f"camera_{i}_detected.jpg", diagnostic_image(rect,delta,info['candidates'][i],line,point))
        diagnostics[i]['shaft_found']=bool(info['candidates'][i])
        diagnostics[i]['selected']=i in ids
        diagnostics[i]['candidates']=len(info['candidates'][i])
    result = dict(id=name, truth=dict(label=label, score=score, pixel=truth_pixel.tolist(),
                                     x_mm=float(xy[0]), y_mm=float(xy[1])), cameras=ids,
                  note=reason, detection=None, error_mm=None, correct=False,
                  views=diagnostics)
    if point is not None:
        result["detection"] = board.score(point)
        result["detection"]["pixel"] = point.tolist()
        result["error_mm"] = float(np.linalg.norm(point - truth_pixel) / board.ppm)
        result["correct"] = result["detection"]["label"] == label and result["detection"]["score"] == score
    combined = np.mean(rectified, axis=0).astype(np.uint8)
    for i, line in enumerate(lines):
        combined = draw_axis(combined, line, COLORS[i])
    # White cross is truth; green circle is estimator output. Truth enters only here.
    tp = tuple(np.rint(truth_pixel).astype(int))
    cv.drawMarker(combined, tp, (255, 255, 255), cv.MARKER_CROSS, 20, 2)
    if point is not None:
        cv.circle(combined, tuple(np.rint(point).astype(int)), 10, (90, 255, 130), 2)
    caption = f"Expected {label} = {score}; "
    caption += f"detected {result['detection']['label']}; error {result['error_mm']:.2f} mm" if point is not None else f"REJECTED: {reason}"
    combined = panel(combined, "ACTUAL ESTIMATOR OUTPUT", caption, width=980, height=1060)
    write_image(folder / "intersection.jpg", combined)
    write_image(folder / "inputs.jpg", np.hstack(camera_rows))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    board = Board()
    texture = board_texture(board)
    maps = distortion_map(K, D, WIDTH, HEIGHT)
    cameras, calibration_rows = [], []
    calibration = []
    for i, position in enumerate(POSITIONS):
        ideal, true_H, _ = render_camera(texture, board, position, dart=False,
                                        width=WIDTH, height=HEIGHT, focal=FOCAL)
        raw, jpg = observed(ideal, maps, 100 + i)
        (OUT / f"camera_{i}_empty.jpg").write_bytes(jpg.tobytes())
        lens_data = dict(K=K.tolist(), D=D.tolist(), image_size=[WIDTH, HEIGHT])
        lens = Lens(raw.shape, lens_data)
        corrected = lens.apply(raw)
        # Detector sees only corrected pixels, geometry and camera-up assumption.
        # true_H is withheld and used ONLY to evaluate recovered calibration.
        preliminary = nominal_homography(board, K, position)
        write_image(OUT / f"camera_{i}_preliminary.jpg", board.grid(warp(corrected, preliminary, board)))
        H, report = automatic_homography(corrected, board, preliminary, K)
        pose = report.get("pose_initialization")
        pose_H = homography_from_pose(np.array(pose["rvec"]), np.array(pose["tvec_mm"]), K, board) if pose else preliminary
        write_image(OUT / f"camera_{i}_pose.jpg", board.grid(warp(corrected, pose_H, board)))
        error = np.linalg.norm(project(board.landmarks(), H @ np.linalg.inv(true_H)) - board.landmarks(), axis=1)
        reference_name = f"camera_{i}_reference.png"
        write_image(OUT / reference_name, corrected)
        calibration.append(dict(H=H.tolist(), shape=[HEIGHT, WIDTH], lens=lens_data,
                                reference=reference_name, harris=report, pose_K=K.tolist()))
        camera = dict(position_mm=position, H=H, lens=lens,
                      background=gray_image(warp(corrected, H, board)))
        cameras.append(camera)
        binary = binarize(cv.cvtColor(corrected, cv.COLOR_BGR2GRAY))
        cv.imwrite(str(OUT / f"camera_{i}_adaptive.png"), binary)
        roi = cv.warpPerspective(board.mask(1.04), np.linalg.inv(H), (WIDTH, HEIGHT), flags=cv.INTER_NEAREST)
        masks = board_color_masks(corrected)
        for name, mask in masks.items():
            write_image(OUT / f"camera_{i}_hsv_{name}.png", mask)
        overlay = corrected.copy()
        for feature_image in [binary, *masks.values()]:
            corners = cv.goodFeaturesToTrack(feature_image, 3000, .005, 3, mask=roi,
                                            blockSize=3, useHarrisDetector=True, k=.04)
            if corners is not None:
                for p in corners.reshape(-1, 2):
                    cv.circle(overlay, tuple(np.rint(p).astype(int)), 2, (90, 255, 130), -1)
        for p in report["matched_source_px"]:
            cv.circle(overlay, tuple(np.rint(p).astype(int)), 5, (0, 210, 255), 2)
        write_image(OUT / f"camera_{i}_harris.jpg", overlay)
        grid = board.grid(warp(corrected, H, board))
        write_image(OUT / f"camera_{i}_rectified.jpg", grid)
        report.update(camera=i, median_error_px=float(np.median(error)), max_error_px=float(error.max()))
        # Separate raw, binary, detected corners, rectified output for inspection.
        calibration_rows.append(np.hstack([
            panel(raw, f"CAMERA {i + 1}: INPUT", "Injected mild distortion + noise", COLORS[i], 480, 380),
            panel(binary, "ADAPTIVE THRESHOLD", f"Gaussian adaptive, block {adaptive_block_size(binary.shape)}, C = 5", COLORS[i], 480, 380),
            panel(overlay, "HARRIS CORNERS", f"{report['inliers']} model inliers", COLORS[i], 480, 380),
            panel(grid, "AUTOMATIC RECTIFICATION", f"GT landmark median error {np.median(error):.2f} px", COLORS[i], 480, 380),
        ]))
        np.savez(OUT / f"camera_{i}_intrinsics.npz", K=K, D=D, image_size=[WIDTH, HEIGHT])
    cv.imwrite(str(OUT / "calibration_pipeline.jpg"), np.vstack(calibration_rows))
    config = dict(version=1, size=board.size, radii=board.radii, cameras=calibration)
    (OUT / "calibration.json").write_text(json.dumps(config, indent=2) + "\n")
    results = [evaluate_case(*case, board, texture, cameras, maps, 200 + j * 10)
               for j, case in enumerate(CASES)]
    # Negative test: three visibly inconsistent new shafts must NOT become a score.
    disjoint = []
    for i, (radius, angle) in enumerate([(103, 0), (166, 234), (135, 90)]):
        p = board.point(radius, angle)
        xy = (p - board.center) / board.ppm
        ideal, _, _ = render_camera(texture, board, POSITIONS[i], tip=[xy[0], -xy[1], 0],
                                    width=WIDTH, height=HEIGHT, focal=FOCAL)
        raw, _ = observed(ideal, maps, 700 + i)
        rect = warp(cameras[i]["lens"].apply(raw), cameras[i]["H"], board)
        disjoint.append(change_mask(gray_image(rect), cameras[i]["background"], board.mask(1.08), 28))
    negative_point, _, negative_reason, _, _ = detect_dart(disjoint,board)
    scene = dict(units="mm", board_plane="z=0, x right, y up, z toward player",
                 resolution=[WIDTH, HEIGHT], diagonal_fov_degrees=DFOV,
                 specification_source="https://support.logi.com/hc/en-ca/articles/360023307774-C920-C-Webcam-Technical-Specifications",
                 camera_positions_mm=POSITIONS, mount_radius_mm=MOUNT_RADIUS_MM,
                 standoff_mm=STANDOFF_MM, look_at_mm=[0, 0, 0], K=K.tolist(), D=D.tolist(),
                 assumptions="User supplied 14.75-inch radius and approximately 8.5-inch stand-off. Azimuths estimated from photo. Distortion illustrative. K/D and nominal pose are supplied; corrected H and hit are estimated. In these fixed cases the nominal pose equals the rendered pose; Monte Carlo perturbs it.")
    report = dict(scene=scene, calibration=[c["harris"] for c in calibration], cases=results,
                  correct=sum(r["correct"] for r in results), total=len(results),
                  detected=sum(r["detection"] is not None for r in results),
                  max_error_mm=max((r["error_mm"] for r in results if r["error_mm"] is not None), default=None),
                  negative_test=dict(rejected=negative_point is None, reason=negative_reason),
                  limitations=["Clean synthetic scene; no hands, overlapping darts or realistic shadows.",
                               "Known injected lens coefficients; this does not demonstrate blind lens calibration.",
                               "One static dart per case; continuous video event timing not tested here.",
                               "Camera-up and 20-at-top assumed; mounted cameras must supply an orientation cue."])
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    # FOV sensitivity frames: 65/90 degrees are hypothetical, 78 is C920 nominal.
    fov_panels = []
    for fov in [65, 78, 90]:
        focal = math.hypot(WIDTH, HEIGHT) / (2 * math.tan(math.radians(fov / 2)))
        Ks = K.copy(); Ks[0, 0] = Ks[1, 1] = focal
        image, _, _ = render_camera(texture, board, POSITIONS[0], dart=False,
                                    width=WIDTH, height=HEIGHT, focal=focal)
        distorted, _ = observed(image, distortion_map(Ks, D, WIDTH, HEIGHT), 800 + fov)
        write_image(OUT / f"fov_{fov}.jpg", distorted)
        fov_panels.append(panel(distorted, f"{fov} DEG DIAGONAL FOV", "C920 nominal" if fov == 78 else "Hypothetical sensitivity test", width=640, height=440))
    write_image(OUT / "fov_comparison.jpg", np.hstack(fov_panels))
    print(json.dumps({k: report[k] for k in ["correct", "total", "detected", "max_error_mm", "negative_test"]}, indent=2))
    for r in results:
        print(r["id"], r["detection"]["label"] if r["detection"] else "REJECTED", r["error_mm"], r["note"])


if __name__ == "__main__":
    main()

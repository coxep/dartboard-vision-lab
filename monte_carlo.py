#!/usr/bin/env python3
"""Seeded engineering stress test; priors are NOT measured C920 population data.

python monte_carlo.py --trials 100 --workers 4 --seed 20261110
Synthetic K/D are supplied exactly to the detector; homographies/hits are withheld.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from benchmark_c920 import (WIDTH, HEIGHT, POSITIONS, distortion_map, observed)
from dartboard import (Board, Lens, automatic_homography, change_mask, gray_image,
                       intersect_lines, project, shaft_line, warp, nominal_homography)
from render_synthetic import board_texture, render_camera

PRIORS = dict(
    provenance="Engineering stress ranges, NOT empirically measured C920 coefficient distributions",
    extrinsics=dict(position_uniform_ball_radius_mm=50.8,
                    look_at_xy_sigma_mm=8, roll_sigma_deg=1.5, roll_clip_deg=4),
    intrinsics=dict(diagonal_fov_uniform_deg=[75, 81], principal_point_sigma_px=[8, 6],
                    k1_uniform=[-.08, .02], k2_uniform=[-.02, .04],
                    p1_p2_uniform=[-.001, .001], k3=0),
    target=dict(radius_uniform_mm=[0, 182], angle_uniform_deg=[0, 360]),
    image=dict(gaussian_noise_sigma=.8, blur_sigma=.45, jpeg_quality=94),
    known_to_detector="Exact synthetic K/D, board dimensions, NOMINAL mount centers and camera-up assumption",
    withheld_from_detector="Sampled true camera pose, ground truth homographies and dart position")


def trial(task):
    trial_id, seed = task
    cv.setNumThreads(1)
    rng = np.random.default_rng(seed)
    cv.setRNGSeed(int(seed % (2**31-1)))
    board, lines, calibration, params = Board(), [], [], []
    texture = board_texture(board)
    radius, angle = rng.uniform(0, 182), rng.uniform(0, 360)
    truth = board.point(radius, angle)
    xy = (truth - board.center) / board.ppm
    truth_score = board.score(truth)
    result = dict(trial=trial_id, seed=int(seed), truth=truth_score,
                  truth_pixel=truth.tolist(), outcome="rejected", reason="", cameras=[],
                  estimate=None, error_mm=None, calibration=calibration, parameters=params)
    for i, nominal in enumerate(POSITIONS):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = np.asarray(nominal) + direction * 50.8 * rng.random() ** (1 / 3)
        look_at = [float(rng.normal(0, 8)), float(rng.normal(0, 8)), 0.]
        roll = float(np.clip(rng.normal(0, 1.5), -4, 4))
        fov = float(rng.uniform(75, 81))
        focal = math.hypot(WIDTH, HEIGHT) / (2 * math.tan(math.radians(fov / 2)))
        K = np.array([[focal, 0, WIDTH / 2 + rng.normal(0, 8)],
                      [0, focal, HEIGHT / 2 + rng.normal(0, 6)], [0, 0, 1.]])
        D = np.array([rng.uniform(-.08, .02), rng.uniform(-.02, .04),
                      rng.uniform(-.001, .001), rng.uniform(-.001, .001), 0.])
        params.append(dict(position_mm=position.tolist(), look_at_mm=look_at, roll_deg=roll,
                           diagonal_fov_deg=fov, K=K.tolist(), D=D.tolist()))
        maps = distortion_map(K, D, WIDTH, HEIGHT)
        lens = Lens((HEIGHT, WIDTH, 3), dict(K=K.tolist(), D=D.tolist(), image_size=[WIDTH, HEIGHT]))
        empty, H_truth, _ = render_camera(texture, board, position, dart=False,
                                          width=WIDTH, height=HEIGHT, focal=focal,
                                          look_at=look_at, roll=roll)
        # render_camera centers its K; translate principal point to the sampled K.
        shift = np.float32([[1, 0, K[0, 2] - WIDTH / 2], [0, 1, K[1, 2] - HEIGHT / 2]])
        empty = cv.warpAffine(empty, shift, (WIDTH, HEIGHT), borderMode=cv.BORDER_REFLECT)
        shift3 = np.vstack([shift, [0, 0, 1]])
        H_truth = H_truth @ np.linalg.inv(shift3)
        empty, _ = observed(empty, maps, int(seed + i * 2))
        empty = lens.apply(empty)
        try:
            H, info = automatic_homography(empty, board, nominal_homography(board, K, nominal))
        except (ValueError, cv.error, np.linalg.LinAlgError) as exc:
            result["reason"] = f"camera {i}: {exc}"
            result["outcome"] = "calibration_failed"
            return result
        error = np.linalg.norm(project(board.landmarks(), H @ np.linalg.inv(H_truth)) - board.landmarks(), axis=1)
        calibration.append(dict(camera=i, inliers=info["inliers"], median_px=float(np.median(error)),
                                max_px=float(error.max()), template_support=info["binary_template_support"]))
        dart, _, _ = render_camera(texture, board, position, tip=[xy[0], -xy[1], 0],
                                   width=WIDTH, height=HEIGHT, focal=focal,
                                   look_at=look_at, roll=roll)
        dart = cv.warpAffine(dart, shift, (WIDTH, HEIGHT), borderMode=cv.BORDER_REFLECT)
        dart, _ = observed(dart, maps, int(seed + i * 2 + 1))
        baseline = gray_image(warp(empty, H, board))
        image = gray_image(warp(lens.apply(dart), H, board))
        lines.append(shaft_line(change_mask(image, baseline, board.mask(1.08), 28), board))
    point, cameras, reason = intersect_lines(lines, tolerance=5, support_distance=16)
    result["cameras"], result["reason"] = cameras, reason
    if point is None:
        return result
    if np.linalg.norm(point - board.center) > board.radii[-1] * board.ppm * 1.08:
        result["reason"] = "intersection outside observable board area"
        return result
    estimate = board.score(point)
    result["estimate"] = estimate
    result["error_mm"] = float(np.linalg.norm(point - truth) / board.ppm)
    result["outcome"] = "correct" if estimate["label"] == truth_score["label"] else "wrong"
    result["requires_review"] = bool(len(cameras) < 3 or estimate["boundary_review"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--output", default="c920_benchmark/monte_carlo.json")
    args = p.parse_args()
    if args.trials < 1 or args.workers < 1:
        p.error("Trials and workers must be positive")
    seeds = np.random.default_rng(args.seed).integers(1, 2**30, args.trials)
    tasks = [(i, int(seed)) for i, seed in enumerate(seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(trial, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"{len(rows)}/{args.trials}: {dict(Counter(r['outcome'] for r in rows))}", flush=True)
    counts = Counter(r["outcome"] for r in rows)
    detected = [r for r in rows if r["estimate"] is not None]
    errors = [r["error_mm"] for r in detected]
    summary = dict(trials=args.trials, counts=dict(counts), estimates=len(detected),
                   estimate_coverage=len(detected) / args.trials,
                   correct_estimates=counts["correct"], wrong_estimates=counts["wrong"],
                   correct_score_fraction_of_estimates=counts["correct"] / len(detected) if detected else None,
                   review_flagged_estimates=sum(r["requires_review"] for r in detected),
                   three_view_estimates=sum(len(r["cameras"]) == 3 for r in detected),
                   position_error_median_mm=float(np.median(errors)) if errors else None,
                   position_error_p95_mm=float(np.percentile(errors, 95)) if errors else None,
                   position_error_max_mm=max(errors) if errors else None)
    result = dict(seed=args.seed, priors=PRIORS, summary=summary, trials=rows,
                  limitations=["Synthetic-only; these coefficient/extrinsic priors are assumed, not measured.",
                               "Exact lens K/D supplied. No test of blind lens calibration.",
                               "Each trial independently calibrates an empty board and detects one static dart.",
                               "No hands, occlusions, overlapping darts, realistic shadows or event timing."])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

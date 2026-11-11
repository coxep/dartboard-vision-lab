#!/usr/bin/env python3
"""Render/replay one recorded Monte Carlo setup and save the actual CV stages.

Camera truth creates the input and evaluates error only. The calibrator receives
pixels, exact synthetic K/D and nominal mounts, never the sampled true pose.
"""
import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from benchmark_c920 import WIDTH, HEIGHT, POSITIONS, distortion_map, observed, write_image
from dartboard import Board, Lens, automatic_homography, nominal_homography, warp, project
from pose_alignment import homography_from_pose
from render_synthetic import board_texture, render_camera, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='c920_benchmark/calibration_sweep_before_pnp.json')
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--output', default='calibration_evidence')
    args = parser.parse_args()
    cv.setNumThreads(1)
    row = json.loads(Path(args.report).read_text())['trials'][args.trial]
    cv.setRNGSeed(int(row['seed'] % (2**31-1)))
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=True)
    board = Board(); texture = board_texture(board)
    results, strips = [], []
    for i, parameters in enumerate(row['parameters']):
        K, D = np.array(parameters['K']), np.array(parameters['D'])
        ideal, truth, _ = render_camera(texture, board, parameters['position_mm'], dart=False,
                                        width=WIDTH, height=HEIGHT, focal=K[0,0],
                                        look_at=parameters['look_at_mm'], roll=parameters['roll_deg'])
        shift = np.float32([[1,0,K[0,2]-WIDTH/2], [0,1,K[1,2]-HEIGHT/2]])
        ideal = cv.warpAffine(ideal, shift, (WIDTH, HEIGHT), borderMode=cv.BORDER_REFLECT)
        truth = truth@np.linalg.inv(np.vstack([shift, [0,0,1]]))
        raw, jpg = observed(ideal, distortion_map(K,D,WIDTH,HEIGHT), row['seed']+i*2)
        (folder/f'camera_{i}_raw.jpg').write_bytes(jpg.tobytes())
        corrected = Lens(raw.shape, dict(K=K.tolist(), D=D.tolist(), image_size=[WIDTH, HEIGHT])).apply(raw)
        prior = nominal_homography(board, K, POSITIONS[i])
        H, info = automatic_homography(corrected, board, prior, K)
        before = np.linalg.norm(project(board.landmarks(), prior@np.linalg.inv(truth))-board.landmarks(), axis=1)
        after = np.linalg.norm(project(board.landmarks(), H@np.linalg.inv(truth))-board.landmarks(), axis=1)
        preliminary = board.grid(warp(corrected, prior, board))
        association = warp(corrected, prior, board)
        pose = info['pose_initialization']
        if pose is not None:
            pose_H = homography_from_pose(np.array(pose['rvec']), np.array(pose['tvec_mm']), K, board)
            fitted = board.grid(warp(corrected, pose_H, board))
            detected = project(pose['matched_source_px'], prior)
            model = board.landmarks()[pose['matched_model_ids']]
            for a,b in zip(model,detected):
                cv.line(association, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (0,210,255), 1, cv.LINE_AA)
                cv.circle(association, tuple(np.rint(b).astype(int)), 3, (90,255,130), -1)
        else:
            fitted = preliminary.copy()
        rectified = board.grid(warp(corrected, H, board))
        for name, image in [('prior',preliminary),('association',association),('pose',fitted),('final',rectified)]:
            write_image(folder/f'camera_{i}_{name}.jpg',image)
        strips.append(np.hstack([panel(image, title, caption, width=400, height=440) for image,title,caption in [
            (raw,f'CAMERA {i+1}: INPUT','Distorted JPEG, sampled pose withheld'),
            (preliminary,'NOMINAL POSE',f'Median landmark error: {np.median(before):.2f} board px'),
            (association,'ASSOCIATION','Green: detected features; yellow: model pairing'),
            (fitted,'SEEDED PNP','Physical pose after association / refinement loop'),
            (rectified,'DENSE REFINEMENT',f'Median: {np.median(after):.2f} px; max: {after.max():.2f} px')]]))
        results.append(dict(camera=i, prior_median_px=float(np.median(before)),
                            final_median_px=float(np.median(after)), final_max_px=float(after.max()),
                            previous_outcome=row['calibration'][i], calibration=info))
    write_image(folder/'pipeline.jpg',np.vstack(strips))
    (folder/'report.json').write_text(json.dumps(dict(trial=args.trial, seed=row['seed'], cameras=results,
        note='Every stage displays warped observed pixels. Geometry overlay is only an overlay.'),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('calibration','previous_outcome')} for r in results],indent=2))

if __name__=='__main__':
    main()

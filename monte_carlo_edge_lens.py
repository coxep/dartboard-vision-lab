#!/usr/bin/env python3
"""Replay recorded randomized poses with all true lens coefficients withheld."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from benchmark_c920 import WIDTH,HEIGHT,POSITIONS,observed,distortion_map
from dartboard import Board,nominal_homography,project
from render_synthetic import board_texture,render_camera
from edge_calibration import calibrate_unknown_radial,distort_pixels


def view(task):
    trial,seed,camera,p=task
    cv.setNumThreads(1);cv.setRNGSeed(seed%(2**31-1))
    b=Board();K=np.array(p['K']);truth_D=np.array(p['D'])
    ideal,truth_H,_=render_camera(board_texture(b),b,p['position_mm'],dart=False,
                                  width=WIDTH,height=HEIGHT,focal=K[0,0],look_at=p['look_at_mm'],roll=p['roll_deg'])
    shift=np.float32([[1,0,K[0,2]-WIDTH/2],[0,1,K[1,2]-HEIGHT/2]])
    ideal=cv.warpAffine(ideal,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
    truth_H=truth_H@np.linalg.inv(np.vstack([shift,[0,0,1]]))
    raw,_=observed(ideal,distortion_map(K,truth_D,WIDTH,HEIGHT),seed+camera*2)
    row=dict(trial=trial,camera=camera,seed=seed,parameters=p,success=False)
    try:
        H,lens,info,_,_=calibrate_unknown_radial(raw,b,K,nominal_homography(b,K,POSITIONS[camera]))
    except (ValueError,cv.error,np.linalg.LinAlgError) as exc:
        row['reason']=str(exc);return row
    angle=np.arange(257)*2*np.pi/257
    probes=b.center+np.c_[np.cos(angle),np.sin(angle)]*np.sqrt(np.linspace(.01,.99,257))[:,None]*b.radii[-1]*b.ppm
    raw_points=distort_pixels(project(probes,np.linalg.inv(truth_H)),K,truth_D)
    estimate=np.array(lens['D'])
    corrected=cv.undistortPoints(raw_points.reshape(-1,1,2),K,estimate,P=K).reshape(-1,2)
    before=np.linalg.norm(project(raw_points,np.array(info['joint_edge_fit']['seed_H']))-probes,axis=1)
    after=np.linalg.norm(project(corrected,H)-probes,axis=1)
    row.update(success=True,before_median_px=float(np.median(before)),before_max_px=float(before.max()),
               after_median_px=float(np.median(after)),after_max_px=float(after.max()),
               estimated_D=estimate.tolist(),k1_absolute_error=float(abs(estimate[0]-truth_D[0])),
               edge_median_source_px=info['joint_edge_fit']['median_source_px'],
               circle_detections=info['postfit_circle_detections'])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setups',type=int,default=20);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--source',default='c920_benchmark/calibration_sweep_hsv.json')
    p.add_argument('--output',default='edge_lens_evidence/monte_carlo.json')
    args=p.parse_args()
    source=json.loads(Path(args.source).read_text());tasks=[]
    for trial in source['trials'][:args.setups]:
        tasks.extend((trial['trial'],trial['seed'],i,params) for i,params in enumerate(trial['parameters']))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(view,tasks):
            rows.append(row)
            if len(rows)%10==0:print(f'{len(rows)}/{len(tasks)} views; {sum(r["success"] for r in rows)} accepted',flush=True)
    accepted=[r for r in rows if r['success']]
    summary=dict(views=len(rows),accepted=len(accepted),rejected=len(rows)-len(accepted),
                 median_view_error_before_px=float(np.median([r['before_median_px'] for r in accepted])),
                 median_view_error_after_px=float(np.median([r['after_median_px'] for r in accepted])),
                 worst_probe_before_px=max(r['before_max_px'] for r in accepted),worst_probe_after_px=max(r['after_max_px'] for r in accepted),
                 improved_view_medians=sum(r['after_median_px']<r['before_median_px'] for r in accepted),
                 worse_view_medians=sum(r['after_median_px']>=r['before_median_px'] for r in accepted),
                 accepted_views_above_2px=sum(r['after_max_px']>2 for r in accepted),
                 median_k1_absolute_error=float(np.median([r['k1_absolute_error'] for r in accepted])))
    result=dict(summary=summary,views=rows,
                known='Exact K, nominal mounts, board geometry. All distortion priors zero.',
                withheld='All sampled true distortion coefficients and camera poses, true H and independent probes.',
                method='Coarse initialization, then raw-edge joint H+k1; k2/p1/p2/k3 fixed zero. Post-fit Hough circles do not feed optimizer.',
                limitations='Synthetic replay of the first recorded setups. Internal acceptance gates do not guarantee ground-truth accuracy. No darts or temporal tracking tested.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

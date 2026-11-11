#!/usr/bin/env python3
"""Test joint radial lens / plane fitting with distortion withheld from calibration."""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,nominal_homography,automatic_homography,refine_plane_template,warp,project,text
from benchmark_c920 import WIDTH,HEIGHT,K,D,POSITIONS,observed,distortion_map,write_image
from render_synthetic import board_texture,render_camera,panel
from edge_calibration import joint_edge_fit,distort_pixels,detect_rectified_circles


def main():
    cv.setNumThreads(1);cv.setRNGSeed(0)
    b=Board();texture=board_texture(b)
    out=Path('edge_lens_evidence');out.mkdir(exist_ok=True)
    # Same mild illustrative C920-like distortion as the earlier fixed benchmark.
    truth_D=D.copy()
    results=[];rows=[]
    for i,position in enumerate(POSITIONS):
        ideal,H_truth,_=render_camera(texture,b,position,dart=False,width=WIDTH,height=HEIGHT,focal=K[0,0])
        raw,jpg=observed(ideal,distortion_map(K,truth_D,WIDTH,HEIGHT),100+i)
        (out/f'camera_{i}_raw.jpg').write_bytes(jpg.tobytes())
        prior=nominal_homography(b,K,position)
        try:seed,initial=automatic_homography(raw,b,prior,K)
        except (ValueError,cv.error):seed,initial=refine_plane_template(raw,prior,b)
        H,estimate,info,edges=joint_edge_fit(raw,b,K,seed)
        corrected=Lens(raw.shape,dict(K=K.tolist(),D=estimate.tolist(),image_size=[WIDTH,HEIGHT])).apply(raw)
        before=warp(raw,seed,b);after=warp(corrected,H,b)
        circles,overlay=detect_rectified_circles(after,b)
        probes=b.center+np.c_[np.cos(np.arange(257)*2*np.pi/257),np.sin(np.arange(257)*2*np.pi/257)]*np.sqrt(np.linspace(.01,.99,257))[:,None]*b.radii[-1]*b.ppm
        image_points=distort_pixels(project(probes,np.linalg.inv(H_truth)),K,truth_D)
        undistorted=cv.undistortPoints(image_points.reshape(-1,1,2),K,estimate,P=K).reshape(-1,2)
        before_error=np.linalg.norm(project(image_points,seed)-probes,axis=1)
        after_error=np.linalg.norm(project(undistorted,H)-probes,axis=1)
        edge_overlay=raw.copy();edge_overlay[edges[0]>0]=(0,210,255);edge_overlay[edges[1]>0]=(80,255,130)
        for name,image in [('edges',edge_overlay),('before',before),('after',after),('circles',overlay)]:write_image(out/f'camera_{i}_{name}.jpg',image)
        result=dict(camera=i,truth_D_evaluation_only=truth_D.tolist(),estimated_D=estimate.tolist(),
                    before_probe_median_px=float(np.median(before_error)),before_probe_max_px=float(before_error.max()),
                    after_probe_median_px=float(np.median(after_error)),after_probe_max_px=float(after_error.max()),
                    fit=info,circle_detector=circles)
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('fit','circle_detector')}),flush=True)
        rows.append(np.hstack([panel(image,title,caption,width=430,height=480) for image,title,caption in [
            (edge_overlay,f'CAMERA {i+1}: OBSERVED EDGES','Yellow: HSV rings; green: adaptive wedge edges'),
            (before,'DISTORTION ASSUMED ZERO',f'H-only probe median {np.median(before_error):.2f}px; max {before_error.max():.2f}px'),
            (after,'JOINT H + K1 DISTORTION',f'Probe median {np.median(after_error):.2f}px; max {after_error.max():.2f}px'),
            (overlay,'PAIRED HSV BOUNDARY CIRCLES','Free circle fits to observed samples; no nominal redraw')]]))
    write_image(out/'comparison.jpg',np.vstack(rows))
    (out/'report.json').write_text(json.dumps(dict(cameras=results,
        known_to_fit='Exact K, board geometry, nominal camera pose. Distortion prior is zero.',
        withheld='All true distortion coefficients, true H, independent probe locations.',
        limitation='Fixed synthetic poses only. H+k1 estimated; k2/p1/p2/k3 fixed to zero. Paired HSV boundary circles are a post-fit consistency diagnostic, not independent accuracy ground truth.'),indent=2)+'\n')

if __name__=='__main__':main()

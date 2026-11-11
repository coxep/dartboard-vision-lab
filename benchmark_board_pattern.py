#!/usr/bin/env python3
"""Fresh grayscale pattern measurements: moving cameras, loss, and acquisition.

True H and true camera motion enter rendering and evaluation only. Lens K/D and
the nominal mount belief are supplied; no appearance reference is supplied.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import cv2 as cv
import numpy as np
from board_pattern import PatternDetector,GeometryTracker,draw_pattern
from dartboard import Board,Lens,project,nominal_homography,warp,text
from benchmark_c920 import WIDTH,HEIGHT,K,D,POSITIONS,observed,distortion_map,write_image
from benchmark_detection import camera_parameters
from render_synthetic import board_texture,render_camera,panel


def metric(H,truth,b):
    angles=np.arange(257)*2*np.pi/257
    radii=np.sqrt(np.linspace(.01,.99,257))*b.radii[-1]*b.ppm
    points=b.center+np.c_[np.cos(angles),np.sin(angles)]*radii[:,None]
    error=np.linalg.norm(project(points,H@np.linalg.inv(truth))-points,axis=1)
    return dict(probe_median_px=float(np.median(error)),probe_max_px=float(error.max()))


def pose_metric(H,truth,K,b):
    from pose_alignment import pose_from_homography
    r,t=pose_from_homography(H,K,b);rt,tt=pose_from_homography(truth,K,b)
    R=cv.Rodrigues(r)[0];Rt=cv.Rodrigues(rt)[0]
    angle=np.degrees(np.arccos(np.clip((np.trace(R@Rt.T)-1)/2,-1,1)))
    return dict(rotation_error_deg=float(angle),camera_center_error_mm=float(np.linalg.norm(-R.T@t+Rt.T@tt)))


def cold_trial(task):
    index,seed=task;cv.setNumThreads(1);cv.setRNGSeed(int(seed%(2**31-1)))
    rng=np.random.default_rng(seed);rng.uniform(0,182);rng.uniform(0,360)
    parameters=camera_parameters(rng,True);b=Board();texture=board_texture(b);detector=PatternDetector(b);rows=[]
    for i,p in enumerate(parameters):
        k,d=p['K'],p['D']
        image,truth,_=render_camera(texture,b,p['position'],dart=False,width=WIDTH,height=HEIGHT,
                                    focal=k[0,0],look_at=p['look_at'],roll=p['roll'])
        shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
        image=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
        truth=truth@np.linalg.inv(np.vstack([shift,[0,0,1]]))
        raw,_=observed(image,distortion_map(k,d,WIDTH,HEIGHT),seed+i*2)
        corrected=Lens(raw.shape,dict(K=k.tolist(),D=d.tolist(),image_size=[WIDTH,HEIGHT])).apply(raw)
        gray=cv.cvtColor(corrected,cv.COLOR_BGR2GRAY)
        H,report=detector.acquire(gray,nominal_homography(b,k,POSITIONS[i]),k)
        row=dict(trial=index,camera=i,found=H is not None,reason=report['reason'],elapsed_ms=report['elapsed_ms'],
                 acquisition=report.get('acquisition'),inliers=report.get('inliers'),seed=seed)
        if H is not None:
            row.update(metric(H,truth,b),**pose_metric(H,truth,k,b))
        rows.append(row)
    return rows


def sequence(output,frames):
    cv.setNumThreads(1);cv.setRNGSeed(0)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    b=Board();texture=board_texture(b);maps=distortion_map(K,D,WIDTH,HEIGHT);rows=[];montages=[]
    for camera,nominal in enumerate(POSITIONS):
        tracker=GeometryTracker(None,nominal_homography(b,K,nominal),b,K)
        lens=Lens((HEIGHT,WIDTH,3),dict(K=K.tolist(),D=D.tolist(),image_size=[WIDTH,HEIGHT]))
        for frame in range(frames):
            blank=40<=frame<45;partial=20<=frame<30
            # Stay roughly within two inches of nominal, including a later jump.
            offset=np.array([12*np.sin(frame/18),6*np.sin(frame/14),3*np.sin(frame/11)])
            if frame>=45:offset+=np.array([28.,-14.,8.])
            position=np.array(nominal)+offset
            current=texture.copy()
            if partial:cv.rectangle(current,(320,170),(450,320),(25,25,25),-1)
            if frame>=30:
                # Fresh texture and exposure: a saved appearance is not available.
                rng=np.random.default_rng(10000+camera*100+frame)
                current=np.uint8(np.clip(current.astype(float)*(.8+.15*np.sin(frame/6))+rng.normal(0,5,current.shape),0,255))
            image,truth,_=render_camera(current,b,position,dart=frame>=12,width=WIDTH,height=HEIGHT,focal=K[0,0])
            raw,jpg=observed(image,maps,2000+camera*100+frame)
            if blank:
                raw[:]=0
                _,jpg=cv.imencode('.jpg',raw,[cv.IMWRITE_JPEG_QUALITY,94])
            corrected=lens.apply(raw);gray=cv.cvtColor(corrected,cv.COLOR_BGR2GRAY)
            prior=tracker.H.copy();start=time.perf_counter();tracker.track(gray)
            recovered=False
            if not tracker.valid and not blank:
                tracker.update(gray);recovered=tracker.valid
            elapsed=(time.perf_counter()-start)*1000
            report=tracker.report
            row=dict(camera=camera,frame=frame,blank=blank,partial_occlusion=partial,found=tracker.valid,
                     reacquired=recovered,elapsed_ms=elapsed,reason=tracker.reason,inliers=report.get('inliers'),
                     median_residual_px=report.get('median_residual_px'))
            if tracker.valid:
                row.update(metric(tracker.H,truth,b),**pose_metric(tracker.H,truth,K,b))
            rows.append(row)
            if frame in (0,12,24,40,45,frames-1):
                rect=warp(gray,tracker.H,b);overlay=draw_pattern(rect,b,report,tracker.H)
                text(overlay,'DETECTED' if tracker.valid else 'LOST',(18,30),
                     color=(60,255,100) if tracker.valid else (40,40,255))
                (out/f'camera_{camera}_frame_{frame:03d}_input.jpg').write_bytes(jpg.tobytes())
                write_image(out/f'camera_{camera}_frame_{frame:03d}_graph.jpg',overlay)
                before=warp(gray,prior,b)
                montage=np.hstack([panel(raw,'CAMERA INPUT',f'Camera {camera+1}, frame {frame}; mild lens distortion',width=430,height=460),
                    panel(before,'PRIOR PLANE PROJECTION','Actual grayscale pixels, before this frame\'s measurement',width=430,height=460),
                    panel(overlay,'FRESH JUNCTION GRAPH',f"{report.get('inliers',0)} inliers; {tracker.reason[:48]}",width=430,height=460),
                    panel(rect,'RECTIFIED OBSERVED PIXELS',f"{elapsed:.1f}ms; "+(f"max GT probe {row['probe_max_px']:.2f}px" if tracker.valid else 'invalid: scoring paused'),width=430,height=460)])
                write_image(out/f'camera_{camera}_frame_{frame:03d}_pipeline.jpg',montage)
                if frame in (24,45):montages.append(montage)
    write_image(out/'sequence_comparison.jpg',np.vstack(montages))
    return rows


def summarize(rows):
    good=[r for r in rows if r['found']];times=[r['elapsed_ms'] for r in good]
    result=dict(views=len(rows),found=len(good),rejected=len(rows)-len(good),
                median_detection_ms=float(np.median(times)) if times else None,
                p95_detection_ms=float(np.percentile(times,95)) if times else None,
                worst_detection_ms=max(times,default=None),
                median_view_probe_error_px=float(np.median([r['probe_median_px'] for r in good])) if good else None,
                worst_probe_error_px=max((r['probe_max_px'] for r in good),default=None),
                accepted_above_2px=sum(r['probe_max_px']>2 for r in good))
    if good and 'rotation_error_deg' in good[0]:
        for key in ('rotation_error_deg','camera_center_error_mm'):
            result['median_'+key]=float(np.median([r[key] for r in good]))
            result['worst_'+key]=max(r[key] for r in good)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='pattern_evidence');p.add_argument('--frames',type=int,default=60)
    p.add_argument('--cold-trials',type=int,default=20);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261114);p.add_argument('--skip-sequence',action='store_true')
    p.add_argument('--skip-acquisition',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.frames<1 or args.cold_trials<1 or args.workers<1:p.error('Positive frame/trial/worker counts required')
    if not args.skip_sequence:
        rows=sequence(out,args.frames);s=summarize(rows)
        s['expected_blank_views']=sum(r['blank'] for r in rows)
        s['false_accepts_on_blank']=sum(r['blank'] and r['found'] for r in rows)
        s['visible_board_misses']=sum(not r['blank'] and not r['found'] for r in rows)
        (out/'sequence.json').write_text(json.dumps(dict(summary=s,frames=rows,
            timing='Grayscale detector/tracker CPU time, one OpenCV thread. Rendering, decoding, lens correction and capture excluded. Not end-to-end FPS.',
            limitations='Synthetic sequence; known K/D, nominal mount prior. Flat patch occlusion; no real hand or depth-tested flight occlusion.'),indent=2)+'\n')
        print('SEQUENCE',json.dumps(s),flush=True)
    if args.skip_acquisition:return
    seeds=np.random.default_rng(args.seed).integers(1,2**30,args.cold_trials);rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(cold_trial,[(i,int(s)) for i,s in enumerate(seeds)]):
            rows+=result;print(f'Acquisition {len(rows)//3}/{args.cold_trials}: {sum(r["found"] for r in rows)}/{len(rows)}',flush=True)
    summary=summarize(rows)
    report=dict(seed=args.seed,summary=summary,views=rows,known_to_detector='Exact K/D and nominal mount orientation/position; no saved board photograph.',
                withheld='Sampled true pose and ground-truth homography; used only for rendering/evaluation.',
                priors='Positions sampled within a 50.8mm ball; aim sigma8mm; roll clipped4deg; FOV75-81deg; existing Monte Carlo Brown distortion ranges.',
                timing='Each worker uses one OpenCV thread; acquisition wall time includes local search and any whole-pattern/PnP fallback. Concurrent workers contend for CPU.')
    (out/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

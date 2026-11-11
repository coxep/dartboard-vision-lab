#!/usr/bin/env python3
"""Frozen current grayscale calibration: independent +/-2-inch pose trials.

Actual poses/H enter rendering and evaluation only. Lens K/D are supplied to
undistortion; the detector receives nominal H/K, nominal wire model and pixels.
Joint multi-camera calibration is NOT implemented or simulated by this harness.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,project,warp,nominal_homography,gray_image,change_mask
from board_pattern import PatternDetector,draw_pattern
from render_synthetic import board_texture,render_camera,panel
from benchmark_c920 import WIDTH,HEIGHT,K,POSITIONS,CASES,observed,distortion_map,write_image
from benchmark_detection import score_result
from dart_detection import detect_dart


def initialize(pose_only=False):
    global BOARD,TEXTURE,DETECTORS,PRIORS,PROBES,POSE_ONLY
    POSE_ONLY=pose_only
    cv.setNumThreads(1)
    BOARD=Board(radii=(97.4,107.4,160.,170.));TEXTURE=board_texture(BOARD,wires=False)
    DETECTORS=[PatternDetector(BOARD,dict(camera_center_mm=C,height_mm=1.,width_mm=.8)) for C in POSITIONS]
    PRIORS=[nominal_homography(BOARD,K,C) for C in POSITIONS]
    yy,xx=np.mgrid[-170:171:10,-170:171:10];xy=np.c_[xx.ravel(),yy.ravel()]
    xy=xy[(xy*xy).sum(axis=1)<=170**2]
    perimeter=np.array([BOARD.point(170,a) for a in np.arange(0,360,2)])
    PROBES=np.vstack([BOARD.center+xy*BOARD.ppm*np.array([1,-1]),perimeter]).astype(np.float32)


def ball(rng,radius=50.8):
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    return direction*radius*rng.random()**(1/3)


def plane_error(H,source,visible):
    errors=np.linalg.norm(project(source,H)-PROBES,axis=1)/BOARD.ppm
    return dict(median_visible_mm=float(np.median(errors[visible])),
                p95_visible_mm=float(np.percentile(errors[visible],95)),
                max_visible_mm=float(errors[visible].max()),max_full_plane_mm=float(errors.max()))


def score(masks,truth):
    if any(m is None for m in masks):return dict(outcome='calibration_rejected',estimate=None,error_mm=None)
    point,ids,reason,_,details=detect_dart(masks,BOARD)
    row=score_result(point,ids,reason,BOARD,truth)
    row['candidate_counts']=details.get('candidate_counts')
    return row


def trial(index,seed,save_folder=None):
    cv.setRNGSeed(int(seed%(2**31-1)));rng=np.random.default_rng(seed)
    pose_rng=np.random.default_rng(seed+174603)
    _,radius,angle,*_=CASES[index%len(CASES)]
    truth=BOARD.point(radius,angle)
    tip_xy=(truth-BOARD.center)/BOARD.ppm*np.array([1,-1])
    dart=dict(direction=[float(rng.uniform(-.25,.25)),float(rng.uniform(-.25,.25)),1.],
              length=float(rng.uniform(65,80)),flight_width=float(rng.uniform(3,6)),
              flight_color=rng.integers(40,235,3).tolist(),barrel_color=rng.integers(70,180,3).tolist())
    wire=dict(height_mm=float(rng.uniform(.8,1.2)),width_mm=float(rng.uniform(.6,1.)))
    if POSE_ONLY:wire=dict(height_mm=1.,width_mm=.8)
    rows=[];estimated_masks=[];oracle_masks=[];tiles=[]
    for i,nominal in enumerate(POSITIONS):
        offset=ball(pose_rng);C=np.asarray(nominal)+offset
        aim=[float(pose_rng.normal(0,8)),float(pose_rng.normal(0,8)),0.]
        roll=float(np.clip(pose_rng.normal(0,1.5),-4,4));fov=float(rng.uniform(75,81))
        f=math.hypot(WIDTH,HEIGHT)/(2*math.tan(math.radians(fov/2)))
        k=np.array([[f,0,WIDTH/2+rng.normal(0,8)],[0,f,HEIGHT/2+rng.normal(0,6)],[0,0,1.]])
        d=np.array([rng.uniform(-.08,.02),rng.uniform(-.02,.04),rng.uniform(-.001,.001),rng.uniform(-.001,.001),0.])
        gamma=float(rng.uniform(.85,1.2));gain=float(rng.uniform(.75,1.2))
        white=rng.uniform(.92,1.08,3)
        if POSE_ONLY:
            fov=78.;f=float(K[0,0]);k=K.copy();d=np.array([-.045,.012,.0007,-.0005,0.])
            gamma=1.;gain=1.;white=np.ones(3)
        options=dict(width=WIDTH,height=HEIGHT,focal=f,look_at=aim,roll=roll,wire_geometry=wire)
        clear,truth_H,_=render_camera(TEXTURE,BOARD,C,dart=False,**options)
        changed,_,_=render_camera(TEXTURE,BOARD,C,tip=[*tip_xy,0.],dart_config=dart,**options)
        shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
        truth_H=truth_H@np.linalg.inv(np.vstack([shift,[0,0,1]]))
        maps=distortion_map(k,d,WIDTH,HEIGHT)
        lens=Lens(clear.shape,dict(K=k.tolist(),D=d.tolist(),image_size=[WIDTH,HEIGHT]))
        def capture(image,n):
            shifted=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
            lit=np.uint8(np.clip(255*(shifted.astype(np.float32)/255)**gamma*gain*white,0,255))
            raw,_=observed(lit,maps,seed+i*11+n)
            return raw,lens.apply(raw)
        raw,reference=capture(clear,0);_,changed=capture(changed,1)
        start=time.perf_counter()
        H,report=DETECTORS[i].acquire(gray_image(reference),PRIORS[i],K=K)
        elapsed=(time.perf_counter()-start)*1000
        source=project(PROBES,np.linalg.inv(truth_H))
        footprint=lens.apply(np.full((HEIGHT,WIDTH),255,np.uint8))
        visible=cv.remap(footprint,source[:,0,None].astype(np.float32),source[:,1,None].astype(np.float32),cv.INTER_LINEAR).ravel()>254
        if not visible.any():raise RuntimeError('No evaluation probes visible')
        row=dict(camera=i,offset_mm=offset.tolist(),offset_radius_mm=float(np.linalg.norm(offset)),
                 true_camera_mm=C.tolist(),aim_mm=aim,roll_deg=roll,fov_deg=fov,K=k.tolist(),D=d.tolist(),
                 gamma=gamma,gain=gain,white_balance=white.tolist(),accepted=H is not None,
                 reason=report['reason'],acquisition=report.get('acquisition'),elapsed_ms=elapsed,visible_probe_fraction=float(visible.mean()),
                 initial=plane_error(PRIORS[i],source,visible),H=H.tolist() if H is not None else None,
                 truth_H_evaluation_only=truth_H.tolist())
        if H is not None:row['estimated']=plane_error(H,source,visible)
        rows.append(row)
        for target,h in ((estimated_masks,H),(oracle_masks,truth_H)):
            if h is None:target.append(None);continue
            rect=warp(changed,h,BOARD);base=warp(reference,h,BOARD)
            target.append(change_mask(gray_image(rect),gray_image(base),BOARD.mask(1.08),28))
        if save_folder is not None:
            out=Path(save_folder);out.mkdir(parents=True,exist_ok=True)
            write_image(out/f'camera_{i}_input.jpg',raw)
            final=warp(reference,H,BOARD) if H is not None else np.zeros((BOARD.size,BOARD.size,3),np.uint8)
            checked=draw_pattern(final,BOARD,report,H) if H is not None else final
            subtitle=f"max visible error {row['estimated']['max_visible_mm']:.3f} mm" if H is not None else report['reason'][:75]
            tiles.append(np.hstack([panel(raw,f'CAMERA {i+1}',f'Offset {np.linalg.norm(offset):.1f} mm; raw distorted input',width=400,height=420),
                panel(warp(reference,PRIORS[i],BOARD),'INITIAL BELIEF','Nominal centre, FOV 78 deg, aim at bull',width=400,height=420),
                panel(checked,'ACCEPTED FIT' if H is not None else 'REJECTED',subtitle,width=400,height=420),
                panel(warp(reference,truth_H,BOARD),'EXACT H: EVALUATION ONLY','Not passed to the detector',width=400,height=420)]))
    result=dict(trial=index,seed=seed,true_wire=wire,dart_config=dart,target=BOARD.score(truth),views=rows,
                estimated_score=score(estimated_masks,truth),oracle_score=score(oracle_masks,truth))
    if save_folder is not None:write_image(Path(save_folder)/'pipeline.jpg',np.vstack(tiles))
    return result


def summary(trials):
    views=[v for t in trials for v in t['views']];good=[v for v in views if v['accepted']]
    errors=np.array([v['estimated']['max_visible_mm'] for v in good])
    times=[v['elapsed_ms'] for v in views]
    answer=dict(triplets=len(trials),views=len(views),accepted_views=len(good),rejected_views=len(views)-len(good),
        all_three_accepted=sum(all(v['accepted'] for v in t['views']) for t in trials),
        camera_acceptance=[dict(camera=i,accepted=sum(v['accepted'] for v in views if v['camera']==i),total=sum(v['camera']==i for v in views)) for i in range(3)],
        initial_median_max_visible_mm=float(np.median([v['initial']['max_visible_mm'] for v in views])),
        accepted_median_max_visible_mm=float(np.median(errors)) if len(errors) else None,
        accepted_p95_max_visible_mm=float(np.percentile(errors,95)) if len(errors) else None,
        accepted_worst_max_visible_mm=float(errors.max()) if len(errors) else None,
        accepted_above_1mm=int((errors>1).sum()),rejection_reasons=dict(Counter(v['reason'] for v in views if not v['accepted'])),
        median_acquisition_ms=float(np.median(times)),p95_acquisition_ms=float(np.percentile(times,95)))
    for mode in ('estimated_score','oracle_score'):
        rows=[t[mode] for t in trials];valid=[r for r in rows if r.get('error_mm') is not None]
        answer[mode]=dict(outcomes=dict(Counter(r['outcome'] for r in rows)),reported=len(valid),
            median_error_mm=float(np.median([r['error_mm'] for r in valid])) if valid else None,
            p95_error_mm=float(np.percentile([r['error_mm'] for r in valid],95)) if valid else None,
            worst_error_mm=max((r['error_mm'] for r in valid),default=None),
            review_flagged=sum(r.get('requires_review',False) for r in rows),
            wrong_unflagged=sum(r['outcome']=='wrong' and not r.get('requires_review',False) for r in rows))
    return answer


def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    views=[v for t in report['trials'] for v in t['views']];good=[v for v in views if v['accepted']];s=report['summary']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax in axes:ax.grid(alpha=.2)
    for i,color in enumerate(('#227c9d','#17a589','#cc7722')):
        rows=[v for v in good if v['camera']==i]
        axes[0].scatter([v['offset_radius_mm'] for v in rows],[v['estimated']['max_visible_mm'] for v in rows],s=16,alpha=.65,color=color,label=f'Camera {i+1}')
    axes[0].axvline(50.8,color='gray',ls=':');axes[0].axhline(1,color='red',ls='--',lw=1)
    axes[0].set(xlabel='Camera-centre displacement (mm)',ylabel='Max visible plane error per accepted view (mm)',title=f"Accepted {s['accepted_views']}/{s['views']} views; rejected {s['rejected_views']}")
    axes[0].legend(fontsize=8)
    for name,values,color in [('Initial belief',[v['initial']['max_visible_mm'] for v in views],'#999999'),('Accepted fit',[v['estimated']['max_visible_mm'] for v in good],'#227c9d')]:
        if values:
            x=np.sort(values);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=name,color=color)
    axes[1].set(xlabel='Max visible plane error (mm)',ylabel='Fraction of views',title='Initial belief and accepted calibration');axes[1].set_xscale('log');axes[1].legend(fontsize=8)
    score_summary=s['estimated_score'];counts=score_summary['outcomes']
    labels=['correct','wrong','rejected','calibration_rejected']
    axes[2].bar(range(4),[counts.get(k,0) for k in labels],color=['#17a589','#c0392b','#cc7722','#999999'])
    axes[2].set_xticks(range(4),['Correct','Wrong','Dart\nrejected','Calibration\nrejected']);axes[2].set(ylabel='Three-camera trials',title='Settled-frame dart results: all trials')
    fig.suptitle('Current independent grayscale calibration: uniform 50.8 mm pose ball\nSynthetic raised wires; known lens correction; no joint calibration'+(' | pose-only control' if report['pose_only'] else ' | mixed stress'),fontsize=12)
    fig.savefig(out/'summary.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=100);p.add_argument('--seed',type=int,default=2026111107)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--output',default='monte_carlo_current')
    p.add_argument('--examples',type=int,default=3)
    p.add_argument('--pose-only',action='store_true',help='Hold lens/photometry/wire dimensions nominal; retain the same sampled poses as the mixed-stress run')
    args=p.parse_args()
    if args.trials<1 or args.workers<1:p.error('positive counts required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    seeds=np.random.default_rng(args.seed).integers(1,2**30,args.trials).tolist();results=[];start=time.perf_counter()
    # New output files on every run; an interrupted run retains completed trials.
    with (out/'trials.jsonl').open('w') as log,ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.pose_only,)) as pool:
        futures={pool.submit(trial,i,int(seed)):(i,seed) for i,seed in enumerate(seeds)}
        for f in as_completed(futures):
            row=f.result();results.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if len(results)%5==0 or len(results)==args.trials:
                print(f"{len(results)}/{args.trials} triplets: {sum(v['accepted'] for t in results for v in t['views'])}/{3*len(results)} views accepted",flush=True)
    results.sort(key=lambda r:r['trial'])
    report=dict(seed=args.seed,pose_only=args.pose_only,trials=results,summary=summary(results),wall_seconds=time.perf_counter()-start,
        algorithm_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        algorithm_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('board_pattern.py','template_registration.py','dartboard.py','wire_model.py','dart_detection.py')},
        sampling=dict(nuisance_mode='Nominal fixed lens/photometry/wires' if args.pose_only else 'Randomized lens/photometry/wires',camera_centre='Independent uniform 3D balls, radius 50.8mm about nominal centres',
            aim='Normal(0,8mm) per board-plane axis; board upright',roll='Normal(0,1.5deg), clipped +/-4deg',
            fov='Uniform diagonal 75-81deg; principal-point sigma 8px/6px',
            distortion='Brown k1[-.08,.02], k2[-.02,.04], p1,p2[-.001,.001], k3=0',
            wire='True height[.8,1.2]mm, width[.6,1]mm; model assumes 1/.8mm and nominal centre',
            photometry='Gamma[.85,1.2], gain[.75,1.2], RGB gains[.92,1.08], mild vignetting, blur, noise, JPEG94',
            darts='Eight fixed target types cycle; tilt x/y[-.25,.25], length[65,80]mm, flight width[3,6]mm'),
        detector_inputs='Corrected grayscale pixels, nominal H and K from FOV78deg/centre/aim, nominal raised-wire geometry. The nominal K supports broad acquisition, matching calibrate CLI; no true pose or H supplied.',
        lens_assumption='Exact synthetic K/D supplied to undistortion. Unknown-lens estimation is not tested.',
        metrics='Plane errors in mm over visible grid/perimeter probes; max per view. Rejections retained. Oracle scoring uses true H for evaluation only.',
        timing='Concurrent acquisition wall time per camera, one OpenCV thread per worker; excludes rendering/undistortion, not live FPS.',
        limitations=['Independent per-camera calibration, not the proposed joint solver.','One textured synthetic board; assumed lens/wire ranges, not measured C920 data.','Static before/after pairs, no live event timing, hands or full depth-tested occlusion.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');plot(report,out)
    if args.examples:
        initialize(args.pose_only)
        rejected=[r for r in results if any(not v['accepted'] for v in r['views'])]
        any_good=[r for r in results if any(v['accepted'] for v in r['views'])]
        good=[r for r in results if all(v['accepted'] for v in r['views'])]
        selected=[]
        if rejected:selected.append(('rejected',rejected[0]))
        if good:
            ranked=sorted(good,key=lambda r:max(v['estimated']['max_visible_mm'] for v in r['views']))
            selected.append(('median',ranked[len(ranked)//2]))
        if any_good:
            worst=max(any_good,key=lambda r:max(v['estimated']['max_visible_mm'] for v in r['views'] if v['accepted']))
            selected.append(('worst_accepted_view',worst))
        for label,row in selected[:args.examples]:
            replay=trial(row['trial'],row['seed'],out/label)
            assert [v['accepted'] for v in replay['views']]==[v['accepted'] for v in row['views']]
        report['examples']={label:dict(trial=row['trial'],seed=row['seed'],image=f'{label}/pipeline.jpg') for label,row in selected[:args.examples]}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':main()

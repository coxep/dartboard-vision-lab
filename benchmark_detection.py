#!/usr/bin/env python3
"""Paired scoring ablation: legacy/new detector x truth/estimated rectification.

Ground-truth H enters only the clearly labelled oracle diagnostic and evaluation.
The new detector itself receives only masks and board size. Known K/D are supplied
to both detectors. All modes see the same decoded camera images.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,automatic_homography,change_mask,gray_image,warp,project,nominal_homography
from benchmark_c920 import WIDTH,HEIGHT,FOCAL,K,D,POSITIONS,CASES,distortion_map,observed,write_image
from render_synthetic import board_texture,render_camera,panel,COLORS
from detection_baseline import shaft_line as legacy_axis,intersect_lines as legacy_intersect
from dart_detection import detect_dart,diagnostic_image


MODES=('legacy_oracle','legacy_estimated','new_oracle','new_estimated')


def camera_parameters(rng,randomized):
    result=[]
    for nominal in POSITIONS:
        if not randomized:
            result.append(dict(position=np.array(nominal),look_at=[0.,0.,0.],roll=0.,K=K.copy(),D=D.copy()))
            continue
        direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        position=np.asarray(nominal)+direction*50.8*rng.random()**(1/3)
        look_at=[float(rng.normal(0,8)),float(rng.normal(0,8)),0.]
        roll=float(np.clip(rng.normal(0,1.5),-4,4))
        focal=math.hypot(WIDTH,HEIGHT)/(2*math.tan(math.radians(rng.uniform(75,81)/2)))
        k=np.array([[focal,0,WIDTH/2+rng.normal(0,8)],[0,focal,HEIGHT/2+rng.normal(0,6)],[0,0,1.]])
        d=np.array([rng.uniform(-.08,.02),rng.uniform(-.02,.04),rng.uniform(-.001,.001),rng.uniform(-.001,.001),0.])
        result.append(dict(position=position,look_at=look_at,roll=roll,K=k,D=d))
    return result


def score_result(point,ids,reason,board,truth):
    if point is not None and np.linalg.norm(point-board.center)>board.radii[-1]*board.ppm*1.08:
        point=None;reason='intersection outside observable scoring area'
    result=dict(outcome='rejected',reason=reason,cameras=ids,estimate=None,error_mm=None)
    if point is not None:
        estimate=board.score(point);target=board.score(truth)
        result.update(outcome='correct' if estimate['label']==target['label'] else 'wrong',estimate=estimate,
                      pixel=point.tolist(),error_mm=float(np.linalg.norm(point-truth)/board.ppm),
                      requires_review=len(ids)<3 or estimate['boundary_review'])
    return result


def trial(task):
    index,seed,fixed,stress,output,save_images=task
    cv.setNumThreads(1);cv.setRNGSeed(int(seed%(2**31-1)))
    rng=np.random.default_rng(seed);board=Board();texture=board_texture(board)
    if fixed:
        name,radius,angle,*_=CASES[index]
    else:name=f'trial_{index:03d}';radius,angle=rng.uniform(0,182),rng.uniform(0,360)
    truth=board.point(radius,angle);xy=(truth-board.center)/board.ppm
    parameters=camera_parameters(rng,not fixed)
    dart_rng=np.random.default_rng(seed+77777)
    config={};existing=[]
    if stress:
        config=dict(direction=[float(dart_rng.uniform(-.45,.45)),float(dart_rng.uniform(-.45,.45)),1.],
                    length=float(dart_rng.uniform(60,90)),flight_width=float(dart_rng.uniform(3,8)),
                    flight_color=[int(x) for x in dart_rng.integers(40,235,3)],
                    barrel_color=[int(x) for x in dart_rng.integers(70,180,3)])
        # Baseline darts appear unchanged in both input frames, not painted out.
        for _ in range(int(dart_rng.integers(0,3))):
            p=board.point(dart_rng.uniform(20,165),dart_rng.uniform(0,360))
            q=(p-board.center)/board.ppm
            existing.append(dict(tip=[float(q[0]),float(-q[1]),0.],direction=[.12,-.18,1.]))
    masks={key:[] for key in ('oracle','estimated')};images={key:[] for key in masks}
    calibration=[];raw_pairs=[]
    for i,par in enumerate(parameters):
        k,d=par['K'],par['D']
        options=dict(width=WIDTH,height=HEIGHT,focal=k[0,0],look_at=par['look_at'],roll=par['roll'])
        clear,H_truth,_=render_camera(texture,board,par['position'],dart=False,**options)
        shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
        H_truth=H_truth@np.linalg.inv(np.vstack([shift,[0,0,1]]))
        maps=distortion_map(k,d,WIDTH,HEIGHT)
        lens=Lens((HEIGHT,WIDTH,3),dict(K=k.tolist(),D=d.tolist(),image_size=[WIDTH,HEIGHT]))
        def capture(image,noise_seed):
            translated=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
            raw,jpg=observed(translated,maps,noise_seed)
            return raw,lens.apply(raw),jpg
        clear_raw,clear,clear_jpg=capture(clear,seed+i*2)
        if existing:
            reference,_,_=render_camera(texture,board,par['position'],dart=False,existing_darts=existing,**options)
            reference_raw,reference,reference_jpg=capture(reference,seed+i*2)
        else:reference_raw,reference,reference_jpg=clear_raw,clear,clear_jpg
        changed,_,_=render_camera(texture,board,par['position'],tip=[xy[0],-xy[1],0.],dart_config=config,
                                   existing_darts=existing,**options)
        changed_raw,changed,changed_jpg=capture(changed,seed+i*2+1)
        raw_pairs.append((reference_raw,changed_raw,changed_jpg))
        H=None
        try:
            H,info=automatic_homography(clear,board,nominal_homography(board,k,POSITIONS[i]),k)
            error=np.linalg.norm(project(board.landmarks(),H@np.linalg.inv(H_truth))-board.landmarks(),axis=1)
            calibration.append(dict(camera=i,success=True,median_px=float(np.median(error)),max_px=float(error.max()),H=H.tolist(),oracle_H_evaluation_only=H_truth.tolist()))
        except (ValueError,cv.error,np.linalg.LinAlgError) as exc:
            calibration.append(dict(camera=i,success=False,reason=str(exc)))
        for key,h in [('oracle',H_truth),('estimated',H)]:
            if h is None:masks[key].append(None);images[key].append(None);continue
            rect=warp(changed,h,board);base=warp(reference,h,board)
            mask=change_mask(gray_image(rect),gray_image(base),board.mask(1.08),28)
            masks[key].append(mask);images[key].append(rect)
    results={};evidence=[]
    folder=Path(output)/name
    for mode in MODES:
        key=mode.split('_')[1]
        if any(m is None for m in masks[key]):
            results[mode]=dict(outcome='calibration_failed',reason='one or more views failed calibration',estimate=None,error_mm=None,cameras=[])
            continue
        if mode.startswith('legacy'):
            lines=[legacy_axis(m,board) for m in masks[key]]
            point,ids,reason=legacy_intersect(lines,tolerance=5,support_distance=16)
            candidate_counts=[int(l is not None) for l in lines]
        else:
            point,ids,reason,lines,details=detect_dart(masks[key],board)
            candidate_counts=details['candidate_counts']
        results[mode]=score_result(point,ids,reason,board,truth)
        results[mode]['candidate_counts']=candidate_counts
        if save_images and mode in ('new_oracle','new_estimated'):
            tiles=[]
            for i,(rect,mask,line) in enumerate(zip(images[key],masks[key],lines)):
                candidates=details['candidates'][i]
                diagnostic=diagnostic_image(rect,mask,candidates,line,point)
                write_image(folder/f'{mode}_camera_{i}_mask.png',mask)
                write_image(folder/f'{mode}_camera_{i}_axes.jpg',diagnostic)
                tiles.append(panel(diagnostic,f"{'EXACT H' if key=='oracle' else 'ESTIMATED H'}: CAMERA {i+1}",
                                   f"{len(candidates)} candidates; {'used' if i in ids else 'unused'}; {reason}",width=430,height=480))
            combined=np.mean(images[key],axis=0).astype(np.uint8)
            for i,line in enumerate(lines):
                if line is not None:
                    cv.line(combined,tuple(np.rint(line['origin']-1200*line['direction']).astype(int)),
                            tuple(np.rint(line['origin']+1200*line['direction']).astype(int)),COLORS[i],1,cv.LINE_AA)
            cv.drawMarker(combined,tuple(np.rint(truth).astype(int)),(255,255,255),cv.MARKER_CROSS,18,2)
            if point is not None:cv.circle(combined,tuple(np.rint(point).astype(int)),8,(90,255,100),2)
            r=results[mode];caption=f"{r['outcome']}; error {r['error_mm']:.2f}mm" if point is not None else reason
            tiles.append(panel(combined,'JOINT INTERSECTION',caption,width=430,height=480))
            comparison=np.hstack(tiles)
            write_image(folder/f'{mode}_pipeline.jpg',comparison)
            evidence.append(str(Path(name)/f'{mode}_pipeline.jpg'))
    if save_images:
        raw_tiles=[]
        for i,(reference,changed,jpg) in enumerate(raw_pairs):
            (folder/f'camera_{i}_input.jpg').write_bytes(jpg.tobytes())
            raw_tiles.append(panel(changed,f'CAMERA {i+1}: DECODED INPUT','Known lens correction; actual pixel detection follows',width=430,height=340))
        write_image(folder/'inputs.jpg',np.hstack(raw_tiles))
    return dict(trial=index,id=name,seed=seed,truth=board.score(truth),truth_pixel=truth.tolist(),
                calibration=calibration,results=results,evidence=evidence,
                scene=dict(dart_config=config,existing_darts=existing,cameras=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for p in parameters]))


def summarize(rows):
    result={}
    for mode in MODES:
        r=[x['results'][mode] for x in rows];counts=Counter(x['outcome'] for x in r)
        accepted=[x for x in r if x['estimate'] is not None];errors=[x['error_mm'] for x in accepted]
        result[mode]=dict(trials=len(r),counts=dict(counts),coverage=len(accepted)/len(r),
                          median_error_mm=float(np.median(errors)) if errors else None,
                          p95_error_mm=float(np.percentile(errors,95)) if errors else None,
                          max_error_mm=max(errors,default=None),three_views=sum(len(x['cameras'])==3 for x in accepted),
                          review_flagged=sum(x['requires_review'] for x in accepted),
                          wrong_review_flagged=sum(x['outcome']=='wrong' and x['requires_review'] for x in accepted),
                          wrong_without_review=sum(x['outcome']=='wrong' and not x['requires_review'] for x in accepted),
                          rejection_reasons=dict(Counter(x['reason'] for x in r if x['estimate'] is None)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed',action='store_true');p.add_argument('--trials',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261110);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stress',action='store_true',help='Vary dart tilt, length, colours, flight size and 0-2 existing darts')
    p.add_argument('--output',default='detection_evidence/randomized');p.add_argument('--save-images',type=int,default=5)
    args=p.parse_args()
    if args.trials<1 or args.workers<1:p.error('Positive trials/workers required')
    count=len(CASES) if args.fixed else args.trials
    seeds=np.random.default_rng(args.seed).integers(1,2**30,count)
    tasks=[(i,int(seed),args.fixed,args.stress,args.output,i<args.save_images) for i,seed in enumerate(seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(trial,tasks,chunksize=1):
            rows.append(r)
            print(f"{len(rows)}/{count}: "+str({m:r['results'][m]['outcome'] for m in MODES}),flush=True)
    report=dict(seed=args.seed,fixed=args.fixed,stress=args.stress,summary=summarize(rows),trials=rows,
                conventions='All errors are board-plane millimetres. Oracle truth-H diagnostic is not deployable calibration. Estimated modes use actual automatic calibration.',
                limitations=['Exact synthetic K/D supplied. One board texture, mild blur/noise/JPEG; no photorealistic shadows or hands.',
                             'The stress renderer uses painter-order darts and pixel-width shafts; it is not a depth-tested cylinder renderer.',
                             'Static empty/reference/changed pairs do not validate live event timing or real-camera accuracy.'])
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()

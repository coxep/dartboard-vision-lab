#!/usr/bin/env python3
"""Replay every rejected Monte Carlo case; log gates without changing the solver."""
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2 as cv
import numpy as np
import monte_carlo_current as mc
from dartboard import Lens,gray_image,warp,project,change_mask
from dart_detection import detect_dart,diagnostic_image
from benchmark_c920 import WIDTH,HEIGHT,K,POSITIONS,observed,distortion_map,write_image
from render_synthetic import render_camera,panel

ROOT=Path('monte_carlo_current')


def instrument(detector,board):
    """Wrap existing calls. Capture returned/in-place ECC iterates and gate values."""
    import template_registration as tr
    import corner_geometry as cg
    events=[];state={};original_ecc=cv.findTransformECC
    originals={name:getattr(tr,name) for name in ('coarse_grayscale_seed','refine_appearance')}
    original_fit=cg.fit_corner_homography;original_measure=detector._measure
    def ecc(*args,**kwargs):
        event=dict(stage=state['stage'],scale=args[0].shape[1]/board.size)
        try:
            correlation,local=original_ecc(*args,**kwargs)
            event['correlation']=float(correlation)
            return correlation,local
        except cv.error as exc:
            event['error']=str(exc);raise
        finally:
            S=np.diag([event['scale'],event['scale'],1.])
            W=np.linalg.inv(S)@args[2]@S
            try:
                candidate=np.linalg.inv(W)@state['seed']
                event['max_update_px']=float(np.linalg.norm(project(board.landmarks(),candidate@np.linalg.inv(state['seed']))-board.landmarks(),axis=1).max())
                state['last_H']=candidate
            except np.linalg.LinAlgError:pass
            events.append(event)
    def make_wrapper(name):
        def wrapped(image,seed,*args,**kwargs):
            state.update(stage=name,seed=np.asarray(seed).copy())
            try:return originals[name](image,seed,*args,**kwargs)
            except (cv.error,ValueError,np.linalg.LinAlgError) as exc:
                events.append(dict(stage=name,gate_error=str(exc)));raise
        return wrapped
    def fitted(*args,**kwargs):
        result=original_fit(*args,**kwargs);state['fit']=result[0];return result
    def measured(gray,H,*args,**kwargs):
        state['fit']=None
        result,report=original_measure(gray,H,*args,**kwargs)
        if result is None:
            event=dict(stage='junction_validation',reason=report['reason'],candidates=report.get('candidates'),inliers=report.get('inliers'))
            if state['fit'] is not None:
                candidate=state['fit']@H;inverse=np.linalg.inv(candidate)
                perimeter=np.array([board.point(board.radii[-1],a) for a in np.linspace(0,360,80,endpoint=False)])
                den=np.c_[perimeter,np.ones(len(perimeter))]@inverse[2]
                p=project(perimeter,inverse)
                inside=(p[:,0]>=0)&(p[:,0]<gray.shape[1])&(p[:,1]>=0)&(p[:,1]<gray.shape[0])
                event.update(perimeter_inside_fraction=float(inside.mean()),horizon_crosses=bool(den.min()*den.max()<=0),projected_area_px2=abs(float(cv.contourArea(p.astype(np.float32)))))
                if report['reason']=='insufficient visible board or degenerate perspective':state['visibility_H']=candidate
            events.append(event)
        return result,report
    cv.findTransformECC=ecc
    for name in originals:setattr(tr,name,make_wrapper(name))
    cg.fit_corner_homography=fitted;detector._measure=measured
    def restore():
        cv.findTransformECC=original_ecc;cg.fit_corner_homography=original_fit;detector._measure=original_measure
        for name,fn in originals.items():setattr(tr,name,fn)
    return events,state,restore


def render(row,view,board,texture):
    i=view['camera'];k=np.array(view['K']);d=np.array(view['D']);seed=row['seed']
    options=dict(width=WIDTH,height=HEIGHT,focal=k[0,0],look_at=view['aim_mm'],roll=view['roll_deg'],wire_geometry=row['true_wire'])
    radius,angle=mc.CASES[row['trial']%len(mc.CASES)][1:3]
    truth=board.point(radius,angle);tip=(truth-board.center)/board.ppm*np.array([1,-1])
    clear,_,_=render_camera(texture,board,view['true_camera_mm'],dart=False,**options)
    after,_,_=render_camera(texture,board,view['true_camera_mm'],tip=[*tip,0.],dart_config=row['dart_config'],**options)
    shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
    maps=distortion_map(k,d,WIDTH,HEIGHT);lens=Lens(clear.shape,dict(K=k.tolist(),D=d.tolist(),image_size=[WIDTH,HEIGHT]))
    def capture(image,n):
        shifted=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
        lit=np.uint8(np.clip(255*(shifted.astype(np.float32)/255)**view['gamma']*view['gain']*np.array(view['white_balance']),0,255))
        raw,_=observed(lit,maps,seed+i*11+n)
        return raw,lens.apply(raw)
    raw,clear=capture(clear,0);_,after=capture(after,1)
    return raw,clear,after,truth


def inspect(task):
    mode,row=task;mc.initialize(mode=='pose_only');board=mc.BOARD
    cv.setRNGSeed(int(row['seed']%(2**31-1)))
    folder=ROOT/mode/'rejections'/f"trial_{row['trial']:03d}";folder.mkdir(parents=True,exist_ok=True)
    result=dict(mode=mode,trial=row['trial'],target=row['target'],estimated_score=row['estimated_score'],oracle_score=row['oracle_score'],calibration=[])
    masks=[];clears=[];afters=[]
    for view in row['views']:
        i=view['camera'];raw,clear,after,truth=render(row,view,board,mc.TEXTURE)
        if view['accepted']:
            if row['estimated_score']['outcome']=='calibration_rejected':
                replay_H,_=mc.DETECTORS[i].acquire(gray_image(clear),mc.PRIORS[i],K=K)
                assert replay_H is not None,(mode,row['trial'],i,'replay changed calibration outcome')
            H=np.array(view['H']);clears.append(warp(clear,H,board));afters.append(warp(after,H,board))
            masks.append(change_mask(gray_image(afters[-1]),gray_image(clears[-1]),board.mask(1.08),28))
            continue
        events,state,restore=instrument(mc.DETECTORS[i],board)
        try:H,report=mc.DETECTORS[i].acquire(gray_image(clear),mc.PRIORS[i],K=K)
        finally:restore()
        assert H is None,(mode,row['trial'],i,'replay changed calibration outcome')
        assert report['reason']==view['reason'],(mode,row['trial'],i,'replay changed rejection reason')
        attempt=state.get('visibility_H',state.get('last_H'))
        attempted=warp(clear,attempt,board) if attempt is not None and np.isfinite(attempt).all() else np.zeros((board.size,board.size,3),np.uint8)
        tiles=[panel(raw,'RAW CAMERA INPUT',f"Camera {i+1}; translation {view['offset_radius_mm']:.1f} mm",width=500,height=500),
            panel(warp(clear,mc.PRIORS[i],board),'INITIAL BELIEF','Actual pixels, nominal homography',width=500,height=500),
            panel(attempted,'FIT BEFORE VISIBILITY REJECTION' if 'visibility_H' in state else 'LAST ECC ATTEMPT: UNACCEPTED','Diagnostic iterate, not a valid calibration',width=500,height=500),
            panel(warp(clear,np.array(view['truth_H_evaluation_only']),board),'EXACT H: EVALUATION ONLY','Truth rectification, not passed to the detector',width=500,height=500)]
        name=f'camera_{i}_calibration.jpg';write_image(folder/name,np.hstack(tiles))
        write_image(folder/f'camera_{i}_input.jpg',raw)
        result['calibration'].append(dict(camera=i,reason=view['reason'],replay_reason=report['reason'],recovery_failures=report.get('recovery_failures',[]),events=events,image=str((folder/name).relative_to(ROOT))))
    if row['estimated_score']['outcome']=='rejected':
        point,ids,reason,selected,details=detect_dart(masks,board)
        replay=mc.score_result(point,ids,reason,board,truth)
        assert replay['outcome']==row['estimated_score']['outcome'] and replay['reason']==row['estimated_score']['reason'],(mode,row['trial'],replay)
        tiles=[];closeups=[];support=[]
        for i,(before,after,mask,candidates) in enumerate(zip(clears,afters,masks,details['candidates'])):
            axes=diagnostic_image(after,mask,candidates,selected[i],point)
            cv.drawMarker(axes,tuple(np.rint(truth).astype(int)),(255,255,255),cv.MARKER_CROSS,18,2)
            write_image(folder/f'camera_{i}_after.jpg',after)
            write_image(folder/f'camera_{i}_mask.png',mask)
            closeup=[]
            for picture,title in ((after,'AFTER: TIP REGION'),(cv.cvtColor(mask,cv.COLOR_GRAY2BGR),'DIFFERENCE MASK'),(axes,'AXES + TRUTH CROSS')):
                patch=cv.getRectSubPix(picture,(161,161),tuple(map(float,truth)))
                patch=cv.resize(patch,(400,400),interpolation=cv.INTER_NEAREST)
                closeup.append(panel(patch,f'CAMERA {i+1}: {title}','Crop centred on truth for diagnosis only',width=400,height=450))
            closeups.append(np.hstack(closeup))
            tiles.append(np.hstack([panel(before,f'CAMERA {i+1}: BEFORE','Estimated H; observed pixels',width=400,height=420),
                panel(after,'AFTER','Settled synthetic dart',width=400,height=420),
                panel(cv.cvtColor(mask,cv.COLOR_GRAY2BGR),'DIFFERENCE MASK',f'{len(candidates)} axis candidates',width=400,height=420),
                panel(axes,'AXIS CANDIDATES','Yellow: candidates; white cross: truth only',width=400,height=420)]))
            y,x=np.where(mask>0);pixels=np.c_[x,y]
            support.append(dict(camera=i,axis_candidates=len(candidates),closest_changed_to_truth_mm=float(np.linalg.norm(pixels-truth,axis=1).min()/board.ppm) if len(pixels) else None,
                closest_axis_to_truth_mm=min((float(abs(c['normal']@truth-c['constant'])/board.ppm) for c in candidates),default=None)))
        name='dart_rejection.jpg';write_image(folder/name,np.vstack(tiles))
        write_image(folder/'tip_closeups.jpg',np.vstack(closeups))
        result['dart']=dict(reason=reason,candidate_counts=details['candidate_counts'],hypotheses=details.get('hypotheses',0),truth_support_evaluation_only=support,image=str((folder/name).relative_to(ROOT)),closeup_image=str((folder/'tip_closeups.jpg').relative_to(ROOT)))
    (folder/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"{mode} trial {row['trial']}: diagnostics saved",flush=True)
    return result


def main():
    tasks=[]
    for mode in ('pose_only','mixed_stress'):
        report=json.loads((ROOT/mode/'report.json').read_text())
        tasks.extend((mode,row) for row in report['trials'] if row['estimated_score']['outcome'] in ('calibration_rejected','rejected'))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(inspect,tasks))
    (ROOT/'rejections.json').write_text(json.dumps(dict(cases=results),indent=2)+'\n')
    print(f'Wrote all {len(results)} rejected trial diagnostics.',flush=True)


if __name__=='__main__':main()

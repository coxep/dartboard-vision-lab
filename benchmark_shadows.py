#!/usr/bin/env python3
"""Paired foreground-stage test: faint soft shadows, frozen estimated homographies."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2 as cv
import numpy as np
import monte_carlo_current as mc
from benchmark_c920 import WIDTH,HEIGHT,observed,distortion_map,write_image
from dartboard import Board,Lens,gray_image,change_mask,warp
from dart_detection import detect_dart,diagnostic_image
from render_synthetic import board_texture,render_camera,panel
from shadow_difference import shadow_aware_masks
from soft_shadows import shadowed_texture


def trial(task):
    index,row,output=task;cv.setNumThreads(1)
    board=Board(radii=(97.4,107.4,160.,170.));texture=board_texture(board,wires=False)
    style=index//8 if index<32 else (index-32)//8
    target_index=index%8;radius,angle=mc.CASES[target_index][1:3]
    truth=board.point(radius,angle);xy=(truth-board.center)/board.ppm*np.array([1,-1])
    dart=dict(row['dart_config'],tip=[*xy,0.])
    if style%2:dart.update(shaft_color=[70,70,70],barrel_color=[95,95,95],flight_color=[100,100,100])
    old=dict(tip=[-60.,65.,0.],direction=[-.15,.1,1.],length=72.,flight_width=5.,flight_color=[90,110,130])
    added=index<32
    before_darts=[] if added else [old]
    after_darts=[dart] if added else [old]
    lights=[dict(position_mm=[-500.,600.,1100.],radius_mm=100.,strength=.22)]
    if style==1:lights=[dict(position_mm=[500.,350.,900.],radius_mm=80.,strength=.25)]
    if style==2:lights=[dict(position_mm=[-450.,400.,1000.],radius_mm=90.,strength=.16),dict(position_mm=[400.,-350.,1200.],radius_mm=120.,strength=.14)]
    if style==3:lights=[dict(position_mm=[-400.,-500.,1000.],radius_mm=150.,strength=.30)]
    previous_lights=[dict(light,position_mm=(np.array(light['position_mm'])+[350.,-200.,0.]).tolist()) for light in lights] if not added else lights
    before_texture,before_field=shadowed_texture(texture,board,before_darts,previous_lights,1000+index)
    after_texture,after_field=shadowed_texture(texture,board,after_darts,lights,1000+index)
    images=[];backgrounds=[];visibility=[]
    for view in row['views']:
        i=view['camera'];k=np.array(view['K']);d=np.array(view['D']);H=np.array(view['H'])
        opts=dict(width=WIDTH,height=HEIGHT,focal=k[0,0],look_at=view['aim_mm'],roll=view['roll_deg'],wire_geometry=row['true_wire'],dart=False)
        before,_,_=render_camera(before_texture,board,view['true_camera_mm'],existing_darts=before_darts,**opts)
        after,_,_=render_camera(after_texture,board,view['true_camera_mm'],existing_darts=after_darts,**opts)
        shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
        maps=distortion_map(k,d,WIDTH,HEIGHT);lens=Lens(before.shape,dict(K=k.tolist(),D=d.tolist(),image_size=[WIDTH,HEIGHT]))
        def capture(image,n):
            shifted=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
            lit=np.uint8(np.clip(255*(shifted.astype(np.float32)/255)**view['gamma']*view['gain']*np.array(view['white_balance']),0,255))
            raw,_=observed(lit,maps,row['seed']+i*11+n)
            return warp(lens.apply(raw),H,board)
        backgrounds.append(capture(before,0));images.append(capture(after,1))
        visibility.append(warp(lens.apply(np.full((HEIGHT,WIDTH),255,np.uint8)),H,board))
    old_masks=[change_mask(gray_image(a),gray_image(b),board.mask(1.08),28) for a,b in zip(images,backgrounds)]
    new_masks,shadow=shadow_aware_masks(images,backgrounds,visibility,board.mask(1.08),28)
    results={};diagnostics={}
    for name,masks in [('gray',old_masks),('shadow_aware',new_masks)]:
        point,ids,reason,selected,info=detect_dart(masks,board)
        if added:result=mc.score_result(point,ids,reason,board,truth)
        else:result=dict(outcome='no_detection' if point is None else 'false_hit',reason=reason,pixel=point.tolist() if point is not None else None)
        result['candidate_counts']=info['candidate_counts'];results[name]=result
        diagnostics[name]=(point,selected,info)
    folder=Path(output)/f'trial_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(3):
        point,selected,info=diagnostics['shadow_aware'];axes=diagnostic_image(images[i],new_masks[i],info['candidates'][i],selected[i],point)
        rows.append(np.hstack([panel(backgrounds[i],f'CAMERA {i+1}: BEFORE','Actual pixels; frozen estimated H',width=400,height=420),
            panel(images[i],'AFTER','New dart + soft shadow' if added else 'Same dart; light direction changed',width=400,height=420),
            panel(old_masks[i],'GRAYSCALE DIFFERENCE',results['gray']['outcome'],width=400,height=420),
            panel(new_masks[i],'SHADOW-AWARE DIFFERENCE',results['shadow_aware']['outcome'],width=400,height=420),
            panel(axes,'REMAINING AXIS CANDIDATES',results['shadow_aware']['reason'][:65],width=400,height=420)]))
        write_image(folder/f'camera_{i}_suppressed.png',shadow['suppressed'][i])
        if index in (0,7,8,16,24,32,40):
            write_image(folder/f'camera_{i}_before.jpg',backgrounds[i])
            write_image(folder/f'camera_{i}_after.jpg',images[i])
            write_image(folder/f'camera_{i}_gray_mask.png',old_masks[i])
            write_image(folder/f'camera_{i}_shadow_mask.png',new_masks[i])
    write_image(folder/'pipeline.jpg',np.vstack(rows))
    write_image(folder/'plane_shadow.jpg',np.uint8(after_field*255))
    print(f"{index+1}/48: {results['gray']['outcome']} → {results['shadow_aware']['outcome']}",flush=True)
    return dict(trial=index,pose_trial=row['trial'],added_dart=added,neutral_dart=bool(style%2 and added),target=board.score(truth) if added else None,
        light_before=previous_lights,light_after=lights,min_board_transmission=float(after_field.min()),results=results,
        suppressed_changed_pixels=[int(cv.countNonZero(m)) for m in shadow['suppressed']],
        image=f'trial_{index:03d}/pipeline.jpg',assumptions='Estimated H from accepted mixed-stress Monte Carlo; no calibration rerun. Known lens correction. Shared board-plane area-light shadow, ambient floor; dart/wire self-shading not modelled.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='shadow_evidence');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    rows=json.loads(Path('monte_carlo_current/mixed_stress/report.json').read_text())['trials']
    good=[r for r in rows if all(v['accepted'] for v in r['views'])]
    tasks=[(i,next(r for r in good if r['trial']%8==i%8),args.output) for i in range(48)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(trial,tasks))
    from collections import Counter
    summary={mode:{kind:dict(Counter(r['results'][mode]['outcome'] for r in results if r['added_dart']==added)) for kind,added in [('new_dart',True),('shadow_only',False)]} for mode in ('gray','shadow_aware')}
    report=dict(summary=summary,trials=results,algorithm_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('shadow_difference.py','soft_shadows.py','dart_detection.py','render_synthetic.py','benchmark_shadows.py')},fixture_design='Eight fixed accepted mixed-stress pose/target setups repeated across four lighting/material conditions; plus sixteen moving-shadow-only pairs. Foreground evaluation, no calibration refit.',limitations=['48 deterministic fixtures, not a new Monte Carlo acceptance claim.','32 added darts and 16 moving-shadow-only pairs; 8 repeated fixed targets.','Shadows are neutral multiplicative board-plane attenuation, 12 sampled light points, with 14–30% source strengths and a high ambient floor.','No coloured-light reflections, dart self-shading, people or camera movement within an event.','HSV/RGB tests and cross-view agreement can still confuse certain material changes with shadows. Real footage remains necessary.'])
    (Path(args.output)/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

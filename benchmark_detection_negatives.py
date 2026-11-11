#!/usr/bin/env python3
"""Image-based negative controls, using saved estimated calibration, not truth H."""
from collections import Counter
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,gray_image,change_mask,warp
from dart_detection import detect_dart,diagnostic_image
from benchmark_c920 import K,D,WIDTH,HEIGHT,FOCAL,POSITIONS,observed,distortion_map,write_image
from render_synthetic import board_texture,render_camera,panel


def main():
    cv.setNumThreads(1);cv.setRNGSeed(1234)
    out=Path('detection_evidence/negatives');out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('c920_benchmark/calibration.json').read_text())
    board=Board();texture=board_texture(board);maps=distortion_map(K,D,WIDTH,HEIGHT)
    lens=Lens((HEIGHT,WIDTH,3),dict(K=K.tolist(),D=D.tolist(),image_size=[WIDTH,HEIGHT]))
    backgrounds=[]
    for i,position in enumerate(POSITIONS):
        empty,_,_=render_camera(texture,board,position,dart=False,width=WIDTH,height=HEIGHT,focal=FOCAL)
        raw,_=observed(empty,maps,100+i)
        H=np.array(config['cameras'][i]['H'])
        backgrounds.append(gray_image(warp(lens.apply(raw),H,board)))
    rows=[];rng=np.random.default_rng(20261113)
    groups=[('no_change',20),('disjoint_cameras',20),('two_new_darts',10),('planar_shadow',5),('brightness_change',5)]
    for kind,count in groups:
        for j in range(count):
            points=[board.point(rng.uniform(30,150),rng.uniform(0,360)) for _ in range(3)]
            def dart(point):
                xy=(point-board.center)/board.ppm
                return dict(tip=[float(xy[0]),float(-xy[1]),0.])
            masks=[];images=[]
            shadow=texture.copy()
            if kind=='planar_shadow':
                y,x=np.mgrid[:board.size,:board.size]
                v=np.exp(-((x-rng.uniform(200,700))/90)**2-((y-rng.uniform(200,700))/35)**2)
                shadow=np.uint8(shadow*(1-.7*v[...,None]))
            for i,position in enumerate(POSITIONS):
                options=dict(width=WIDTH,height=HEIGHT,focal=FOCAL)
                image,_,_=render_camera(shadow,board,position,dart=False,
                    existing_darts=[dart(points[0]),dart(points[1])] if kind=='two_new_darts' else
                                   [dart(points[i])] if kind=='disjoint_cameras' else [],**options)
                if kind=='brightness_change':image=np.uint8(np.clip(image.astype(int)+35,0,255))
                raw,_=observed(image,maps,8000+len(rows)*3+i)
                rect=warp(lens.apply(raw),np.array(config['cameras'][i]['H']),board)
                images.append(rect)
                masks.append(change_mask(gray_image(rect),backgrounds[i],board.mask(1.08),28))
            point,ids,reason,lines,info=detect_dart(masks,board)
            r=dict(kind=kind,trial=j,estimated=point is not None,cameras=ids,reason=reason,
                   pixel=None if point is None else point.tolist(),candidate_counts=info['candidate_counts'])
            if j==0 or point is not None:
                name=f'{kind}_{j:02d}.jpg'
                tiles=[panel(diagnostic_image(images[i],masks[i],info['candidates'][i],lines[i],point),
                             f'{kind.upper()}: C{i+1}',reason,width=430,height=480) for i in range(3)]
                write_image(out/name,np.hstack(tiles));r['image']=name
            rows.append(r)
    summary={kind:dict(trials=count,false_estimates=sum(r['estimated'] for r in rows if r['kind']==kind),
                       rejection_reasons=dict(Counter(r['reason'] for r in rows if r['kind']==kind and not r['estimated'])))
             for kind,count in groups}
    report=dict(summary=summary,trials=rows,limitation='Synthetic negative controls. Planar Gaussian shadow and brightness steps are simple image perturbations, not realistic hands or lighting.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

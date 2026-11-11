#!/usr/bin/env python3
"""Save exact input JPEG bytes and a close-up of the retained wrong-score case.

Re-render only the saved input samples from their recorded scene parameters.
Detection results are not recomputed or changed by this presentation step.
"""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,warp,text
from benchmark_c920 import WIDTH,HEIGHT,distortion_map,observed,write_image
from render_synthetic import board_texture,render_camera,panel


def main():
    cv.setNumThreads(1);root=Path('detection_evidence');b=Board();texture=board_texture(b)
    for name in ('randomized','stress_holdout'):
        report=json.loads((root/name/'report.json').read_text())
        for row in report['trials']:
            folder=root/name/row['id']
            if not (folder/'camera_0_input.jpg').exists():continue
            xy=(np.array(row['truth_pixel'])-b.center)/b.ppm
            before=[];after=[]
            for i,par in enumerate(row['scene']['cameras']):
                k=np.array(par['K']);d=np.array(par['D'])
                options=dict(width=WIDTH,height=HEIGHT,focal=k[0,0],look_at=par['look_at'],roll=par['roll'])
                frames=[]
                for changed in (False,True):
                    image,_,_=render_camera(texture,b,par['position'],dart=changed,tip=[xy[0],-xy[1],0.],
                        dart_config=row['scene']['dart_config'],existing_darts=row['scene']['existing_darts'],**options)
                    shift=np.float32([[1,0,k[0,2]-WIDTH/2],[0,1,k[1,2]-HEIGHT/2]])
                    image=cv.warpAffine(image,shift,(WIDTH,HEIGHT),borderMode=cv.BORDER_REFLECT)
                    raw,jpg=observed(image,distortion_map(k,d,WIDTH,HEIGHT),row['seed']+i*2+int(changed))
                    frames.append(raw)
                    if changed:(folder/f'camera_{i}_input.jpg').write_bytes(jpg.tobytes())
                before.append(panel(frames[0],f'CAMERA {i+1}: REFERENCE','Existing darts included in baseline',width=430,height=340))
                after.append(panel(frames[1],f'CAMERA {i+1}: NEW DART','Decoded JPEG supplied to the detector',width=430,height=340))
            write_image(folder/'before_after.jpg',np.vstack([np.hstack(before),np.hstack(after)]))
    # Inspect actual rectified pixels around the sector-boundary error.
    report=json.loads((root/'stress_holdout/report.json').read_text())
    for row in report['trials']:
        result=row['results']['new_estimated']
        if result['outcome']!='wrong':continue
        folder=root/'stress_holdout'/row['id'];par=row['scene']['cameras'][0]
        raw=cv.imread(str(folder/'camera_0_input.jpg'))
        lens=Lens(raw.shape,dict(K=par['K'],D=par['D'],image_size=[WIDTH,HEIGHT]))
        rect=warp(lens.apply(raw),np.array(row['calibration'][0]['H']),b)
        truth=np.array(row['truth_pixel']);estimate=np.array(result['pixel'])
        lo=np.rint(truth).astype(int)-20;crop=rect[lo[1]:lo[1]+40,lo[0]:lo[0]+40]
        crop=cv.resize(crop,(480,480),interpolation=cv.INTER_NEAREST)
        annotated=crop.copy()
        for point,color in [(truth,(255,255,255)),(estimate,(60,60,255))]:
            p=np.rint((point-lo+.5)*12).astype(int)
            cv.drawMarker(annotated,tuple(p),color,cv.MARKER_CROSS,12,1)
        image=np.hstack([panel(crop,'ACTUAL PIXELS: 12X','Nearest-neighbour enlargement; no model board redraw',width=520,height=570),
                         panel(annotated,'WHITE: TRUTH / RED: ESTIMATE',
                               f"{row['truth']['label']} -> {result['estimate']['label']}; {result['error_mm']:.3f}mm; REVIEW",width=520,height=570)])
        write_image(folder/'boundary_closeup.jpg',image)


if __name__=='__main__':main()

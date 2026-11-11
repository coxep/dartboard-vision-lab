#!/usr/bin/env python3
"""Small, reproducible raised-wire experiment; renderer truth is evaluation-only.

Nominal mount centres seed registration. True displaced centres produce pixels;
only the evaluator uses their homographies. No measured C920 intrinsics claimed.
"""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,project,warp
from board_pattern import PatternDetector,draw_pattern
from render_synthetic import board_texture,render_camera
from wire_model import mesh,export_obj
from template_registration import contrast_normalized


def run(folder='raised_wire_evidence'):
    cv.setNumThreads(1);cv.setRNGSeed(3)
    out=Path(folder);out.mkdir(exist_ok=True,parents=True)
    b=Board(radii=(97.4,107.4,160,170));texture=board_texture(b,wires=False)
    centres=json.loads(Path('measured_setup/scene.json').read_text())['positions_mm']
    vertices,faces=mesh(b);export_obj(out/'spider_mm.obj',vertices,faces)
    rows=[];panels=[]
    for i,nominal in enumerate(centres):
        geometry=dict(height_mm=1.,width_mm=.8,camera_center_mm=nominal)
        detector=PatternDetector(b,geometry)
        _,prior,_=render_camera(texture,b,nominal,dart=False,width=1920,height=1080,focal=1300)
        for scene,displacement in [('nominal',[0,0,0]),('moved',[20,-15,12]),('lighting',[-12,18,-8]),('occlusion',[0,0,0])]:
            C=np.asarray(nominal)+displacement
            image,truth,_=render_camera(texture,b,C,dart=True,width=1920,height=1080,focal=1300,
                                       wire_geometry=geometry)
            if scene=='lighting':
                yy,xx=np.mgrid[:1080,:1920];gain=.45+.6*xx/1920+.25*yy/1080
                image=np.uint8(np.clip(255*(image.astype(float)/255)**1.25*gain[...,None]+7,0,255))
            if scene=='occlusion':
                region=b.center+b.ppm*np.array([[-100,-30],[-50,-30],[-50,-100],[-100,-100]])
                cv.fillConvexPoly(image,np.rint(project(region,np.linalg.inv(truth))).astype(np.int32),(20,20,20))
            H,report=detector.acquire(image,prior)
            error=None
            if H is not None:
                # Held-out interior probes, not detected training junctions.
                xy=np.array([(x,y) for x in np.linspace(-145,145,13) for y in np.linspace(-145,145,13) if x*x+y*y<160**2])
                probes=b.center+b.ppm*xy*np.array([1,-1])
                errors=np.linalg.norm(project(probes,H@np.linalg.inv(truth))-probes,axis=1)/b.ppm
                error=dict(median_mm=float(np.median(errors)),max_mm=float(errors.max()))
            rows.append(dict(camera=i,scene=scene,accepted=H is not None,error=error,report=report))
            print(i,scene,H is not None,error,report['reason'],flush=True)
            prefix=out/f'camera_{i}_{scene}'
            cv.imwrite(str(prefix)+'_input.jpg',image)
            truth_rect=warp(image,truth,b)
            cv.imwrite(str(prefix)+'_truth.jpg',truth_rect)
            truth_gray=cv.cvtColor(truth_rect,cv.COLOR_BGR2GRAY)
            cv.imwrite(str(prefix)+'_truth_gray.jpg',truth_gray)
            initial=warp(image,prior,b)
            gray=cv.cvtColor(initial,cv.COLOR_BGR2GRAY)
            normalized=np.uint8(np.clip(128+42*contrast_normalized(gray),0,255))
            cv.imwrite(str(prefix)+'_normalized.jpg',normalized)
            if H is not None:
                rect=warp(image,H,b)
                overlay=draw_pattern(rect,b,report,H)
                cv.imwrite(str(prefix)+'_rectified.jpg',rect)
                cv.imwrite(str(prefix)+'_boundaries.jpg',overlay)
                if scene=='nominal':
                    strip=np.hstack([cv.resize(image,(640,360)),cv.resize(truth_rect,(360,360)),cv.resize(rect,(360,360)),cv.resize(overlay,(360,360))])
                    header=np.full((40,strip.shape[1],3),28,np.uint8)
                    for x,title in [(10,f'Camera {i+1}: rendered input'),(650,'Exact H (evaluation only)'),(1010,'Detected H'),(1370,'Observed boundary checks')]:
                        cv.putText(header,title,(x,26),cv.FONT_HERSHEY_SIMPLEX,.5,(210,230,210),1,cv.LINE_AA)
                    panels.append(np.vstack([header,strip]))
    (out/'validation.json').write_text(json.dumps(dict(wire_model='triangular prism, 1mm high / 0.8mm base; assumed',
        renderer='pinhole textured plane + depth-sorted 3D triangles; no cast shadows or physical lens calibration',
        camera_size=[1920,1080],focal_px=1300,nominal_centres_mm=centres,results=rows),indent=2)+'\n')
    if panels:cv.imwrite(str(out/'three_views.jpg'),np.vstack(panels))
    return rows

if __name__=='__main__':run()

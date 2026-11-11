#!/usr/bin/env python3
"""Render three successive darts and average actual estimated-H rectifications.

No true homography is passed to calibration/tracking. Valid image footprints
exclude warp padding from the mean; each observed camera has equal weight.
"""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,warp
from board_pattern import PatternDetector,GeometryTracker
from render_synthetic import board_texture,render_camera,panel


def average_views(images,masks):
    valid=np.stack(masks).astype(bool)
    count=valid.sum(axis=0)
    total=(np.stack(images).astype(np.float32)*valid[...,None]).sum(axis=0)
    mean=total/np.maximum(count[...,None],1)
    assert np.isfinite(mean).all()
    return np.uint8(np.clip(np.rint(mean),0,255)),count


def run(folder='three_dart_evidence'):
    cv.setNumThreads(1);cv.setRNGSeed(7)
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    board=Board(radii=(97.4,107.4,160,170));texture=board_texture(board,wires=False)
    nominal=json.loads(Path('measured_setup/scene.json').read_text())['positions_mm']
    offsets=[[12,-8,5],[-10,8,-4],[6,-10,4]]
    darts=[dict(tip=[0.,103.,0.],direction=[.12,-.18,1.],flight_color=[60,160,235]),
           dict(tip=[-76.,-89.,0.],direction=[-.14,.12,1.],flight_color=[190,180,45]),
           dict(tip=[116.,21.,0.],direction=[.05,.15,1.],flight_color=[100,70,225])]
    cameras=[];records=[]
    for i,centre in enumerate(nominal):
        geometry=dict(camera_center_mm=centre,height_mm=1.,width_mm=.8)
        options=dict(width=1920,height=1080,focal=1300,wire_geometry=geometry)
        actual=np.asarray(centre)+offsets[i]
        _,prior,_=render_camera(texture,board,centre,dart=False,**options)
        empty,_,_=render_camera(texture,board,actual,dart=False,**options)
        H,report=PatternDetector(board,geometry).acquire(empty,prior)
        if H is None:raise RuntimeError(f'Camera {i} calibration rejected: {report["reason"]}')
        tracker=GeometryTracker(None,H,board,wire_geometry=geometry)
        cameras.append((actual,options,tracker))
    for count in range(1,4):
        rectified=[];masks=[];rows=[]
        for i,(centre,options,tracker) in enumerate(cameras):
            image,_,_=render_camera(texture,board,centre,dart=False,existing_darts=darts[:count],**options)
            tracker.track(image)
            if not tracker.valid:raise RuntimeError(f'{count} darts / camera {i}: {tracker.reason}')
            rect=warp(image,tracker.H,board)
            footprint=warp(np.full(image.shape[:2],255,np.uint8),tracker.H,board)>254
            cv.imwrite(str(out/f'after_{count}_camera_{i}_input.jpg'),image)
            cv.imwrite(str(out/f'after_{count}_camera_{i}_rectified.jpg'),rect)
            rectified.append(rect);masks.append(footprint)
            rows.append(dict(camera=i,H=tracker.H.tolist(),valid=tracker.valid))
        average,coverage=average_views(rectified,masks)
        cv.imwrite(str(out/f'average_after_{count}.jpg'),average)
        cv.imwrite(str(out/f'coverage_after_{count}.png'),np.uint8(coverage*85))
        names=['Upper left','Upper right','Bottom']
        views=[panel(im,names[i],f'{count} dart(s); actual camera pixels, estimated H',width=660,height=720)
               for i,im in enumerate(rectified)]
        views.append(panel(average,'THREE-CAMERA AVERAGE',f'{count} dart(s); equal weight per valid camera pixel',width=660,height=720))
        cv.imwrite(str(out/f'comparison_after_{count}.jpg'),np.vstack([np.hstack(views[:2]),np.hstack(views[2:])]))
        records.append(dict(darts=count,cameras=rows,coverage_pixels={str(i):int((coverage==i).sum()) for i in range(4)}))
        print(f'{count} darts: 3/3 camera alignments valid; averaged actual rectified pixels',flush=True)
    (out/'scene.json').write_text(json.dumps(dict(nominal_centres_mm=nominal,actual_offsets_mm=offsets,
        assumed_wire_height_mm=1.,assumed_wire_base_width_mm=.8,render_size=[1920,1080],focal_px=1300,
        darts=darts,records=records,average='sum(rectified * valid_footprint) / number_of_valid_views; equal weights',
        caveat='Synthetic pinhole renderer with depth-sorted wire triangles and painter-order darts; no measured C920 optics or full depth-tested occlusion.'),indent=2)+'\n')
    (out/'README.md').write_text('''# Three darts / three-camera average

Run `python render_three_darts.py` from the repository root.
Each successive image contains the previous darts plus one new dart. Each camera
is aligned from its nominal mount belief and then continuously remeasured using
the current grayscale detector. True camera homographies are not supplied.

`average_after_N.jpg` averages the three actual estimated-H rectifications after
N darts. Equal weights apply wherever a camera supplies valid pixels. Warp
padding is excluded; `coverage_after_N.png` encodes 0/1/2/3 views as 0/85/170/255.
`comparison_after_N.jpg` places all three rectifications beside their average.
The raw per-camera inputs and rectifications are also saved.

The board surface and dart tips align. Raised shafts/flights retain view-dependent
projections, producing faint multiple images in the average. Averaging alone is
not the hit estimator. The scene uses assumed wire dimensions and a simplified
3D dart renderer; it does not establish real-camera performance.
''')

if __name__=='__main__':run()

#!/usr/bin/env python3
"""Validate observed threshold/HSV masks on real photos; no camera truth supplied.

The no-click ellipse/bull bootstrap fits each photo independently. Only board
crops and derived outputs are saved. Geometry-aligned purity is a consistency
measure, NOT an independently measured calibration error.
"""
import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board, automatic_homography, binarize, board_color_masks, warp, project, ellipse_homography, text
from threshold_evidence import wedge_accuracy
from render_synthetic import panel
from benchmark_c920 import write_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos',nargs='+')
    parser.add_argument('--output',default='real_photo_validation')
    args=parser.parse_args()
    cv.setNumThreads(1);cv.setRNGSeed(0)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    board=Board();reports=[]
    for i,path in enumerate(args.photos):
        frame=cv.imread(path)
        if frame is None:raise ValueError(f'Cannot read photo {i+1}')
        accepted,reason=True,None
        try:
            H,info=automatic_homography(frame,board)
        except (ValueError,cv.error,np.linalg.LinAlgError) as exc:
            accepted,reason=False,str(exc)
            print('Photo',i+1,'REJECTED:',exc,flush=True)
            # Use the coarse candidate for a DISPLAY CROP only, not acceptance.
            try:
                H,info=ellipse_homography(frame,board)
            except (ValueError,cv.error,np.linalg.LinAlgError):
                reports.append(dict(photo=i+1,calibrated=False,reason=reason))
                continue
        perimeter=np.array([board.point(185,a) for a in np.linspace(0,360,120,endpoint=False)])
        p=project(perimeter,np.linalg.inv(H))
        lo=np.maximum(np.floor(p.min(axis=0)).astype(int),[0,0]);hi=np.minimum(np.ceil(p.max(axis=0)).astype(int),[frame.shape[1],frame.shape[0]])
        crop=frame[lo[1]:hi[1],lo[0]:hi[0]]
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        old=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,5)
        new=binarize(gray);masks=board_color_masks(frame)
        images=[('source',crop),('before',old[lo[1]:hi[1],lo[0]:hi[0]]),
                ('adaptive',new[lo[1]:hi[1],lo[0]:hi[0]]),
                ('hsv_red',masks['red'][lo[1]:hi[1],lo[0]:hi[0]]),
                ('hsv_green',masks['green'][lo[1]:hi[1],lo[0]:hi[0]])]
        rectified=warp(frame,H,board)
        plane_binary=binarize(cv.cvtColor(rectified,cv.COLOR_BGR2GRAY))
        overlay=board.grid(rectified)
        if not accepted:
            text(overlay,"REJECTED CANDIDATE",at=(25,35),color=(30,30,255))
        images += [('rectified',rectified),('rectified_grid',overlay),('plane_binary',plane_binary)]
        for name,image in images:write_image(folder/f'photo_{i+1}_{name}.png',image)
        output=np.hstack([panel(image,title,'Observed photo pixels',width=400,height=460) for image,title in [
            (crop,f'REAL PHOTO {i+1}'),(images[1][1],'OLD BLOCK 31'),(images[2][1],'LARGE WINDOW'),
            (images[3][1],'HSV RED'),(images[4][1],'HSV GREEN'),(overlay,'ACCEPTED BOARD FIT' if accepted else 'REJECTED CANDIDATE')]])
        write_image(folder/f'photo_{i+1}_comparison.jpg',output)
        score=wedge_accuracy(plane_binary,board) if accepted else None
        reports.append(dict(photo=i+1,calibrated=accepted,reason=reason,inliers=info['inliers'],
                            fit_median_canonical_px=info['fit_median_px'],
                            binary_template_support=info['binary_template_support'],
                            corner_inputs=info.get('corner_inputs'),
                            wedge_consistency=score, H=H.tolist(),
                            note='Real photo, no manual points, no supplied lens K/D. Fit residual and wedge consistency are internal checks; no independent geometric ground truth.'))
        print('Photo',i+1, {k:v for k,v in reports[-1].items() if k not in ('H','wedge_consistency')},'wedge',score['correct_fraction'] if score else None,flush=True)
    (folder/'report.json').write_text(json.dumps(dict(photos=reports,independent_geometric_ground_truth=False),indent=2)+'\n')

if __name__=='__main__':main()

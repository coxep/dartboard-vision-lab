#!/usr/bin/env python3
"""Reproduce colour-validated corner fits on a real crop and three camera views."""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,automatic_homography,harris_refine,warp,project,text,nominal_homography
from render_synthetic import board_texture,render_camera,panel
from benchmark_c920 import K,D,POSITIONS,WIDTH,HEIGHT,observed,distortion_map,write_image
from edge_calibration import detect_rectified_circles
from corner_geometry import colour_corner_scores


def inspect(frame,board,H,folder,name):
    rectified=warp(frame,H,board)
    correction,report=harris_refine(rectified,np.eye(3),board)
    if not report['refined']:raise ValueError(report)
    overlay=rectified.copy()
    for p in report['rejected_colour_source_px']:
        cv.drawMarker(overlay,tuple(np.rint(p).astype(int)),(40,40,240),cv.MARKER_TILTED_CROSS,6,1)
    for p,q in zip(report['matched_source_px'],report['matched_target_px']):
        p,q=np.array(p),np.array(q)
        cv.circle(overlay,tuple(np.rint(p).astype(int)),3,(70,255,70),1,cv.LINE_AA)
        cv.line(overlay,tuple(np.rint(p).astype(int)),tuple(np.rint(p+5*(q-p)).astype(int)),(255,220,40),1)
    text(overlay,f"{report['inliers']} accepted; residual arrows x5",(18,30),scale=.6)
    fitted=warp(rectified,correction,board)
    circles,circle_overlay=detect_rectified_circles(fitted,board)
    for label,image in [('corners',overlay),('fitted',fitted),('circles',circle_overlay)]:
        write_image(folder/f'{name}_{label}.jpg',image)
    # Keep full accepted correspondences; bound the diagnostics to this board crop.
    report['circle_measurements']=circles
    report['corner_H']= (correction@H).tolist()
    return report,overlay,circle_overlay,correction


def main():
    cv.setNumThreads(1);cv.setRNGSeed(0)
    out=Path('corner_evidence');out.mkdir(exist_ok=True)
    # The saved fixture is a board crop derived from the supplied original photo.
    frame=cv.imread('real_photo_validation/photo_1_source.png')
    rows=[];real=[];synthetic=[]
    for name,radii in [('8mm',(99.,107.,162.,170.)),('strip10mm',(97.4,107.4,160.,170.))]:
        b=Board(radii=radii)
        H,initial=automatic_homography(frame,b)
        r,overlay,circles,_=inspect(frame,b,H,out,'real_'+name)
        r.update(profile=name,radii_mm=radii,dense_edge_support=initial['binary_template_support'])
        real.append(r)
        rows.append(np.hstack([panel(image,title,caption,width=600,height=650) for image,title,caption in [
            (overlay,f'ORIGINAL PHOTO: {name}',f"Green: {r['inliers']} consensus matches; red: colour-rejected candidates"),
            (circles,'MEASURED COLOUR BOUNDARIES',f"Corner fit median {r['fit_median_px']:.2f}px; free circle fits, no nominal redraw")]]))
    write_image(out/'real_comparison.jpg',np.vstack(rows))
    rows=[];b=Board();texture=board_texture(b)
    for i,position in enumerate(POSITIONS):
        ideal,truth,_=render_camera(texture,b,position,dart=False,width=WIDTH,height=HEIGHT,focal=K[0,0])
        raw,_=observed(ideal,distortion_map(K,D,WIDTH,HEIGHT),100+i)
        frame=Lens(raw.shape,dict(K=K.tolist(),D=D.tolist(),image_size=[WIDTH,HEIGHT])).apply(raw)
        H,_=automatic_homography(frame,b,nominal_homography(b,K,position),K)
        r,overlay,circles,correction=inspect(frame,b,H,out,f'camera_{i}')
        rng=np.random.default_rng(600+i)
        angle=rng.uniform(0,2*np.pi,300);radius=np.sqrt(rng.uniform(.01,.95,300))*320
        probes=b.center+np.c_[np.cos(angle),np.sin(angle)]*radius[:,None]
        image_points=project(probes,np.linalg.inv(truth))
        before=np.linalg.norm(project(image_points,H)-probes,axis=1)
        after=np.linalg.norm(project(image_points,correction@H)-probes,axis=1)
        r.update(camera=i,dense_probe_median_px=float(np.median(before)),
                 corner_probe_median_px=float(np.median(after)),corner_probe_max_px=float(after.max()))
        synthetic.append(r)
        rows.append(np.hstack([panel(image,title,caption,width=600,height=650) for image,title,caption in [
            (overlay,f'CAMERA {i+1}: COLOUR VALIDATION',f"{r['inliers']} matched corners; median fitting residual {r['fit_median_px']:.2f}px"),
            (circles,'ACTUAL PIXELS: CORNER-FIT RECTIFICATION',f"Withheld probe median: dense {np.median(before):.2f}px -> corners {np.median(after):.2f}px")]]))
    write_image(out/'synthetic_comparison.jpg',np.vstack(rows))
    ids=np.arange(80);swapped=ids.copy()
    swapped[:20]+=20;swapped[20:40]-=20;swapped[40:60]+=20;swapped[60:]-=20
    a,w=colour_corner_scores(texture,np.eye(3),b,b.landmarks(),ids,ids)
    x,y=colour_corner_scores(texture,np.eye(3),b,b.landmarks(),ids,swapped)
    report=dict(real=real,synthetic=synthetic,topology_control=dict(correct_accepted=int(((a>=.72)&(w>=.44)).sum()),
                swapped_accepted=int(((x>=.72)&(y>=.44)).sum()),total=80),
                limitation='Known K/D supplied only for synthetic views. Real band profiles are alternative hypotheses, not measured physical dimensions. Colour boundaries differ from wire apexes. Real fit residuals are internal consistency, not physical ground-truth error. Dense final calibration remains the default; corner refinement is compared separately here.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(real=[dict(profile=r['profile'],inliers=r['inliers'],fit=r['fit_median_px']) for r in real],
                         synthetic=[{k:r[k] for k in ('camera','inliers','dense_probe_median_px','corner_probe_median_px','corner_probe_max_px')} for r in synthetic]),indent=2))


if __name__=='__main__':main()

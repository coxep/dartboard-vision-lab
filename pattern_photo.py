"""Check grayscale junction detection on a photo with a saved orientation prior.

No intrinsic calibration is inferred from this single photograph. The prior H
is supplied explicitly; both radius profiles and failures are retained.
"""
import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from board_pattern import PatternDetector,draw_pattern
from dartboard import Board,warp
from render_synthetic import panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('photo');p.add_argument('--prior',default='real_photo_validation/report.json')
    p.add_argument('--output',default='pattern_evidence/real');args=p.parse_args()
    image=cv.imread(args.photo)
    if image is None:raise ValueError('Photo could not be read')
    H=np.asarray(json.loads(Path(args.prior).read_text())['photos'][0]['H'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY);rows=[];panels=[]
    from boundary_audit import audit_boundaries,boundary_geometry_valid
    for radii in ((99,107,162,170),(97.4,107.4,160,170)):
        b=Board(radii=radii);fitted,report=PatternDetector(b).detect(gray,H,iterations=2)
        boundaries,boundary_image=audit_boundaries(warp(image,fitted if fitted is not None else H,b),b)
        geometry_valid=fitted is not None and boundary_geometry_valid(boundaries,b.size/900)
        rows.append(dict(radii_mm=radii,found=fitted is not None,geometry_valid=geometry_valid,
                         H=fitted.tolist() if fitted is not None else None,report=report,boundaries=boundaries))
        cv.imwrite(str(out/('boundaries_8mm.jpg' if radii[0]==99 else 'boundaries_strip10mm.jpg')),boundary_image)
        rect=warp(gray,fitted if fitted is not None else H,b)
        overlay=draw_pattern(rect,b,report,fitted if fitted is not None else H)
        panels.append(panel(b.grid(overlay),f'{radii}: '+('GEOMETRY VERIFIED' if geometry_valid else 'GEOMETRY NOT VERIFIED'),
                            f"{report.get('inliers',0)} junctions, {report.get('rings',0)} rings; "+
                            (f"median residual {report['median_residual_px']:.2f}px" if fitted is not None else report['reason']),
                            width=640,height=700))
    cv.imwrite(str(out/'profiles.jpg'),np.hstack(panels))
    # Separate, explicitly labelled unknown-lens experiment. Do not pass these
    # guessed intrinsics into the physical-pose accuracy benchmark.
    from boundary_audit import refine_labelled_boundaries
    b=Board(radii=(97.4,107.4,160,170))
    focal=np.hypot(*image.shape[:2])/(2*np.tan(np.deg2rad(78/2)))
    K=np.array([[focal,0,image.shape[1]/2],[0,focal,image.shape[0]/2],[0,0,1.]])
    seed=np.asarray(rows[1]['H']) if rows[1]['found'] else H
    fitted,lens,fit,overlay=refine_labelled_boundaries(image,b,seed,K)
    fit.update(H=fitted.tolist(),lens=lens,intrinsics_source='Assumed centred principal point and 78-degree diagonal FOV; NOT measured phone intrinsics.')
    cv.imwrite(str(out/'boundaries_refined.jpg'),overlay)
    before=cv.imread(str(out/'boundaries_strip10mm.jpg'))
    cv.imwrite(str(out/'boundary_comparison.jpg'),np.hstack([
        panel(before,'BEFORE: INCOMPLETE GRAYSCALE FIT','Magenta: model; cyan: observed paint-boundary samples',width=640,height=700),
        panel(overlay,'AFTER: SEPARATE LABELLED EDGE / LENS FIT','Fixed strip-10 profile; internal edge check only; K assumed',width=640,height=700)]))
    (out/'report.json').write_text(json.dumps(dict(
        input='User-uploaded close-up, converted to grayscale; board crop only is published.',
        supplied='Previously saved photo-specific H; no K/D. Not a cold-acquisition test.',
        limitation='Ring profiles are nominal alternatives, not measurements of this board. Residual is an internal fit statistic, not geometric ground truth.',
        profiles=rows,labelled_boundary_fit=fit),indent=2)+'\n')
    print([(r['radii_mm'],r['found'],r['report'].get('inliers'),r['report'].get('median_residual_px')) for r in rows])


if __name__=='__main__':main()

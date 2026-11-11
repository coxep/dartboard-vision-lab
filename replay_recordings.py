#!/usr/bin/env python3
"""Inspect synchronized reference/settled event windows from three video files.

This validates supplied event windows, not automatic throw timing. Each camera's
frame numbers can differ to accommodate measured synchronization offsets.
Optional measured board-plane truth is used only for metrics after detection.
"""
import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board,Lens,Tracker,change_mask,gray_image,warp,project
from dart_detection import detect_dart,diagnostic_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs=3,required=True)
    p.add_argument('--calibration',required=True)
    p.add_argument('--events',required=True,help='JSON list of reference_frames and settled_frames, each a length-3 integer list')
    p.add_argument('--output',default='recording_validation')
    args=p.parse_args();cv.setNumThreads(1)
    calibration_path=Path(args.calibration).resolve()
    config=json.loads(calibration_path.read_text());events=json.loads(Path(args.events).read_text())
    if config.get('version')!=1 or len(config['cameras'])!=3:raise ValueError('Three-camera version-1 calibration required')
    board=Board(config['size'],tuple(config['radii']));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    caps=[cv.VideoCapture(str(path)) for path in args.sources];results=[]
    try:
        if not all(c.isOpened() for c in caps):raise ValueError('Could not open all three recordings')
        def frame(camera,index):
            if not isinstance(index,int) or index<0:raise ValueError('Frame indices must be nonnegative integers')
            caps[camera].set(cv.CAP_PROP_POS_FRAMES,index)
            ok,image=caps[camera].read()
            if not ok:raise ValueError(f'Cannot read camera {camera}, frame {index}')
            return image
        for index,event in enumerate(events):
            if any(len(event[key])!=3 for key in ('reference_frames','settled_frames')):raise ValueError('Three frame indices required per event')
            result=dict(event=index,outcome='rejected',reason='',estimate=None,error_mm=None)
            images=[];masks=[];failure=None
            for i,entry in enumerate(config['cameras']):
                base=frame(i,event['reference_frames'][i]);changed=frame(i,event['settled_frames'][i])
                if list(base.shape[:2])!=entry['shape'] or changed.shape!=base.shape:raise ValueError('Recording resolution differs from calibration')
                lens=Lens(base.shape,entry['lens']);base=lens.apply(base);changed=lens.apply(changed)
                reference=cv.imread(str(calibration_path.parent/entry['reference']))
                if reference is None:raise ValueError('Missing saved calibration reference')
                tracker=Tracker(reference,entry['H'],board,entry.get('pose_K'))
                tracker.update(base)
                if not tracker.valid:failure=f'camera {i}: reference alignment failed';break
                old_H=tracker.H.copy();base_plane=warp(base,old_H,board)
                tracker.update(changed)
                if not tracker.valid:failure=f'camera {i}: settled alignment failed';break
                drift=np.linalg.norm(project(board.anchors(),old_H@np.linalg.inv(tracker.H))-board.anchors(),axis=1).max()
                if drift>3*board.size/900:failure=f'camera {i}: moved during event; rebase required';break
                rect=warp(changed,tracker.H,board)
                mask=change_mask(gray_image(rect),gray_image(base_plane),board.mask(1.08),28)
                if np.count_nonzero(mask)>np.count_nonzero(board.mask(1.08))*.12:
                    failure=f'camera {i}: large scene change/occlusion';break
                images.append(rect);masks.append(mask)
            if failure:
                result['reason']=failure;results.append(result);continue
            point,ids,reason,selected,info=detect_dart(masks,board)
            result.update(cameras=ids,reason=reason,candidate_counts=info['candidate_counts'])
            if point is not None and np.linalg.norm(point-board.center)<=board.radii[-1]*board.ppm*1.08:
                result.update(outcome='estimated',estimate=board.score(point),pixel=point.tolist())
                result['requires_review']=bool(len(ids)<3 or result['estimate']['boundary_review'])
                if 'truth_xy_mm' in event:
                    truth=board.center+np.array(event['truth_xy_mm'])*board.ppm
                    result['error_mm']=float(np.linalg.norm(point-truth)/board.ppm)
                if 'truth_label' in event:
                    result['outcome']='correct' if result['estimate']['label']==event['truth_label'] else 'wrong'
            elif point is not None:
                result['reason']='intersection outside observable scoring area'
            folder=out/f'event_{index:03d}';folder.mkdir(exist_ok=True)
            for i in range(3):
                cv.imwrite(str(folder/f'camera_{i}_mask.png'),masks[i])
                cv.imwrite(str(folder/f'camera_{i}_axes.jpg'),diagnostic_image(images[i],masks[i],info['candidates'][i],selected[i],point))
            results.append(result)
    finally:
        for c in caps:c.release()
    report=dict(events=results,truth_usage='Optional measured truth only evaluates the result; never enters detection.',
                limitation='Supplied event windows; synchronization offsets and physical ground truth are user annotations. Does not validate automatic live throw timing.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

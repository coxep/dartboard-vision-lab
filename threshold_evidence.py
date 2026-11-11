#!/usr/bin/env python3
"""Compare the old texture-scale threshold with the current wedge-scale threshold.

All images are computed from the saved, observed camera pixels. Known board
geometry supplies ONLY the evaluation mask; it never fills or repaints wedges.
"""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from dartboard import Board, adaptive_block_size, binarize, warp
from render_synthetic import panel
from benchmark_c920 import write_image


def wedge_accuracy(binary, board):
    y,x = np.mgrid[:board.size, :board.size]
    x,y = (x-board.center[0])/board.ppm, (y-board.center[1])/board.ppm
    radius = np.hypot(x,y)
    angle = np.degrees(np.arctan2(x,-y)) % 360
    sector = ((angle+9)//18).astype(int)%20
    # Exclude bull, coloured bands and wire/sector boundary neighborhoods.
    mask = (radius>25)&(radius<155)&((radius<95)|(radius>111))&(np.abs((angle+9)%18-9)<6)
    expected = np.where(sector%2==0,0,255)
    results = [dict(sector=s, expected=int(0 if s%2==0 else 255),
                    correct_fraction=float((binary[mask&(sector==s)]==expected[mask&(sector==s)]).mean())) for s in range(20)]
    return dict(correct_fraction=float((binary[mask]==expected[mask]).mean()),
                worst_sector_fraction=min(r['correct_fraction'] for r in results),
                per_sector=results, evaluated_pixels=int(mask.sum()))


def main():
    cv.setNumThreads(1)
    folder=Path('threshold_evidence');folder.mkdir(exist_ok=True)
    config=json.loads(Path('c920_benchmark/calibration.json').read_text())
    board=Board();rows=[];reports=[]
    for i, entry in enumerate(config['cameras']):
        source=cv.imread(f'c920_benchmark/camera_{i}_reference.png')
        gray=cv.cvtColor(source,cv.COLOR_BGR2GRAY)
        old=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,5)
        new=binarize(gray)
        H=np.array(entry['H'])
        # Nearest sampling keeps evaluation images genuinely binary.
        def binary_warp(image):
            return cv.warpPerspective(image,H,(board.size,board.size),flags=cv.INTER_NEAREST)
        old_plane,new_plane=binary_warp(old),binary_warp(new)
        plane_input=cv.cvtColor(warp(source,H,board),cv.COLOR_BGR2GRAY)
        plane_binary=binarize(plane_input)
        for name,image in [('before',old),('after',new),('before_plane',old_plane),('after_plane',new_plane),('plane_threshold',plane_binary)]:
            write_image(folder/f'camera_{i}_{name}.png',image)
        rows.append(np.hstack([panel(image,title,caption,width=500,height=570) for image,title,caption in [
            (old_plane,f'CAMERA {i+1}: BEFORE','Gaussian adaptive: block 31, C=5; no preblur'),
            (new_plane,'AFTER',f'Gaussian adaptive: block {adaptive_block_size(gray.shape)}, C=5; 5x5 preblur'),
            (plane_binary,'THRESHOLD IN BOARD PLANE',f'Block {adaptive_block_size(plane_input.shape)}; actual validation input')]]))
        reports.append(dict(camera=i, source_block=adaptive_block_size(gray.shape),
                            plane_block=adaptive_block_size(plane_input.shape),
                            before=wedge_accuracy(old_plane,board), after=wedge_accuracy(new_plane,board),
                            threshold_in_plane=wedge_accuracy(plane_binary,board)))
    write_image(folder/'comparison.jpg',np.vstack(rows))
    report=dict(method='Gaussian adaptive, C=5, block=nearest odd to half the shorter image dimension; Gaussian preblur 5x5 sigma1.2',
                evaluation='Single-score wedge interiors only: radius 25-155mm excluding 95-111mm and within 6 degrees of sector centers. Colours, wire boundaries and bull excluded.',
                note='Fixed synthetic camera inputs only; this measures binary wedge classification, not pose or dart accuracy.',cameras=reports)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(camera=r['camera'],before=r['before']['correct_fraction'],after=r['after']['correct_fraction'],plane=r['threshold_in_plane']['correct_fraction']) for r in reports],indent=2))

if __name__=='__main__':main()

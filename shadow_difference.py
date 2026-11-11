"""Colour-consistent, shared-plane shadow rejection for aligned camera images.

HSV gates dimming candidates; hue alone is never a foreground detector. Require
compatible dimming at the SAME board coordinate in at least two cameras. Raised
dart shafts generally project to different board coordinates across views.
This is a conservative heuristic, not a physical proof that a pixel is a shadow.
"""
import cv2 as cv
import numpy as np


def shadow_aware_masks(images,backgrounds,valid_masks,board_mask,threshold=28):
    if len(images)<2 or not len(images)==len(backgrounds)==len(valid_masks):
        raise ValueError('At least two aligned colour views with matching backgrounds and visibility masks required')
    candidates=[];ratios=[];visible=[];changes=[]
    for image,background,valid in zip(images,backgrounds,valid_masks):
        if image.shape!=background.shape or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Matching BGR colour images required')
        current=cv.GaussianBlur(image,(3,3),0);reference=cv.GaussianBlur(background,(3,3),0)
        hsv=cv.cvtColor(current,cv.COLOR_BGR2HSV).astype(np.float32)
        old=cv.cvtColor(reference,cv.COLOR_BGR2HSV).astype(np.float32)
        ratio=hsv[...,2]/np.maximum(old[...,2],1)
        dh=np.abs(hsv[...,0]-old[...,0]);dh=np.minimum(dh,180-dh)
        # On neutral/dark areas hue is noisy; use scaled RGB consistency instead.
        predicted=reference.astype(np.float32)*ratio[...,None]
        chroma=np.max(np.abs(current.astype(np.float32)-predicted),axis=2)/np.maximum(old[...,2],35)
        hue_reliable=(hsv[...,1]>40)&(old[...,1]>40)
        colour_consistent=(chroma<.10)&(np.abs(hsv[...,1]-old[...,1])<35)&(~hue_reliable|(dh<10))
        seen=(valid>254)&(board_mask>0)
        candidate=seen&(old[...,2]>35)&(ratio>=.50)&(ratio<.99)&colour_consistent
        candidates.append(candidate);ratios.append(ratio);visible.append(seen)
        gray=cv.cvtColor(current,cv.COLOR_BGR2GRAY);base=cv.cvtColor(reference,cv.COLOR_BGR2GRAY)
        changes.append((cv.absdiff(gray,base)>=threshold)&seen)
    visible=np.asarray(visible);candidates=np.asarray(candidates);ratios=np.asarray(ratios)
    # A shaft can occlude a genuine shadow in one view. Estimate dimming from
    # agreeing pairs, then test EACH view against it; a single global veto would
    # copy the other cameras' shaft-shaped holes into every foreground mask.
    field=np.ones(board_mask.shape,np.float32);best=np.zeros(board_mask.shape,np.float32)
    import itertools
    for i,j in itertools.combinations(range(len(images)),2):
        mean=(ratios[i]+ratios[j])*.5
        pair=candidates[i]&candidates[j]&(np.abs(ratios[i]-ratios[j])<.04)
        support=(candidates&(np.abs(ratios-mean)<.04)).sum(axis=0)
        quality=np.where(pair,support+mean*.01,0)
        update=quality>best;field[update]=mean[update];best[update]=quality[update]
    common=best>0
    results=[];suppressed=[]
    for changed,candidate,ratio in zip(changes,candidates,ratios):
        shadow=common&candidate&(np.abs(ratio-field)<.04)
        foreground=np.uint8(changed&~shadow)*255
        foreground=cv.morphologyEx(foreground,cv.MORPH_CLOSE,np.ones((3,3),np.uint8))
        results.append(foreground)
        suppressed.append(np.uint8(changed&shadow)*255)
    return results,dict(common_shadow=np.uint8(common)*255,suppressed=suppressed,
        method='HSV and scaled RGB consistency plus paired-camera planar dimming agreement, tested per view',
        limitation='Coloured illumination, very dark shadows and a new object with matching appearance in every view can remain ambiguous.')

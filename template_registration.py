"""Direct observed-pattern registration and explicit ring-boundary measurements.

The objective is image appearance in canonical board coordinates, not distances
between detected corners and model corners. A homography seed may come from a
geometric detector. Neither camera intrinsics nor pose decomposition is required.
Input should be undistorted if significant lens curvature is present: a 3x3 H
cannot represent a general lens-correction warp.
"""
import cv2 as cv
import numpy as np


def contrast_normalized(gray,scale=1.):
    smooth=cv.GaussianBlur(gray.astype(np.float32),(0,0),max(.6,scale))
    mean=cv.GaussianBlur(smooth,(0,0),9*scale)
    variance=cv.GaussianBlur(smooth*smooth,(0,0),9*scale)-mean*mean
    return np.clip((smooth-mean)/np.sqrt(np.maximum(variance,12.**2)),-3,3).astype(np.float32)


def coarse_grayscale_seed(image,seed,board,wire_geometry=None):
    """Direct grayscale pyramid registration; no thresholded input or contours."""
    from wire_model import canonical_template
    gray=image if image.ndim==2 else cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    observed=cv.warpPerspective(gray,seed,(board.size,board.size))
    template=canonical_template(board,wire_geometry)
    ref=contrast_normalized(template,board.size/900)
    observed=contrast_normalized(observed,board.size/900)
    valid=cv.warpPerspective(np.full(gray.shape,255,np.uint8),seed,(board.size,board.size),flags=cv.INTER_NEAREST)
    yy,xx=np.mgrid[:board.size,:board.size]
    radius=np.hypot(xx-board.center[0],yy-board.center[1])/board.ppm
    mask=np.uint8((valid>250)&(radius>24)&(radius<board.radii[-1]*1.04))*255
    W=np.eye(3,dtype=np.float32);history=[]
    for scale in (.25,.5):
        shape=(round(board.size*scale),round(board.size*scale));S=np.diag([scale,scale,1.]).astype(np.float32)
        local=S@W@np.linalg.inv(S)
        correlation,local=cv.findTransformECC(cv.resize(ref,shape,interpolation=cv.INTER_AREA),
            cv.resize(observed,shape,interpolation=cv.INTER_AREA),local,cv.MOTION_HOMOGRAPHY,
            (cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,40,1e-5),
            inputMask=cv.resize(mask,shape,interpolation=cv.INTER_NEAREST),gaussFiltSize=5)
        W=np.linalg.inv(S)@local@S;history.append(dict(scale=scale,correlation=float(correlation)))
    if correlation<.75:raise ValueError('Grayscale board template lacks correlation')
    return np.linalg.inv(W)@seed,dict(objective='locally contrast-normalized grayscale board-template pyramid',history=history)


def refine_appearance(image,seed,board,wire_geometry=None):
    """Register actual observed board pixels to the fixed canonical template."""
    from dartboard import project
    from wire_model import canonical_template
    gray=image if image.ndim==2 else cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    observed=cv.warpPerspective(gray,seed,(board.size,board.size))
    valid=cv.warpPerspective(np.full(gray.shape,255,np.uint8),seed,(board.size,board.size),flags=cv.INTER_NEAREST)
    template=canonical_template(board,wire_geometry)
    # Include pixels on BOTH sides of the outer double boundary. A mask cropped
    # inside the scoring radius omits the very edge we need to align.
    roi=np.uint8((valid>250)&(board.mask(1.04)>0))*255
    # Select neighbourhoods from known metric geometry, not segmented pixels.
    yy,xx=np.mgrid[:board.size,:board.size]
    delta=np.stack([xx,yy],axis=-1)-board.center
    radius=np.linalg.norm(delta,axis=-1)
    angle=np.arctan2(delta[...,0],-delta[...,1])
    spoke_angle=(angle+np.deg2rad(9))%np.deg2rad(18)
    spoke_distance=radius*np.sin(np.minimum(spoke_angle,np.deg2rad(18)-spoke_angle))
    ring_distance=np.min(abs(radius[...,None]-np.array([6.35,15.9,*board.radii])*board.ppm),axis=-1)
    support=np.uint8((ring_distance<7*board.size/900)|
                     ((spoke_distance<7*board.size/900)&(radius>15.9*board.ppm)))*255
    reference=contrast_normalized(template,board.size/900)
    observed=contrast_normalized(observed,board.size/900)
    W=np.eye(3,dtype=np.float32);history=[]
    for level,scale in enumerate((.5,1.)):
        shape=(round(board.size*scale),round(board.size*scale))
        S=np.diag([scale,scale,1.]).astype(np.float32)
        local=S@W@np.linalg.inv(S)
        mask=roi if level==0 else (roi&support)
        correlation,local=cv.findTransformECC(
            cv.resize(reference,shape,interpolation=cv.INTER_AREA),
            cv.resize(observed,shape,interpolation=cv.INTER_AREA),local,cv.MOTION_HOMOGRAPHY,
            (cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,40,1e-5),
            inputMask=cv.resize(mask,shape,interpolation=cv.INTER_NEAREST),gaussFiltSize=5)
        W=np.linalg.inv(S)@local@S
        history.append(dict(scale=scale,correlation=float(correlation)))
    candidate=np.linalg.inv(W)@seed
    delta=project(board.landmarks(),candidate@np.linalg.inv(seed))-board.landmarks()
    if not np.isfinite(candidate).all() or history[-1]['correlation']<.7 or np.linalg.norm(delta,axis=1).max()>12*board.size/900:
        raise ValueError('Direct template appearance fit lacks correlation or exceeds local capture range')
    return candidate,dict(objective='locally contrast-normalized board appearance, then ring/spoke template neighbourhoods',
        optimizer='multiscale ECC, homography',history=history,
        correspondence_refit_after_registration=False,outer_boundary_in_objective=True)


def measure_ring_boundaries(image,H,board,wire_geometry=None):
    """Detect edge-normal transitions, independently for every named boundary.

    Fit a photometric step plus a narrow wire-highlight nuisance term to sampled
    image profiles. This reports where the observed edge lies. Ring radii never
    float, and the measurements are validation, not fake 2D point correspondences.
    """
    from dartboard import project
    scale=board.size/900
    plane=cv.warpPerspective(image,H,(board.size,board.size))
    if plane.ndim==2:plane=plane[...,None]
    if plane.shape[2]==4:plane=plane[...,:3]
    valid=cv.warpPerspective(np.full(image.shape[:2],255,np.uint8),H,(board.size,board.size),flags=cv.INTER_NEAREST)
    angles=np.deg2rad((18*np.arange(20)[:,None]+[-6.,-3.,0.,3.,6.]).ravel())
    normal=np.c_[np.sin(angles),-np.cos(angles)]
    tangent=np.c_[-normal[:,1],normal[:,0]]
    radii=np.array([6.35,15.9,*board.radii])*board.ppm
    names=['inner bull','outer bull','treble inner','treble outer','double inner','double outer']
    raised=wire_geometry and wire_geometry.get('height_mm',0)>0
    reference=None
    if raised:
        from wire_model import canonical_template
        reference=canonical_template(board,wire_geometry).astype(np.float32)
    reports=[]
    for ring,(radius,name) in enumerate(zip(radii,names)):
        gap=np.min(abs(np.delete(radii,ring)-radius))
        gate=min(6*scale,.4*gap)
        offsets=np.arange(-gate-4*scale,gate+4.01*scale,.5*scale)
        shifts=np.arange(-gate,gate+.01*scale,.5*scale)
        # Free left/right brightness and wire highlight, shared edge location.
        x=(offsets[None,:]-shifts[:,None])/scale
        basis=np.stack([np.ones_like(x),np.tanh(x/1.1),np.exp(-.5*(x/.65)**2)],axis=2)
        inverse=np.linalg.pinv(basis)
        profiles=[];visibility=[]
        for lateral in (-1.,0.,1.):
            xy=board.center+(radius+offsets[None,:,None])*normal[:,None,:]+lateral*scale*tangent[:,None,:]
            values=cv.remap(plane.astype(np.float32),xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv.INTER_LINEAR)
            if values.ndim==2:values=values[...,None]
            profiles.append(values)
            visibility.append(cv.remap(valid,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv.INTER_NEAREST).min(axis=1)>250)
        observed=np.mean(profiles,axis=0)
        visible=np.logical_and.reduce(visibility)
        acceptance=.22
        if raised:
            # Match the complete height-aware local template, rather than
            # assigning the wire-highlight apex to a z=0 step/corner.
            samples=board.center+(radius+offsets[None,None,:,None]-shifts[None,:,None,None])*normal[:,None,None,:]
            target=cv.remap(reference,samples[...,0].reshape(len(angles),-1).astype(np.float32),
                            samples[...,1].reshape(len(angles),-1).astype(np.float32),cv.INTER_LINEAR).reshape(len(angles),len(shifts),len(offsets))
            values=observed@np.array([.114,.587,.299]) if observed.shape[-1]==3 else observed[...,0]
            values-=values.mean(axis=1,keepdims=True);target-=target.mean(axis=2,keepdims=True)
            norm=np.linalg.norm(values,axis=1);target_norm=np.linalg.norm(target,axis=2)
            correlation=np.einsum('nst,nt->ns',target,values)/np.maximum(target_norm*norm[:,None],1e-8)
            score=1-correlation
            score[(target_norm/np.sqrt(len(offsets))<8)|(norm[:,None]/np.sqrt(len(offsets))<8)]=np.inf
            acceptance=.15
        else:
        # axes: profile, shift hypothesis, coefficient, colour channel
            coefficients=np.einsum('skt,ntc->nskc',inverse,observed)
            predicted=np.einsum('stk,nskc->nstc',basis,coefficients)
            rms=np.sqrt(np.mean((predicted-observed[:,None])**2,axis=(2,3)))
            contrast=2*np.sqrt(np.mean(coefficients[:,:,1]**2,axis=2))
            score=rms/np.maximum(contrast,1)
            score[(contrast<18)|(contrast>240)]=np.inf
        best=score.argmin(axis=1);row=np.arange(len(angles))
        delta=shifts[best].copy()
        for i,j in enumerate(best):
            if 0<j<len(shifts)-1 and np.isfinite(score[i,j-1:j+2]).all():
                left,middle,right=score[i,j-1:j+2];denom=left-2*middle+right
                if denom>1e-8:delta[i]+=.5*(left-right)/denom*.5*scale
        good=visible&(score[row,best]<acceptance)&(abs(delta)<gate-.25*scale)
        points=board.center+(radius+delta[good,None])*normal[good]
        offsets_good=delta[good]
        quadrants=len(set((np.degrees(angles[good])%360//90).astype(int)))
        supported=bool(good.sum()>=12 and quadrants>=3 and good.sum()>=.4*max(int(visible.sum()),1))
        reports.append(dict(boundary=name,radius_mm=float(radius/board.ppm),found=supported,
            samples=int(good.sum()),visible_profiles=int(visible.sum()),quadrants=quadrants,
            search_halfwidth_px=float(gate),median_offset_px=float(np.median(offsets_good)) if len(points) else None,
            p95_absolute_offset_px=float(np.percentile(abs(offsets_good),95)) if len(points) else None,
            source_px=project(points,np.linalg.inv(H)).tolist() if len(points) else [],
            location_definition='surface boundary inferred from height-aware image profile' if raised else 'photometric transition near the model boundary',
            angles_deg=(np.degrees(angles[good])%360).tolist()))
    return reports


def verify_scoring_boundaries(reports,board):
    """All four treble/double edges need direct observed support, including outer."""
    scale=board.size/900
    return all(r['found'] and abs(r['median_offset_px'])<2.5*scale
               and r['p95_absolute_offset_px']<4*scale for r in reports[2:])

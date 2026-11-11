"""Experimental joint plane-homography / radial-distortion edge fit.

K stays fixed. k1 is estimated by default; optional k2 can overfit a single view.
Other distortion terms remain at their supplied priors.
Circle detection is a separate post-fit diagnostic on observed rectified pixels.
"""
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


def distort_pixels(pixels, K, D):
    """Forward Brown-Conrady distortion, in the original image pixel coordinates."""
    K,D=np.asarray(K,float),np.asarray(D,float).ravel()
    x=(pixels[:,0]-K[0,2])/K[0,0]
    y=(pixels[:,1]-K[1,2])/K[1,1]
    k1,k2,p1,p2,k3=D
    r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    return np.c_[K[0,0]*xd+K[0,2],K[1,1]*yd+K[1,2]]


def joint_edge_fit(frame, board, K, seed_H, distortion_prior=None, radial_terms=1):
    """Fit model edges directly to RAW, distorted camera pixels.

    H maps undistorted pixels to the board. No circle detections or image truth
    enter this solve. An existing pose/template initializer supplies seed_H.
    """
    from dartboard import binarize, board_color_masks
    if radial_terms not in (1,2):raise ValueError("radial_terms must be 1 or 2")
    K=np.asarray(K,float)
    prior=np.zeros(5)
    if distortion_prior is not None:
        values=np.asarray(distortion_prior,float).ravel()
        if len(values) not in (4,5):raise ValueError('Joint edge fitting currently supports 4/5-term Brown distortion priors')
        prior[:len(values)]=values
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    colours=board_color_masks(frame)
    ring_mask=cv.morphologyEx(colours['red']|colours['green'],cv.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Separate observed fields prevent ring samples being attracted to spokes.
    raw_edges=[cv.Canny(ring_mask,50,150), cv.Canny(binarize(gray),50,150)]
    radius=board.radii[-1]
    circles=np.array([board.point(r,a) for r in (6.35,15.9,*board.radii)
                      for a in np.linspace(0,360,240,endpoint=False)])
    spokes=np.array([board.point(r,-9+18*s) for s in range(20)
                     for r in np.linspace(22,radius-2,70)])
    model=np.vstack([circles,spokes])
    groups=np.r_[np.zeros(len(circles),int),np.ones(len(spokes),int)]
    normalized=(model-board.center)/(radius*board.ppm)
    points=np.c_[normalized,np.ones(len(model))]
    S=np.array([[radius*board.ppm,0,board.center[0]],
                [0,radius*board.ppm,board.center[1]],[0,0,1.]])
    base=np.linalg.inv(seed_H)@S
    def correction(p):
        return np.array([[1+p[0],p[1],p[2]],[p[3],1+p[4],p[5]],[p[6],p[7],1.]])
    def predict(p, selected):
        q=selected@(base@correction(p)).T
        D=prior.copy();D[:radial_terms]=p[8:]*.1
        return distort_pixels(q[:,:2]/q[:,2:],K,D)
    p=np.r_[np.zeros(8),prior[:radial_terms]/.1]
    lower=np.r_[np.full(8,-.2),np.full(radial_terms,-2.5)]
    upper=-lower
    history=[]
    for scale in (.5,1.):
        width,height=round(frame.shape[1]*scale),round(frame.shape[0]*scale)
        fields=[]
        for edge in raw_edges:
            small=cv.resize(edge,(width,height),interpolation=cv.INTER_AREA)
            small=np.uint8(small>20)*255
            distance=cv.distanceTransform(255-small,cv.DIST_L2,5)
            fields.append(cv.GaussianBlur(distance,(3,3),.5).astype(float))
        predicted=predict(p,points)*scale
        keep=((predicted[:,0]>4)&(predicted[:,0]<width-5)&(predicted[:,1]>4)&(predicted[:,1]<height-5))
        if keep.mean()<.65:raise ValueError('Too little model visible for joint lens/plane fit')
        active=points[keep];active_groups=groups[keep]
        def residual(parameters,regularize=True):
            q=predict(parameters,active)*scale
            x=np.clip(q[:,0],0,width-1.001);y=np.clip(q[:,1],0,height-1.001)
            xi,yi=np.floor(x).astype(int),np.floor(y).astype(int)
            dx,dy=x-xi,y-yi
            value=np.zeros(len(q))
            for g,field in enumerate(fields):
                v=((1-dx)*(1-dy)*field[yi,xi]+dx*(1-dy)*field[yi,xi+1]
                   +(1-dx)*dy*field[yi+1,xi]+dx*dy*field[yi+1,xi+1])
                value[active_groups==g]=v[active_groups==g]
            # Do not reward coordinates that leave the image after initialization.
            value+=np.linalg.norm(q-np.c_[x,y],axis=1)
            if regularize:
                return np.r_[value,.05*parameters[:8],.3*(parameters[8:]-prior[:radial_terms]/.1)]
            return value/scale
        fit=least_squares(residual,p,bounds=(lower,upper),loss='soft_l1',f_scale=1.,
                          max_nfev=160,x_scale='jac',ftol=1e-6,xtol=1e-6,gtol=1e-6)
        p=fit.x
        errors=residual(p,False)
        history.append(dict(scale=scale,evaluations=fit.nfev,median_source_px=float(np.median(errors)),
                            p90_source_px=float(np.percentile(errors,90)),k1=float(p[8]*.1),k2=float(p[9]*.1) if radial_terms==2 else float(prior[1])))
    D=prior.copy();D[:radial_terms]=p[8:]*.1
    H=S@np.linalg.inv(base@correction(p))
    info=dict(method='raw observed HSV-ring / adaptive-spoke edges: joint H + radial coefficients',
              estimated=['homography','k1']+(['k2'] if radial_terms==2 else []),
              fixed=['K','p1','p2','k3']+(['k2'] if radial_terms==1 else []),
              distortion_prior=prior.tolist(),D=D.tolist(),history=history,
              seed_H=np.asarray(seed_H).tolist(),H=H.tolist(),
              samples=int(keep.sum()),median_source_px=float(np.median(errors)),p90_source_px=float(np.percentile(errors,90)))
    if not np.isfinite(H).all() or info['median_source_px']>3 or info['p90_source_px']>8:
        raise ValueError('Joint edge fit did not reach acceptable edge support')
    return H,D,info,raw_edges


def detect_rectified_circles(rectified,board):
    """Hough gradient detector on actual image intensities, with radius search bands.

    This is a post-fit diagnostic, not an input to joint_edge_fit, and not an
    independent accuracy measurement against physical ground truth.
    """
    gray=cv.GaussianBlur(cv.cvtColor(rectified,cv.COLOR_BGR2GRAY),(5,5),1.2)
    overlay=rectified.copy();results=[]
    for radius in (15.9,*board.radii):
        expected=radius*board.ppm
        circles=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,dp=1.,minDist=30,
                               param1=90,param2=12 if expected<50 else 18,minRadius=max(4,round(expected-5)),
                               maxRadius=round(expected+5))
        if circles is None:
            results.append(dict(nominal_radius_mm=radius,found=False));continue
        candidates=circles[0]
        candidates=candidates[np.linalg.norm(candidates[:,:2]-board.center,axis=1)<25]
        if not len(candidates):
            results.append(dict(nominal_radius_mm=radius,found=False));continue
        circle=min(candidates,key=lambda c:np.linalg.norm(c[:2]-board.center)+.5*abs(c[2]-expected))
        x,y,r=map(float,circle)
        consistent=np.linalg.norm(circle[:2]-board.center)<=4 and abs(r-expected)<=3
        colour=(40,220,255) if consistent else (40,40,255)
        cv.circle(overlay,(round(x),round(y)),round(r),colour,2,cv.LINE_AA)
        cv.drawMarker(overlay,(round(x),round(y)),(40,220,255),cv.MARKER_CROSS,10,1)
        results.append(dict(nominal_radius_mm=radius,found=True,consistent=bool(consistent),center_px=[x,y],radius_px=r,
                            center_offset_px=float(np.linalg.norm(circle[:2]-board.center)),
                            radius_offset_px=float(r-expected)))
    return results,overlay


def calibrate_unknown_radial(raw,board,K,nominal_H=None,distortion_prior=None):
    """Opt-in experimental path: edges fit H+k1; circles only inspect the output."""
    from dartboard import (Lens,automatic_homography,refine_plane_template,warp,
                           harris_refine,binary_template_support,project)
    K=np.asarray(K,float)
    prior=np.zeros(5)
    if distortion_prior is not None:
        values=np.asarray(distortion_prior,float).ravel()
        if len(values) not in (4,5):raise ValueError('Joint edge fitting currently supports 4/5-term Brown distortion priors')
        prior[:len(values)]=values
    lens_data=dict(K=K.tolist(),D=prior.tolist(),image_size=[raw.shape[1],raw.shape[0]])
    preliminary=Lens(raw.shape,lens_data).apply(raw)
    try:
        seed,initialization=automatic_homography(preliminary,board,nominal_H,K)
    except (ValueError,cv.error):
        if nominal_H is None:raise
        seed,initialization=refine_plane_template(preliminary,nominal_H,board)
    H,D,fit,edges=joint_edge_fit(raw,board,K,seed,prior)
    lens_data['D']=D.tolist()
    corrected=Lens(raw.shape,lens_data).apply(raw)
    rectified=warp(corrected,H,board)
    _,report=harris_refine(rectified,np.eye(3),board)
    support=binary_template_support(corrected,H,board)
    if not report['refined'] or support<.6:
        raise ValueError('Joint edge/lens fit failed final distributed-feature/edge verification')
    report['matched_source_px']=project(report['matched_source_px'],np.linalg.inv(H)).tolist()
    report['binary_template_support']=support
    report['harris_role']='validation only; joint edge/lens H retained'
    report['joint_edge_fit']=fit
    circles,overlay=detect_rectified_circles(rectified,board)
    report['postfit_circle_detections']=circles
    report['circle_role']='post-fit Hough diagnostic, not used by edge optimizer'
    return H,lens_data,report,corrected,overlay

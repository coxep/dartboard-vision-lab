"""Measure discrete paint-boundary offsets, without fitting circles or changing H.

This is an independent diagnostic for colour images. Paint edges are not assumed
to coincide exactly with wire apexes. Search gates cannot certify a wrong model.
"""
import cv2 as cv
import numpy as np


def audit_boundaries(rectified,board):
    from dartboard import board_color_masks
    masks=board_color_masks(rectified)
    coloured=(masks['red']|masks['green']).astype(np.float32)/255
    angles=np.deg2rad((18*np.arange(20)[:,None]+[-5.,0.,5.]).ravel())
    normals=np.c_[np.sin(angles),-np.cos(angles)]
    specs=[('inner bull',6.35,masks['red'].astype(np.float32)/255,-1),
           ('outer bull',15.9,coloured,-1),
           ('treble inner',board.radii[0],coloured,1),('treble outer',board.radii[1],coloured,-1),
           ('double inner',board.radii[2],coloured,1),('double outer',board.radii[3],coloured,-1)]
    overlay=rectified.copy();reports=[]
    radii=np.array([s[1] for s in specs])*board.ppm
    for ring,(label,radius,mask,sign) in enumerate(specs):
        # An association window must never reach the neighboring boundary's
        # window. Patch contexts may overlap; boundary identities must not.
        separation=np.min(abs(np.delete(radii,ring)-radii[ring]))
        gate=min(9.*board.size/900,.45*separation)
        offsets=np.arange(-gate,gate+.001,.125*board.size/900)
        xy=board.center+(radius*board.ppm+offsets[None,:,None])*normals[:,None,:]
        profile=cv.remap(mask,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv.INTER_LINEAR)
        found=[];measured=[]
        for ray,normal in zip(profile,normals):
            candidates=np.flatnonzero((sign*np.diff(ray)>.01)&((ray[:-1]-.5)*(ray[1:]-.5)<=0))
            if not len(candidates):continue
            positions=offsets[candidates]+(.5-ray[candidates])/(ray[candidates+1]-ray[candidates])*(offsets[1]-offsets[0])
            offset=float(positions[np.argmin(abs(positions))]);found.append(offset)
            point=board.center+(radius*board.ppm+offset)*normal;measured.append(point)
            cv.circle(overlay,tuple(np.rint(point).astype(int)),2,(255,220,20),-1,cv.LINE_AA)
        expected=board.center+radius*board.ppm*np.c_[np.sin(np.linspace(0,2*np.pi,721)), -np.cos(np.linspace(0,2*np.pi,721))]
        cv.polylines(overlay,[np.rint(expected).astype(np.int32)],True,(220,40,220),1,cv.LINE_AA)
        reports.append(dict(boundary=label,radius_mm=radius,samples=len(found),coverage=len(found)/len(normals),
            disjoint_search_halfwidth_px=float(gate),
            median_signed_offset_px=float(np.median(found)) if found else None,
            p95_absolute_offset_px=float(np.percentile(abs(np.asarray(found)),95)) if found else None,
            observed_px=np.asarray(measured).tolist(),offsets_px=found))
    return reports,overlay


def boundary_geometry_valid(reports,scale=1.):
    """Every band edge must be supported; a low 3-ring corner residual is insufficient.

    Paint/wire discrepancies are tolerated by 2px median / 4px p95. This is a
    diagnostic acceptance rule, not a certificate of measured metric accuracy.
    """
    return all(r['coverage']>=.65 and r['median_signed_offset_px'] is not None
               and abs(r['median_signed_offset_px'])<=2*scale
               and r['p95_absolute_offset_px']<=4*scale for r in reports)


def refine_labelled_boundaries(frame,board,initial_H,K,iterations=4,initial_D=None):
    """Experimental H+k1 fit to separately identified, discrete colour edges.

    Each observed point keeps its own ring identity. Fit distances to those
    FIXED metric boundaries, never to the union of all nearby template edges.
    There are no free fitted circle radii. K is a supplied fixed parameterization;
    if it is guessed, the recovered k1 is not a measured camera calibration.
    """
    from scipy.optimize import least_squares
    from dartboard import Lens,project,warp
    from edge_calibration import distort_pixels
    K=np.asarray(K,np.float64);H=np.asarray(initial_H,np.float64).copy();D=np.zeros(5)
    if initial_D is not None:
        supplied=np.asarray(initial_D,np.float64).ravel()
        if len(supplied) not in (4,5):raise ValueError('Labelled-boundary fit supports 4/5-term Brown D')
        D[:len(supplied)]=supplied
    radius=board.radii[-1]*board.ppm
    S=np.array([[radius,0,board.center[0]],[0,radius,board.center[1]],[0,0,1.]])
    history=[]
    def correction(p):
        return np.array([[1+p[0],p[1],p[2]],[p[3],1+p[4],p[5]],[p[6],p[7],1.]])
    for iteration in range(iterations):
        data=dict(K=K.tolist(),D=D.tolist(),image_size=[frame.shape[1],frame.shape[0]])
        clear=Lens(frame.shape,data).apply(frame)
        boundaries,_=audit_boundaries(warp(clear,H,board),board)
        if any(r['coverage']<.5 for r in boundaries):raise ValueError('Too little distinct boundary support')
        sources=[];targets=[]
        for boundary in boundaries:
            # Float64 is essential for the finite-difference lens derivative.
            undistorted=project(boundary['observed_px'],np.linalg.inv(H)).astype(np.float64)
            sources.extend(distort_pixels(undistorted,K,D))
            targets.extend([boundary['radius_mm']*board.ppm]*len(undistorted))
        sources=np.asarray(sources,np.float64);targets=np.asarray(targets)
        def residual(p):
            d=D.copy();d[0]=p[8]
            u=cv.undistortPoints(sources.reshape(-1,1,2),K,d,P=K).reshape(-1,2)
            candidate=S@correction(p)@np.linalg.inv(S)@H
            q=np.c_[u,np.ones(len(u))]@candidate.T
            q=q[:,:2]/q[:,2:]
            # Boundary normal distances, with a weak orientation/lens prior.
            return np.r_[np.linalg.norm(q-board.center,axis=1)-targets,.02*p[:8],.05*p[8]]
        fit=least_squares(residual,np.r_[np.zeros(8),D[0]],loss='soft_l1',f_scale=1.,max_nfev=100,
            bounds=(np.r_[np.full(8,-.15),-.6],np.r_[np.full(8,.15),.6]))
        H=S@correction(fit.x)@np.linalg.inv(S)@H;D[0]=fit.x[8]
        history.append(dict(iteration=iteration+1,k1=float(D[0]),samples=len(sources),evaluations=fit.nfev))
    data=dict(K=K.tolist(),D=D.tolist(),image_size=[frame.shape[1],frame.shape[0]])
    rectified=warp(Lens(frame.shape,data).apply(frame),H,board)
    boundaries,overlay=audit_boundaries(rectified,board)
    valid=bool(boundary_geometry_valid(boundaries,board.size/900) and abs(D[0])<.59)
    return H,data,dict(geometry_valid=valid,method='labelled disjoint boundary association + robust H/k1 fit',
        boundaries=boundaries,history=history,fitted=['H (8 DOF)','k1'],fixed=['K','k2','p1','p2','k3','all metric boundary radii'],
        limitation='Internal paint-edge consistency only; no free-radius circle detector, measured intrinsics, physical accuracy truth or independent validation.'),overlay

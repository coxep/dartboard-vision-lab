"""Candidate shaft axes and cross-view consensus from changed image pixels.

No camera truth, rendered dart axis, or target coordinate enters this module.
"""
import itertools
import math
import cv2 as cv
import numpy as np


def axis_candidates(mask, board, maximum=5):
    """Fit thin cross-section centres, preserving separate fragment hypotheses.

    Flights and barrels can dominate silhouette area. Each elongated connected
    component supplies an orientation; narrow cross sections from all aligned
    pixels supply the axis fit. All changed pixels remain available to check
    whether a proposed hit is actually observed, including small tip fragments.
    """
    scale=board.size/900
    y,x=np.where(mask>0)
    if len(x)<12:return []
    all_points=np.float32(np.c_[x,y])
    n,labels,stats,_=cv.connectedComponentsWithStats(mask,8)
    proposals=[]
    for i in sorted(range(1,n),key=lambda i:stats[i,cv.CC_STAT_AREA],reverse=True)[:30]:
        if stats[i,cv.CC_STAT_AREA]<max(8,8*scale*scale):continue
        yy,xx=np.where(labels==i);p=np.float32(np.c_[xx,yy])
        values,vectors=np.linalg.eigh(np.cov(p.T))
        if values[-1]<(5*scale)**2 or values[-1]/max(values[0],.1)<3:continue
        proposals.append((p.mean(0),vectors[:,-1]))
    candidates=[]
    for origin,direction in proposals:
        normal=np.array([-direction[1],direction[0]])
        across=(all_points-origin)@normal
        local=all_points[abs(across)<12*scale]
        along=(local-origin)@direction
        bins=np.floor(along/(2*scale)).astype(int)
        centers=[];widths=[]
        for index in np.unique(bins):
            p=local[bins==index];cross=(p-origin)@normal
            if len(p)<2:continue
            lo,hi=np.percentile(cross,[5,95])
            centers.append(origin+direction*np.median((p-origin)@direction)+normal*(lo+hi)/2)
            widths.append(hi-lo)
        if len(centers)<6:continue
        centers=np.float32(centers);widths=np.asarray(widths)
        narrow=widths<=max(3*scale,min(7*scale,float(np.percentile(widths,55))*1.5))
        p=centers[narrow]
        if len(p)<6:continue
        vx,vy,px,py=cv.fitLine(p,cv.DIST_HUBER,0,.001,.001).ravel()
        direction=np.array([vx,vy],float);origin=np.array([px,py],float)
        normal=np.array([-vy,vx],float)
        residual=abs((p-origin)@normal)
        keep=residual<=max(1.5*scale,3*float(np.median(residual)))
        p=p[keep]
        if len(p)<6:continue
        vx,vy,px,py=cv.fitLine(p,cv.DIST_HUBER,0,.001,.001).ravel()
        direction=np.array([vx,vy],float);origin=np.array([px,py],float)
        normal=np.array([-vy,vx],float)
        residual=abs((p-origin)@normal)
        length=float(np.ptp((p-origin)@direction))
        rms=float(np.sqrt(np.mean(residual**2)))
        if length<12*scale or rms>2*scale:continue
        support=all_points[abs((all_points-origin)@normal)<max(3*scale,3*rms)]
        candidate=dict(normal=normal,constant=float(normal@origin),origin=origin,
                       direction=direction,points=support,fit_points=p,
                       length_px=length,rms_px=rms,quality=float(length/(1+rms/scale)),
                       support_points=all_points)
        duplicate=next((c for c in candidates if abs(c['direction']@direction)>.9995
                        and abs(c['normal']@origin-c['constant'])<1.5*scale),None)
        if duplicate is None:candidates.append(candidate)
        elif candidate['quality']>duplicate['quality']:
            candidates=[c for c in candidates if c is not duplicate];candidates.append(candidate)
    return sorted(candidates,key=lambda c:c['quality'],reverse=True)[:maximum]


def combine_candidates(candidates, tolerance=3., support_distance=12., minimum_angle=15.):
    """Enumerate pair hypotheses, then require a unique cross-view consensus.

    Every available camera with usable axes must support the result. Inconsistent
    views are reported rather than silently picking one of three arbitrary pairs.
    Two-view solutions remain available when a view has no usable shaft at all.
    """
    present=[i for i,c in enumerate(candidates) if c]
    if len(present)<2:return None,[], 'fewer than two usable shaft axes', [None]*len(candidates),{}
    hypotheses=[]
    for i,j in itertools.combinations(present,2):
        for a,b in itertools.product(candidates[i],candidates[j]):
            A=np.array([a['normal'],b['normal']])
            if abs(np.linalg.det(A))<math.sin(math.radians(minimum_angle)):continue
            point=np.linalg.solve(A,[a['constant'],b['constant']])
            selected={};cost=0.
            for camera in present:
                options=[]
                for axis in candidates[camera]:
                    distance=abs(axis['normal']@point-axis['constant'])
                    pixels=axis.get('support_points',axis['points'])
                    nearest=float(np.linalg.norm(pixels-point,axis=1).min())
                    # A tip can match the underlying board's brightness in one
                    # view. Permit a bounded extension there; require close
                    # observed hit support in at least two views.
                    reach=max(support_distance, min(6*support_distance,.35*axis.get('length_px',0)))
                    if distance<=tolerance and nearest<=reach:
                        options.append((distance+2*axis.get('rms_px',0),axis))
                if options:
                    value,axis=min(options,key=lambda x:x[0]);selected[camera]=axis;cost+=value
            if len(selected)<2:continue
            axes=list(selected.values());A=np.array([c['normal'] for c in axes]);b=np.array([c['constant'] for c in axes])
            if np.linalg.cond(A)>10:continue
            point=np.linalg.lstsq(A,b,rcond=None)[0]
            if any(abs(c['normal']@point-c['constant'])>tolerance for c in axes):continue
            close=sum(np.linalg.norm(c.get('support_points',c['points'])-point,axis=1).min()<=support_distance for c in axes)
            if close<2:continue
            hypotheses.append((len(selected),cost,point,selected))
    if not hypotheses:return None,[], 'no supported nonparallel axis intersection',[None]*len(candidates),{}
    hypotheses.sort(key=lambda h:(-h[0],h[1]))
    best=hypotheses[0]
    if best[0]<len(present):
        return None,[], 'usable camera axes disagree',[None]*len(candidates),dict(hypotheses=len(hypotheses))
    rivals=[h for h in hypotheses[1:] if h[0]==best[0] and np.linalg.norm(h[2]-best[2])>2*support_distance
            and h[1]<=best[1]+tolerance]
    if rivals:return None,[], 'multiple separated supported intersections',[None]*len(candidates),dict(hypotheses=len(hypotheses))
    ids=sorted(best[3]);selected=[best[3].get(i) for i in range(len(candidates))]
    return best[2],ids,'three views' if len(ids)>=3 else 'two views; reduced confidence',selected,dict(
        hypotheses=len(hypotheses),axis_residual_px=[float(abs(c['normal']@best[2]-c['constant'])) for c in best[3].values()])


def detect_dart(masks,board):
    candidates=[axis_candidates(m,board) for m in masks]
    scale=board.size/900
    point,ids,reason,selected,info=combine_candidates(candidates,3*scale,12*scale)
    info['candidate_counts']=[len(c) for c in candidates]
    return point,ids,reason,selected,dict(info,candidates=candidates)


def diagnostic_image(rectified,mask,candidates,selected=None,point=None):
    """Orange change pixels, cyan fit samples, green selected axis."""
    out=rectified.copy();out[mask>0]=(.35*out[mask>0]+.65*np.array([40,100,240])).astype(np.uint8)
    for c in candidates:
        color=(80,220,250)
        cv.line(out,tuple(np.rint(c['origin']-1200*c['direction']).astype(int)),
                tuple(np.rint(c['origin']+1200*c['direction']).astype(int)),color,1,cv.LINE_AA)
    if selected is not None:
        for p in selected.get('fit_points',[]):cv.circle(out,tuple(np.rint(p).astype(int)),1,(255,230,40),-1)
        cv.line(out,tuple(np.rint(selected['origin']-1200*selected['direction']).astype(int)),
                tuple(np.rint(selected['origin']+1200*selected['direction']).astype(int)),(90,255,100),1,cv.LINE_AA)
    if point is not None:cv.circle(out,tuple(np.rint(point).astype(int)),8,(255,255,255),2)
    return out

"""Fresh dartboard-pattern measurements in grayscale, independent of board texture.

Inspired by the multiscale local responses and ordered-pattern checks used by
OpenCV's checkerboard detectors. This is an original dartboard-specific detector,
not a port of the symmetric X-corner test or rectangular checkerboard topology.

Input pixels must already be lens-corrected. A plane/orientation prior supplies
the search region and resolves the board's repeated sector pattern. Cold/broad
acquisition and small per-frame updates are separate operations.
"""
import math
import time
import cv2 as cv
import numpy as np


def gray_image(image):
    if image.ndim==2:return image
    if image.shape[2]==4:return cv.cvtColor(image,cv.COLOR_BGRA2GRAY)
    return cv.cvtColor(image,cv.COLOR_BGR2GRAY)


class PatternDetector:
    def __init__(self,board):
        self.board=board
        self.targets=board.landmarks()
        self.scale=board.size/900
        self.patch=max(7,round(11*self.scale))
        # Continuous metric templates centred exactly on each known junction.
        # Three-region T junctions are retained; checkerboard X symmetry is not
        # imposed. Intensities describe classes, never a saved board photograph.
        offsets=np.arange(-self.patch,self.patch+1)
        yy,xx=np.meshgrid(offsets,offsets,indexing='ij')
        xy=self.targets[:,None,None,:]+np.stack([xx,yy],axis=-1)
        delta=(xy-board.center)/board.ppm
        radius=np.linalg.norm(delta,axis=-1)
        angle=np.degrees(np.arctan2(delta[...,0],-delta[...,1]))
        sector=((angle+9)%360//18).astype(int)
        # Mutually exclusive radial intervals. Red/green have similar grayscale
        # brightness: do not manufacture a strong corner at the outer double
        # boundary. Per-boundary counts expose that observability limitation.
        regions=np.searchsorted(np.asarray(board.radii),radius,side='right')
        model=np.where(sector%2==0,40.,190.)
        band=(regions==1)|(regions==3)
        model[band]=103.
        model[regions==4]=30.
        self.templates=[]
        for sigma in (1.6*self.scale,.8*self.scale):
            self.templates.append([cv.GaussianBlur(m.astype(np.float32),(0,0),sigma) for m in model])

    def _measure(self,gray,H,search):
        from dartboard import project
        board=self.board
        plane=cv.warpPerspective(gray,H,(board.size,board.size))
        valid=cv.warpPerspective(np.full(gray.shape,255,np.uint8),H,(board.size,board.size),flags=cv.INTER_NEAREST)
        search=max(2,round(search*self.scale))
        measured=[];ids=[];scores=[]
        smooth=[cv.GaussianBlur(plane.astype(np.float32),(0,0),s*self.scale) for s in (1.6,.8)]
        for index,point in enumerate(self.targets):
            center=point.copy();score=0.
            for level in range(2):
                window=search if level==0 else max(2,round(2*self.scale))
                side=2*(self.patch+window)+1
                patch=cv.getRectSubPix(smooth[level],(side,side),tuple(map(float,center)))
                if patch.std()<7:break
                response=cv.matchTemplate(patch,self.templates[level][index],cv.TM_CCOEFF_NORMED)
                _,score,_,peak=cv.minMaxLoc(response)
                px,py=peak
                if not np.isfinite(score) or score<.65:break
                shift=np.array([px-window,py-window],float)
                for axis,(a,b) in enumerate(((px,py),(py,px))):
                    profile=response[b,:] if axis==0 else response[:,b]
                    if 0<a<len(profile)-1:
                        left,middle,right=map(float,profile[a-1:a+2]);denom=left-2*middle+right
                        if denom< -1e-6:shift[axis]+=np.clip(.5*(left-right)/denom,-.75,.75)
                center=center+shift
            # The test is on the measured patch, with actual source visibility.
            x,y=np.rint(center).astype(int)
            if score>=.78 and 0<=x<board.size and 0<=y<board.size and valid[y,x]>250:
                measured.append(center);ids.append(index);scores.append(score)
        if len(ids)<20:return None,dict(reason='too few compatible grayscale junctions',candidates=len(ids))
        points=np.float32(measured);ids=np.array(ids,int);scores=np.array(scores)
        correction,inliers=cv.findHomography(points,self.targets[ids],cv.RANSAC,2.2*self.scale)
        if correction is None:return None,dict(reason='junction consensus failed',candidates=len(ids))
        good=inliers.ravel().astype(bool)
        # Cyclic ring neighbours and radial chains form the model graph. RANSAC
        # picks consensus; a dense, distributed subgraph is required to accept it.
        kept=set(ids[good].tolist())
        ring_edges=sum((r*20+(s+1)%20) in kept for r,s in (divmod(i,20) for i in kept))
        radial_edges=sum((i+20) in kept for i in kept if i<60)
        spokes=len({i%20 for i in kept});rings=len({i//20 for i in kept})
        quadrants=len({(i%20)//5 for i in kept})
        report=dict(candidates=len(ids),inliers=int(good.sum()),ring_links=ring_edges,radial_links=radial_edges,
                    spokes=spokes,rings=rings,quadrants=quadrants,
                    observed_plane_px=points.tolist(),model_ids=ids.tolist(),inlier_mask=good.tolist(),
                    correlation=scores.tolist())
        if good.sum()<20 or spokes<9 or rings<3 or quadrants<4 or ring_edges+radial_edges<20:
            return None,dict(report,reason='junction graph lacks distributed support')
        from corner_geometry import fit_corner_homography
        correction,fit=fit_corner_homography(points[good],self.targets[ids[good]],correction,board)
        fitted=project(points[good],correction)
        error=np.linalg.norm(fitted-self.targets[ids[good]],axis=1)
        up=project([board.center,board.center+[0,-100]],correction)
        vector=up[1]-up[0]
        rotation=math.degrees(math.atan2(vector[0],-vector[1]))
        if abs(rotation)>9 or np.percentile(error,95)>2.5*self.scale:
            return None,dict(report,reason='junction fit violates orientation/residual checks')
        candidate=correction@H
        # Do not accept homographies whose horizon crosses the scoring surface.
        perimeter=np.array([board.point(board.radii[-1],a) for a in np.linspace(0,360,80,endpoint=False)])
        inverse=np.linalg.inv(candidate)
        denominator=np.c_[perimeter,np.ones(len(perimeter))]@inverse[2]
        visible=project(perimeter,inverse)
        inside=((visible[:,0]>=0)&(visible[:,0]<gray.shape[1])&(visible[:,1]>=0)&(visible[:,1]<gray.shape[0]))
        if denominator.min()*denominator.max()<=0 or inside.mean()<.70 or abs(cv.contourArea(visible.astype(np.float32)))<4000:
            return None,dict(report,reason='insufficient visible board or degenerate perspective')
        report.update(reason='fresh grayscale junction graph',median_residual_px=float(np.median(error)),
                      p95_residual_px=float(np.percentile(error,95)),median_correlation=float(np.median(scores[good])),
                      source_px=project(points[good],np.linalg.inv(H)).tolist(),
                      target_px=self.targets[ids[good]].tolist(),accepted_ids=ids[good].tolist(),least_squares=fit)
        report['inliers_per_boundary']=np.bincount(ids[good]//20,minlength=4).tolist()
        return candidate,report

    def physical_fit(self,H,prior_H,report,K):
        """Fit measured undistorted pixels to metric Z=0 points, with fixed K.

        The unconstrained H only supplies consensus/diagnostics. The returned H
        is derived from a proper rotation and translation; no free H refinement
        follows this step.
        """
        from dartboard import project
        from pose_alignment import (plane_coordinates,pose_from_homography,
            homography_from_pose,fit_seeded_pose,pose_is_plausible)
        K=np.asarray(K,np.float64)
        pixels=np.asarray(report['source_px'],np.float64)
        targets=np.asarray(report['target_px'],np.float64)
        xy=project(targets,plane_coordinates(self.board)).astype(np.float64)
        objects=np.c_[xy,np.zeros(len(xy))]
        r0,t0=pose_from_homography(prior_H,K,self.board)
        center0=(-cv.Rodrigues(r0)[0].T@t0).ravel()
        result=fit_seeded_pose(objects,pixels,K,r0,t0,threshold=3.)
        if result is None:return None,dict(report,reason='seeded metric PnP failed')
        _,r,t,good,errors=result
        ids=np.asarray(report['accepted_ids'])[good];kept=set(ids.tolist())
        ring_links=sum((i//20*20+(i+1)%20) in kept for i in kept)
        radial_links=sum(i+20 in kept for i in kept if i<60)
        support=dict(inliers=len(ids),rings=len(set(ids//20)),spokes=len(set(ids%20)),
                     quadrants=len(set((ids%20)//5)),ring_links=ring_links,radial_links=radial_links)
        if (len(ids)<20 or support['rings']<3 or support['spokes']<9 or support['quadrants']<4
                or ring_links+radial_links<20 or not pose_is_plausible(r,t,center0,objects)):
            return None,dict(report,reason='metric pose lacks distributed support or violates mount prior')
        physical=homography_from_pose(r,t,K,self.board)
        residual=np.linalg.norm(project(pixels[good],physical)-targets[good],axis=1)
        if np.percentile(residual,95)>2.5*self.scale:
            return None,dict(report,reason='metric pose fails canonical residual check')
        # Preserve a same-correspondence free-H comparison for independent GT evaluation.
        report.update(support,unconstrained_H=H.tolist(),source_px=pixels[good].tolist(),
            target_px=targets[good].tolist(),accepted_ids=ids.tolist(),
            median_residual_px=float(np.median(residual)),p95_residual_px=float(np.percentile(residual,95)),
            median_source_residual_px=float(np.median(errors[good])),
            pose=dict(rvec=r.ravel().tolist(),tvec_mm=t.ravel().tolist(),
                      camera_center_mm=(-cv.Rodrigues(r)[0].T@t).ravel().tolist(),
                      fixed_intrinsics=True,degrees_of_freedom=6),
            reason='fresh grayscale junction graph + seeded metric PnP/LM')
        report['inliers_per_boundary']=np.bincount(ids//20,minlength=4).tolist()
        return physical,report

    def detect(self,image,prior_H,search=6.,iterations=1,colour=False,K=None):
        if iterations<1:raise ValueError('iterations must be positive')
        start=time.perf_counter();gray=gray_image(image)
        H=np.asarray(prior_H,float).copy();report={};found=False
        if gray.std()<5:
            return None,dict(found=False,reason='image lacks contrast',elapsed_ms=(time.perf_counter()-start)*1000)
        for _ in range(iterations):
            candidate,report=self._measure(gray,H,search)
            if candidate is None:break
            if K is not None:
                candidate,report=self.physical_fit(candidate,H,report,K)
                if candidate is None:break
            H=candidate;found=True
        if found and candidate is None:found=False
        if found and colour and image.ndim==3:
            # Colour is supplementary evidence, never required for grayscale.
            from corner_geometry import colour_corner_scores
            points=np.array(report['source_px'],np.float32)
            agreement,_=colour_corner_scores(image,H,self.board,points,np.arange(len(points)),report['accepted_ids'])
            report['colour_agreement']=float(np.mean(agreement))
        report.update(found=found,elapsed_ms=(time.perf_counter()-start)*1000,
                      method='multiscale grayscale junction templates + cyclic/radial graph + robust least squares',
                      fit_model='seeded metric PnP + LM (6 DOF)' if K is not None else 'unconstrained homography (8 DOF)',
                      appearance_reference_used=False)
        return H if found else None,report

    def stabilize(self,image,H,report,K=None):
        """Require successive fresh fits to agree before trusting a large update."""
        from dartboard import project
        for step in range(4):
            candidate,report=self.detect(image,H,search=6,iterations=1,K=K)
            if candidate is None:return None,report
            shifts=np.linalg.norm(project(self.targets,candidate@np.linalg.inv(H))-self.targets,axis=1)
            shift=float(shifts.max())
            H=candidate
            if shift<.9*self.scale and np.median(shifts)<.45*self.scale:
                report['stabilization_passes']=step+1
                report['last_update_max_px']=float(shift)
                report['last_update_median_px']=float(np.median(shifts))
                return H,report
        return None,dict(report,found=False,reason='pattern fit did not converge under fresh measurements')

    def acquire(self,image,prior_H,K=None):
        """Try a wider local graph search, then grayscale whole-pattern seeds.

        Orientation/mount prior is required. This is reacquisition in its region,
        not unconstrained discovery of an arbitrarily oriented board in a room.
        """
        start=time.perf_counter()
        H,report=self.detect(image,prior_H,search=18,iterations=2,K=K)
        if H is not None:
            H,report=self.stabilize(image,H,report,K)
            if H is not None:
                report.update(acquisition='wide junction search',elapsed_ms=(time.perf_counter()-start)*1000)
                return H,report
        if gray_image(image).std()<5:return None,report
        from dartboard import refine_plane_template,ideal_board_template
        frame=cv.cvtColor(gray_image(image),cv.COLOR_GRAY2BGR)
        candidates=[np.asarray(prior_H,float)]
        if K is not None:
            from pose_alignment import seeded_pose_hypotheses
            try:
                hypotheses,_=seeded_pose_hypotheses(frame,self.board,K,prior_H,ideal_board_template(self.board))
                candidates=[h for h,info in hypotheses if info['correlation']>.75]+candidates
            except (ValueError,cv.error,np.linalg.LinAlgError):pass
        failures=[]
        for seed in candidates:
            try:
                coarse,_=refine_plane_template(frame,seed,self.board)
                H,report=self.detect(image,coarse,search=6,iterations=2,K=K)
                if H is not None:
                    H,report=self.stabilize(image,H,report,K)
                    if H is not None:
                        report.update(acquisition='grayscale whole-pattern recovery + junction graph',elapsed_ms=(time.perf_counter()-start)*1000)
                        return H,report
            except (ValueError,cv.error,np.linalg.LinAlgError) as exc:failures.append(str(exc))
        report.update(found=False,acquisition='failed',elapsed_ms=(time.perf_counter()-start)*1000,recovery_failures=failures)
        return None,report


class GeometryTracker:
    """Compatibility API: every update remeasures the pattern; no saved texture."""
    def __init__(self,reference,H,board,K=None):
        self.board=board;self.H=np.asarray(H,float).copy();self.reference_H=self.H.copy()
        self.K=K;self.detector=PatternDetector(board);self.valid=False
        self.reason='awaiting pattern detection';self.report={}

    def _update(self,frame,recover):
        from dartboard import project
        start=time.perf_counter();previous_valid=self.valid
        H,report=self.detector.detect(frame,self.H,search=6,iterations=1,K=self.K)
        if H is not None:
            shift=np.linalg.norm(project(self.board.landmarks(),H@np.linalg.inv(self.H))-self.board.landmarks(),axis=1).max()
            if not previous_valid or shift>.75*self.board.size/900 or report['inliers']<40:
                H,report=self.detector.stabilize(frame,H,report,self.K)
        if H is None and recover:H,report=self.detector.acquire(frame,self.H,self.K)
        report['elapsed_ms']=(time.perf_counter()-start)*1000
        self.report=report
        if H is None:
            self.valid=False;self.reason=report['reason'];return False
        drift=np.linalg.norm(project(self.board.landmarks(),H@np.linalg.inv(self.H))-self.board.landmarks(),axis=1).max()
        self.H=H;self.valid=True
        self.reason=f"{report['inliers']} fresh junctions; {report['median_residual_px']:.2f}px residual"
        return bool(drift>.75*self.board.size/900)

    def track(self,frame):return self._update(frame,False)
    def update(self,frame):return self._update(frame,True)


def draw_pattern(rectified,board,report,H):
    """Draw actual observed junctions and their accepted model-graph links."""
    from dartboard import project
    out=cv.cvtColor(rectified,cv.COLOR_GRAY2BGR) if rectified.ndim==2 else rectified.copy()
    if not report.get('found'):return out
    points=project(report['source_px'],H)
    ids=report['accepted_ids'];lookup={i:p for i,p in zip(ids,points)}
    for i,p in lookup.items():
        r,s=divmod(i,20)
        for j in (r*20+(s+1)%20,i+20 if r<3 else -1):
            if j in lookup:cv.line(out,tuple(np.rint(p).astype(int)),tuple(np.rint(lookup[j]).astype(int)),(110,200,60),1,cv.LINE_AA)
        cv.circle(out,tuple(np.rint(p).astype(int)),3,(60,255,130),1,cv.LINE_AA)
        q=board.landmarks()[i]
        cv.line(out,tuple(np.rint(p).astype(int)),tuple(np.rint(p+5*(q-p)).astype(int)),(255,210,40),1,cv.LINE_AA)
    return out

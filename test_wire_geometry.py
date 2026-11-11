"""Independent projection/parallax and raised-wire image regressions."""
import unittest
import cv2 as cv
import numpy as np
from dartboard import Board,project
from render_synthetic import board_texture,render_camera,camera_matrix,project3
from wire_model import apparent_plane,surface_footpoint
from board_pattern import PatternDetector


class WireGeometryTests(unittest.TestCase):
    def test_mixed_surface_and_raised_points_match_independent_camera_projection(self):
        b=Board();C=np.array([-285.,243.,216.])
        _,H,_=render_camera(board_texture(b),b,C,dart=False)
        points=np.array([[0,0,0],[0,0,1],[120,-50,.7],[-70,90,1.2]])
        image_points=project3(points,camera_matrix(C))
        observed=(project(image_points,H)-b.center)/b.ppm*np.array([1,-1])
        np.testing.assert_allclose(observed,apparent_plane(points,C),atol=1e-4)
        recovered=surface_footpoint(observed,points[:,2],C)
        np.testing.assert_allclose(recovered,points[:,:2],atol=1e-4)
        self.assertGreater(np.linalg.norm(observed[1]-points[1,:2]),1.7)
        np.testing.assert_allclose(observed[0],points[0,:2],atol=1e-4)

    def test_raised_wire_alignment_with_inexact_camera_centre_and_lighting(self):
        cv.setNumThreads(1);cv.setRNGSeed(3)
        b=Board(radii=(97.4,107.4,160,170));texture=board_texture(b,wires=False)
        nominal=np.array([-285.,243.,216.])
        geometry=dict(height_mm=1.,width_mm=.8,camera_center_mm=nominal.tolist())
        _,prior,_=render_camera(texture,b,nominal,dart=False)
        image,truth,_=render_camera(texture,b,nominal+[15,-12,8],wire_geometry=geometry)
        yy,xx=np.mgrid[:image.shape[0],:image.shape[1]]
        image=np.uint8(np.clip(image.astype(float)*(.55+.5*xx/image.shape[1]+.2*yy/image.shape[0])[...,None]+7,0,255))
        fitted,report=PatternDetector(b,geometry).acquire(cv.cvtColor(image,cv.COLOR_BGR2GRAY),prior)
        self.assertIsNotNone(fitted,report)
        probes=b.center+b.ppm*np.array([[-130,10],[110,-30],[20,80],[-40,-120],[0,0]])
        error=np.linalg.norm(project(probes,fitted@np.linalg.inv(truth))-probes,axis=1)/b.ppm
        self.assertLess(float(error.max()),.7)
        self.assertTrue(all(r['found'] for r in report['boundaries'][2:]))

    def test_missing_outer_double_boundary_is_rejected(self):
        from template_registration import measure_ring_boundaries,verify_scoring_boundaries
        b=Board(radii=(97.4,107.4,160,170));texture=board_texture(b,wires=False)
        # Continue each coloured double band beyond its expected outer edge.
        yy,xx=np.mgrid[:b.size,:b.size];delta=np.stack([xx,yy],axis=-1)-b.center
        radius=np.linalg.norm(delta,axis=-1)/b.ppm
        direction=delta/np.maximum(radius[...,None]*b.ppm,1)
        source=b.center+165*b.ppm*direction
        extension=cv.remap(texture,source[...,0].astype(np.float32),source[...,1].astype(np.float32),cv.INTER_LINEAR)
        texture[(radius>168)&(radius<188)]=extension[(radius>168)&(radius<188)]
        reports=measure_ring_boundaries(texture,np.eye(3),b)
        self.assertTrue(reports[4]['found'])
        self.assertFalse(reports[5]['found'])
        self.assertFalse(verify_scoring_boundaries(reports,b))

if __name__=='__main__':unittest.main()

"""Geometry-only detection tests, including loss and reference independence."""
import unittest
import cv2 as cv
import numpy as np
from board_pattern import PatternDetector,GeometryTracker
from dartboard import Board,project
from render_synthetic import board_texture,render_camera


class PatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(1);cv.setRNGSeed(0)
        cls.board=Board();cls.texture=board_texture(cls.board)

    def test_fresh_grayscale_measurements_do_not_need_a_reference(self):
        b=self.board;position=[-285,243,216]
        image,H,_=render_camera(self.texture,b,position,dart=False)
        tracker=GeometryTracker(None,H,b)
        for frame in range(8):
            texture=self.texture.copy()
            # Independent texture/exposure every frame; no saved descriptors.
            rng=np.random.default_rng(100+frame)
            texture=np.uint8(np.clip(texture.astype(float)*(.75+.04*frame)+rng.normal(0,6,texture.shape),0,255))
            image,truth,_=render_camera(texture,b,np.array(position)+[1.5*frame,.2*frame,0],dart=frame>4)
            tracker.track(cv.cvtColor(image,cv.COLOR_BGR2GRAY))
            self.assertTrue(tracker.valid,tracker.reason)
            self.assertFalse(tracker.report['appearance_reference_used'])
            error=np.linalg.norm(project(b.landmarks(),tracker.H@np.linalg.inv(truth))-b.landmarks(),axis=1)
            self.assertLess(float(error.max()),2.)

    def test_missing_board_is_invalid_and_returning_board_is_remeasured(self):
        b=self.board;image,H,_=render_camera(self.texture,b,[285,243,216],dart=False)
        tracker=GeometryTracker(None,H,b);tracker.update(image);self.assertTrue(tracker.valid)
        before=tracker.H.copy()
        tracker.update(np.zeros_like(image));self.assertFalse(tracker.valid)
        np.testing.assert_array_equal(tracker.H,before)
        moved,truth,_=render_camera(self.texture,b,[315,233,221],dart=False)
        tracker.update(cv.cvtColor(moved,cv.COLOR_BGR2GRAY));self.assertTrue(tracker.valid,tracker.reason)
        error=np.linalg.norm(project(b.landmarks(),tracker.H@np.linalg.inv(truth))-b.landmarks(),axis=1)
        self.assertLess(float(error.max()),2.)

    def test_partial_occlusion_leaves_a_distributed_graph(self):
        b=self.board;image,H,_=render_camera(self.texture,b,[0,-375,216],dart=False)
        texture=self.texture.copy();cv.rectangle(texture,(320,170),(450,320),(25,25,25),-1)
        image,truth,_=render_camera(texture,b,[0,-375,216],dart=False)
        fitted,report=PatternDetector(b).detect(cv.cvtColor(image,cv.COLOR_BGR2GRAY),H)
        self.assertIsNotNone(fitted,report)
        self.assertGreaterEqual(report['rings'],3);self.assertEqual(report['quadrants'],4)
        error=np.linalg.norm(project(b.landmarks(),fitted@np.linalg.inv(truth))-b.landmarks(),axis=1)
        self.assertLess(float(error.max()),2.)

    def test_edges_noise_and_cartesian_checkerboard_do_not_form_the_board(self):
        b=self.board;_,H,_=render_camera(self.texture,b,[285,243,216],dart=False)
        yy,xx=np.mgrid[:720,:960]
        rng=np.random.default_rng(9)
        examples=[np.uint8(rng.integers(0,256,(720,960))),
                  np.uint8(((xx//35+yy//35)%2)*180+30),
                  np.uint8((xx>400)*200+20)]
        detector=PatternDetector(b)
        for image in examples:
            fitted,report=detector.detect(image,H)
            self.assertIsNone(fitted,report)

    def test_direct_template_fit_is_independent_of_intrinsics(self):
        b=self.board
        _,prior,_=render_camera(self.texture,b,[-285,243,216],dart=False)
        image,truth,_=render_camera(self.texture,b,[-276,238,220],dart=True)
        H,report=PatternDetector(b).acquire(cv.cvtColor(image,cv.COLOR_BGR2GRAY),prior)
        self.assertIsNotNone(H,report)
        self.assertFalse(report['appearance']['correspondence_refit_after_registration'])
        self.assertEqual(report['corner_role'],'validation only; sparse refit discarded')
        error=np.linalg.norm(project(b.landmarks(),H@np.linalg.inv(truth))-b.landmarks(),axis=1)
        self.assertLess(float(error.max()),2.)
        self.assertTrue(all(r['found'] for r in report['boundaries'][2:]))

    def test_discrete_boundary_check_rejects_wrong_ring_widths(self):
        from boundary_audit import audit_boundaries,boundary_geometry_valid
        b=Board(radii=(97.4,107.4,160,170));image=board_texture(b)
        good,_=audit_boundaries(image,b)
        bad,_=audit_boundaries(image,Board())
        self.assertTrue(boundary_geometry_valid(good))
        self.assertFalse(boundary_geometry_valid(bad))
        radii=np.array([x['radius_mm'] for x in bad])*b.ppm
        for i in range(len(radii)-1):
            self.assertLess(bad[i]['disjoint_search_halfwidth_px']+bad[i+1]['disjoint_search_halfwidth_px'],radii[i+1]-radii[i])


if __name__=='__main__':unittest.main()

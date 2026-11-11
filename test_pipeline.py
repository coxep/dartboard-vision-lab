"""Geometry and failure-handling checks; synthetic benchmarks report accuracy separately."""
import unittest
import numpy as np
import cv2 as cv

from dartboard import (Board, Tracker, automatic_homography, gray_image,
                       change_mask, intersect_lines, project, shaft_line, warp,
                       nominal_homography, binarize, board_color_masks)
from render_synthetic import board_texture, render_camera, camera_matrix


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(1)
        cv.setRNGSeed(0)
        cls.board = Board()
        cls.texture = board_texture(cls.board)

    def test_binary_wedge_interiors_survive_texture_and_lighting(self):
        # Test the requested behavior, including a smooth illumination gradient.
        b = self.board
        gray = cv.cvtColor(self.texture, cv.COLOR_BGR2GRAY)
        lighting = np.linspace(.65, 1.2, b.size)[None, :]
        observed = np.uint8(np.clip(gray*lighting, 0, 255))
        binary = binarize(observed)
        yy, xx = np.mgrid[:b.size, :b.size]
        x,y = (xx-b.center[0])/b.ppm, (yy-b.center[1])/b.ppm
        radius = np.hypot(x,y)
        angle = np.degrees(np.arctan2(x,-y)) % 360
        sector = ((angle+9)//18).astype(int)%20
        margin = np.abs((angle+9)%18-9) < 6
        interior = (radius>25)&(radius<155)&((radius<95)|(radius>111))&margin
        for s in range(20):
            values = binary[interior&(sector==s)]
            correct = values== (0 if s%2==0 else 255)
            self.assertGreater(float(correct.mean()), .995, f"sector {s}")

    def test_hsv_ring_masks_handle_red_wrap_and_achromatic_wedges(self):
        # A patch per hue, including both ends of the red hue range and shadows.
        hsv = np.zeros((40, 240, 3), np.uint8)
        colours = [(0,220,170),(179,220,170),(60,220,170),(90,220,70),
                   (0,0,190),(0,0,40)]
        for i, colour in enumerate(colours):
            hsv[:,i*40:(i+1)*40] = colour
        masks = board_color_masks(cv.cvtColor(hsv,cv.COLOR_HSV2BGR))
        for i in range(6):
            p=(20,20+i*40)
            self.assertEqual(int(masks['red'][p]),255 if i in (0,1) else 0)
            self.assertEqual(int(masks['green'][p]),255 if i in (2,3) else 0)

    def test_real_board_photo_rectifies_without_manual_points(self):
        from pathlib import Path
        from threshold_evidence import wedge_accuracy
        image=cv.imread(str(Path(__file__).parent/'real_photo_validation/photo_1_source.png'))
        self.assertIsNotNone(image)
        H,report=automatic_homography(image,self.board)
        self.assertGreaterEqual(report['inliers'],40)
        self.assertGreater(report['binary_template_support'],.8)
        binary=binarize(cv.cvtColor(warp(image,H,self.board),cv.COLOR_BGR2GRAY))
        self.assertGreater(wedge_accuracy(binary,self.board)['correct_fraction'],.99)

    def test_known_scoring_locations(self):
        b = self.board
        for radius, angle, label, score in [(130, 0, 'S20', 20), (130, 90, 'S6', 6),
                                            (130, 180, 'S3', 3), (130, 270, 'S11', 11),
                                            (103, 0, 'T20', 60), (166, 234, 'D16', 32),
                                            (0, 0, 'DB', 50), (11, 0, 'SB', 25), (178, 54, 'MISS', 0)]:
            with self.subTest(label=label):
                result = b.score(b.point(radius, angle))
                self.assertEqual((result['label'], result['score']), (label, score))
        self.assertTrue(b.score(b.point(162.5, 180))['boundary_review'])

    def test_no_click_calibration_of_three_perspectives(self):
        b = self.board
        for position in [[-340, 240, 270], [340, 240, 270], [0, -390, 270]]:
            image, true_H, _ = render_camera(self.texture, b, position, dart=False)
            H, report = automatic_homography(image, b)
            error = np.linalg.norm(project(b.landmarks(), H @ np.linalg.inv(true_H)) - b.landmarks(), axis=1)
            self.assertTrue(report['refined'])
            self.assertLess(float(np.median(error)), 3.)
            self.assertLess(float(error.max()), 6.)

    def test_nominal_projection_matches_pinhole_render(self):
        b, position = self.board, [-285., 243., 216.]
        image, true_H, _ = render_camera(self.texture, b, position, dart=False)
        K = np.array([[650., 0, 480.], [0, 650., 360.], [0, 0, 1.]])
        seed = nominal_homography(b, K, position)
        error = np.linalg.norm(project(b.landmarks(), seed @ np.linalg.inv(true_H)) - b.landmarks(), axis=1)
        self.assertLess(float(error.max()), .001)

    def test_continuous_tracking_follows_camera_movement(self):
        b, base = self.board, np.array([-340., 240., 270.])
        reference, initial_H, _ = render_camera(self.texture, b, base, dart=False)
        K = np.array([[650., 0, 480.], [0, 650., 360.], [0, 0, 1.]])
        tracker = Tracker(reference, initial_H, b, K)
        tracker.update(reference)
        self.assertTrue(tracker.valid)
        for frame in range(1, 13):
            image, true_H, _ = render_camera(self.texture, b, base + [2*frame, .6*frame, .3*frame], dart=False)
            tracker.track(image)
            self.assertTrue(tracker.valid)
            error = np.linalg.norm(project(b.landmarks(), tracker.H @ np.linalg.inv(true_H)) - b.landmarks(), axis=1)
            self.assertLess(float(error.max()), 3.)

    def test_plane_template_corrects_inaccurate_pose_prior(self):
        b, nominal = self.board, np.array([-340., 240., 270.])
        image, true_H, _ = render_camera(self.texture, b, nominal + [10, 6, 4], dart=False)
        K = np.array([[650., 0, 480.], [0, 650., 360.], [0, 0, 1.]])
        prior = nominal_homography(b, K, nominal)
        before = np.linalg.norm(project(b.landmarks(), prior @ np.linalg.inv(true_H)) - b.landmarks(), axis=1)
        H, info = automatic_homography(image, b, prior)
        after = np.linalg.norm(project(b.landmarks(), H @ np.linalg.inv(true_H)) - b.landmarks(), axis=1)
        self.assertLess(float(np.median(after)), float(np.median(before)))
        self.assertLess(float(np.median(after)), 2.)
        self.assertIn('validation only', info['harris_role'])

    def test_seeded_pnp_rejects_correspondence_outliers(self):
        from pose_alignment import pose_from_homography, fit_seeded_pose, homography_from_pose
        b = self.board
        K = np.array([[650., 0, 480.], [0, 650., 360.], [0, 0, 1.]])
        true_H = nominal_homography(b, K, [-280., 240., 220.])
        seed = nominal_homography(b, K, [-295., 246., 226.])
        xy = (b.landmarks()-b.center)/b.ppm
        objects = np.c_[xy[:, 0], -xy[:, 1], np.zeros(len(xy))].astype(float)
        pixels = project(b.landmarks(), np.linalg.inv(true_H)).astype(float)
        rng = np.random.default_rng(92)
        pixels += rng.normal(0, .15, pixels.shape)
        bad = rng.choice(len(pixels), 20, replace=False)
        pixels[bad] = rng.uniform([100, 100], [850, 600], (len(bad), 2))
        r, t = pose_from_homography(seed, K, b)
        fit = fit_seeded_pose(objects, pixels, K, r, t)
        self.assertIsNotNone(fit)
        _, r, t, inliers, _ = fit
        self.assertLessEqual(int(inliers[bad].sum()), 1)
        self.assertGreaterEqual(int(inliers.sum()), 59)
        H = homography_from_pose(r, t, K, b)
        error = np.linalg.norm(project(b.landmarks(), H@np.linalg.inv(true_H))-b.landmarks(), axis=1)
        self.assertLess(float(error.max()), 1.)

    def test_pose_association_recovers_oblique_bottom_camera(self):
        b = self.board
        K = np.array([[1000., 0, 720.], [0, 1000., 405.], [0, 0, 1.]])
        nominal = np.array([0., -374.65, 215.9])
        image, true_H, _ = render_camera(self.texture, b, nominal+[45., -12., 13.],
                                         dart=False, width=1440, height=810, focal=1000.,
                                         look_at=[-18., 9., 0], roll=1.5)
        seed = nominal_homography(b, K, nominal)
        H, report = automatic_homography(image, b, seed, K)
        error = np.linalg.norm(project(b.landmarks(), H@np.linalg.inv(true_H))-b.landmarks(), axis=1)
        self.assertTrue(report['pose_initializer_used'])
        self.assertLess(float(error.max()), 1.5)

    def test_three_rendered_dart_silhouettes_intersect_at_tip(self):
        b, lines = self.board, []
        for position in [[-340, 240, 270], [340, 240, 270], [0, -390, 270]]:
            empty, H, _ = render_camera(self.texture, b, position, dart=False)
            dart, _, _ = render_camera(self.texture, b, position)
            mask = change_mask(gray_image(warp(dart, H, b)), gray_image(warp(empty, H, b)), b.mask(1.08), 28)
            lines.append(shaft_line(mask, b))
        point, views, _ = intersect_lines(lines)
        self.assertIsNotNone(point)
        self.assertEqual(len(views), 3)
        self.assertLess(float(np.linalg.norm(point - b.point(103, 0)) / b.ppm), 1.)

    def test_reject_missing_and_parallel_lines(self):
        self.assertIsNone(intersect_lines([None, None, None])[0])
        line = dict(normal=np.array([0., 1.]), constant=100.,
                    points=np.float32([[100, 100], [200, 100]]))
        other = dict(normal=np.array([0., 1.]), constant=200.,
                     points=np.float32([[100, 200], [200, 200]]))
        self.assertIsNone(intersect_lines([line, other, None])[0])


if __name__ == '__main__':
    unittest.main()

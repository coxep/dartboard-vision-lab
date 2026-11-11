"""Geometry and failure-handling checks; synthetic benchmarks report accuracy separately."""
import unittest
import numpy as np
import cv2 as cv

from dartboard import (Board, Tracker, automatic_homography, gray_image,
                       change_mask, intersect_lines, project, shaft_line, warp,
                       nominal_homography)
from render_synthetic import board_texture, render_camera, camera_matrix


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(1)
        cv.setRNGSeed(0)
        cls.board = Board()
        cls.texture = board_texture(cls.board)

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
        tracker = Tracker(reference, initial_H, b)
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

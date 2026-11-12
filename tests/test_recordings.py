import csv

import cv2 as cv
import numpy as np

from dartboard_vision.recordings import Recording


def test_pairing_skips_incomplete_frames_and_applies_explicit_rotation(tmp_path):
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cv.imwrite(str(tmp_path / "first.png"), image)
    cv.imwrite(str(tmp_path / "last.png"), image + 20)
    with (tmp_path / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["camera", "file", "t_s", "whole"])
        for camera in ("left", "right"):
            writer.writerows(
                [
                    [camera, "first.png", 0, 1],
                    [camera, "", 0.5, 0],
                    [camera, "last.png", 1, 1],
                ]
            )
    recording = Recording(tmp_path, ["left", "right"], rotation=180)
    frames, selected = recording.read(0.6)
    assert [r["t_s"] for r in selected] == [1.0, 1.0]
    for frame in frames:
        np.testing.assert_array_equal(frame, cv.rotate(image + 20, cv.ROTATE_180))

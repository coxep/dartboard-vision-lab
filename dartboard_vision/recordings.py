"""Timestamp pairing for a directory of recorded camera images and frames.csv."""

import csv
from pathlib import Path

import cv2 as cv
import numpy as np


class Recording:
    """Index metadata once and decode the nearest whole frame from each view."""

    def __init__(self, directory, cameras, rotation=0):
        self.directory = Path(directory)
        rotations = {
            0: None,
            90: cv.ROTATE_90_CLOCKWISE,
            180: cv.ROTATE_180,
            270: cv.ROTATE_90_COUNTERCLOCKWISE,
        }
        if rotation not in rotations:
            raise ValueError("Rotation must be 0, 90, 180 or 270 degrees")
        self.rotation = rotations[rotation]
        self.cameras = tuple(cameras)
        with (self.directory / "frames.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.streams = []
        for camera in self.cameras:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["camera"] == camera
                    and r.get("whole", "1") == "1"
                    and r["file"]
                ),
                key=lambda r: float(r["t_s"]),
            )
            if not selected:
                raise ValueError(f"No whole frames for camera {camera}")
            self.streams.append(selected)
        self.times = [
            np.array([float(r["t_s"]) for r in rows]) for rows in self.streams
        ]
        self._cache = {}

    def read(self, t_s):
        images, inputs = [], []
        for camera, rows, times in zip(self.cameras, self.streams, self.times):
            index = int(np.argmin(abs(times - t_s)))
            row = rows[index]
            key = (camera, index)
            if key not in self._cache:
                image = cv.imread(str(self.directory / row["file"]))
                if image is None:
                    raise FileNotFoundError(self.directory / row["file"])
                if self.rotation is not None:
                    image = cv.rotate(image, self.rotation)
                self._cache = {k: v for k, v in self._cache.items() if k[0] != camera}
                self._cache[key] = image
            images.append(self._cache[key])
            inputs.append(dict(camera=camera, file=row["file"], t_s=float(row["t_s"])))
        return images, inputs

"""Recompute contact samples using explicit calibration and saved tracked maps.

This is a recording replay, not automatic calibration or live capture. Nothing
is fitted to expected dart positions, and numeric scoring stays disabled.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .geometry import Board
from .masks import MaskSettings
from .recordings import Recording
from .session import DartboardSession
from .tracking import PyramidTracker


def read_json(path):
    return json.loads(Path(path).read_text())


def replay(
    dataset,
    calibration,
    tracking,
    mask_settings,
    output,
    rotation=0,
    clear_control=False,
):
    """Replay saved maps; save per-camera masks as well as contact/round state."""
    board = Board(**calibration["board"])
    if board.size != 900:
        raise ValueError("The current saved-map replay uses a 900-pixel board plane")
    models = calibration["models"]
    cameras = calibration["cameras"]
    samples = tracking["frames"]
    settings = [MaskSettings(**item) for item in mask_settings]
    if not samples or len(models) != len(cameras) or len(settings) != len(models):
        raise ValueError("Need saved samples and one model/settings entry per camera")
    if any(len(s["homographies"]) != len(models) for s in samples):
        raise ValueError("Each tracked sample needs one homography per camera")
    if any(b["t_s"] <= a["t_s"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Tracked samples must have strictly increasing timestamps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recording = Recording(dataset, cameras, rotation)
    first, _ = recording.read(samples[0]["t_s"])
    trackers = [PyramidTracker(im, model) for im, model in zip(first, models)]
    empty, empty_valid = [], []
    for tracker, image, homography in zip(trackers, first, samples[0]["homographies"]):
        tracker.H = np.asarray(homography)
        plane, valid = tracker.rectify(image)
        empty.append(plane)
        empty_valid.append(valid)
    session = DartboardSession(empty, empty_valid, settings, board=board)
    records = []
    for index, sample in enumerate(samples):
        raw, inputs = recording.read(sample["t_s"])
        planes, visibility = [], []
        for tracker, image, homography in zip(trackers, raw, sample["homographies"]):
            tracker.H = np.asarray(homography)
            plane, valid = tracker.rectify(image)
            planes.append(plane)
            visibility.append(valid)
        started = time.perf_counter()
        record, masks = session.process(planes, visibility, sample["t_s"])
        record["processing_ms"] = 1000 * (time.perf_counter() - started)
        record["inputs"] = inputs
        records.append(record)
        np.savez_compressed(output / f"masks_{index:02d}.npz", masks=np.asarray(masks))
        strength = session.display_strength
        glow = np.uint8(np.clip((strength - 1.5) / 3.5, 0, 1) * 255)
        if not cv.imwrite(str(output / f"glow_{index:02d}.png"), glow):
            raise OSError("Could not save HSV display response")
    controls = []
    if clear_control:
        for index in range(3):
            control, _ = session.process(
                empty, empty_valid, samples[-1]["t_s"] + index + 1
            )
            controls.append(control)
    report = dict(
        samples=records,
        clear_controls=controls,
        clear_control_is_recorded_removal=False,
        clear_control_description="Replay the initial empty images; not recorded dart removal.",
        empty_reference_fixed=True,
        calibration_preserved_on_clear=True,
        scoring_orientation_verified=False,
        confirmation_frames=2,
        clear_confirmation_frames=3,
        clear_foreground_fraction=0.001,
        history=session.state.history,
        live_capture=False,
        calibration_rerun=False,
        tracking_rerun=False,
        per_camera_masks_saved=True,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--mask-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument(
        "--clear-control",
        action="store_true",
        help="Also test reset using initial empty images, separately from footage",
    )
    args = parser.parse_args()
    cv.setNumThreads(1)
    report = replay(
        args.dataset,
        read_json(args.calibration),
        read_json(args.tracking),
        read_json(args.mask_settings),
        args.output,
        args.rotation,
        args.clear_control,
    )
    print(
        json.dumps(
            dict(
                samples=len(report["samples"]),
                final_hits=len(report["samples"][-1]["round"]["hits"]),
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()

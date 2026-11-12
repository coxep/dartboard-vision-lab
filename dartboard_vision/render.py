"""Render recorded camera averages with mask-selected darts and a contact inset."""

import argparse
import json
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

from .fusion import CachedFusion, build_confidence, overlay_masked_views
from .geometry import Board
from .presentation import compose, mark_hits, template_board
from .recordings import Recording
from .replay import read_json
from .tracking import PyramidTracker


def render(
    dataset,
    calibration,
    tracking,
    session_directory,
    output,
    rotation=0,
    fps=30.0,
    weights=None,
):
    board = Board(**calibration["board"])
    if board.size != 900:
        raise ValueError("The current video layout uses a 900-pixel board canvas")
    output, session_directory = Path(output), Path(session_directory)
    output.mkdir(parents=True, exist_ok=True)
    fits = tracking["frames"]
    samples = read_json(session_directory / "report.json")["samples"]
    if not fits or not samples:
        raise ValueError("Need tracked fits and contact samples")
    fit_times = [r["t_s"] for r in fits]
    hit_times = [r["t_s"] for r in samples]
    recording = Recording(dataset, calibration["cameras"], rotation)
    initial, _ = recording.read(fit_times[0])
    models = calibration["models"]
    trackers = [PyramidTracker(image, model) for image, model in zip(initial, models)]
    if weights:
        with np.load(weights) as data:
            confidence = data["confidence"]
    else:
        confidence = build_confidence(initial, models, board)[0]
    template = template_board(board)
    video = output / "template_hit_inset.mp4"
    writer = cv.VideoWriter(
        str(video), cv.VideoWriter_fourcc(*"avc1"), fps, (board.size, board.size)
    )
    if not writer.isOpened():
        raise RuntimeError("H.264 video encoder unavailable")
    fusion, masks, mask_index = None, None, None
    count = round(fit_times[-1] * fps) + 1
    records = []

    def rectify(index, raw, fit_index):
        tracker = trackers[index]
        tracker.H = np.asarray(fits[fit_index]["homographies"][index])
        return tracker.rectify(raw[index])

    try:
        with ThreadPoolExecutor(max_workers=len(trackers)) as pool:
            for frame in range(count):
                t_s = frame / fps
                fit_index = max(0, bisect_right(fit_times, t_s + 1e-9) - 1)
                hit_index = bisect_right(hit_times, t_s + 1e-9) - 1
                contacts = samples[hit_index]["round"]["hits"] if hit_index >= 0 else []
                if hit_index >= 0 and hit_index != mask_index:
                    with np.load(
                        session_directory / f"masks_{hit_index:02d}.npz"
                    ) as data:
                        masks = data["masks"]
                    mask_index = hit_index
                raw, _ = recording.read(t_s)
                views = list(
                    pool.map(lambda i: rectify(i, raw, fit_index), range(len(trackers)))
                )
                planes, valid = [p for p, _ in views], [v for _, v in views]
                if fusion is None:
                    fusion = CachedFusion(confidence, valid)
                average = fusion.blend(planes, valid)
                composite = (
                    overlay_masked_views(average, planes, masks, valid)
                    if masks is not None
                    else average
                )
                canvas = compose(composite, template, contacts, t_s)
                writer.write(canvas)
                active = (
                    (np.asarray(masks) > 0) & np.asarray(valid)
                    if masks is not None
                    else None
                )
                records.append(
                    dict(
                        frame=frame,
                        t_s=t_s,
                        fit_t_s=fit_times[fit_index],
                        contact_t_s=hit_times[hit_index] if hit_index >= 0 else None,
                        count=len(contacts),
                        mask_pixels=int(np.any(active, axis=0).sum())
                        if active is not None
                        else 0,
                    )
                )
                if frame == count - 1:
                    for name, image in {
                        "poster.jpg": canvas,
                        "average_only.jpg": average,
                        "masked_composite.jpg": composite,
                        "camera_and_template.jpg": np.hstack(
                            [
                                mark_hits(composite, contacts),
                                mark_hits(template, contacts),
                            ]
                        ),
                        "template_hits.png": mark_hits(template, contacts, 28),
                    }.items():
                        if not cv.imwrite(str(output / name), image):
                            raise OSError(f"Could not save {name}")
                    support = active.sum(axis=0)
                    cv.imwrite(str(output / "mask_support.png"), np.uint8(support * 85))
                if frame % 90 == 0:
                    print(f"Rendered {frame}/{count}", flush=True)
    finally:
        writer.release()
    cap = cv.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != count:
        raise RuntimeError(f"Decoded {decoded} of {count} output frames")
    report = dict(
        output_frames=count,
        decoded_frames=decoded,
        video_fps=fps,
        duration_s=count / fps,
        calibration_rerun=False,
        tracking_rerun=False,
        detection_rerun=False,
        video_is_live_capture=False,
        final_contact_count=len(samples[-1]["round"]["hits"]),
        compositing="Original current-camera average outside masks; opaque equal average of only valid mask-contributing current camera pixels inside masks",
        empty_reference_blended=False,
        masks_held_between_detection_samples=True,
        contact_updates_hz=1 / (hit_times[-1] - hit_times[-2])
        if len(hit_times) > 1
        else None,
        frame_records=records,
        limitation="Offline replay with saved map/contact samples. Masks and homographies are held between samples. Shadows and transient motion can remain in masks. Contact candidates are not verified scores.",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "calibration", "tracking", "session", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    cv.setNumThreads(1)
    report = render(
        args.dataset,
        read_json(args.calibration),
        read_json(args.tracking),
        args.session,
        args.output,
        args.rotation,
        args.fps,
        args.weights,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "frame_records"}))


if __name__ == "__main__":
    main()

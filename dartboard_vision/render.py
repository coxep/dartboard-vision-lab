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
from .masks import MaskSettings, dart_mask, evidence_score
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
    mask_settings=None,
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
    settings = [MaskSettings(**s) for s in mask_settings] if mask_settings else None
    if settings is None or len(settings) != len(initial):
        raise ValueError("Supply one frozen mask-settings entry per camera")
    seeded_models = [
        dict(model, inverse_homography_normalized=H)
        for model, H in zip(models, fits[0]["homographies"])
    ]
    trackers = [
        PyramidTracker(image, model) for image, model in zip(initial, seeded_models)
    ]
    empty, initial_valid = zip(
        *(tracker.rectify(image) for tracker, image in zip(trackers, initial))
    )
    y, x = np.mgrid[: board.size, : board.size]
    roi = (
        np.hypot(x - board.center[0], y - board.center[1])
        <= board.radii[-1] * board.ppm
    )
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
    fusion = None
    count = round((fit_times[-1] - fit_times[0]) * fps) + 1
    records = []
    previous_average = None

    def rectify(index, raw):
        tracker = trackers[index]
        alignment = tracker.update(raw[index])
        plane, visible = tracker.rectify(raw[index])
        support = visible & initial_valid[index] & roi
        mask, info = dart_mask(empty[index], plane, support, settings[index])
        strength = evidence_score(info["fields"], settings[index])
        return plane, visible, mask, strength, alignment

    try:
        with ThreadPoolExecutor(max_workers=len(trackers)) as pool:
            for frame in range(count):
                t_s = fit_times[0] + frame / fps
                hit_index = bisect_right(hit_times, t_s + 1e-9) - 1
                contacts = samples[hit_index]["round"]["hits"] if hit_index >= 0 else []
                raw, inputs = recording.read(t_s)
                views = list(pool.map(lambda i: rectify(i, raw), range(len(trackers))))
                planes = [view[0] for view in views]
                valid = [view[1] for view in views]
                masks = [view[2] for view in views]
                if fusion is None:
                    fusion = CachedFusion(confidence, valid)
                average = fusion.blend(planes, valid)
                composite = overlay_masked_views(average, planes, masks, valid)
                canvas = compose(composite, template, contacts, t_s)
                writer.write(canvas)
                active = (np.asarray(masks) > 0) & np.asarray(valid)
                records.append(
                    dict(
                        frame=frame,
                        t_s=t_s,
                        fit_t_s=t_s,
                        mask_t_s=t_s,
                        inputs=inputs,
                        alignment=[view[4] for view in views],
                        homographies=[tracker.H.tolist() for tracker in trackers],
                        background_frame_change=float(
                            np.mean(np.abs(average.astype(float) - previous_average))
                        )
                        if previous_average is not None
                        else 0.0,
                        contact_t_s=hit_times[hit_index] if hit_index >= 0 else None,
                        count=len(contacts),
                        mask_pixels=int(np.any(active, axis=0).sum())
                        if active is not None
                        else 0,
                    )
                )
                previous_average = average.astype(float)
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
                    for camera, view in zip(calibration["cameras"], views):
                        cv.imwrite(str(output / f"{camera}_current.png"), view[0])
                        cv.imwrite(str(output / f"{camera}_mask.png"), view[2])
                        np.savez_compressed(
                            output / f"{camera}_strength.npz", strength=view[3]
                        )
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
        tracking_rerun=True,
        detection_rerun=False,
        mask_extraction_rerun=True,
        contact_detection_rerun=False,
        video_is_live_capture=False,
        final_contact_count=len(samples[-1]["round"]["hits"]),
        compositing="Original current-camera average outside masks; opaque equal average of only valid mask-contributing current camera pixels inside masks",
        empty_reference_blended=False,
        masks_held_between_detection_samples=False,
        mask_updates_hz=fps,
        alignment_updates_hz=fps,
        contact_updates_hz=1 / (hit_times[-1] - hit_times[-2])
        if len(hit_times) > 1
        else None,
        frame_records=records,
        limitation="Offline replay of timestamp-paired recordings, not connected live cameras. Alignment and masks are refreshed on every rendered frame; a failed local alignment retains its previous accepted map. Missing source frames repeat their nearest recorded frame. Contact markers retain the existing approximately 2 Hz results and are not verified scores.",
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
    parser.add_argument("--mask-settings", type=Path, required=True)
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
        read_json(args.mask_settings),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "frame_records"}))


if __name__ == "__main__":
    main()

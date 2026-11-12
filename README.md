# Dartboard Vision Lab

Three-camera board-plane alignment, HSV dart evidence and multiview contact
estimation. The maintained code is in [`dartboard_vision/`](dartboard_vision/).

[Current visual walkthrough](https://dartboard-vision-lab.spunky-cabin-7615.chatgpt.site/)

## Install and check

Python 3.10 or newer:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check dartboard_vision tests
ruff format --check dartboard_vision tests
```

Use `opencv-contrib-python`, which supplies the thinning operation used by the
shaft detector. Install one OpenCV distribution in your environment; do not mix
it with `opencv-python`. A deployment without GUI support can use the matching
contrib headless build instead.

## Replay the supplied recording

```sh
dartboard-replay --dataset /path/to/darts_sample_3cams_3darts \
  --calibration examples/recorded_rig/calibration.json \
  --tracking examples/recorded_rig/tracking.json \
  --mask-settings examples/recorded_rig/mask_settings.json \
  --rotation 180 --output /tmp/dartboard-replay
```

This reuses saved calibration/tracked maps and recomputes HSV masks, contact
candidates and round state. It writes a JSON report, per-camera binary masks and
display evidence. Paths, camera order, rotation and estimated models are explicit
inputs; no recording path is embedded in the package. See the
[example configuration](examples/recorded_rig/README.md).

Render the average, mask-selected foreground and board inset:

```sh
dartboard-render --dataset /path/to/darts_sample_3cams_3darts \
  --calibration examples/recorded_rig/calibration.json \
  --tracking examples/recorded_rig/tracking.json \
  --mask-settings examples/recorded_rig/mask_settings.json \
  --session /tmp/dartboard-replay --rotation 180 --output /tmp/dartboard-video
```

The video updates alignment and HSV masks on every rendered frame. It uses the
same current camera pixels inside each view's mask and averages only contributing
views at overlaps. Outside the masks it keeps the current-camera average. The
tracking file supplies the initial maps and replay time window; its later maps
are not held during rendering. Only contact markers reuse the saved roughly
2 Hz contact results. This is offline video rendering, not a live speed claim.

## Use the components

- `alignment.FastJointRegions`: refine a seeded three-view solution against
  painted board geometry and cross-view appearance.
- `PyramidTracker`: fast homography updates with the estimated lens map frozen.
- `DartboardSession`: take aligned images, detect contacts and maintain the round.
- `fusion.CachedFusion`: blend the current camera images with cached weights.
- `fusion.overlay_masked_views`: use each camera's dart mask to select its image
  pixels; average only contributing views wherever masks overlap.

The empty reference is used for detection, not added to the current camera
average. The default round state confirms new hits over two samples, keeps hits
through misses, and clears them after three valid empty-image checks. Scoring
requires a verified orientation and an explicit scoring callback.

## Scope and limits

This remains a prototype validated on one real recording. Contact coordinates
are estimates, not independently measured accuracy. The fast tracker needs a
close initial alignment; automatic reacquisition after a large camera move is
unfinished. Initial joint refinement is separate from per-frame tracking. The
current joint solver uses three cameras and a 900×900 plane. No fixed focal
length or physical mount pose is required by the maintained alignment code.

[Architecture and data flow](docs/architecture.md) describes model conventions,
module boundaries, references and display semantics. Numerical tests cover the
joint Jacobians, distortion derivatives, HSV/shadow behavior, contact geometry,
round lifecycle, visibility and compositing.

Earlier synthetic scripts remain at the repository root for reproduction; their
commands and claims are documented separately in [LEGACY_README.md](LEGACY_README.md).
Local footage and experiment artifacts stay under `evaluation_runs/` and are
excluded from the installed package.

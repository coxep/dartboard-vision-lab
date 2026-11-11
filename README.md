# Dartboard Vision Lab

An OpenCV prototype for continuous dartboard calibration, three-camera
rectification and dart scoring.

[View the current pipeline](https://dartboard-vision-lab.spunky-cabin-7615.chatgpt.site/)

## Current approach

Each camera is aligned independently to the same metric board plane. The script
remeasures the board pattern every frame and combines the dart observations
across cameras to estimate the hit.

1. **Correct lens distortion.** Apply the camera’s calibrated intrinsic matrix K
   and distortion coefficients D before estimating the board alignment.
2. **Seed the plane projection.** Use the approximate mount pose or the previous
   accepted pose to establish scale, orientation and a search region.
3. **Detect fresh board junctions.** Match multiscale grayscale ring/spoke
   templates, refine their locations to subpixel precision, and check the cyclic
   and radial neighbour structure against known board geometry.
4. **Fit the geometry.** RANSAC removes inconsistent correspondences. With K,
   seeded iterative solvePnP and LM fit the measured pixels to millimetre board
   coordinates at Z=0; the homography is derived from that physical pose. Without
   K, use an eight-parameter homography.
5. **Rectify observed pixels.** Warp each actual image into the common plane.
   Fresh detection runs continuously; larger movement triggers broader recovery.
   Scoring pauses when alignment is lost and the temporal background is refreshed
   when the alignment changes.
6. **Locate and score the dart.** Threshold settled-frame changes, retain shaft
   candidates, and jointly select axes across the cameras. Their supported
   intersection gives the hit, which is mapped to a sector and scoring band.
   Close wire calls and ambiguous intersections require review.

The continuous detector accepts grayscale input and does not need a saved board
photograph, ORB descriptors or optical flow. Colour masks can supplement boundary
checks. Rectified images always contain warped input pixels. There is no joint
camera bundle adjustment; the shaft-intersection stage combines the views.

## Install

Python 3.10 or newer. Use a desktop OpenCV build for the live display.

```sh
python -m pip install -r requirements.txt
```

## Camera setup and board geometry

Input order is **upper left, upper right, bottom**. The board is upright with 20
at the top, and camera images are upright.

The nominal setup in `measured_setup/scene.json` places each lens approximately
14.75 inches (374.65 mm) from the bull across the board plane and 8.5 inches
(215.9 mm) in front of the scoring surface. Camera azimuths are approximate.
Update those positions to match the mounts. Coordinates are x right, y up and
z toward the player; the scoring plane is Z=0.

The initial mount/orientation prior resolves the repeating sector pattern.
Recovery searches around that belief; this is not unconstrained discovery of an
arbitrarily oriented board in a room.

Match the model to the physical boundaries used for calibration:

- `--board-profile strip10`: nominal strip/blade profile, radii
  97.4 / 107.4 / 160 / 170 mm.
- `--radii TREBLE_IN TREBLE_OUT DOUBLE_IN DOUBLE_OUT`: supply measured radii
  instead. These override the named profile.

Paint edges, wire apexes and wire outside edges are distinct physical references.
An 8 mm inside-to-inside measurement is not the same as 10 mm apex-to-apex strip
spacing. The script retains `legacy8` as its default for the existing synthetic
fixtures; choose the board profile explicitly for your setup.

## Lens calibration

Supply one OpenCV pinhole-calibration NPZ per camera. Each file contains:

- `K`: 3×3 intrinsic matrix.
- `D`: OpenCV distortion coefficients.
- `image_size`: `[width, height]` used for that calibration.

Keep source order, resolution, crop and focus/zoom consistent with calibration.
Lens parameters stay fixed when only the camera mount moves. The supplied
synthetic coefficients are illustrative, not measurements of your C920 cameras.
The OpenCV fisheye model is not supported by this loader.

## Run

Calibrate the board, inspect continuous rectification alone, then run scoring:

```sh
python dartboard.py calibrate --sources 0 1 2 --board-profile strip10 --geometry-prior measured_setup/scene.json --intrinsics cam0.npz cam1.npz cam2.npz
python dartboard.py track-board --sources 0 1 2 --log board_tracking.jsonl
python dartboard.py run --sources 0 1 2
```

Sources can be USB indices, network stream URLs or video paths. Local cameras
must be attached to the computer running the script; network streams can be
remote. No hardware is accessed until you run a camera/stream command.

For saved images, use `--images left.jpg right.jpg bottom.jpg`; add `--once` to
`track-board` or `run`. `--headless` hides the dashboard. Ctrl+C stops execution.
`--manual` enables an optional four-click calibration fallback.

Start with an empty board. Throw **one dart at a time**, then wait for the result.
Existing darts remain part of the reference state; simultaneous additions are
ambiguous and may be rejected. Pause before retrieving darts.

| Key | Action |
| --- | --- |
| P | Pause for dart retrieval |
| R | Refresh the background and resume |
| G | Toggle the scoring grid |
| S | Save a snapshot |
| Q | Quit |

The scoring command writes estimates and review/rejection reasons to JSONL, with
frames and masks under `dartboard_output/`. The calibration-only command reports
fresh detections, loss and recovery without scoring.

## Replay recordings

Use the saved calibration with three recordings and explicit reference/settled
frame indices for each throw. Per-camera indices account for synchronization
offsets. See `examples/replay_events.json`.

```sh
python replay_recordings.py --sources left.mp4 right.mp4 bottom.mp4 --calibration calibration.json --events examples/replay_events.json --output recording_validation
```

Optional truth labels or hit coordinates are evaluated after detection and are
not supplied to the estimator. This tool replays event windows; it does not
establish automatic live throw timing.

## Validation status

The current implementation has been exercised on synthetic camera inputs with
known lens parameters and an original-photo boundary check. Synchronized
recordings from the actual C920 setup are still needed to assess real-camera
accuracy and live scoring reliability. Synthetic inputs use a textured board
plane and a simplified projected 3D dart with lens distortion, blur, noise and
JPEG compression; they do not reproduce real hands, lighting or depth occlusion.

Run the regression checks with:

```sh
python -m unittest -q test_pipeline test_detection test_recording_replay test_board_pattern
```

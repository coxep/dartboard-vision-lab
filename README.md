# Dartboard Vision Lab

Experimental OpenCV script for three-camera board rectification, continuous tracking,
shaft-axis intersection and per-dart score estimates. The published tests include
failures and wrong estimates. This is not yet reliable unattended scoring.

Public pipeline and raw frames: https://dartboard-vision-lab.spunky-cabin-7615.chatgpt.site/

## Install

Python 3.10 or newer, with a desktop OpenCV build for the live dashboard:

```sh
python -m pip install -r requirements.txt
```

## Use cameras or streams

Camera order is upper left, upper right, bottom. Sources may be USB indices, stream
URLs or video paths. No hardware is accessed until you run the commands.

```sh
python dartboard.py calibrate --sources 0 1 2 --geometry-prior measured_setup/scene.json
python dartboard.py run --sources 0 1 2
```

Use the same source order, resolution, lens settings and crop for calibration and
operation. For URLs, replace `0 1 2` with your three stream URLs. Local USB cameras
must be attached to the computer running the script; network streams can be remote.
Change the nominal positions in `measured_setup/scene.json` to match your mounts.

Your initial measurements are encoded in that scene: 14.75 inches (374.65 mm) from
bull to lens across the board plane; approximately 8.5 inches (215.9 mm) lens
stand-off. Board coordinates are x right, y up, z toward the player; the board
surface is z=0. Camera azimuths are approximate, based on the setup photograph.
Camera images are assumed upright and the board has 20 at the top.

For real lens correction, pass one OpenCV pinhole calibration NPZ per camera:

```sh
python dartboard.py calibrate --sources 0 1 2 --geometry-prior measured_setup/scene.json --intrinsics cam0.npz cam1.npz cam2.npz
```

Each NPZ contains `K` (3×3 camera matrix), `D` (OpenCV distortion coefficients) and
`image_size` (`[width, height]`). Synthetic coefficients in this repository are
illustrative; they are not measurements of your individual Logitech C920 cameras.
Fisheye-model coefficients are not supported. Lock focus/zoom after calibration.

Automatic calibration requires no clicked points. With a geometry prior and camera
matrix K, it uses this sequence:

1. Warp observed pixels using the nominal camera pose.
2. Detect Shi–Tomasi corners in that preliminary plane image. Try the current pose
   and six bounded mount offsets (±35 mm along each axis).
3. Alternate one-to-one feature assignment (Hungarian and greedy alternatives)
   with seeded iterative `solvePnP`, shrinking the association gate over five
   iterations. Local quadrant appearance helps distinguish ring/spoke crossings.
4. Use planar homography RANSAC to find consistent correspondence subsets, then
   seeded PnP and LM to refine their physical poses. This deliberately does not
   rely on `solvePnPRansac`'s planar minimal-set behavior. Keep only distributed
   inliers, positive depth, and poses within 85 mm of the initial mount belief.
5. Rank pose hypotheses by observed whole-board image correlation. Refine the best
   candidates using multiscale ECC and metric wire-edge alignment. The final dense
   refinement has eight homography parameters; it is not a six-parameter PnP pose.
6. Validate with adaptive-binary Harris features and the full wire template.
   Check final orientation against the original belief. Retain the previous
   dense-only initializer as an explicit fallback if pose hypotheses fail.

The geometry model is separate from the textured renderer. All rectified outputs
contain warped input pixels, never a replacement rendering. K and lens distortion
are fixed during these fits; this is not blind intrinsic calibration. Without a
mount prior, the script falls back to color-assisted ellipse/bull initialization.
`--manual` enables an optional four-click fallback. Nominal radii are configurable
with `--radii 99 107 162 170`.

The live script tracks board texture each frame with forward/backward optical flow
and periodically re-registers to an immutable ORB appearance reference. If K was
saved during calibration, tracked/reference correspondences refine the previous
physical camera pose using seeded PnP. Older calibration files retain homography
tracking. It refreshes the temporal baseline when a camera moves and pauses scoring
if registration fails. These are independent per-camera plane fits; hit estimation
combines usable shaft axes. It is not a joint camera bundle adjustment.

Start with an empty board. Throw one dart, then wait for the estimate. Keys:
`P` pauses for dart retrieval, `R` refreshes the background and resumes, `G` toggles
the grid, `S` saves a snapshot, `Q` quits. Pause **before** removing darts: an
absolute difference cannot reliably distinguish adding from removing a dart.
`--headless` hides the dashboard. Ctrl+C stops. Use `--once` to save one rectified
set without scoring. Estimates and review/rejection reasons go to JSONL, with
diagnostic frames and masks under `dartboard_output/`.

## Reproduce the synthetic experiments

```sh
python render_synthetic.py
python benchmark_c920.py
python monte_carlo.py --trials 100 --workers 4 --seed 20261110
python -m unittest -v test_pipeline.py

# Calibration-only ablation and recovery evidence
OPENBLAS_NUM_THREADS=1 python monte_carlo.py --trials 100 --workers 4 --calibration-only --no-pose --output c920_benchmark/calibration_sweep_before_pnp.json
OPENBLAS_NUM_THREADS=1 python monte_carlo.py --trials 100 --workers 4 --calibration-only --output c920_benchmark/calibration_sweep.json
OPENBLAS_NUM_THREADS=1 python monte_carlo.py --trials 20 --workers 4 --seed 20261111 --calibration-only --output c920_benchmark/calibration_holdout.json
python calibration_evidence.py
```

The renderer projects a 900×900 textured board plane and simple 3D dart into
1920×1080 C920-like frames. It uses the published 78-degree diagonal field of view,
illustrative Brown–Conrady distortion, blur, Gaussian noise and JPEG compression.
There is no 4K intermediate. The detector rectifies actual input pixels with
`remap`/`warpPerspective`; it never substitutes a newly rendered clean board.

Synthetic K/D and the nominal mounting geometry are supplied to the detector.
Monte Carlo true camera positions are sampled within a 50.8 mm ball around each
nominal mount, with additional aim/roll and optical variation. Sampled true poses,
homographies and dart hit locations are withheld. The model compares recovered
landmarks and hit locations with ground truth only during evaluation.

The ranges in `monte_carlo.py` are engineering stress assumptions, not measured C920
population distributions. Test reports retain every trial, including calibration
failures, rejected intersections and wrong score estimates. Position-error
statistics include wrong-score estimates. Review flags are tolerances and checks,
not calibrated statistical confidence intervals.

`initial_geometry_report.json` and `initial_geometry_monte_carlo.json` retain the
earlier geometry-based experiment. It produced 5 correct estimates and 3 rejections
on 8 fixed targets; the original 100-trial run produced 18 correct estimates,
3 wrong estimates, 43 intersection rejections and 36 calibration failures. Later
reports use the corrected mount dimensions and/or optimizer and must be read as
separate experiments. The website reports which experiment each image belongs to.

## Seeded pose experiment

With the same 100 randomized setups (300 views, seed 20261110), the dense-only
baseline calibrated 284/300 views and all three cameras in 84/100 setups. The
seeded pose/reassociation initializer calibrated **297/300 views**, with all three
cameras in **97/100 setups**. Of the 297 accepted views, 296 used a pose hypothesis;
one used the explicit dense-only fallback. Median per-view landmark error was
0.605 canonical pixels; the worst individual landmark error was 0.996 pixels and
the worst interior-probe error was 0.901 pixels. The remaining failures were bottom
camera views in trials 72, 78 and 92. No accepted view exceeded the 2-pixel
landmark/interior-probe evaluation threshold.

A separate seed (20261111), using the same scene generator and stress ranges,
calibrated all 60 views in 20 setups. Its worst landmark error was 0.989 pixels.
This is synthetic evidence with exact K/D, not a real-camera validation or a test
of different board textures and lighting.

Reports: `c920_benchmark/calibration_sweep_before_pnp.json`,
`c920_benchmark/calibration_sweep.json`, and `c920_benchmark/calibration_holdout.json`.
`calibration_evidence/pipeline.jpg` shows actual pixels at every stage of trial 0,
including the previously failing bottom view. The nine unit/integration checks
include 25% corrupted correspondences and a 12-frame moving-camera sequence.

## Important practical limits

The latest measured-setup fixed-target run produced 4 correct estimates and
4 rejections; estimated-hit errors were 0.296–0.364 mm. The dense template mapping
had roughly 0.54–0.65 canonical-pixel median landmark error in those fixed views.
The historical scoring run, before the seeded pose loop, with each camera sampled within a 50.8 mm ball of its
nominal position, produced 14 correct estimates, 2 wrong estimates, 23 rejected
intersections and 61 calibration failures. There were 16 estimates (16% coverage);
median position error was 0.267 mm, 95th percentile 24.128 mm and worst 36.555 mm.
The apparently small median must not obscure the wrong estimates and low coverage.
See `c920_benchmark/report.json` and `c920_benchmark/monte_carlo.json` for every case.

Hands, dart overlap, realistic shadows, changing illumination and automatic dart
removal are not solved. Shaft segmentation and calibration failures remain material.
Two-view estimates and near-wire calls need review. A camera movement concurrent
with a throw can cause that event to be skipped during baseline refresh.
The static benchmarks do not exercise the full live event state machine.
The included tracking test separately checks a sequence of changing camera poses.

USB `grab`/`retrieve` provides approximate synchronization, not hardware sync;
network buffering and blocking capture depend on the OpenCV backend. The script
waits for a settled scene before evaluating a dart. It records per-dart estimates,
not a full 301/501 game.

Logitech source: https://support.logi.com/hc/en-ca/articles/360023307774-C920-C-Webcam-Technical-Specifications
OpenCV: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html
Camera models: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

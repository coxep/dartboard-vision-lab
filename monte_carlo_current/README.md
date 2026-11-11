# Current two-inch Monte Carlo

This evaluates the existing independent grayscale calibration, not a joint
multi-camera optimizer. Camera poses/H are withheld from estimation.

## Results

| Condition | Accepted views | Complete triplets | Median maximum error | P95 maximum error | Worst maximum error |
| --- | --- | --- | --- | --- | --- |
| Pose only | 297/300 | 97/100 | 0.165 mm | 0.297 mm | 0.336 mm |
| Mixed lens/lighting/wires | 295/300 | 95/100 | 0.200 mm | 0.431 mm | 0.563 mm |

Each error is the maximum over visible independent board-plane probes in a
view; percentiles summarize accepted views only. No accepted view exceeded
1 mm. Initial-belief median maximum errors were about 39 mm. This is a current
baseline, not a paired comparison against an earlier architecture.

Pose-only scoring: 96 correct, 1 dart rejection, 3 calibration rejections, no
wrong scores. Mixed stress: 87 correct, 8 dart rejections, 5 calibration
rejections, no wrong scores. Seven correct pose-only estimates and eight correct
mixed-stress estimates were flagged for review. Real-video performance remains
unknown. These eight target types do not measure accuracy uniformly near wires.

Median acquisition times were 3.79/4.06 seconds per camera for pose-only/mixed
stress, with p95 11.20/11.94 seconds under six concurrent single-thread workers.
These are cold-acquisition times excluding rendering and lens correction, not
tracking speed or a real-time FPS claim. Caching and shared-appearance joint
refinement are proposed optimizations, not included in this benchmark.

## All rejection evidence

`python inspect_mc_rejections.py` reconstructs the rejected scenes and wraps the
unchanged solver calls to log intermediate ECC/gate values. It asserts that
replayed calibration and dart outcomes match. `rejections.json` indexes every
case; original-resolution inputs, masks, truth-centred diagnostic crops and full
optimizer traces accompany the images.

Eight camera fits failed across the two conditions (the same pose can fail in
both): six ECC non-convergence errors, one correlation/update guard, and one
visibility guard. Pose-only trial 12 reached correlation 0.493 (<0.700) with a
110.34 px local update (>12 px); its coarse and junction fallbacks also failed.
Mixed-stress trial 0 camera 2 had correlation about 0.9705 and 67 junction
inliers, but only 68.75% of outer-perimeter samples were inside the image,
below the 70% guard. The latter is an overly conservative coverage decision for
an otherwise strongly supported candidate, not evidence of a bad alignment.
The frozen counts are not retroactively changed by this diagnosis.

Dart rejection messages: two fewer-than-two-axis cases, four incomplete
cross-view-consensus cases, three unsupported-intersection cases. Seven of the
eight mixed-stress dart rejections persist with exact H. Masks frequently lose
near-tip shaft support, so a usable fitted line alone does not guarantee an
accepted intersection. Ground-truth support distances are evaluation only.

## Reproduce

The original algorithm commit is recorded in each report (`3cdcbb7`). The later
optional shadow filter affects only the run command when explicitly enabled;
it is not used by `monte_carlo_current.py` or its published results.

```
MPLCONFIGDIR=/tmp/dartboard-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python monte_carlo_current.py --trials 100 --seed 2026111107 --workers 3 --pose-only --output monte_carlo_current/pose_only
MPLCONFIGDIR=/tmp/dartboard-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python monte_carlo_current.py --trials 100 --seed 2026111107 --workers 3 --output monte_carlo_current/mixed_stress
```

Both conditions use the same 100 random three-camera poses. Each translation is
independent and uniform in a three-dimensional ball of radius 50.8 mm about its
nominal centre, not a Gaussian or a +/-50.8 mm cube. Aim varies with sigma 8 mm on
the board plane; roll varies with sigma 1.5 degrees, clipped to +/-4 degrees.
The camera model starts from the nominal centre, bull aim, zero roll and 78-degree
diagonal FOV. Nominal K supports the same broad acquisition fallback as the CLI.
The final alignment remains a free eight-parameter appearance homography.

The pose-only condition holds lens, photometry and wire dimensions nominal.
Mixed stress additionally randomizes FOV, principal point, Brown distortion,
exposure, gamma, RGB balance, and wire dimensions. Both conditions use mild
noise, blur, vignetting and JPEG compression. Exact synthetic K/D are supplied
to undistortion. This is not a test of unknown-lens estimation or measured C920
lens parameters. One textured board and a simplified 3D wire/dart renderer are
used; cast shadows, real hands and full depth-tested occlusion are absent.

Calibration error is measured in millimetres using independent grid/perimeter
probes. Every accepted view contributes its maximum error over visible probes.
Rejected views and entire rejected triplets are retained. The report also stores
full-plane extrapolation error. Error percentiles condition on acceptance.

One dart is rendered in a settled before/after pair for every triplet, cycling
through eight target types. The normal dart estimator receives masks generated
with the estimated H, and scoring pauses if any calibration is rejected. A
separate exact-H diagnostic helps isolate dart-detector failures. True H is used
only for rendering/evaluation and that explicitly labelled diagnostic. Static
pairs do not validate live event timing.

`report.json` records seed, all poses, assumptions, source commit and algorithm
hashes. `trials.jsonl` retains completed trials during execution. `summary.png`
plots all outcomes. Representative cases are replayed by seed: one rejected
triplet, a median complete triplet, and the triplet containing the worst accepted
view, including an incomplete triplet if applicable. The CV implementation is
unchanged during the experiment; the preceding three-trial smoke test used a
different seed and is not included in the published Monte Carlo statistics.

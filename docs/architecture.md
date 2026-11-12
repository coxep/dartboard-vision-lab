# Current pipeline

The maintained implementation is `dartboard_vision/`. It does not import the
historical scripts, inspect a particular recording, open cameras or write files
at import time.

| Module | Responsibility |
| --- | --- |
| `geometry` | Metric board dimensions, scoring zones and painted-region template |
| `photometry` | Local contrast normalization and four color-class probabilities |
| `alignment` | Joint template and cross-camera appearance refinement, analytic Jacobians |
| `tracking` | Small pyramidal ECC updates with a fixed per-camera distortion map |
| `masks` | Empty/current HSV differences, illumination suppression and binary evidence |
| `contacts` | Shaft proposals, robust line fits and multiview tip intersections |
| `session` | Per-sample detection, hit association, confirmation and round reset |
| `fusion` | Cached camera averaging and mask-selected foreground compositing |
| `recordings` | Timestamp pairing and explicit input rotation |
| `replay` | Command-line replay using supplied calibration and tracked maps |
| `presentation` | Board template, X markers and video inset styling |
| `render` | Mask-selected camera compositing and offline video encoding |

## Calibration and tracking

Joint alignment uses known painted-region geometry as unary constraints and
shared image appearance as pair constraints. Robust residual weights reduce
outlier influence. Each view has a normalized inverse homography and a nonlinear
image warp: division lambda, optional radial fourth-power residual, and optional
tangential p1/p2. These are residual image-warp coefficients, not a recovered
physical camera model. The distortion center and normalization radius are
explicit parts of each model.

Initial joint refinement and per-frame tracking are separate operations. The
fast tracker keeps lens terms fixed and updates homography against a per-camera
initial empty image in the common board plane. RGB output is sampled directly
from the raw camera using the combined map. A failed local update is held and
requests reacquisition; automatic general reacquisition is unfinished.

The current optimizer and saved-map format use a 900×900 board canvas. Board
dimensions and detector settings are explicit configuration; the example models
are estimates for the supplied recording, not defaults for another camera rig.
The joint solver currently expects exactly three views. A homography seed is
required; this package does not claim unconstrained board discovery.

## Detection and round state

`DartboardSession` takes rectified current images, visibility masks and immutable
empty references. HSV evidence is computed per camera. Shaft-center proposals
must have strong image support and plausible terminal evidence; intersections
need support from at least two cameras. Hungarian association maintains hit
identities. The default confirms a new contact over two samples and retains
existing contacts through misses. Three valid clear-image checks reset the
round; a missed contact alone does not. Calibration and the empty references
survive reset. Numeric scores require an explicitly supplied, verified scoring
callback.

## Display

The background is the visibility-normalized, optionally weighted average of the
current views. For mask-selected display, at each board pixel:

```
n = sum(mask[c] & visible[c])
foreground = sum(image[c] * (mask[c] & visible[c])) / n
output = foreground if n > 0 else background
```

Mask contributions have equal weights; the original empty image is never added
to this foreground. The masks select actual camera pixels, so darts are no
longer diluted by views that show only the board at that position. This display
operation does not change the contact estimator. Shadow false positives and
mask timing errors remain possible.

The video renderer tracks alignment and recomputes each camera's mask on every
output frame using the same timestamp-paired RGB images for background and
foreground. Empty references stay fixed for differencing. A failed alignment
keeps its last accepted map; missing source frames naturally repeat. Saved
contact markers still update at their original cadence, separately from masks.

## Historical material

Root-level scripts and `LEGACY_README.md` are the earlier synthetic prototype.
`evaluation_runs/` contains local raw data, snapshots and experiment drivers.
Compatibility imports there point at the maintained package rather than keeping
second copies of its algorithms. Neither experiment artifacts nor raw recordings
are dependencies of the installed package.

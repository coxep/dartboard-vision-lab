# Fresh pattern-detection evidence

`sequence.json` and `camera_*` contain the final known-K/D grayscale moving-camera
experiment. Each pipeline image shows actual observed pixels and measured graph
overlays. The input JPEG is black during the deliberately absent-board frames.

`holdout/acquisition.json` contains all 300 final cold-acquisition views, seed
20261115, including the three rejected views. True pose/H are evaluator-only.
`same_points_free_H` evaluates the unconstrained fit from the same candidate
correspondences before final PnP. Robust inlier selection may differ between fits.

`homography_baseline/` preserves earlier H-only sequence/acquisition reports and
the smaller development-seed acquisition run. They are historical comparisons,
not extra validation of the final pose-derived implementation.

`scoring_fixed/` reruns the eight fixed targets with the new calibration and the
actual joint shaft-axis detector. Exact-H modes are oracle diagnostics only.

`real/` processes the original uploaded photo with an explicitly supplied prior H.
Neither nominal radius profile alone passes the full-boundary audit. A separate
labelled H/k1 fit improves six fixed model boundaries. Its K is assumed from a
centred principal point and 78-degree diagonal FOV; no real phone/C920 intrinsics
were measured. Internal edge consistency does not establish physical accuracy.
Only rectified board crops and derived overlays are included.

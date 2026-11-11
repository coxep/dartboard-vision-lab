# Raised-wire and exact-homography images

Run `python validate_raised_wires.py` from the repository root.

The renderer projects a textured board plane and an actual triangular wire mesh
with a simplified 3D dart. It uses pinhole projection at 1920×1080, focal length
1300 px, and the nominal mounts in measured_setup/scene.json. These are assumed
optics, not measured Logitech C920 parameters. Wire height/base width are assumed
1.0/0.8 mm. Triangles are depth sorted; cast shadows and full depth rendering are
not modelled.

Each camera has nominal, moved, lighting and occlusion cases. The detector gets
only pixels, the nominal mount belief and assumed wire dimensions. Actual moved
camera positions and exact homographies are used by the renderer/evaluator only.
Nominal fixtures intentionally have an exact nominal seed; displaced fixtures
check refinement from an inexact seed.

- `*_input.jpg`: rendered image.
- `*_binary.jpg`: large-window adaptive threshold of the initial plane warp.
- `*_rectified.jpg`: actual pixels warped by the estimated homography.
- `*_boundaries.jpg`: independent measured junction/edge-profile checks.
- `*_truth.jpg`: actual pixels warped by the exact board-plane homography.
- `*_truth_binary.jpg`: thresholding after the exact transform.
- `three_views.jpg`: nominal input / exact H / estimated H / validation panels.
- `spider_mm.obj`: 3D spider in millimetres, surface Z=0.
- `validation.json`: evaluation on independent interior plane probes, not junctions.

Exact-H panels preserve dart occlusion, cropping, resampling and elevated-wire
parallax. Their purpose is to distinguish image/model limitations from estimated
alignment errors. Ground truth is never passed to detection for the moved cases.

`scoring/` is a separate eight-target regression using the existing flat-spider,
known-lens-distortion fixture and the current direct-template calibration. It is
not raised-wire scoring evidence. Its images supply the website scoring example.
Real synchronized camera footage remains unvalidated.

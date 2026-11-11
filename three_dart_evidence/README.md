# Three darts / three-camera average

Run `python render_three_darts.py` from the repository root.
Each successive image contains the previous darts plus one new dart. Each camera
is aligned from its nominal mount belief and then continuously remeasured using
the current grayscale detector. True camera homographies are not supplied.

`average_after_N.jpg` averages the three actual estimated-H rectifications after
N darts. Equal weights apply wherever a camera supplies valid pixels. Warp
padding is excluded; `coverage_after_N.png` encodes 0/1/2/3 views as 0/85/170/255.
`comparison_after_N.jpg` places all three rectifications beside their average.
The raw per-camera inputs and rectifications are also saved.

The board surface and dart tips align. Raised shafts/flights retain view-dependent
projections, producing faint multiple images in the average. Averaging alone is
not the hit estimator. The scene uses assumed wire dimensions and a simplified
3D dart renderer; it does not establish real-camera performance.

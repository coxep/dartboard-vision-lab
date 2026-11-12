# Supplied recording example

These files reproduce the latest saved calibration and tracked maps for
`darts_sample_3cams_3darts`. They contain estimated results, not measured ground
truth or generic camera defaults. The footage itself is not included.

- `calibration.json`: board dimensions, camera order and three image-warp models.
- `tracking.json`: timestamped inverse homographies from the existing tracker.
- `mask_settings.json`: camera thresholds derived from the early empty pair.

Run from the repository root after installation:

```sh
dartboard-replay --dataset /path/to/darts_sample_3cams_3darts \
  --calibration examples/recorded_rig/calibration.json \
  --tracking examples/recorded_rig/tracking.json \
  --mask-settings examples/recorded_rig/mask_settings.json \
  --rotation 180 --output /tmp/dartboard-replay
```

The rotation is explicit because these particular frames were recorded upside
down. Replay pairs each timestamp to the nearest whole frame in each camera;
it does not imply synchronized shutters. Add `--clear-control` to exercise reset
with the initial empty images separately from the recording. This is not a
recorded dart-removal event. Replay uses saved maps and recomputes HSV/contact
evidence; it does not rerun initial calibration or temporal tracking.

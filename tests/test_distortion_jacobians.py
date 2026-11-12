def test_distortion_jacobians():
    import numpy as np

    from dartboard_vision.alignment import offset_mapping, source_mapping

    rng = np.random.default_rng(71)
    q = np.c_[rng.uniform(-0.8, 0.8, (61, 2)), np.ones(61)]
    model = dict(
        division_lambda=0.04,
        residual_radial4=-0.003,
        normalization_radius_px=700.0,
        distortion_centre_px=[640.0, 360.0],
    )
    p = np.array([0.5, 0.01, 0.08, -0.015, 0.3, -0.04, 0.01, -0.005, 0.001, 0.002])
    base, dx, dy = source_mapping(q, p, model)
    for k in range(10):
        eps = 1e-7
        probe = p.copy()
        probe[k] += eps
        numeric = (source_mapping(q, probe, model, derivatives=False) - base) / eps
        np.testing.assert_allclose(
            numeric, np.stack([dx[:, k], dy[:, k]], axis=-1), rtol=2e-4, atol=1e-4
        )
    H = np.r_[p[:8], 1.0].reshape(3, 3)
    offset = np.array([0.2, -0.3, 0.001])
    xy, dx, dy = offset_mapping(q, offset, H, model)
    for k in range(3):
        eps = 1e-5 if k < 2 else 1e-7
        probe = offset.copy()
        probe[k] += eps
        numeric = (offset_mapping(q, probe, H, model, derivatives=False) - xy) / eps
        np.testing.assert_allclose(
            numeric, np.stack([dx[:, k], dy[:, k]], axis=-1), rtol=2e-4, atol=1e-4
        )
    print(
        "PASS: all 10 H/radial derivatives and subsequent frozen-radial offset tracking derivatives."
    )
    model["tangential"] = [0.002, -0.001]
    p = np.r_[p, 0.0003, -0.0004]
    base, dx, dy = source_mapping(q, p, model)
    for k in range(12):
        eps = 1e-7
        probe = p.copy()
        probe[k] += eps
        numeric = (source_mapping(q, probe, model, derivatives=False) - base) / eps
        np.testing.assert_allclose(
            numeric, np.stack([dx[:, k], dy[:, k]], axis=-1), rtol=2e-4, atol=1e-4
        )
    xy, dx, dy = offset_mapping(q, offset, H, model)
    for k in range(3):
        eps = 1e-5 if k < 2 else 1e-7
        probe = offset.copy()
        probe[k] += eps
        numeric = (offset_mapping(q, probe, H, model, derivatives=False) - xy) / eps
        np.testing.assert_allclose(
            numeric, np.stack([dx[:, k], dy[:, k]], axis=-1), rtol=2e-4, atol=1e-4
        )
    print(
        "PASS: all 12 map derivatives including p1/p2, plus frozen-tangential tracking derivatives."
    )

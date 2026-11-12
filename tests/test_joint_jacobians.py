def test_joint_jacobians():
    import numpy as np

    from dartboard_vision.alignment import blocks, warp_samples

    rng = np.random.default_rng(8)
    field = rng.normal(size=(300, 300, 5))
    q = np.c_[rng.uniform(-0.5, 0.5, (53, 2)), np.ones(53)]
    p = np.array([0.5, 0.002, 0.003, -0.001, 0.498, 0.004, 0.001, -0.002])
    model = dict(
        division_lambda=0.06,
        normalization_radius_px=450.0,
        distortion_centre_px=[150.0, 150.0],
    )
    values, D = warp_samples(field, q, p, model, scale=1.0)
    for k in range(8):
        eps = 1e-7
        probe = p.copy()
        probe[k] += eps
        numeric = (
            warp_samples(field, q, probe, model, scale=1.0, derivatives=False) - values
        ) / eps
        np.testing.assert_allclose(D[:, :, k], numeric, rtol=4e-3, atol=2e-2)
    n = 13
    v = rng.normal(size=(3, n, 5))
    D = rng.normal(size=(3, n, 5, 8))
    target = rng.normal(size=(n, 4))
    uw = rng.uniform(0.1, 1, (3, n))
    pw = [rng.uniform(0.1, 1, n) for _ in range(3)]
    r, J = blocks(v, D, target, uw, pw)
    for i in range(3):
        for k in range(8):
            eps = 1e-7
            probe = v.copy()
            probe[i] += eps * D[i, :, :, k]
            numeric = (blocks(probe, None, target, uw, pw) - r) / eps
            np.testing.assert_allclose(numeric, J[:, i * 8 + k], rtol=1e-5, atol=1e-7)
    print(
        "PASS: fixed radial source map/image derivatives and all 24 coupled residual Jacobian columns match finite differences."
    )
    from dartboard_vision.alignment import plane_features

    size = 90
    y, x = np.mgrid[:size, :size]
    mask = (np.hypot(x - 45, y - 45) < 34) & (np.hypot(x - 45, y - 45) > 7)
    gray = rng.uniform(20, 210, (size, size))
    D = rng.normal(size=(size, size, 8))
    features, analytic = plane_features(gray, D, mask, 0.3)
    # Outside-ROI text and its derivative must have exactly zero influence.
    changed = gray.copy()
    changed[~mask] = rng.uniform(-10000, 10000, np.sum(~mask))
    changed_D = D.copy()
    changed_D[~mask] *= 10000
    other, other_D = plane_features(changed, changed_D, mask, 0.3)
    np.testing.assert_allclose(features, other, atol=0, rtol=0)
    np.testing.assert_allclose(analytic, other_D, atol=0, rtol=0)
    for k in range(8):
        eps = 1e-4
        numeric = (
            plane_features(gray + eps * D[:, :, k], None, mask, 0.3) - features
        ) / eps
        np.testing.assert_allclose(analytic[:, :, :, k], numeric, rtol=1e-3, atol=1e-5)
    print(
        "PASS: masked plane normalization/Scharr derivatives; changing any outside-ROI pixel or derivative has zero effect."
    )
    from dartboard_vision.alignment import offset_samples

    seed = np.r_[p, 1.0].reshape(3, 3)
    offset = np.array([0.12, -0.23])
    values, D = offset_samples(field, q, offset, seed, model, 1.0, (0, 0))
    for k in range(2):
        eps = 1e-5
        probe = offset.copy()
        probe[k] += eps
        numeric = (
            offset_samples(field, q, probe, seed, model, 1.0, (0, 0), False) - values
        ) / eps
        np.testing.assert_allclose(D[:, :, k], numeric, rtol=2e-3, atol=1e-4)
    print(
        "PASS: the two constant board-image offset derivatives match source sampling finite differences."
    )
    offset = np.array([0.12, -0.23, 0.001])
    values, D = offset_samples(field, q, offset, seed, model, 1.0, (0, 0))
    for k in range(3):
        eps = 1e-7 if k == 2 else 1e-5
        probe = offset.copy()
        probe[k] += eps
        numeric = (
            offset_samples(field, q, probe, seed, model, 1.0, (0, 0), False) - values
        ) / eps
        np.testing.assert_allclose(D[:, :, k], numeric, rtol=3e-3, atol=1e-3)
    print(
        "PASS: the optional single radial coefficient derivative matches finite differences; it is only enabled for initial refinement."
    )
    from dartboard_vision.alignment import classify

    features = rng.normal(size=(17, 2))
    D = rng.normal(size=(17, 2, 3))
    photo = (rng.normal(size=(4, 2)), np.array([0.4, 0.7]))
    values, analytic = classify(features, D, photo)
    for k in range(3):
        eps = 1e-6
        numeric = (classify(features + eps * D[:, :, k], None, photo) - values) / eps
        np.testing.assert_allclose(analytic[:, :, k], numeric, rtol=1e-3, atol=5e-5)
    print(
        "PASS: sparse colour probabilities and their chain-rule derivatives match finite differences."
    )
    full = np.r_[seed.ravel()[:8], 0.001]
    values, D = warp_samples(field, q, full, model, scale=1.0)
    for k in range(9):
        eps = 1e-7
        probe = full.copy()
        probe[k] += eps
        numeric = (
            warp_samples(field, q, probe, model, scale=1.0, derivatives=False) - values
        ) / eps
        np.testing.assert_allclose(D[:, :, k], numeric, rtol=4e-3, atol=2e-2)
    print(
        "PASS: initial H plus one radial coefficient has nine verified analytic derivatives per camera."
    )
    v = rng.normal(size=(3, 13, 7))
    D = rng.normal(size=(3, 13, 7, 2))
    target = rng.normal(size=(13, 4))
    uw = np.ones((3, 13))
    pw = [np.ones(13) for _ in range(3)]
    base, J = blocks(v, D, target, uw, pw)
    for i in range(3):
        for k in range(2):
            probe = v.copy()
            probe[i] += 1e-7 * D[i, :, :, k]
            numeric = (blocks(probe, None, target, uw, pw) - base) / 1e-7
            np.testing.assert_allclose(numeric, J[:, i * 2 + k], rtol=1e-5, atol=1e-7)
    print(
        "PASS: six-variable colour/intensity/Scharr joint blocks couple the correct cameras."
    )

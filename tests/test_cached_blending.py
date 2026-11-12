def test_cached_blending():
    import numpy as np

    from dartboard_vision.fusion import CachedFusion, fuse

    rng = np.random.default_rng(59)
    planes = rng.integers(0, 256, (3, 60, 80, 3), dtype=np.uint8)
    confidence = rng.uniform(0, 3, (3, 60, 80))
    valid = rng.random((3, 60, 80)) > 0.2
    valid[:, :4, :4] = False
    cache = CachedFusion(confidence, valid)
    actual = cache.blend(planes)
    expected = fuse(planes, valid, confidence)
    assert np.max(abs(actual.astype(int) - expected.astype(int))) <= 1
    assert np.all(actual[:4, :4] == 0)
    np.testing.assert_allclose(
        cache.shares.sum(axis=0)[np.any(valid, axis=0)], 1.0, atol=2e-7
    )
    valid[0, 5:20] = False
    actual = cache.blend(planes, valid)
    expected = fuse(planes, valid, confidence)
    assert np.max(abs(actual.astype(int) - expected.astype(int))) <= 1
    assert np.all(cache.shares[~valid] == 0)
    print(
        "PASS: cached float32 fusion agrees within one intensity level, renormalizes visibility changes, excludes invalid views and returns black with no support."
    )

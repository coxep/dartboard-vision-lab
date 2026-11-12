def test_hsv_and_contact_geometry():
    import cv2 as cv
    import numpy as np

    from dartboard_vision.contacts import intersections, shaft_lines, terminal_evidence
    from dartboard_vision.masks import dart_mask, differences

    cv.setNumThreads(1)
    y, x = np.mgrid[:900, :900]
    background = np.full((900, 900, 3), [80, 145, 180], np.uint8)
    seen = np.ones((900, 900), bool)
    shade = 1 - 0.35 * np.exp(-((x - 450) ** 2 + (y - 450) ** 2) / (2 * 140**2))
    shadowed = np.uint8(background.astype(float) * shade[..., None])
    mask, _ = dart_mask(background, shadowed, seen)
    assert np.count_nonzero(mask) / mask.size < 0.001, "smooth shadow labelled as dart"
    after = shadowed.copy()
    cv.line(after, (350, 400), (450, 600), (20, 20, 220), 5)
    mask, _ = dart_mask(background, after, seen)
    expected = np.zeros((900, 900), np.uint8)
    cv.line(expected, (350, 400), (450, 600), 255, 5)
    assert np.mean(mask[expected > 0] > 0) > 0.8, "coloured shaft lost with shadow"

    # HSV hue wraps at 360 degrees; red near either side must stay nearby.
    a = np.full((900, 900, 3), [90, 130, 170], np.uint8)
    b = a.copy()
    for image, hue in [(a, 179), (b, 1)]:
        patch = np.full((20, 20, 3), [hue, 220, 180], np.uint8)
        image[20:40, 20:40] = cv.cvtColor(patch, cv.COLOR_HSV2BGR)
    f = differences(a, b, seen)
    assert np.max(f["hue_degrees"][22:38, 22:38]) < 6, (
        "hue wrap treated as a large change"
    )

    points = np.array([[250, 250], [550, 250], [450, 600]])
    axes = []
    for delta in [(-60, -140), (140, -40)]:
        mask = np.zeros((900, 900), np.uint8)
        for p in points:
            cv.line(mask, tuple(p), tuple(p + delta), 255, 5)
        axes.append(
            shaft_lines(mask, np.where(mask > 0, 3.0, 0.0).astype(np.float32))[0]
        )
    hits, _ = intersections(axes, np.array([450, 450]), 420)
    assert len(hits) == 3
    predicted = np.array([h["point_pixels"] for h in hits])
    assert (
        np.max(np.linalg.norm(predicted[:, None] - points[None], axis=2).min(axis=0))
        < 2
    )
    # A broken Hough segment inside a longer observed shaft is not a physical tip.
    axis = axes[0][0]
    mid = axis["ends"].mean(0)
    assert not terminal_evidence(axis, mid)[0]
    print(
        "HSV hue wrap, soft shadow suppression, shaft retention, three contacts and interior-fragment rejection passed."
    )

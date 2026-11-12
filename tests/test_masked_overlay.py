import numpy as np

from dartboard_vision.fusion import overlay_masked_views


def test_only_masked_visible_views_contribute():
    background = np.full((2, 4, 3), 10, np.uint8)
    planes = np.array([np.full_like(background, value) for value in (60, 120, 240)])
    masks = np.zeros((3, 2, 4), np.uint8)
    masks[0, 0, :3] = 255
    masks[1, 0, 1:3] = 255
    masks[2, 0, 2] = 255
    masks[2, 1, 0] = 255
    visible = np.ones_like(masks, dtype=bool)
    visible[2, 1, 0] = False
    result = overlay_masked_views(background, planes, masks, visible)
    np.testing.assert_array_equal(result[0, :, 0], [60, 90, 140, 10])
    np.testing.assert_array_equal(result[1], background[1])
    np.testing.assert_array_equal(background, 10)


def test_empty_masks_leave_original_unchanged():
    rng = np.random.default_rng(9)
    background = rng.integers(0, 256, (12, 15, 3), dtype=np.uint8)
    planes = np.zeros((3, 12, 15, 3), np.uint8)
    masks = np.zeros((3, 12, 15), np.uint8)
    np.testing.assert_array_equal(
        overlay_masked_views(background, planes, masks), background
    )

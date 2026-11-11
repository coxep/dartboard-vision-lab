"""Colour-topology association and explicit metric-plane corner fitting.

Colours disambiguate the two sides of a band; they cannot identify the sector
number modulo the repeating pattern. The existing upright pose prior does that.
"""
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


def model_classes(points, board):
    """0 dark, 1 light, 2 red, 3 green, in canonical board coordinates."""
    x, y = ((np.asarray(points) - board.center) / board.ppm).T
    radius = np.hypot(x, y)
    sector = ((np.degrees(np.arctan2(x, -y)) + 9) % 360 // 18).astype(int)
    labels = sector % 2
    ti, to, di, do = board.radii
    band = ((radius >= ti) & (radius <= to)) | ((radius >= di) & (radius <= do))
    labels[band] += 2
    labels[radius > do] = 0
    labels[radius <= 15.9] = 3
    labels[radius <= 6.35] = 2
    return labels


def colour_corner_scores(frame, seed, board, corners, candidate_ids, model_ids):
    """Test four region patches around each observed corner, avoiding the wire.

    Sample positions follow the local radial/tangential basis of the associated
    model corner, but are centred on the DETECTION, not the nominal coordinate.
    A wrong inward/outward band assignment therefore fails the colour test.
    Returns a mean agreement and the weakest of the four region agreements.
    """
    from dartboard import binarize, board_color_masks, project
    classes = (binarize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) > 0).astype(np.uint8)
    masks = board_color_masks(frame)
    classes[masks['red'] > 0] = 2
    classes[masks['green'] > 0] = 3
    ids = np.asarray(model_ids, int)
    nominal = board.landmarks()[ids]
    radial = (nominal-board.center)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    tangent = np.c_[-radial[:, 1], radial[:, 0]]
    # Nine samples in each of four patches, all distances in board millimetres.
    offsets = np.array([(rs*(2.5+dr), ts*(3.5+dt))
                        for rs in (-1, 1) for ts in (-1, 1)
                        for dr in (-.6, 0, .6) for dt in (-.6, 0, .6)])
    delta = board.ppm*(radial[:, None]*offsets[None, :, :1]
                       + tangent[:, None]*offsets[None, :, 1:])
    expected = model_classes((nominal[:, None]+delta).reshape(-1, 2), board)
    centers = project(corners[np.asarray(candidate_ids, int)], seed)
    source = project((centers[:, None]+delta).reshape(-1, 2), np.linalg.inv(seed))
    xy = np.rint(source).astype(int)
    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < frame.shape[1])
             & (xy[:, 1] >= 0) & (xy[:, 1] < frame.shape[0]))
    observed = np.full(len(xy), 255, np.uint8)
    observed[valid] = classes[xy[valid, 1], xy[valid, 0]]
    agreement = (observed == expected).reshape(-1, 4, 9).mean(axis=2)
    return agreement.mean(axis=1), agreement.min(axis=1)


def fit_corner_homography(source, target, seed, board):
    """Robust least squares on fixed, validated correspondences after consensus.

    Optimise a normalized plane correction in float64. The returned transform
    maps original source pixels to the known metric board coordinates.
    """
    from dartboard import project
    scale = board.radii[-1]*board.ppm
    S = np.array([[scale, 0, board.center[0]], [0, scale, board.center[1]], [0, 0, 1.]])
    src = (project(source, seed).astype(float)-board.center)/scale
    dst = (np.asarray(target, float)-board.center)/scale
    homogeneous = np.c_[src, np.ones(len(src))]

    def matrix(p):
        return np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [p[6], p[7], 1.]])

    def residual(p):
        q = homogeneous @ matrix(p).T
        return ((q[:, :2]/q[:, 2:]-dst)*scale).ravel()

    before = residual(np.zeros(8)).reshape(-1, 2)
    fit = least_squares(residual, np.zeros(8), loss='soft_l1', f_scale=1.,
                        max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    H = S @ matrix(fit.x) @ np.linalg.inv(S) @ seed
    error = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    return H, dict(method='RANSAC consensus then explicit soft-L1 least squares on known corner coordinates',
                   correspondences=len(source), evaluations=fit.nfev,
                   initial_rmse_px=float(np.sqrt(np.mean(before**2)*2)),
                   rmse_px=float(np.sqrt(np.mean(error**2))),
                   median_px=float(np.median(error)), p95_px=float(np.percentile(error, 95)))

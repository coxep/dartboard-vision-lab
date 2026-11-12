"""Direct joint H tracking: abstract paint regions and current raw image texture.

Feature maps are prepared once in SOURCE coordinates. Only raw-source samples
are used during optimization, avoiding changes in interpolation blur when H is
updated. Default tracking uses six image-offset parameters and a frozen radial
coefficient. Optional post-joint calibration can add a residual radial-r4 term.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np

from ._seed import SeedRectifier as JointTracker
from .geometry import BOARD, R, geometry_fields
from .photometry import fit_photometry, probabilities

PAIRS = ((0, 1), (0, 2), (1, 2))


def sample_field(field, xy):
    h, w = field.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    ix = x.astype(int)
    iy = y.astype(int)
    dx = (x - ix)[:, None]
    dy = (y - iy)[:, None]
    a = field[iy, ix]
    b = field[iy, ix + 1]
    c = field[iy + 1, ix]
    d = field[iy + 1, ix + 1]
    value = (
        (1 - dx) * (1 - dy) * a + dx * (1 - dy) * b + (1 - dx) * dy * c + dx * dy * d
    )
    return value, (1 - dy) * (b - a) + dy * (d - c), (1 - dx) * (c - a) + dx * (d - b)


def source_mapping(q, p, model, scale=0.5, origin=(0.0, 0.0), derivatives=True):
    H = np.r_[p[:8], 1.0].reshape(3, 3)
    v = q @ H.T
    z = v[:, 2]
    u = v[:, :2] / z[:, None]
    lam = model["division_lambda"] + (float(p[8]) if len(p) > 8 else 0.0)
    r2 = np.sum(u * u, axis=1)
    root = np.sqrt(np.maximum(1 - 4 * lam * r2, 1e-8))
    factor = 2 / (1 + root)
    distorted = u * factor[:, None]
    beta = model.get("residual_radial4", 0.0) + (float(p[9]) if len(p) > 9 else 0.0)
    extra = None
    tangent = np.array(model.get("tangential", [0.0, 0.0]), dtype=float) + (
        np.asarray(p[10:12]) if len(p) > 11 else 0.0
    )
    p1, p2 = tangent
    if beta or np.any(tangent) or len(p) > 9:
        base = distorted
        rd2 = np.sum(base * base, axis=1)
        extra = 1 + beta * rd2**2
        distorted = base * extra[:, None]
        bx, by = base.T
        distorted = (
            distorted
            + np.c_[
                2 * p1 * bx * by + p2 * (rd2 + 2 * bx * bx),
                p1 * (rd2 + 2 * by * by) + 2 * p2 * bx * by,
            ]
        )
    xy = (
        (
            distorted * model["normalization_radius_px"]
            + np.array(model["distortion_centre_px"])
            + 0.5
        )
        * scale
        - 0.5
        - np.array(origin)
    )
    if not derivatives:
        return xy
    a = q / z[:, None]
    zero = np.zeros_like(a)
    du = np.c_[a, zero, -u[:, 0, None] * a[:, :2]]
    dv = np.c_[zero, a, -u[:, 1, None] * a[:, :2]]
    df = 4 * lam / (root * (1 + root) ** 2)
    dx = (factor + 2 * df * u[:, 0] ** 2)[:, None] * du + (2 * df * u[:, 0] * u[:, 1])[
        :, None
    ] * dv
    dy = (2 * df * u[:, 0] * u[:, 1])[:, None] * du + (factor + 2 * df * u[:, 1] ** 2)[
        :, None
    ] * dv
    if len(p) > 8:
        df_lambda = 4 * r2 / (root * (1 + root) ** 2)
        dx = np.c_[dx, u[:, 0] * df_lambda]
        dy = np.c_[dy, u[:, 1] * df_lambda]
    if extra is not None:
        old_dx, old_dy = dx, dy
        cross = 4 * beta * rd2 * bx * by + 2 * p1 * bx + 2 * p2 * by
        dx = (extra + 4 * beta * rd2 * bx**2 + 2 * p1 * by + 6 * p2 * bx)[
            :, None
        ] * old_dx + cross[:, None] * old_dy
        dy = (
            cross[:, None] * old_dx
            + (extra + 4 * beta * rd2 * by**2 + 6 * p1 * by + 2 * p2 * bx)[:, None]
            * old_dy
        )
        if len(p) > 9:
            dx = np.c_[dx, base[:, 0] * rd2**2]
            dy = np.c_[dy, base[:, 1] * rd2**2]
        if len(p) > 11:
            dx = np.c_[dx, 2 * bx * by, rd2 + 2 * bx * bx]
            dy = np.c_[dy, rd2 + 2 * by * by, 2 * bx * by]
    return (
        xy,
        dx * (scale * model["normalization_radius_px"]),
        dy * (scale * model["normalization_radius_px"]),
    )


def warp_samples(field, q, p, model, scale=0.5, origin=(0.0, 0.0), derivatives=True):
    if not derivatives:
        return sample_field(field, source_mapping(q, p, model, scale, origin, False))[0]
    xy, dx, dy = source_mapping(q, p, model, scale, origin, True)
    v, gx, gy = sample_field(field, xy)
    return v, gx[:, :, None] * dx[:, None, :] + gy[:, :, None] * dy[:, None, :]


def offset_mapping(
    q, offset, seed, model, scale=0.5, origin=(0.0, 0.0), derivatives=True
):
    shifted = q.copy()
    shifted[:, :2] += np.array(offset[:2])[None, :] / R
    v = shifted @ seed.T
    z = v[:, 2]
    u = v[:, :2] / z[:, None]
    lam = model["division_lambda"] + (float(offset[2]) if len(offset) > 2 else 0.0)
    r2 = np.sum(u * u, axis=1)
    root = np.sqrt(np.maximum(1 - 4 * lam * r2, 1e-8))
    factor = 2 / (1 + root)
    base = u * factor[:, None]
    distorted = base
    beta = model.get("residual_radial4", 0.0)
    p1, p2 = model.get("tangential", [0.0, 0.0])
    extra = None
    if beta or p1 or p2:
        bx, by = base.T
        rd2 = bx * bx + by * by
        extra = 1 + beta * rd2**2
        distorted = (
            base * extra[:, None]
            + np.c_[
                2 * p1 * bx * by + p2 * (rd2 + 2 * bx * bx),
                p1 * (rd2 + 2 * by * by) + 2 * p2 * bx * by,
            ]
        )
    xy = (
        (
            distorted * model["normalization_radius_px"]
            + np.array(model["distortion_centre_px"])
            + 0.5
        )
        * scale
        - 0.5
        - np.array(origin)
    )
    if not derivatives:
        return xy
    du = (seed[0, :2][None, :] - u[:, 0, None] * seed[2, :2][None, :]) / (
        z[:, None] * R
    )
    dv = (seed[1, :2][None, :] - u[:, 1, None] * seed[2, :2][None, :]) / (
        z[:, None] * R
    )
    df = 4 * lam / (root * (1 + root) ** 2)
    dx = (factor + 2 * df * u[:, 0] ** 2)[:, None] * du + (2 * df * u[:, 0] * u[:, 1])[
        :, None
    ] * dv
    dy = (2 * df * u[:, 0] * u[:, 1])[:, None] * du + (factor + 2 * df * u[:, 1] ** 2)[
        :, None
    ] * dv
    if len(offset) > 2:
        df_lambda = 4 * r2 / (root * (1 + root) ** 2)
        dx = np.c_[dx, u[:, 0] * df_lambda]
        dy = np.c_[dy, u[:, 1] * df_lambda]
    if extra is not None:
        cross = 4 * beta * rd2 * bx * by + 2 * p1 * bx + 2 * p2 * by
        old_dx, old_dy = dx, dy
        dx = (extra + 4 * beta * rd2 * bx * bx + 2 * p1 * by + 6 * p2 * bx)[
            :, None
        ] * old_dx + cross[:, None] * old_dy
        dy = (
            cross[:, None] * old_dx
            + (extra + 4 * beta * rd2 * by * by + 6 * p1 * by + 2 * p2 * bx)[:, None]
            * old_dy
        )
    return (
        xy,
        dx * (scale * model["normalization_radius_px"]),
        dy * (scale * model["normalization_radius_px"]),
    )


def offset_samples(field, q, offset, seed, model, scale, origin, derivatives=True):
    if not derivatives:
        return sample_field(
            field, offset_mapping(q, offset, seed, model, scale, origin, False)
        )[0]
    xy, dx, dy = offset_mapping(q, offset, seed, model, scale, origin)
    value, gx, gy = sample_field(field, xy)
    return value, gx[:, :, None] * dx[:, None, :] + gy[:, :, None] * dy[:, None, :]


def colour_features(image):
    smooth = cv.GaussianBlur(image.astype(np.float32), (0, 0), 0.6)
    b, g, r = cv.split(smooth)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.stack([np.log(gray + 8), (r - g) / (r + g + b + 24)], axis=-1)


def plane_features(gray, derivative, mask, scale=1 / 3, support_cache=None):
    """Masked board-plane normalization and Scharr; no outside-ROI samples."""
    m = mask.astype(np.float64)
    if support_cache is not None and (
        "mask" not in support_cache or not np.array_equal(mask, support_cache["mask"])
    ):
        support_cache.clear()
        support_cache["mask"] = mask.copy()

    def average(value, sigma):
        numerator = cv.GaussianBlur(
            value * m if value.ndim == 2 else value * m[..., None], (0, 0), sigma
        )
        if support_cache is not None and sigma in support_cache:
            denominator = support_cache[sigma]
        else:
            denominator = np.maximum(cv.GaussianBlur(m, (0, 0), sigma), 1e-8)
            if support_cache is not None:
                support_cache[sigma] = denominator
        return (
            numerator / denominator
            if value.ndim == 2
            else numerator / denominator[..., None]
        )

    g = average(gray, max(0.6, scale))
    mean = average(g, 9 * scale)
    variance = average(g * g, 9 * scale) - mean * mean
    den = np.sqrt(np.maximum(variance, 144.0))
    f = (g - mean) / den
    active = (abs(f) < 3) & mask
    f = np.clip(f, -3, 3)
    f[~mask] = 0
    if support_cache is not None and "stencil" in support_cache:
        gradient_valid = support_cache["stencil"]
    else:
        gradient_valid = cv.erode(np.uint8(mask), np.ones((3, 3), np.uint8)) > 0
        if support_cache is not None:
            support_cache["stencil"] = gradient_valid

    def scharr(v, dx, dy):
        return cv.Scharr(v, cv.CV_64F, dx, dy, scale=1 / 32)

    features = np.stack(
        [f, scharr(f, 1, 0) * gradient_valid, scharr(f, 0, 1) * gradient_valid], axis=-1
    )
    if derivative is None:
        return features
    dg = average(derivative, max(0.6, scale))
    dm = average(dg, 9 * scale)
    dv = average(2 * g[..., None] * dg, 9 * scale) - 2 * mean[..., None] * dm
    df = (dg - dm) / den[..., None] - (g - mean)[..., None] * np.where(
        (variance > 144.0)[..., None], 0.5 * dv / den[..., None] ** 3, 0
    )
    df *= active[..., None]
    D = np.stack(
        [
            df,
            scharr(df, 1, 0) * gradient_valid[..., None],
            scharr(df, 0, 1) * gradient_valid[..., None],
        ],
        axis=2,
    )
    return features, D


def classify(features, D, photo):
    p = probabilities(features, photo)
    if D is None:
        return p
    centers, scale = photo
    dl = -np.sum(
        ((features[:, None, :] - centers) / scale**2)[:, :, :, None] * D[:, None, :, :],
        axis=2,
    )
    return p, p[:, :, None] * (dl - np.sum(p[:, :, None] * dl, axis=1)[:, None, :])


def robust_cost(r, scale=0.5):
    return float(np.sum(scale * scale * (np.sqrt(1 + (r / scale) ** 2) - 1)))


def paint_features(values, D, support, texture_weight):
    """Keep unary colours; add pairwise colours and secondary texture.

    Unknown outer-board paint labels never enter the pairwise colour factors.
    No ring/spoke or observed-edge exclusions are introduced.
    """
    s = support[None, :, None]
    weight = np.sqrt(texture_weight)
    v = np.concatenate(
        [values[:, :, :4], values[:, :, :4] * s, values[:, :, 4:] * weight], axis=2
    )
    if D is None:
        return v, None
    return v, np.concatenate(
        [D[:, :, :4], D[:, :, :4] * s[..., None], D[:, :, 4:] * weight], axis=2
    )


def paint_robust(r, n, scale=0.5):
    """One soft-L1 weight per colour/texture vector, rather than channel.

    RMS vector magnitude preserves the original per-channel residual scale.
    Cost and IRLS weights describe the same differentiable objective.
    """
    weights = np.empty_like(r)
    cost = 0.0
    offset = 0

    def assign(v, w):
        nonlocal cost
        mean = np.mean(v * v, axis=-1)
        root = np.sqrt(1 + mean / scale**2)
        cost += float(np.sum(v.shape[-1] * scale**2 * (root - 1)))
        w[:] = 1 / root[..., None]

    for i in range(3):
        count = 4 * n
        assign(
            r[offset : offset + count].reshape(n, 4),
            weights[offset : offset + count].reshape(n, 4),
        )
        offset += count
    for k in range(3):
        count = 7 * n
        v = r[offset : offset + count].reshape(n, 7)
        w = weights[offset : offset + count].reshape(n, 7)
        assign(v[:, :4], w[:, :4])
        assign(v[:, 4:], w[:, 4:])
        offset += count
    assert offset == len(r)
    return cost, weights


def blocks(values, D, target, uw, pw):
    n = values.shape[1]
    res = [((values[i, :, :4] - target) * uw[i, :, None]).ravel() for i in range(3)]
    res.extend(
        ((values[a, :, 4:] - values[b, :, 4:]) * pw[k][:, None]).ravel()
        for k, (a, b) in enumerate(PAIRS)
    )
    r = np.concatenate(res)
    if D is None:
        return r
    active = D.shape[-1]
    J = np.zeros((len(r), 3 * active))
    offset = 0
    for i in range(3):
        J[offset : offset + 4 * n, i * active : (i + 1) * active] = (
            D[i, :, :4, :] * uw[i, :, None, None]
        ).reshape(4 * n, active)
        offset += 4 * n
    for k, (a, b) in enumerate(PAIRS):
        count = n * (values.shape[2] - 4)
        J[offset : offset + count, a * active : (a + 1) * active] = (
            D[a, :, 4:, :] * pw[k][:, None, None]
        ).reshape(count, active)
        J[offset : offset + count, b * active : (b + 1) * active] = (
            -D[b, :, 4:, :] * pw[k][:, None, None]
        ).reshape(count, active)
        offset += count
    return r, J


class FastJointRegions:
    def __init__(
        self,
        frames,
        models,
        board=BOARD,
        size=300,
        stride=3,
        alpha=1.0,
        iterations=3,
        source_scale=0.5,
        spatial_confidence=None,
        texture_outer_radius_mm=None,
        image_masks=None,
        binary_template_weight=False,
        texture_inner_radius_mm=25.0,
        paint_consensus=False,
        texture_weight=1.0,
    ):
        if any(abs(m.get("division_lambda2", 0)) > 0 for m in models):
            raise ValueError(
                "This tracker uses the restored single-term lens calibration"
            )
        self.board = board
        self.size = size
        self.alpha = alpha
        self.iterations = iterations
        self.models = [dict(m) for m in models]
        self.source_scale = source_scale
        if not np.isfinite(texture_weight) or texture_weight < 0:
            raise ValueError("Texture weight must be finite and nonnegative")
        self.paint_consensus = paint_consensus
        self.texture_weight = texture_weight
        self.rectifiers = [
            JointTracker(frame, model, use_klt=False)
            for frame, model in zip(frames, models)
        ]
        labels, distance, known_roi, _ = geometry_fields(size, board)
        roi = known_roi.copy()
        yy, xx = np.mgrid[:size, :size]
        if texture_outer_radius_mm is not None:
            if (
                not np.isfinite(texture_outer_radius_mm)
                or texture_outer_radius_mm < board.radii[-1] + 4
            ):
                raise ValueError("Texture radius must cover the known scoring ROI")
            if (
                not np.isfinite(texture_inner_radius_mm)
                or not 0 <= texture_inner_radius_mm < texture_outer_radius_mm
            ):
                raise ValueError("Invalid texture inner radius")
            radius = np.hypot(xx * 900 / size - 450, yy * 900 / size - 450) / board.ppm
            roi |= (radius >= texture_inner_radius_mm) & (
                radius < texture_outer_radius_mm
            )
        self.fit_mask = roi.copy()
        self.dense_q = np.c_[
            (xx.ravel() * 900 / size - 450) / R,
            (yy.ravel() * 900 / size - 450) / R,
            np.ones(size * size),
        ]
        roi &= (xx % stride == 0) & (yy % stride == 0)
        self.y, self.x = np.nonzero(roi)
        self.template_support = known_roi[self.y, self.x]
        self.camera_masks = np.tile(self.fit_mask, (3, 1, 1))
        if image_masks is not None:
            image_masks = np.asarray(image_masks)
            if image_masks.shape != (3, size, size) or np.any(
                (image_masks != 0) & (image_masks != 1)
            ):
                raise ValueError(
                    "Image masks must be binary maps for the three rectified views"
                )
            self.camera_masks &= image_masks.astype(bool)
        self.sample_masks = self.camera_masks.copy()
        if np.any(self.camera_masks != self.fit_mask[None, :, :]):
            self.sample_masks = np.array(
                [
                    cv.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    for m in self.camera_masks
                ]
            )
        self.q = np.c_[
            (self.x * 900 / size - 450) / R,
            (self.y * 900 / size - 450) / R,
            np.ones(len(self.x)),
        ]
        self.target = np.eye(4)[labels[self.y, self.x]]
        self.spatial_confidence = None
        if spatial_confidence is not None:
            confidence = np.asarray(spatial_confidence, dtype=float)
            if (
                confidence.shape != (3, size, size)
                or not np.all(np.isfinite(confidence))
                or np.any(confidence < 0)
                or np.any(confidence.sum(axis=0) <= 0)
            ):
                raise ValueError(
                    "Confidence must contain three finite nonnegative maps with positive total support"
                )
            self.spatial_confidence = (
                3 * confidence / confidence.sum(axis=0, keepdims=True)
            )
        taper = np.clip((distance - 0.8) / 1.7, 0, 1)
        taper = taper * taper * (3 - 2 * taper)
        self.area_weight = (0.2 + 0.8 * taper)[self.y, self.x] * self.template_support
        if binary_template_weight:
            self.area_weight = self.template_support.astype(float)
        self.photometry = []
        for i, frame in enumerate(frames):
            crop, origin = self.crop(i, frame)
            f = colour_features(crop)
            xy = source_mapping(
                self.q,
                self.rectifiers[i].J.ravel()[:8],
                models[i],
                source_scale,
                origin,
                False,
            )
            sampled = sample_field(f, xy)[0]
            self.photometry.append(
                fit_photometry(
                    sampled,
                    labels[self.y, self.x],
                    (distance[self.y, self.x] > 2.5) & self.template_support,
                )
            )
        self.pool = ThreadPoolExecutor(max_workers=3)
        self._plane_support_cache = [{}, {}, {}]

    def crop(self, i, frame):
        small = cv.resize(
            frame,
            None,
            fx=self.source_scale,
            fy=self.source_scale,
            interpolation=cv.INTER_AREA,
        )
        xy = source_mapping(
            self.q,
            self.rectifiers[i].J.ravel()[:8],
            self.models[i],
            self.source_scale,
            derivatives=False,
        )
        margin = int(np.ceil(4 * 9 * self.source_scale)) + 3
        lo = np.maximum(np.floor(xy.min(axis=0)).astype(int) - margin, 0)
        hi = np.minimum(
            np.ceil(xy.max(axis=0)).astype(int) + margin + 1, small.shape[1::-1]
        )
        return small[lo[1] : hi[1], lo[0] : hi[0]], lo

    def prepare(self, i, frame):
        crop, origin = self.crop(i, frame)
        features = colour_features(crop).astype(np.float32)
        texture = cv.cvtColor(crop, cv.COLOR_BGR2GRAY).astype(np.float32)
        return np.dstack([features, texture]).astype(np.float32), origin

    def update(
        self,
        frames,
        calibrate_distortion=False,
        refine_homography=False,
        radial_camera_mask=None,
        linearization_callback=None,
        calibrate_radial4=False,
        calibrate_tangential=False,
    ):
        if calibrate_tangential and not calibrate_radial4:
            raise ValueError(
                "Tangential calibration requires the post-joint radial calibration pass"
            )
        if calibrate_radial4 and not calibrate_distortion:
            raise ValueError(
                "Higher-order calibration requires the radial calibration pass"
            )
        if radial_camera_mask is not None and (
            not calibrate_distortion or len(radial_camera_mask) != 3
        ):
            raise ValueError(
                "A radial-camera mask requires distortion calibration and one boolean per view"
            )
        start = time.perf_counter()
        prepared = list(self.pool.map(lambda i: self.prepare(i, frames[i]), range(3)))
        prep_s = time.perf_counter() - start
        fit_h = calibrate_distortion or refine_homography
        seed = [t.J.copy() for t in self.rectifiers]
        active = (
            12
            if calibrate_tangential
            else 10
            if calibrate_radial4
            else 9
            if calibrate_distortion
            else 8
            if fit_h
            else 2
        )
        p = (
            np.array([np.r_[x.ravel()[:8], np.zeros(active - 8)] for x in seed])
            if fit_h
            else np.zeros((3, 2))
        )
        uw = np.tile(np.sqrt(self.area_weight), (3, 1))
        pw = [np.full(len(self.q), np.sqrt(self.alpha)) for _ in PAIRS]
        keep = self.sample_masks[:, self.y, self.x]
        uw *= keep
        pw = [weight * (keep[a] & keep[b]) for weight, (a, b) in zip(pw, PAIRS)]
        if self.spatial_confidence is not None:
            c = self.spatial_confidence[:, self.y, self.x]
            uw *= np.sqrt(c)
            pw = [
                np.sqrt(self.alpha * 2 * c[a] * c[b] / np.maximum(c[a] + c[b], 1e-12))
                for a, b in PAIRS
            ]

        def evaluate(state, jac=False):
            def camera(i):
                field, origin = prepared[i]
                if fit_h:
                    colour = warp_samples(
                        field[:, :, :2],
                        self.q,
                        state[i],
                        self.models[i],
                        self.source_scale,
                        origin,
                        jac,
                    )
                    gray = warp_samples(
                        field[:, :, 2:],
                        self.dense_q,
                        state[i],
                        self.models[i],
                        self.source_scale,
                        origin,
                        jac,
                    )
                else:
                    colour = offset_samples(
                        field[:, :, :2],
                        self.q,
                        state[i],
                        seed[i],
                        self.models[i],
                        self.source_scale,
                        origin,
                        jac,
                    )
                    gray = offset_samples(
                        field[:, :, 2:],
                        self.dense_q,
                        state[i],
                        seed[i],
                        self.models[i],
                        self.source_scale,
                        origin,
                        jac,
                    )
                if jac:
                    colour, dc = classify(colour[0], colour[1], self.photometry[i])
                    texture, dt = plane_features(
                        gray[0][:, 0].reshape(self.size, self.size),
                        gray[1][:, 0, :].reshape(self.size, self.size, active),
                        self.camera_masks[i],
                        self.size / 900,
                        self._plane_support_cache[i],
                    )
                    return np.c_[colour, texture[self.y, self.x]], np.concatenate(
                        [dc, dt[self.y, self.x]], axis=1
                    )
                texture = plane_features(
                    gray[:, 0].reshape(self.size, self.size),
                    None,
                    self.camera_masks[i],
                    self.size / 900,
                    self._plane_support_cache[i],
                )
                return np.c_[
                    classify(colour, None, self.photometry[i]), texture[self.y, self.x]
                ]

            sampled = list(self.pool.map(camera, range(3)))
            values = np.array([x[0] for x in sampled]) if jac else np.array(sampled)
            D = np.array([x[1] for x in sampled]) if jac else None
            if self.paint_consensus:
                values, D = paint_features(
                    values, D, self.template_support, self.texture_weight
                )
            return blocks(values, D, self.target, uw, pw)

        def cost_weights(residual):
            if self.paint_consensus:
                return paint_robust(residual, len(self.q))
            return robust_cost(residual), 1 / np.sqrt(1 + (residual / 0.5) ** 2)

        initial_jacobian = self.iterations > 0 or linearization_callback is not None
        if initial_jacobian:
            residual, J = evaluate(p, True)
        else:
            residual = evaluate(p)
        before = cost_weights(residual)[0]
        cost = before
        steps = []
        if linearization_callback is not None:
            if radial_camera_mask is not None:
                for i, enabled in enumerate(radial_camera_mask):
                    if not enabled:
                        J[:, i * active + 8] = 0
            linearization_callback(residual.copy(), J.copy(), p.copy())
        for it in range(self.iterations):
            if it:
                residual, J = evaluate(p, True)
            if radial_camera_mask is not None:
                for i, enabled in enumerate(radial_camera_mask):
                    if not enabled:
                        J[:, i * active + 8] = 0
            w = cost_weights(residual)[1]
            A = J.T @ (w[:, None] * J)
            g = J.T @ (w * residual)
            damping = 1e-3 * np.maximum(np.diag(A), 1e-6)
            delta = np.linalg.solve(A + np.diag(damping), -g).reshape(3, active)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = p + fraction * delta
                if fit_h:
                    movement = []
                    for i in range(3):
                        rel = np.linalg.inv(seed[i]) @ np.r_[
                            candidate[i, :8], 1.0
                        ].reshape(3, 3)
                        q = self.q @ rel.T
                        movement.append(
                            float(
                                np.max(
                                    np.linalg.norm(
                                        q[:, :2] / q[:, 2:] - self.q[:, :2], axis=1
                                    )
                                )
                                * R
                            )
                        )
                    if calibrate_distortion and np.max(abs(candidate[:, 8])) > 0.02:
                        continue
                    if calibrate_radial4 and np.max(abs(candidate[:, 9])) > 0.02:
                        continue
                    if calibrate_tangential and np.max(abs(candidate[:, 10:12])) > 0.02:
                        continue
                else:
                    movement = np.linalg.norm(candidate[:, :2], axis=1)
                if max(movement) > 8:
                    continue
                newcost = cost_weights(evaluate(candidate))[0]
                if np.isfinite(newcost) and newcost < cost:
                    oldcost = cost
                    p = candidate
                    cost = newcost
                    accepted = True
                    steps.append(
                        dict(
                            iteration=it + 1,
                            cost=cost,
                            step_fraction=fraction,
                            max_total_correction_px=max(movement),
                        )
                    )
                    break
            if not accepted:
                break
            if (oldcost - cost) / max(oldcost, 1) < 1e-4:
                break
        for i, rectifier in enumerate(self.rectifiers):
            if fit_h:
                rectifier.J = np.r_[p[i, :8], 1.0].reshape(3, 3)
            if calibrate_distortion:
                self.models[i] = dict(
                    self.models[i],
                    division_lambda=self.models[i]["division_lambda"] + float(p[i, 8]),
                )
                if calibrate_radial4:
                    self.models[i]["residual_radial4"] = self.models[i].get(
                        "residual_radial4", 0.0
                    ) + float(p[i, 9])
                if calibrate_tangential:
                    self.models[i]["tangential"] = (
                        np.array(self.models[i].get("tangential", [0.0, 0.0]))
                        + p[i, 10:12]
                    ).tolist()
                rectifier.lam = self.models[i]["division_lambda"]
                rectifier.model = self.models[i]
                r2 = rectifier.radial_d**2
                rectifier.radial_u = rectifier.radial_d / (1 + rectifier.lam * r2)
            elif not fit_h:
                T = np.eye(3)
                T[:2, 2] = p[i] / R
                rectifier.J = seed[i] @ T
                rectifier.J /= rectifier.J[2, 2]
        return dict(
            elapsed_ms=(time.perf_counter() - start) * 1000,
            prepare_ms=prep_s * 1000,
            objective_before=before,
            objective_after=cost,
            iterations=len(steps),
            steps=steps,
            objective_decreased=cost < before,
            quality_evaluated=False,
            acceptance_definition="Non-increasing source-image objective only; no assertion of geometric/scoring accuracy",
            cross_view_weight=self.alpha,
            spatial_confidence_used=self.spatial_confidence is not None,
            feature_maps_rebuilt_per_derivative=False,
            first_frame_texture_target=False,
            offsets_px=None if fit_h else p.tolist(),
            parameters=3 * active
            - (
                sum(not x for x in radial_camera_mask)
                if radial_camera_mask is not None
                else 0
            ),
            radial_camera_mask=radial_camera_mask,
            homography_refined=fit_h,
            distortion_refined=calibrate_distortion,
            division_lambda=[m["division_lambda"] for m in self.models],
            homographies=[r.J.tolist() for r in self.rectifiers],
            kernel="3x3 Scharr inside scoring ROI",
        )

    def rectify(self, i, frame):
        """Render actual source pixels, including any frozen residual radial term."""
        y, x = np.mgrid[:900, :900]
        q = np.c_[(x.ravel() - 450) / R, (y.ravel() - 450) / R, np.ones(900 * 900)]
        xy = (
            source_mapping(
                q,
                self.rectifiers[i].J.ravel()[:8],
                self.models[i],
                scale=1.0,
                derivatives=False,
            )
            .reshape(900, 900, 2)
            .astype(np.float32)
        )
        valid = (
            (xy[:, :, 0] >= 0)
            & (xy[:, :, 0] < frame.shape[1] - 1)
            & (xy[:, :, 1] >= 0)
            & (xy[:, :, 1] < frame.shape[0] - 1)
        )
        return cv.remap(
            frame,
            xy[:, :, 0],
            xy[:, :, 1],
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        ), valid

    def close(self):
        self.pool.shutdown()

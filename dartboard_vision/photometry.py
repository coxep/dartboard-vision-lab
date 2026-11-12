"""Local contrast and color-class probabilities used during alignment."""

import cv2 as cv
import numpy as np


def contrast_normalized(gray, scale=1.0):
    smooth = cv.GaussianBlur(gray.astype(np.float32), (0, 0), max(0.6, scale))
    mean = cv.GaussianBlur(smooth, (0, 0), 9 * scale)
    variance = cv.GaussianBlur(smooth * smooth, (0, 0), 9 * scale) - mean * mean
    return np.clip(
        (smooth - mean) / np.sqrt(np.maximum(variance, 12.0**2)), -3, 3
    ).astype(np.float32)


def fit_photometry(features, labels, support):
    centers = np.array(
        [np.median(features[(labels == k) & support], axis=0) for k in range(4)]
    )
    error = features - centers[labels]
    scale = np.maximum(1.4826 * np.median(abs(error[support]), axis=0), [0.12, 0.025])
    return centers, scale


def probabilities(features, photo):
    centers, scale = photo
    cost = np.sum(((features[..., None, :] - centers) / scale) ** 2, axis=-1) / 2
    logits = -cost
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(np.maximum(logits, -60))
    return e / e.sum(axis=-1, keepdims=True)

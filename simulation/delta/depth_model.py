"""DA3: one picture in, a distance for every pixel out. Brain side.

DA3METRIC-LARGE is meant to give metres, but on our 120-degree lens its
numbers come out about 1.6x too big (measured against Habitat's depth --
probably a field of view wider than it was trained on). So its scale is not
trusted. Stereo measures real metres wherever it finds a match; DA3 sees
everything, blank walls included. `StereoRuler` compares the two each frame
and rescales DA3 -- dense like DA3, in stereo's units. Only the brain's own
data goes in.
"""

import sys
import types

import cv2
import numpy as np


def _let_da3_import():
    """DA3 imports its file-export helpers at import time. Two of them need
    packages we do not install -- pycolmap (COLMAP files) and moviepy 1.x
    (videos) -- and we never export files. Stand-ins let the import succeed;
    calling those exports would fail loudly, which is what we want."""
    import moviepy

    sys.modules.setdefault("moviepy.editor", moviepy)

    try:
        import pycolmap  # noqa: F401
    except ImportError:
        sys.modules["pycolmap"] = types.ModuleType("pycolmap")


class DepthModel:
    """Raw DA3 depth for one image, resized back to that image's size."""

    def __init__(self, name, device="cuda", process_res=504):
        import os

        import torch

        # DA3 prints three lines per picture at its default INFO level.
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

        _let_da3_import()

        from depth_anything_3.api import DepthAnything3

        self.torch = torch
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(name).to(device).eval()

    def __call__(self, rgb, right_rgb=None):
        """Depth for the LEFT picture. Given `right_rgb` too, DA3 sees both
        cameras (no pose is passed: see experiments/da3_stereo.py)."""
        images = [rgb] if right_rgb is None else [rgb, right_rgb]

        with self.torch.no_grad():
            prediction = self.model.inference(images, process_res=self.process_res)

        raw = prediction.depth[0]
        height, width = rgb.shape[:2]

        return cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)


class StereoFill:
    """Stereo where stereo has an answer; DA3, rescaled to stereo, elsewhere.

    Measured on this rig: stereo's median error is about 1%, DA3's about 8%,
    and DA3's scale is not even constant -- it drifts about 9% from the top
    of the picture to the bottom (where the floor is). One scale number
    cannot fix that, so the scale is a smooth surface over the picture,
        metres per DA3 unit = a + b*row + c*col + d*row^2
    fitted each frame to the pixels where both have a value. Outliers are
    dropped and the fit redone once.

    Returns (fused, da3_in_metres). `scale` holds the latest average, for
    printing.
    """

    def __init__(self, config):
        self.c = config
        self.scale = None
        self._coefficients = None

    def __call__(self, raw, stereo):
        height, width = raw.shape
        rows, cols = np.mgrid[0:height, 0:width]
        v = rows / height - 0.5
        u = cols / width - 0.5

        both = (
            np.isfinite(stereo)
            & (stereo >= self.c["near"])
            & (stereo <= self.c["far"])
            & (raw > 0)
        )

        if both.sum() >= self.c["min_pixels"]:
            picked = np.flatnonzero(both)[:: max(1, int(both.sum()) // 20000)]
            terms = np.stack([np.ones(len(picked)), v.ravel()[picked], u.ravel()[picked], v.ravel()[picked] ** 2], axis=1)
            ratio = stereo.ravel()[picked] / raw.ravel()[picked]

            fit, *_ = np.linalg.lstsq(terms, ratio, rcond=None)
            residual = np.abs(terms @ fit - ratio)
            keep = residual <= 3.0 * np.median(residual) + 1e-6
            fit, *_ = np.linalg.lstsq(terms[keep], ratio[keep], rcond=None)

            if self._coefficients is None:
                self._coefficients = fit
            else:
                self._coefficients = self._coefficients + self.c["smoothing"] * (fit - self._coefficients)

        if self._coefficients is None:
            return None, None

        a, b, c, d = self._coefficients
        scale = a + b * v + c * u + d * v ** 2
        self.scale = float(scale.mean())

        metric = (raw * scale).astype(np.float32)
        fused = np.where(np.isfinite(stereo), stereo, metric).astype(np.float32)

        return fused, metric

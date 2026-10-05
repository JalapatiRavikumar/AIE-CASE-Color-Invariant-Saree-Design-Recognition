"""Synthetic recoloring that preserves spatial structure / luminance layout."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


PALETTES = {
    "red": [(180, 20, 30), (220, 90, 70), (90, 10, 20), (240, 200, 160)],
    "blue": [(20, 40, 150), (70, 110, 210), (10, 20, 80), (180, 210, 240)],
    "green": [(20, 120, 50), (90, 180, 90), (10, 60, 30), (200, 230, 180)],
    "purple": [(90, 30, 140), (160, 90, 200), (40, 10, 70), (220, 190, 240)],
    "gold": [(180, 140, 40), (230, 190, 80), (90, 60, 20), (250, 230, 170)],
    "teal": [(10, 110, 120), (40, 170, 170), (5, 50, 60), (180, 230, 230)],
}


def _to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    s = np.where(maxc == 0, 0, (maxc - minc) / np.clip(maxc, 1e-8, None))
    rc = (maxc - r) / np.clip(maxc - minc, 1e-8, None)
    gc = (maxc - g) / np.clip(maxc - minc, 1e-8, None)
    bc = (maxc - b) / np.clip(maxc - minc, 1e-8, None)
    h = np.zeros_like(maxc)
    h = np.where((maxc == r) & (maxc != minc), (bc - gc), h)
    h = np.where((maxc == g) & (maxc != minc), 2.0 + (rc - bc), h)
    h = np.where((maxc == b) & (maxc != minc), 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0
    h = np.where(maxc == minc, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    rgb = np.zeros_like(hsv)
    masks = [i == k for k in range(6)]
    choices = [
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ]
    for mask, choice in zip(masks, choices):
        rgb[mask] = choice[mask]
    return rgb


def rgb_to_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Approximate luminance channel (not a full CIE LAB conversion)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


@dataclass
class RecolorEngine:
    """Structure-preserving palette transforms used as positive colorways."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def hue_shift(self, image: Image.Image, shift: float) -> Image.Image:
        rgb = _to_numpy(image)
        hsv = rgb_to_hsv(rgb)
        hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
        return _to_pil(hsv_to_rgb(hsv))

    def grayscale(self, image: Image.Image) -> Image.Image:
        rgb = _to_numpy(image)
        y = rgb_to_lab_l(rgb)
        return _to_pil(np.stack([y, y, y], axis=-1))

    def brightness(self, image: Image.Image, factor: float) -> Image.Image:
        rgb = np.clip(_to_numpy(image) * factor, 0.0, 1.0)
        return _to_pil(rgb)

    def contrast(self, image: Image.Image, factor: float) -> Image.Image:
        rgb = _to_numpy(image)
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        return _to_pil(np.clip((rgb - mean) * factor + mean, 0.0, 1.0))

    def channel_perturb(self, image: Image.Image, scale: float = 0.25) -> Image.Image:
        rgb = _to_numpy(image)
        gains = 1.0 + (self.rng.uniform(-scale, scale, size=3).astype(np.float32))
        return _to_pil(np.clip(rgb * gains, 0.0, 1.0))

    def palette_map(self, image: Image.Image, palette_name: str) -> Image.Image:
        """Map luminance bins onto a named palette while keeping edges."""
        palette = np.array(PALETTES[palette_name], dtype=np.float32) / 255.0
        rgb = _to_numpy(image)
        y = rgb_to_lab_l(rgb)
        edges = np.abs(np.gradient(y)[0]) + np.abs(np.gradient(y)[1])
        edges = np.clip(edges / (edges.max() + 1e-8), 0, 1)
        bins = np.linspace(0, 1, len(palette) + 1)
        out = np.zeros_like(rgb)
        for i, color in enumerate(palette):
            mask = (y >= bins[i]) & (y < bins[i + 1] if i < len(palette) - 1 else True)
            out[mask] = color
        # restore high-frequency structure from original luminance
        y_out = rgb_to_lab_l(out)
        scale = (y + 1e-3) / (y_out + 1e-3)
        out = np.clip(out * scale[..., None], 0, 1)
        out = out * (1.0 - 0.35 * edges[..., None]) + rgb * (0.35 * edges[..., None])
        return _to_pil(out)

    def apply_named(self, image: Image.Image, name: str) -> Image.Image:
        mapping = {
            "red": lambda im: self.palette_map(im, "red"),
            "blue": lambda im: self.palette_map(im, "blue"),
            "green": lambda im: self.palette_map(im, "green"),
            "purple": lambda im: self.palette_map(im, "purple"),
            "gold": lambda im: self.palette_map(im, "gold"),
            "grayscale": self.grayscale,
            "bright": lambda im: self.brightness(im, 1.35),
            "dark": lambda im: self.brightness(im, 0.7),
            "contrast": lambda im: self.contrast(im, 1.4),
            "hue": lambda im: self.hue_shift(im, 0.33),
        }
        if name not in mapping:
            raise KeyError(name)
        return mapping[name](image)

    def random_colorway(self, image: Image.Image) -> Image.Image:
        names = list(PALETTES.keys()) + ["grayscale", "hue", "bright", "contrast"]
        name = str(self.rng.choice(names))
        if name in PALETTES:
            return self.palette_map(image, name)
        return self.apply_named(image, "hue" if name == "hue" else name)

    def steal_palette(self, source: Image.Image, palette_donor: Image.Image) -> Image.Image:
        """Apply another image's coarse color distribution onto source structure."""
        src = _to_numpy(source)
        donor = _to_numpy(palette_donor.resize(source.size, Image.Resampling.BILINEAR))
        y = rgb_to_lab_l(src)
        hsv_d = rgb_to_hsv(donor)
        hsv = rgb_to_hsv(src)
        hsv[..., 0] = hsv_d[..., 0]
        hsv[..., 1] = 0.65 * hsv_d[..., 1] + 0.35 * hsv[..., 1]
        hsv[..., 2] = y
        return _to_pil(hsv_to_rgb(hsv))

"""Training and evaluation transforms, including color-invariant augmentations."""

from __future__ import annotations

import io
import random

from typing import Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms

from .preprocess import IMAGENET_MEAN, IMAGENET_STD
from .recolor import PALETTES, RecolorEngine

_engine = RecolorEngine()


class JPEGCompression:
    def __init__(self, min_q: int = 30, max_q: int = 90) -> None:
        self.min_q = min_q
        self.max_q = max_q

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.min_q, self.max_q)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, std: float = 8.0) -> None:
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0.0, self.std, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


class RandomRecolor:
    def __init__(self, p: float = 0.8) -> None:
        self.p = p
        self.engine = RecolorEngine()
        self.names = list(PALETTES.keys()) + ["grayscale", "hue", "bright", "dark", "contrast"]

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        name = random.choice(self.names)
        try:
            return self.engine.apply_named(img, name)
        except Exception:
            return img


class HSVJitter:
    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.2, 1.6))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
        if random.random() < 0.35:
            img = ImageOps.grayscale(img).convert("RGB")
        return img


class MildBlur:
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.4:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.4)))
        return img


def build_train_transform(image_size: int, use_recolor: bool = True, strong_color: bool = True) -> transforms.Compose:
    ops: list[Any] = [
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05), shear=4),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
        MildBlur(),
        GaussianNoise(std=5.0),
        JPEGCompression(),
    ]
    if strong_color:
        ops.append(HSVJitter())
    if use_recolor:
        ops.append(RandomRecolor(p=0.85))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def tensor_from_pil(image: Image.Image, image_size: int):
    return build_eval_transform(image_size)(image.convert("RGB"))

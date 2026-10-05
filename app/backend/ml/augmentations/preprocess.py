"""Training and evaluation transforms, including color-invariant augmentations."""

from __future__ import annotations

from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torchvision import transforms as T

from .recolor import RecolorEngine, rgb_to_hsv, hsv_to_rgb


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_image(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class RandomJPEG:
    def __init__(self, quality_min: int = 30, quality_max: int = 90, p: float = 0.3) -> None:
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        q = int(np.random.randint(self.quality_min, self.quality_max + 1))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, std: float = 0.04, p: float = 0.3) -> None:
        self.std = std
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, self.std, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomHSV:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        arr = np.asarray(image).astype(np.float32) / 255.0
        hsv = rgb_to_hsv(arr)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-0.2, 0.2)) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.4, 1.6), 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.7, 1.3), 0, 1)
        rgb = np.clip(hsv_to_rgb(hsv) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)


class RandomLABTint:
    def __init__(self, p: float = 0.4) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        arr = np.asarray(image).astype(np.float32)
        arr[..., 0] += np.random.uniform(-18, 18)
        arr[..., 1] += np.random.uniform(-18, 18)
        arr[..., 2] += np.random.uniform(-18, 18)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomRecolor:
    def __init__(self, p: float = 0.7, engine: RecolorEngine | None = None) -> None:
        self.p = p
        self.engine = engine or RecolorEngine()

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        return self.engine.random_colorway(image)


class RandomBlur:
    def __init__(self, p: float = 0.25) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.4, 1.6)))


class MaybeGrayscale:
    def __init__(self, p: float = 0.15) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return image
        return ImageOps.grayscale(image).convert("RGB")


def _geometric(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=4),
            T.RandomPerspective(distortion_scale=0.15, p=0.25),
        ]
    )


def train_transform(
    image_size: int = 128,
    color_mode: str = "full",
) -> Callable:
    """color_mode: none | standard | recolor | full."""
    geo = _geometric(image_size)
    pieces: list = [geo]
    if color_mode in {"standard", "recolor", "full"}:
        pieces.extend(
            [
                T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.5, hue=0.15),
                RandomHSV(p=0.45),
                RandomLABTint(p=0.3),
                MaybeGrayscale(p=0.12),
            ]
        )
    if color_mode in {"recolor", "full"}:
        pieces.append(RandomRecolor(p=0.75))
    pieces.extend(
        [
            RandomBlur(p=0.2),
            GaussianNoise(p=0.2),
            RandomJPEG(p=0.25),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return T.Compose(pieces)


def eval_transform(image_size: int = 128, grayscale: bool = False) -> T.Compose:
    ops: list = [
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
    ]
    if grayscale:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return T.Compose(ops)


def tensor_from_pil(image: Image.Image, image_size: int = 128, grayscale: bool = False) -> torch.Tensor:
    return eval_transform(image_size, grayscale=grayscale)(image.convert("RGB"))

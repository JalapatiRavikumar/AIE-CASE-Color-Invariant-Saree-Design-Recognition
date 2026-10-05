"""Handloom Saree Synthetic Dataset Generator.

Generates procedural images of 6 canonical Indian handloom weave motifs:
  1. Ikat        — diagonal bleeding-dye chevrons
  2. Jamdani     — scattered floral butas on a grid
  3. Banarasi    — interlocked brocade paisleys
  4. Patola      — geometric double-ikat diamonds
  5. Kanjivaram  — temple-border checks with zari lines
  6. Chanderi    — sheer diagonal dot grid with scattered stars

Each design is rendered in K_COLORWAYS different colour palettes so the
model must learn the motif, not the palette.

Usage:
    python scripts/generate_synthetic.py \
        --n-designs 60 --colorways 8 \
        --size 224 \
        --output backend/data/raw/synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Colour palettes per saree tradition ──────────────────────────────────────
TRADITION_PALETTES: dict[str, list[tuple[tuple[int,int,int], tuple[int,int,int]]]] = {
    "ikat": [
        ((220, 40, 40),  (240, 200, 100)),   # red-gold
        ((30,  60, 160), (220, 230, 250)),   # blue-white
        ((20, 110,  50), (210, 240, 190)),   # green-cream
        ((130, 30, 130), (240, 200, 240)),   # purple-lavender
        ((200, 130, 10), (255, 245, 180)),   # amber-yellow
        ((10,  110, 120),(180, 235, 235)),   # teal-cyan
        ((180, 50,  10), (240, 200, 160)),   # rust-peach
        ((5,   20,  80), (160, 190, 230)),   # navy-light
    ],
    "jamdani": [
        ((245, 245, 220), (180, 30,  30)),   # cream-red
        ((255, 250, 240), (30,  60, 160)),   # ivory-blue
        ((240, 255, 240), (20,  100, 40)),   # white-green
        ((255, 240, 245), (140, 20, 140)),   # blush-purple
        ((255, 250, 200), (180, 120, 10)),   # cream-gold
        ((230, 250, 255), (10,  110, 120)),  # sky-teal
        ((255, 235, 220), (180, 60,  10)),   # peach-rust
        ((240, 240, 255), (30,  30, 130)),   # lavender-navy
    ],
    "banarasi": [
        ((180, 10,  10), (220, 170, 30)),    # deep-red/gold
        ((10,  20,  90), (200, 160, 30)),    # navy/gold
        ((80,  10, 100), (210, 165, 30)),    # purple/gold
        ((10,  80,  20), (210, 170, 30)),    # bottle-green/gold
        ((120, 10,  10), (190, 140, 25)),    # maroon/gold
        ((30,  60, 140), (205, 160, 25)),    # royal-blue/gold
        ((140, 60,  10), (215, 170, 35)),    # brown/gold
        ((10, 100,  80), (205, 160, 30)),    # teal/gold
    ],
    "patola": [
        ((220, 40,  40), (250, 220, 100)),   # red/yellow
        ((30,  70, 180), (240, 210, 100)),   # blue/yellow
        ((180, 20, 120), (250, 220, 100)),   # magenta/yellow
        ((20, 120,  50), (250, 220, 100)),   # green/yellow
        ((200, 100, 10), (240, 220,  80)),   # orange/yellow
        ((80,  10, 140), (250, 220, 100)),   # violet/yellow
        ((10, 100, 130), (250, 220, 100)),   # cyan/yellow
        ((160, 10,  30), (250, 220, 100)),   # crimson/yellow
    ],
    "kanjivaram": [
        ((160, 10,  10), (210, 160, 25)),    # red/gold
        ((10,  10, 100), (210, 160, 25)),    # dark-blue/gold
        ((80,  10, 100), (210, 160, 25)),    # purple/gold
        ((10,  80,  20), (210, 160, 25)),    # green/gold
        ((120, 10,  10), (185, 140, 20)),    # maroon/gold
        ((10,  60, 140), (200, 155, 20)),    # royal/gold
        ((140, 55,  10), (210, 160, 25)),    # brown/gold
        ((10, 100,  80), (200, 155, 20)),    # teal/gold
    ],
    "chanderi": [
        ((245, 245, 255), (180, 30,  30)),   # white/red
        ((245, 255, 245), (30,  60, 160)),   # white/blue
        ((255, 245, 235), (140, 10, 140)),   # white/purple
        ((255, 250, 235), (180, 130, 10)),   # white/gold
        ((235, 250, 255), (10, 110, 120)),   # white/teal
        ((255, 235, 240), (180, 50,  10)),   # white/rust
        ((240, 240, 255), (30,  30, 120)),   # white/navy
        ((255, 245, 240), (150, 50,  10)),   # white/terracotta
    ],
}

TRADITIONS = list(TRADITION_PALETTES.keys())


# ── Per-tradition drawing functions ──────────────────────────────────────────

def draw_ikat(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
              bg: tuple, fg: tuple) -> None:
    """Diagonal bleeding-dye chevron bands."""
    band = rng.randint(sz // 10, sz // 6)
    for y in range(-sz, sz * 2, band * 2):
        pts = []
        bleed = rng.randint(2, 8)
        for x in range(0, sz + 1, 4):
            pts.append((x, y + int(math.sin(x * 0.08) * bleed)))
        pts += [(sz, y + band), (0, y + band)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=fg)
    # Horizontal accent lines
    for y in range(0, sz, band * 2):
        draw.line([(0, y), (sz, y)], fill=fg, width=2)


def draw_jamdani(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
                 bg: tuple, fg: tuple) -> None:
    """Scattered floral butas (small motifs) on a sheer ground."""
    spacing = rng.randint(sz // 8, sz // 5)
    for gy in range(spacing // 2, sz, spacing):
        for gx in range(spacing // 2, sz, spacing):
            if rng.random() > 0.25:
                r = rng.randint(4, sz // 16)
                # Petal cluster
                n_petals = rng.randint(4, 6)
                for p in range(n_petals):
                    angle = 2 * math.pi * p / n_petals
                    px = gx + int(r * math.cos(angle))
                    py = gy + int(r * math.sin(angle))
                    draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=fg)
                draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=fg)


def draw_banarasi(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
                  bg: tuple, fg: tuple) -> None:
    """Interlocked paisley (mango) brocade motifs."""
    cell = rng.randint(sz // 6, sz // 4)
    for gy in range(0, sz + cell, cell):
        offset = (cell // 2) if ((gy // cell) % 2 == 1) else 0
        for gx in range(-offset, sz + cell, cell):
            # Paisley = elongated drop
            hl = rng.randint(cell // 3, cell // 2)
            hw = rng.randint(cell // 6, cell // 4)
            angle = rng.choice([0, 30, -30, 45, -45])
            rad = math.radians(angle)
            pts = []
            for i in range(20):
                t = 2 * math.pi * i / 20
                ex = hw * math.cos(t)
                ey = hl * (math.sin(t) - 0.4 * math.sin(2 * t))
                rx = gx + ex * math.cos(rad) - ey * math.sin(rad)
                ry = gy + ex * math.sin(rad) + ey * math.cos(rad)
                pts.append((rx, ry))
            if len(pts) >= 3:
                draw.polygon(pts, fill=fg, outline=bg)
            # Inner accent dot
            draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=bg)


def draw_patola(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
                bg: tuple, fg: tuple) -> None:
    """Geometric double-ikat diamond grid."""
    cell = rng.randint(sz // 8, sz // 5)
    for gy in range(-cell, sz + cell, cell):
        for gx in range(-cell, sz + cell, cell):
            half = cell // 2
            pts = [(gx, gy - half), (gx + half, gy),
                   (gx, gy + half), (gx - half, gy)]
            draw.polygon(pts, fill=fg, outline=bg)
            # Small inner diamond
            q = half // 3
            inner = [(gx, gy - q), (gx + q, gy),
                     (gx, gy + q), (gx - q, gy)]
            draw.polygon(inner, fill=bg)


def draw_kanjivaram(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
                    bg: tuple, fg: tuple) -> None:
    """Temple border: alternating solid blocks + fine zari (gold) lines."""
    block = rng.randint(sz // 8, sz // 5)
    # Fill alternating checks
    toggle = False
    for gy in range(0, sz, block):
        for gx in range(0, sz, block):
            color = fg if toggle else bg
            draw.rectangle([gx, gy, gx + block, gy + block], fill=color)
            toggle = not toggle
        toggle = not toggle
    # Overlay fine zari lines (horizontal)
    line_spacing = max(4, block // 4)
    for y in range(0, sz, line_spacing):
        draw.line([(0, y), (sz, y)], fill=fg, width=1)
    # Vertical borders
    border_w = block // 2
    draw.rectangle([0, 0, border_w, sz], fill=fg)
    draw.rectangle([sz - border_w, 0, sz, sz], fill=fg)


def draw_chanderi(draw: ImageDraw.ImageDraw, sz: int, rng: random.Random,
                  bg: tuple, fg: tuple) -> None:
    """Sheer diagonal dot grid with scattered star motifs."""
    spacing = rng.randint(sz // 12, sz // 7)
    # Diagonal dots
    for gy in range(-spacing, sz + spacing, spacing):
        for gx in range(-spacing, sz + spacing, spacing):
            r = rng.randint(1, 3)
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=fg)
    # Scattered 6-point stars
    n_stars = rng.randint(6, 16)
    for _ in range(n_stars):
        cx, cy = rng.randint(0, sz), rng.randint(0, sz)
        r_out = rng.randint(8, 20)
        r_in = r_out // 2
        pts = []
        for i in range(12):
            angle = math.pi * i / 6 - math.pi / 2
            r = r_out if i % 2 == 0 else r_in
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=fg, outline=bg)


DRAW_FN = {
    "ikat":       draw_ikat,
    "jamdani":    draw_jamdani,
    "banarasi":   draw_banarasi,
    "patola":     draw_patola,
    "kanjivaram": draw_kanjivaram,
    "chanderi":   draw_chanderi,
}


# ── Generator ─────────────────────────────────────────────────────────────────

def make_design(tradition: str, design_idx: int, size: int,
                palette_idx: int) -> Image.Image:
    """Render one design × one palette → PIL Image."""
    bg, fg = TRADITION_PALETTES[tradition][palette_idx % len(TRADITION_PALETTES[tradition])]
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    rng = random.Random(design_idx * 31337)   # deterministic per design
    DRAW_FN[tradition](draw, size, rng, bg, fg)
    # Mild Gaussian blur to soften procedural edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Generate handloom saree synthetic images")
    p.add_argument("--n-designs", type=int, default=60,
                   help="Designs per tradition (total = n_designs × 6 traditions)")
    p.add_argument("--colorways", type=int, default=8,
                   help="Colour palettes per design (max 8)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--output", type=str, default="backend/data/raw/synthetic")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    k = min(args.colorways, 8)
    total = 0

    for trad in TRADITIONS:
        palettes = TRADITION_PALETTES[trad]
        for d in range(args.n_designs):
            design_id = f"{trad}_{d:04d}"
            design_dir = output / design_id
            design_dir.mkdir(exist_ok=True)

            chosen = list(range(min(k, len(palettes))))
            for pi in chosen:
                bg, fg = palettes[pi]
                img = make_design(trad, d, args.size, pi)
                img_path = design_dir / f"palette_{pi:02d}.png"
                img.save(str(img_path))

                sidecar = {
                    "design_id": design_id,
                    "style_label": f"{trad}_palette{pi}",
                    "tradition": trad,
                    "source": "synthetic_handloom",
                    "license": "CC0",
                    "attribution": "Procedurally generated handloom saree pattern",
                    "bg_color": list(bg),
                    "fg_color": list(fg),
                }
                img_path.with_suffix(".json").write_text(
                    json.dumps(sidecar, indent=2), encoding="utf-8"
                )
                total += 1

    print(f"\nGenerated {total} images")
    print(f"   {len(TRADITIONS)} traditions x {args.n_designs} designs x {k} colorways")
    print(f"   Traditions: {', '.join(TRADITIONS)}")
    print(f"   Output: {output.resolve()}")


if __name__ == "__main__":
    main()

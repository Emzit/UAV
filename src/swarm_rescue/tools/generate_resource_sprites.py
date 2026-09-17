"""
Generate bomb.png and disposal_center.png for swarm_rescue resources.

Run from repo root:
  .venv/bin/python -m swarm_rescue.tools.generate_resource_sprites
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

RESOURCES = Path(__file__).resolve().parent.parent / "resources"


def _draw_bomb(size: int = 24) -> Image.Image:
    """Round bomb with fuse — readable at 24px (game radius 12)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2 + 1
    r = size // 2 - 3

    # Fuse
    fuse_top = 2
    draw.line([(cx, fuse_top), (cx, cy - r + 2)], fill=(90, 70, 50, 255), width=2)
    draw.ellipse([cx - 2, fuse_top - 1, cx + 2, fuse_top + 3], fill=(255, 180, 40, 255))

    # Body
    body_box = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(body_box, fill=(35, 35, 40, 255))
    draw.arc(body_box, start=200, end=340, fill=(70, 70, 78, 255), width=2)

    # Hazard stripe (map tooling still uses yellow detection)
    stripe_y = cy - 2
    draw.rectangle([cx - r + 3, stripe_y - 2, cx + r - 3, stripe_y + 2], fill=(179, 143, 0, 255))

    # Highlight
    draw.ellipse([cx - r // 2, cy - r // 2, cx - 2, cy - 2], fill=(90, 90, 100, 120))

    # Outline
    draw.ellipse(body_box, outline=(20, 20, 24, 255), width=1)
    return img


def _draw_disposal_center(size: int = 800) -> Image.Image:
    """
    Atlas texture cropped from center by DisposalCenter (800x800 source).
    Red field + central disposal / EOD emblem (not medical cross).
    """
    img = Image.new("RGBA", (size, size), (255, 64, 64, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    # Inner safe zone panel
    panel_r = 72
    draw.ellipse(
        [cx - panel_r, cy - panel_r, cx + panel_r, cy + panel_r],
        fill=(245, 245, 240, 255),
        outline=(180, 30, 30, 255),
        width=4,
    )

    # Containment octagon (disposal pit)
    pit_r = 48
    points = []
    for i in range(8):
        ang = math.pi / 8 + i * math.pi / 4
        points.append(
            (cx + pit_r * math.cos(ang), cy + 6 + pit_r * math.sin(ang))
        )
    draw.polygon(points, fill=(60, 60, 68, 255), outline=(40, 40, 48, 255))

    # Down arrow (deliver bomb here)
    arrow_y = cy - 18
    draw.polygon(
        [
            (cx, arrow_y + 22),
            (cx - 16, arrow_y),
            (cx - 6, arrow_y),
            (cx - 6, arrow_y - 14),
            (cx + 6, arrow_y - 14),
            (cx + 6, arrow_y),
            (cx + 16, arrow_y),
        ],
        fill=(255, 200, 50, 255),
    )

    # Small bomb silhouette at bottom of pit
    bx, by = cx, cy + 22
    br = 14
    draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(30, 30, 35, 255))
    draw.rectangle([bx - br + 3, by - 3, bx + br - 3, by + 1], fill=(179, 143, 0, 255))
    draw.line([(bx, by - br - 6), (bx, by - br + 2)], fill=(80, 60, 45, 255), width=3)

    # Corner hazard ticks on red field
    tick_len = 40
    for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        tx = cx + ox * (panel_r + 28)
        ty = cy + oy * (panel_r + 28)
        draw.line([(tx, ty), (tx + ox * tick_len, ty)], fill=(255, 220, 80, 255), width=5)
        draw.line([(tx, ty), (tx, ty + oy * tick_len)], fill=(255, 220, 80, 255), width=5)

    return img


def main() -> None:
    bomb_path = RESOURCES / "bomb.png"
    disposal_path = RESOURCES / "disposal_center.png"

    _draw_bomb(24).save(bomb_path, "PNG")
    _draw_disposal_center(800).save(disposal_path, "PNG")
    print(f"Wrote {bomb_path}")
    print(f"Wrote {disposal_path}")


if __name__ == "__main__":
    main()

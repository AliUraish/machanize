"""Compose synchronized Front and Wrist images for cloud monitoring."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def compose_front_wrist_jpeg(
    front: Any,
    wrist: Any,
    *,
    sample_id: int,
    timestamp: str,
    size: tuple[int, int] = (480, 360),
    quality: int = 82,
) -> bytes:
    """Return one labeled side-by-side JPEG with a correlation identifier."""

    front_image = ImageOps.fit(_to_image(front), size)
    wrist_image = ImageOps.fit(_to_image(wrist), size)
    canvas = Image.new("RGB", (size[0] * 2, size[1] + 34), "#080a0c")
    canvas.paste(front_image, (0, 24))
    canvas.paste(wrist_image, (size[0], 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), "FRONT", fill="#72f1b8")
    draw.text((size[0] + 10, 7), "WRIST", fill="#72f1b8")
    draw.text((150, 7), f"sample={sample_id}  {timestamp}", fill="white")
    output = BytesIO()
    canvas.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array).convert("RGB")

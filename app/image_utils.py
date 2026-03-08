from __future__ import annotations

import base64
import io

from PIL import Image

from app.errors import INVALID_IMAGE, SAMError


def decode_base64_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise SAMError(INVALID_IMAGE, f"Could not decode image: {exc}")


def encode_image_base64(image: Image.Image, *, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

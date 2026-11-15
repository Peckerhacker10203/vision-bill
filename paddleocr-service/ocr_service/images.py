"""Derived RGB inputs and deliberate PNG views of native output coordinates."""

import base64
import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps

from vision_bill.model.ocr import WorkingImage


@dataclass(frozen=True)
class NormalizedSource:
    path: Path
    transforms: list[dict[str, Any]]
    findings: list[str]


@contextmanager
def normalize_source(source: Path, source_sha256: str) -> Iterator[NormalizedSource]:
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != source_sha256:
        raise ValueError("Original source hash does not match request")
    # Registers HEIF support without importing any inference framework.
    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    with Image.open(BytesIO(original)) as image:
        frame_count = getattr(image, "n_frames", 1)
        image.seek(0)
        orientation = image.getexif().get(274, 1)
        transforms = [
            {
                "operation": "normalize_rgb",
                "exif_orientation": orientation,
                "source_format": image.format,
                "selected_frame": 0,
                "frame_count": frame_count,
            }
        ]
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        with tempfile.TemporaryDirectory(prefix="vision-bill-ocr-") as directory:
            path = Path(directory) / "input.png"
            rgb.save(path, format="PNG")
            yield NormalizedSource(
                path, transforms, ["multiframe_first_frame_only"] if frame_count > 1 else []
            )


def working_image(preprocessor: Any, transforms: list[dict[str, Any]]) -> WorkingImage:
    import numpy as np

    pixels = preprocessor.get("output_img")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
    ):
        raise ValueError("DocPreprocessor output_img must be an actual uint8 BGR image")
    image = Image.fromarray(pixels[:, :, ::-1].copy(), "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    angle = preprocessor.get("angle")
    if angle not in (0, 90, 180, 270):
        raise ValueError("Enabled orientation model must report an actual angle")
    return WorkingImage(
        view_id=uuid4(),
        width=image.width,
        height=image.height,
        sha256=hashlib.sha256(png).hexdigest(),
        png_base64=base64.b64encode(png).decode("ascii"),
        transforms=[
            *transforms,
            {
                "operation": "doc_preprocessor",
                "angle": int(angle),
                "orientation_model": "PP-LCNet_x1_0_doc_ori",
                "unwarping_model": "UVDoc",
                "coordinate_space": "working",
                "inverse_mapping": None,
            },
        ],
        inverse_mapping=None,
    )

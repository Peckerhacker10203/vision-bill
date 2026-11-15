"""Preserve uploaded bytes while deriving RGB inputs and working-space PNG views."""

import base64
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
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
    uploaded_bytes = source.read_bytes()
    actual_sha256 = hashlib.sha256(uploaded_bytes).hexdigest()
    if actual_sha256 != source_sha256:
        raise ValueError("Uploaded source hash differs from the supplied identity")

    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    with TemporaryDirectory(prefix="vision-bill-ocr-") as temporary:
        derived_path = Path(temporary, "input.png")
        with Image.open(BytesIO(uploaded_bytes)) as decoded:
            frame_count = getattr(decoded, "n_frames", 1)
            decoded.seek(0)
            transform = {
                "operation": "normalize_rgb",
                "source_format": decoded.format,
                "exif_orientation": decoded.getexif().get(274, 1),
                "selected_frame": 0,
                "frame_count": frame_count,
            }
            ImageOps.exif_transpose(decoded).convert("RGB").save(derived_path, format="PNG")
        findings = ["multiframe_first_frame_only"] if frame_count > 1 else []
        yield NormalizedSource(derived_path, [transform], findings)


def working_image(preprocessor: Any, transforms: list[dict[str, Any]]) -> WorkingImage:
    import numpy as np

    output = preprocessor.get("output_img")
    if (
        not isinstance(output, np.ndarray)
        or output.ndim != 3
        or output.shape[2] != 3
        or output.dtype != np.uint8
    ):
        raise ValueError("DocPreprocessor output_img must contain uint8 BGR pixels")
    angle = preprocessor.get("angle")
    if angle not in {0, 90, 180, 270}:
        raise ValueError("Enabled document orientation requires a recognized rotation angle")
    rgb = Image.fromarray(np.ascontiguousarray(output[..., ::-1]))
    encoded = BytesIO()
    rgb.save(encoded, format="PNG")
    png_bytes = encoded.getvalue()
    preprocessing_transform = {
        "operation": "doc_preprocessor",
        "angle": int(angle),
        "orientation_model": "PP-LCNet_x1_0_doc_ori",
        "unwarping_model": "UVDoc",
        "coordinate_space": "working",
        "inverse_mapping": None,
    }
    return WorkingImage(
        view_id=uuid4(),
        width=rgb.width,
        height=rgb.height,
        sha256=hashlib.sha256(png_bytes).hexdigest(),
        png_base64=base64.b64encode(png_bytes).decode("ascii"),
        transforms=transforms + [preprocessing_transform],
        inverse_mapping=None,
    )

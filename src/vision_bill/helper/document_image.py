"""Visible document pixels over an explicit white page, after orientation."""

from PIL import Image

NORMALIZER_VERSION = "white-document-rgb-v2"


def document_rgb(image: Image.Image) -> Image.Image:
    """Composite RGBA, LA and palette/color-key transparency without losing ink."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")

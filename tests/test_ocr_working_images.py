import base64
import hashlib
from io import BytesIO

import pytest
from PIL import Image


def test_normalization_preserves_source_and_rotates_exif(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "photo.jpg"
    image = Image.new("RGB", (9, 5), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    original = source.read_bytes()
    with normalize_source(source, hashlib.sha256(original).hexdigest()) as normalized:
        assert Image.open(normalized.path).size == (5, 9)
        assert Image.open(normalized.path).mode == "RGB"
        assert normalized.transforms[0]["exif_orientation"] == 6
        derived = normalized.path
    assert source.read_bytes() == original
    assert not derived.exists()


def test_multiframe_gif_reports_first_frame(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "photo.gif"
    Image.new("RGB", (4, 7), "red").save(
        source, save_all=True, append_images=[Image.new("RGB", (4, 7), "blue")]
    )
    with normalize_source(source, hashlib.sha256(source.read_bytes()).hexdigest()) as normalized:
        assert normalized.transforms[0]["selected_frame"] == 0
        assert normalized.transforms[0]["frame_count"] == 2
        assert normalized.findings == ["multiframe_first_frame_only"]
        assert Image.open(normalized.path).convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_source_hash_mismatch_rejected_before_decode(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "bad.png"
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="source hash"), normalize_source(source, "0" * 64):
        pass


def test_actual_preprocessor_bgr_output_becomes_rgb_png():
    import numpy as np
    from ocr_service.images import working_image

    pixels = np.zeros((8, 3, 3), dtype=np.uint8)
    pixels[:, :, 2] = 255
    view = working_image({"output_img": pixels, "angle": 90}, [{"input": "RGB"}])
    image = Image.open(BytesIO(base64.b64decode(view.png_base64)))
    assert image.size == (view.width, view.height) == (3, 8)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert view.sha256 == hashlib.sha256(base64.b64decode(view.png_base64)).hexdigest()
    assert view.inverse_mapping is None
    assert view.transforms[-1] == {
        "operation": "doc_preprocessor",
        "angle": 90,
        "orientation_model": "PP-LCNet_x1_0_doc_ori",
        "unwarping_model": "UVDoc",
        "coordinate_space": "working",
        "inverse_mapping": None,
    }


def test_working_image_requires_actual_native_output():
    from ocr_service.images import working_image

    with pytest.raises(ValueError, match="output_img"):
        working_image({}, [])


def test_heif_source_uses_actual_decoder_and_preserves_bytes(tmp_path):
    from ocr_service.images import normalize_source
    from pillow_heif import from_pillow

    source = tmp_path / "photo.heic"
    from_pillow(Image.new("RGB", (12, 8), "white")).save(source)
    original = source.read_bytes()
    with normalize_source(source, hashlib.sha256(original).hexdigest()) as normalized:
        with Image.open(normalized.path) as derived:
            assert derived.mode == "RGB"
            assert derived.size == (12, 8)
        assert normalized.transforms[0]["source_format"] == "HEIF"
    assert source.read_bytes() == original


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
@pytest.mark.parametrize("native", [True, False])
def test_transparent_document_preserves_black_ink_on_white_background(tmp_path, mode, native):
    from ocr_service.images import normalize_source
    from vision_bill.model.extraction import ReceiptInput
    from vision_bill.provider.llm.ollama import OllamaProvider

    source = tmp_path / "transparent.png"
    if mode == "RGBA":
        image = Image.new(mode, (2, 1), (0, 0, 0, 0))
        image.putpixel((0, 0), (0, 0, 0, 255))
    elif mode == "LA":
        image = Image.new(mode, (2, 1), (0, 0))
        image.putpixel((0, 0), (0, 255))
    else:
        image = Image.new(mode, (2, 1), 0)
        image.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)
        image.info["transparency"] = 0
        image.putpixel((0, 0), 1)
    image.save(source)
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if native:
        with normalize_source(source, digest) as normalized:
            derived = normalized.path.read_bytes()
            transforms = normalized.transforms
    else:
        _, _, views = OllamaProvider("http://local")._prepare_receipt_input(
            ReceiptInput(mode="direct_vision", image_paths=[str(source)])
        )
        view = views[0]
        derived = base64.b64decode(view.working_image.png_base64)
        transforms = view.working_image.transforms
        assert view.source_sha256 == digest
        assert view.working_image.sha256 == hashlib.sha256(derived).hexdigest()
        assert view.working_image.inverse_mapping is None
    with Image.open(BytesIO(derived)) as rgb:
        assert rgb.size == (2, 1)
        assert [rgb.getpixel((x, 0)) for x in range(2)] == [(0, 0, 0), (255, 255, 255)]
    assert transforms[0]["alpha_background"] == "#ffffff"
    assert source.read_bytes() == original

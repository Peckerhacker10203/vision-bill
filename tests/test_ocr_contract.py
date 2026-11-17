"""The native source and app wheel each carry the same portable OCR contract."""

import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "paddleocr-service"


def test_native_source_imports_without_application(tmp_path):
    isolated = tmp_path / "native"
    shutil.copytree(SERVICE, isolated, ignore=shutil.ignore_patterns("__pycache__"))
    script = """
        import importlib.abc
        import sys
        from pathlib import Path
        class NoApplication(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'vision_bill' or fullname.startswith('vision_bill.'):
                    raise ImportError('native OCR must not import the app')
        sys.meta_path.insert(0, NoApplication())
        sys.path.insert(0, sys.argv[1])
        from ocr_contract import OcrResponse, SourceRef
        from ocr_contract.document_image import NORMALIZER_VERSION, document_rgb
        from ocr_service import api, images, pipeline, qualify, worker
        from ocr_service.profile import content_identity, effective_pipeline_config, load_profile
        from PIL import Image
        assert content_identity(OcrResponse.model_json_schema()) == '86aa8a10686ac01547bba6ef80d85bc0a39d193e6911b4df99d7145ff178701b'
        assert content_identity(SourceRef.model_json_schema()) == '6473c0469cb3b365dca0629f7d2f1202f12a4d6ebaf5cd0c1fa1c5c417210b00'
        config = effective_pipeline_config(load_profile(Path(sys.argv[1]) / 'profile.json'))
        assert content_identity(config) == 'ea5ead7b3e204670985765cd359511db31c077b5308cc1c22e5c083dbdb7239f'
        assert NORMALIZER_VERSION == 'white-document-rgb-v2'
        assert document_rgb(Image.new('RGBA', (1, 1), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
        assert not any(name == 'vision_bill' or name.startswith('vision_bill.') for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), str(isolated)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_app_imports_reexport_canonical_contract_and_normalizer():
    import ocr_contract
    from ocr_contract import document_image
    from vision_bill.helper import document_image as app_image
    from vision_bill.model import ocr

    for name in ocr_contract.__all__:
        assert getattr(ocr, name) is getattr(ocr_contract, name)
    assert app_image.document_rgb is document_image.document_rgb
    assert app_image.NORMALIZER_VERSION == document_image.NORMALIZER_VERSION


def test_app_wheel_installs_neutral_contract(tmp_path):
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "ocr_contract/__init__.py" in archive.namelist()
        assert "ocr_contract/document_image.py" in archive.namelist()
        archive.extractall(tmp_path / "installed")
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent("""
            import json, sys
            sys.path.insert(0, sys.argv[1])
            from ocr_contract import OcrResponse
            from vision_bill.model.ocr import OcrResponse as AppResponse
            assert OcrResponse is AppResponse
            print(json.dumps({'module': OcrResponse.__module__}))
        """), str(tmp_path / "installed")],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["module"].startswith("ocr_contract")

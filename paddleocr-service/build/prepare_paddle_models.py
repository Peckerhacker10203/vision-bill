"""Prepare the four pinned official Paddle model archives for an OCR release."""

import argparse
import json
import os
import re
import tarfile
from http.client import HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, NamedTuple
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ocr_service.provision import (
    Archive,
    FileRecord,
    content_identity,
    file_digest,
    publish_directory,
)

MIB = 1024**2
APACHE_LICENSE_HEADER = b"Apache License\nVersion 2.0, January 2004"
COMPANION_LICENSE_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/"
    "ffb64904d23708863ff5b8da312a5cbd52a7f462/LICENSE"
)
COMPANION_LICENSE_SHA256 = "accc36c817ac5ede473d05732e14afc11cb6e55ef5919bc826b653f631165043"
STATIC_FILES = ("README.md", "inference.json", "inference.pdiparams", "inference.yml")


class Model(NamedTuple):
    repository: str
    revision: str
    files: tuple[str, ...]
    weights: str
    weight_sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/PaddlePaddle/{self.repository}"

    def resolve_url(self, filename: str) -> str:
        return f"{self.url}/resolve/{self.revision}/{quote(filename)}"


MODELS = {
    "PP-DocLayoutV3": Model(
        "PP-DocLayoutV3",
        "241f8bdfc77a7c7bee915a5057aaee58c235a8d3",
        STATIC_FILES,
        "inference.pdiparams",
        "70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137",
    ),
    "PP-LCNet_x1_0_doc_ori": Model(
        "PP-LCNet_x1_0_doc_ori",
        "d3b95a6dff5fe8a94f2748e12b61cb26818a0df8",
        (*STATIC_FILES, "config.json"),
        "inference.pdiparams",
        "e8d6e7c5d264507e40e58a655779059d616b20d7441ea22047d829eb3931989c",
    ),
    "UVDoc": Model(
        "UVDoc",
        "16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3",
        (*STATIC_FILES, "config.json"),
        "inference.pdiparams",
        "810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca",
    ),
    "PaddleOCR-VL-1.6-0.9B": Model(
        "PaddleOCR-VL-1.6",
        "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd",
        (
            "LICENSE",
            "README.md",
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "inference.yml",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
        "model.safetensors",
        "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
    ),
}


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Official asset redirects must preserve HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, destination: Path, limit: int) -> None:
    opener = build_opener(HTTPSOnlyRedirect())
    with opener.open(url, timeout=60) as response:
        if response.status != 200 or "Content-Range" in response.headers:
            raise ValueError(f"Expected a complete HTTP 200 response for {destination.name}")
        content_lengths = response.headers.get_all("Content-Length", [])
        expected = None
        if content_lengths:
            if len(content_lengths) != 1 or not re.fullmatch(r"[0-9]+", content_lengths[0]):
                raise ValueError(
                    f"Official asset has invalid Content-Length for {destination.name}"
                )
            digits = content_lengths[0].lstrip("0") or "0"
            if len(digits) > len(str(limit)) or int(digits) > limit:
                raise ValueError(f"Official asset exceeds size bound for {destination.name}")
            expected = int(digits)

        remaining = limit
        with destination.open("xb") as output:
            while chunk := response.read(min(MIB, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError(f"Official asset exceeds size bound for {destination.name}")
                output.write(chunk)
        received = limit - remaining
        if expected is not None and received != expected:
            raise ValueError(f"Official asset has unexpected length for {destination.name}")
        if not received:
            raise ValueError(f"Official asset is empty for {destination.name}")


def prepare_model(model: Model, tree: Path, archive: Path, source: str) -> Archive:
    for filename in model.files:
        limit = 3 * 1024 * MIB if filename == model.weights else 32 * MIB
        download(model.resolve_url(filename), tree / filename, limit)
    weight_digest = file_digest(tree / model.weights)
    if weight_digest != model.weight_sha256:
        raise ValueError(f"Weight SHA-256 mismatch for {model.repository}")

    license_path = tree / "LICENSE"
    if "LICENSE" not in model.files:
        readme = (tree / "README.md").read_text(encoding="utf-8")
        frontmatter, delimiter, _ = readme.partition("\n---\n")
        if (
            not delimiter
            or not frontmatter.startswith("---\n")
            or not re.search(r"^license: apache-2\.0\s*$", frontmatter, re.MULTILINE)
        ):
            raise ValueError(
                f"Official Apache-2.0 model-card metadata missing for {model.repository}"
            )
        download(COMPANION_LICENSE_URL, license_path, MIB)
        if file_digest(license_path) != COMPANION_LICENSE_SHA256:
            raise ValueError("PaddleX companion license SHA-256 mismatch")
        (tree / "LICENSE-PROVENANCE.txt").write_text(
            f"Model license declaration: {model.resolve_url('README.md')}\n"
            f"Apache-2.0 text: {COMPANION_LICENSE_URL}\n"
            f"Apache-2.0 text SHA-256: {COMPANION_LICENSE_SHA256}\n",
            encoding="utf-8",
        )
    license_lines = license_path.read_bytes().lstrip().splitlines()[:2]
    if b"\n".join(line.strip() for line in license_lines) != APACHE_LICENSE_HEADER:
        raise ValueError(f"Apache-2.0 license text missing for {model.repository}")

    files = {}
    with tarfile.open(archive, mode="x", format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(tree.iterdir()):
            member = tarfile.TarInfo(file.name)
            member.size = file.stat().st_size
            member.mode = 0o644
            digest = weight_digest if file.name == model.weights else file_digest(file)
            with file.open("rb") as stream:
                tar.addfile(member, stream)
            files[file.name] = FileRecord(sha256=digest, size_bytes=member.size)

    return Archive.model_validate(
        {
            "source": source,
            "sha256": file_digest(archive),
            "size_bytes": archive.stat().st_size,
            "unpacked_bytes": sum(row.size_bytes for row in files.values()),
            "files": files,
            "tree_sha256": content_identity({name: row.sha256 for name, row in files.items()}),
            "license": {
                "spdx": "Apache-2.0",
                "text_file": "LICENSE",
                "redistribution_allowed": True,
            },
            "provenance": {"url": model.url, "revision": model.revision},
        }
    )


def prepare(output: Path, archive_base_url: str) -> None:
    base = urlsplit(archive_base_url)
    if (
        base.scheme != "https"
        or not base.hostname
        or base.username is not None
        or base.query
        or base.fragment
        or "?" in archive_base_url
        or "#" in archive_base_url
        or "\\" in archive_base_url
        or any(character.isspace() for character in archive_base_url)
    ):
        raise ValueError("Provide an HTTPS archive base URL with no credentials, query or fragment")
    if os.path.lexists(output):
        raise FileExistsError(f"Output directory already exists: {output}")

    with TemporaryDirectory(prefix=".paddle-models-", dir=output.parent) as scratch:
        staged = Path(scratch) / "output"
        staged.mkdir()
        records = {}
        for name, model in MODELS.items():
            with TemporaryDirectory(dir=scratch) as tree:
                filename = f"{name}.tar"
                record = prepare_model(
                    model,
                    Path(tree),
                    staged / filename,
                    f"{archive_base_url.rstrip('/')}/{filename}",
                )
                records[name] = record.model_dump(mode="json")
        (staged / "models.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        publish_directory(staged, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local output directory")
    parser.add_argument(
        "--archive-base-url",
        required=True,
        help="approved HTTPS destination for the generated archives (no upload is performed)",
    )
    args = parser.parse_args()
    prepare(args.output, args.archive_base_url)
    print(f"Prepared local model archives and {args.output / 'models.json'}.")


if __name__ == "__main__":
    main()

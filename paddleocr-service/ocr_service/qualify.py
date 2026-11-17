"""Executable native release gate. Run in the built image on physical hardware.

No framework import occurs in the coordinator. Inspection, collection and replay
run in separate bounded subprocesses; stdout/stderr never expose fixture content.
"""

import argparse
import importlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import (
    HookCollector,
    common_coverage,
    inspect_artifact,
    parameter_bindings,
    seal_dependencies,
)
from .device_inventory import inspect_device_identity
from .fp32 import audit_dynamic_model, floating_dtype, require_gpu
from .images import normalize_source
from .pipeline import PipelineAdapter, build_pipeline, construct_native, tree_identity
from .precision import ParameterRecord, StaticArtifact
from .profile import (
    MODEL_NAMES,
    RuntimeSettings,
    content_identity,
    file_digest,
    load_profile,
    verify_dependencies,
)
from .supervision import qualification_is_supervised
from .targets import validate_profile

PROFILE = Path(__file__).resolve().parents[1] / "profile.json"
KINDS = {"upright", "rotated", "warped", "table", "formula"}


class QualificationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["upright", "rotated", "warped", "table", "formula"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Require recognized structure for table/formula cases, without logging text.
    required_labels: list[str] = Field(default_factory=list)


class QualificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qualified: bool = False
    accelerator: str
    image_identity: str
    device: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    model_content: dict[str, str] = Field(default_factory=dict)
    profile_id: str | None = None
    runtime_fingerprint: str | None = None
    manifest_sha256: str | None = None
    cases: list[dict[str, Any]] = Field(default_factory=list)
    qualified_targets: list[str] = Field(default_factory=list)
    failure: str | None = None
    stage: str = "inputs"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")
    temporary.replace(path)


def load_cases(path: Path) -> list[QualificationCase]:
    raw = tomllib.loads(path.read_text())
    cases = [QualificationCase.model_validate(row) for row in raw["cases"]]
    if len(cases) != 5 or {case.kind for case in cases} != KINDS:
        raise ValueError("Exactly one real case of every required kind is needed")
    for case in cases:
        resolved = (path.parent / case.path).resolve()
        if not resolved.is_file() or file_digest(resolved) != case.sha256:
            raise ValueError("Qualification input hash mismatch")
        if case.kind in {"table", "formula"} and case.kind not in case.required_labels:
            raise ValueError("Table/formula fixtures require matching recognized labels")
    if len({case.sha256 for case in cases}) != len(cases):
        raise ValueError("Qualification cases must be distinct real inputs")
    return [case.model_copy(update={"path": (path.parent / case.path).resolve()}) for case in cases]


def run_stage(stage: str, payload: dict[str, Any], directory: Path) -> dict[str, Any]:
    incoming, outgoing = directory / "stage-input.json", directory / "stage-output.json"
    write_json(incoming, payload)
    outgoing.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ocr_service.qualify",
            "--stage",
            stage,
            "--stage-input",
            str(incoming),
            "--output",
            str(outgoing),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=600)
    except BaseException as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The child can exit between the deadline and group signal.
        process.wait(timeout=30)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise TimeoutError("Native qualification stage exceeded 600 seconds") from None
        raise
    if code:
        raise subprocess.CalledProcessError(code, process.args)
    if outgoing.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Qualification metadata exceeds bounds")
    result: dict[str, Any] = json.loads(outgoing.read_bytes())
    return result


def device_identity(paddle: Any, runtime: RuntimeSettings, *, compute: bool = True) -> dict[str, Any]:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise ValueError("Architecture spoofing is forbidden")
    result = inspect_device_identity(runtime.accelerator, runtime.gpu_index)
    result["paddle_build"] = str(paddle.version.full_version)
    if runtime.accelerator == "rocm":
        target = result["gfx_target"]
        kernels = json.loads(Path("/opt/ocr/build-evidence/kernels.json").read_text())
        profile = validate_profile(kernels.get("target_profile"))
        if target not in profile["candidate_targets"]:
            raise ValueError("Actual HIP target is outside the candidate target set")
        for library in ("paddle", "rocblas", "miopen"):
            if not kernels[library].get("files"):
                raise ValueError("Required kernel/library inventory is empty")
            if (
                kernels[library].get("offline_status") != "unverified"
                and target not in kernels[library]["targets"]
            ):
                raise ValueError("Required device kernels are missing")
            for filename, digest in kernels[library]["files"].items():
                if file_digest(Path(filename)) != digest:
                    raise ValueError("Required kernel inventory changed")
        result["gfx_target"] = target
        result["kernel_inventory_sha256"] = content_identity(kernels)
        if compute:
            result["compute_probe"] = compute_probe(paddle, runtime.gpu_index)
    return result


def compute_probe(paddle: Any, gpu_index: int) -> dict[str, Any]:
    """Target-specific FP32 math evidence inside the bounded collector process."""
    import numpy as np

    paddle.set_device(f"gpu:{gpu_index}")
    started = time.perf_counter()
    with paddle.no_grad():
        left = paddle.ones([4, 4], dtype="float32")
        right = paddle.ones([4, 4], dtype="float32")
        image = paddle.ones([1, 1, 5, 5], dtype="float32")
        kernel = paddle.ones([1, 1, 3, 3], dtype="float32")
        product = paddle.matmul(left, right)
        convolution = paddle.nn.functional.conv2d(image, kernel)
        for value in (left, right, image, kernel, product, convolution):
            require_gpu(value, gpu_index)
            if floating_dtype(value.dtype) != "float32":
                raise ValueError("Native compute probe requires FP32")
        if not np.all(product.numpy() == 4) or not np.all(convolution.numpy() == 9):
            raise ValueError("Native compute probe result mismatch")
    return {
        "passed": True,
        "precision": "float32",
        "device": f"gpu:{gpu_index}",
        "seconds": time.perf_counter() - started,
        "operations": ["matmul", "conv2d"],
    }


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = RuntimeSettings.model_validate(payload["runtime"])
    verify_dependencies(runtime.expected_packages, payload["dependencies"])
    profile = load_profile(PROFILE)
    native, paddle, runner_type = construct_native(profile, runtime)
    device = device_identity(paddle, runtime)
    preprocessor = native.doc_preprocessor_pipeline
    models = [
        native.layout_det_model,
        preprocessor.doc_ori_classify_model,
        preprocessor.doc_unwarping_model,
    ]
    collectors = []
    for name, model in zip(MODEL_NAMES[:3], models, strict=True):
        runner = model.runner
        expected = {
            "run_mode": "paddle_fp32",
            "device_type": "gpu",
            "device_id": runtime.gpu_index,
            "enable_cinn": False,
        }
        if (
            not isinstance(runner, runner_type)
            or runner._model_name != name
            or any(runner._config.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Collector resolved runner is not native GPU FP32")
        artifact = payload["inventory"][name]
        if (
            Path(runner.model_dir).resolve() != runtime.model_dirs[name].resolve()
            or artifact["graph_file"].rsplit(".", 1)[0] != runner.model_file_prefix
        ):
            raise ValueError("Inspector and native runner artifacts differ")
        collector = HookCollector(runtime.gpu_index)
        runner.predictor.register_input_hook(collector.observe_input)
        runner.predictor.register_output_hook(collector.observe_output)
        collectors.append(collector)
    # This temporary adapter supplies exact kwargs and response normalization only.
    # It is never qualified and its gated predict() is never invoked.
    adapter = PipelineAdapter(
        native, profile, "unsealed", "unsealed", [], gpu_index=runtime.gpu_index
    )
    forwards: list[list[Any]] = [[] for _ in collectors]
    for row in payload["cases"]:
        case = QualificationCase.model_validate(row)
        for _ in range(2):
            for collector in collectors:
                collector.start()
            with normalize_source(case.path, case.sha256) as normalized:
                results = iter(native.predict(str(normalized.path), **adapter.predict_kwargs))
                result = next(results)
                if next(results, None) is not None:
                    raise ValueError("Expected single-image native inference")
                response = adapter.convert_result(
                    result,
                    request_id=UUID(int=0),
                    photo_index=0,
                    source_sha256=case.sha256,
                    transforms=normalized.transforms,
                    findings=normalized.findings,
                )
            if not set(case.required_labels) <= {block.label for block in response.blocks}:
                raise ValueError("Required recognized structure absent")
            audit_dynamic_model(native.vl_rec_model.infer, runtime.gpu_index)
            for index, collector in enumerate(collectors):
                forwards[index].append(collector.finish())
    artifacts = {}
    for index, name in enumerate(MODEL_NAMES[:3]):
        coverage = common_coverage(forwards[index])
        inventory = payload["inventory"][name]
        bindings = parameter_bindings(
            [ParameterRecord.model_validate(row) for row in inventory["parameters"]],
            coverage,
            runtime.gpu_index,
        )
        artifacts[name] = StaticArtifact.model_validate(
            {
                **inventory,
                "coverage": [row.model_dump() for row in coverage],
                "parameter_bindings": bindings,
            }
        ).model_dump()
    return {"static_artifacts": artifacts, "device": device}


def replay(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = RuntimeSettings.model_validate(payload["runtime"])
    started = time.perf_counter()
    adapter = build_pipeline(load_profile(PROFILE), runtime)
    constructed = time.perf_counter()
    adapter.qualify(Path(payload["smoke"]))
    cold_smoke = time.perf_counter() - constructed
    startup = time.perf_counter() - started
    case = QualificationCase.model_validate(payload["case"])
    timings = []
    for _ in range(2):
        started = time.perf_counter()
        response = adapter.predict(
            case.path, request_id=UUID(int=0), photo_index=0, source_sha256=case.sha256
        )
        timings.append(time.perf_counter() - started)
        if not set(case.required_labels) <= {block.label for block in response.blocks}:
            raise ValueError("Required recognized structure absent in replay")
    return {
        "kind": case.kind,
        "source_sha256": case.sha256,
        "startup_seconds": startup,
        "cold_smoke_seconds": cold_smoke,
        "first_case_seconds": timings[0],
        "warm_case_seconds": timings[1],
        "passed": True,
        "static_audits": adapter.evidence,
        "dynamic_fp32_parameters": audit_dynamic_model(
            adapter.pipeline.vl_rec_model.infer, runtime.gpu_index
        ),
        "profile_id": adapter.profile_id,
        "runtime_fingerprint": adapter.runtime_fingerprint,
    }


def validate_replay(result: dict[str, Any]) -> None:
    audits = result.get("static_audits", [])
    if (
        result.get("passed") is not True
        or len(audits) != 3
        or {row.get("model_name") for row in audits} != set(MODEL_NAMES[:3])
        or result.get("dynamic_fp32_parameters", 0) <= 0
    ):
        raise ValueError("Native replay evidence is incomplete")
    for row in audits:
        if any(
            row.get(key, 0) <= 0 for key in ("input_count", "output_count", "floating_parameters")
        ):
            raise ValueError("Native replay audit is empty")
    for key in ("startup_seconds", "cold_smoke_seconds", "first_case_seconds", "warm_case_seconds"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Native replay timing is invalid")


def qualify_runtime(runtime: RuntimeSettings, cases: Path, output: Path) -> QualificationReport:
    report = QualificationReport(
        accelerator=runtime.accelerator, image_identity=runtime.image_identity
    )
    serving = output.with_suffix(".runtime.json")
    try:
        report.stage = "publication_prepare"
        serving.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        report.stage = "inputs"
        inputs = load_cases(cases)
        if not re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", runtime.image_identity):
            raise ValueError("An immutable built image digest is required")
        report.profile = load_profile(PROFILE).model_dump()
        report.stage = "dependencies"
        native_packages = [
            name
            for name in ("paddlepaddle", "paddlepaddle-gpu")
            if any(
                str(dist.metadata["Name"]).lower().replace("_", "-") == name
                for dist in importlib.metadata.distributions()
            )
        ]
        if len(native_packages) != 1:
            raise ValueError("Exactly one native Paddle installation is required")
        expected, dependencies = seal_dependencies(native_packages[0])
        if runtime.expected_packages and runtime.expected_packages != expected:
            raise ValueError("Runtime expected packages differ from installed lock")
        runtime = runtime.model_copy(update={"expected_packages": expected})
        verify_dependencies(expected, dependencies)
        report.model_content = {
            name: tree_identity(runtime.model_dirs[name]) for name in MODEL_NAMES
        }
        smoke = next(case for case in inputs if case.kind == "upright")
        base = {
            "runtime": runtime.model_dump(mode="json"),
            "dependencies": dependencies,
            "cases": [case.model_dump(mode="json") for case in inputs],
        }
        with tempfile.TemporaryDirectory(prefix="ocr-qualify-") as scratch:
            directory = Path(scratch)
            report.stage = "artifact_inspection"
            inventory = {
                name: run_stage("inspect", {**base, "model_name": name}, directory)
                for name in MODEL_NAMES[:3]
            }
            report.stage = "native_collection"
            collected = run_stage("collect", {**base, "inventory": inventory}, directory)
            report.device = collected["device"]
            manifest = {
                "qualified_target": {key: value for key, value in report.device.items() if key != "compute_probe"},
                "dependencies": dependencies,
                "image_identity": runtime.image_identity,
                "paddle_identity": "paddle:" + runtime.expected_paddle_commit,
                "model_content": report.model_content,
                "smoke_sha256": smoke.sha256,
                "static_artifacts": collected["static_artifacts"],
            }
            # Seal candidate evidence before fresh factory validation, never previous success.
            write_json(runtime.qualification_manifest, manifest)
            report.manifest_sha256 = file_digest(runtime.qualification_manifest)
            report.stage = "fresh_replay"
            for case in inputs:
                # Two independent fresh workers check compiled hook identifier stability.
                for repetition in range(2):
                    result = run_stage(
                        "replay",
                        {**base, "smoke": str(smoke.path), "case": case.model_dump(mode="json")},
                        directory,
                    )
                    validate_replay(result)
                    if report.profile_id is not None and (
                        result["profile_id"] != report.profile_id
                        or result["runtime_fingerprint"] != report.runtime_fingerprint
                    ):
                        raise ValueError("Fresh replay identity changed")
                    report.profile_id = result["profile_id"]
                    report.runtime_fingerprint = result["runtime_fingerprint"]
                    report.cases.append({**result, "fresh_worker": repetition})
            if report.model_content != {
                name: tree_identity(runtime.model_dirs[name]) for name in MODEL_NAMES
            }:
                raise ValueError("Model bytes changed during qualification")
            verify_dependencies(expected, dependencies)
        report.stage = "publish_runtime"
        write_json(serving, runtime.model_dump(mode="json"))
        target = report.device.get("gfx_target") or report.device.get("name")
        completed = report.model_copy(
            update={
                "qualified": True,
                "qualified_targets": [str(target)],
                "stage": "complete",
            }
        )
        report.stage = "publish_report"
        write_json(output, completed.model_dump())
        return completed  # Both mandatory publications succeeded before committing success.
    except Exception as exc:  # noqa: BLE001 -- unknown native/build errors fail closed
        report.qualified = False
        report.qualified_targets = []
        report.failure = type(exc).__name__
        # A writer can fail after replacing its destination. Revoke both stale and
        # partially published artifacts before attempting a sanitized failure report.
        serving.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        try:
            write_json(output, report.model_dump())
        except OSError:
            # Unwritable report storage must not turn a failed attempt into success.
            # A post-replacement error can leave this false report, never a true one.
            pass
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=["cuda", "rocm"])
    parser.add_argument("--runtime", type=Path, default=os.environ.get("OCR_RUNTIME_SETTINGS_PATH"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["inspect", "collect", "replay"], help=argparse.SUPPRESS)
    parser.add_argument("--stage-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        payload = json.loads(args.stage_input.read_bytes())
        if args.stage == "inspect":
            runtime = RuntimeSettings.model_validate(payload["runtime"])
            verify_dependencies(runtime.expected_packages, payload["dependencies"])
            paddle = importlib.import_module("paddle")
            if str(paddle.version.commit) != runtime.expected_paddle_commit:
                raise ValueError("Inspector Paddle source identity mismatch")
            name = payload["model_name"]
            result = inspect_artifact(
                runtime.model_dirs[name], name, paddle, "paddle:" + runtime.expected_paddle_commit
            )
        elif args.stage == "collect":
            result = collect(payload)
        else:
            result = replay(payload)
        write_json(args.output, result)
        return
    if not qualification_is_supervised(args.supervisor_fd):
        parser.error(
            "Run full qualification through ocr_service.initialize maintain "
            "or ocr_service.source_maintenance with explicit maintenance intent"
        )
    if not args.runtime or not args.cases or not args.accelerator:
        parser.error("--runtime (or OCR_RUNTIME_SETTINGS_PATH), --accelerator and --cases required")
    runtime = RuntimeSettings.model_validate_json(args.runtime.read_bytes())
    if runtime.accelerator != args.accelerator:
        parser.error("--accelerator differs from explicit runtime settings")
    report = qualify_runtime(runtime, args.cases, args.output)
    raise SystemExit(0 if report.qualified else 1)


if __name__ == "__main__":
    main()

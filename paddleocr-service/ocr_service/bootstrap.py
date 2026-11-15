"""Unsealed, metadata-only inspection and hook collection for the release gate.

This module cannot authorize serving. Its output is replayed by a fresh ordinary
factory/adapter. Unknown CPU float transfers conservatively fail collection.
"""

from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .fp32 import floating_dtype
from .precision import HookRecord, ParameterRecord
from .profile import dependency_identity, file_digest

INSPECTOR_VERSION = "pinned-export-inventory-v1"


def seal_dependencies(native_package: str) -> tuple[dict[str, str], dict[str, Any]]:
    if native_package not in {"paddlepaddle", "paddlepaddle-gpu"}:
        raise ValueError("Unknown native Paddle distribution")
    active = {
        name: set[str]()
        for name in (
            "paddleocr",
            "paddlex",
            native_package,
            "pillow",
            "pillow-heif",
            "numpy",
            "pydantic",
            "packaging",
        )
    }
    active["paddlex"].add("ocr")
    pending = list(active)
    processed: dict[str, frozenset[str]] = {}
    versions: dict[str, str] = {}
    while pending:
        name = pending.pop()
        extras = frozenset(active[name])
        if processed.get(name) == extras:
            continue
        versions[name] = metadata.version(name)
        processed[name] = extras
        for text in metadata.requires(name) or []:
            requirement = Requirement(text)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in extras | {""}
            ):
                continue
            child = str(canonicalize_name(requirement.name))
            if not requirement.specifier.contains(metadata.version(child), prereleases=True):
                raise ValueError("Installed dependency violates its active requirement")
            extra_names = {str(canonicalize_name(extra)) for extra in requirement.extras}
            if child not in active:
                active[child] = extra_names
                pending.append(child)
            elif not extra_names <= active[child]:
                active[child].update(extra_names)
                pending.append(child)
    records = {
        name: {"version": versions[name], "content_sha256": dependency_identity(name)}
        for name in sorted(active)
    }
    return versions, records


class HookCollector:
    """A bounded observation collector, never a synthetic StaticAuditSession."""

    def __init__(self, gpu_index: int, max_records: int = 100_000):
        self.gpu_index = gpu_index
        self.max_records = max_records
        self.observed: set[tuple[str, str, str, str, str]] = set()
        self.violations: set[str] = set()
        self.counts = {"input": 0, "output": 0}
        self.enabled = False

    def start(self) -> None:
        self.observed.clear()
        self.violations.clear()
        self.counts = {"input": 0, "output": 0}
        self.enabled = True

    def observe(self, hook: str, op: str, name: str, tensor: Any) -> None:
        if not self.enabled:
            return
        self.counts[hook] += 1
        try:
            dtype = floating_dtype(tensor.dtype)
            if dtype is None:
                return
            place = tensor.place
            if (
                dtype != "float32"
                or not place.is_gpu_place()
                or place.gpu_device_id() != self.gpu_index
            ):
                self.violations.add("Unproven floating precision or device")
                return
            record = (hook, str(op), str(name), dtype, f"gpu:{self.gpu_index}")
            if len(op) > 1024 or len(name) > 4096:
                self.violations.add("Hook metadata exceeds bounds")
            elif record not in self.observed and len(self.observed) >= self.max_records:
                self.violations.add("Hook inventory exceeds bounds")
            else:
                self.observed.add(record)
        except Exception:  # noqa: BLE001 -- native callbacks must retain failures
            self.violations.add("Unable to inspect hook metadata")

    def observe_input(self, op: str, name: str, tensor: Any) -> None:
        self.observe("input", op, name, tensor)

    def observe_output(self, op: str, name: str, tensor: Any) -> None:
        self.observe("output", op, name, tensor)

    def finish(self) -> list[HookRecord]:
        self.enabled = False
        if (
            self.violations
            or not all(self.counts.values())
            or {row[0] for row in self.observed} != {"input", "output"}
        ):
            raise ValueError("Incomplete or invalid native hook coverage")
        return [
            HookRecord.model_validate(
                dict(zip(("hook", "op", "name", "dtype", "place"), row, strict=True))
            )
            for row in sorted(self.observed)
        ]


def common_coverage(forwards: list[list[HookRecord]]) -> list[HookRecord]:
    if not forwards or not forwards[0] or any(rows != forwards[0] for rows in forwards):
        raise ValueError("Per-forward coverage differs; single-set qualification is unsupported")
    return forwards[0]


def parameter_bindings(
    parameters: list[ParameterRecord], coverage: list[HookRecord], gpu_index: int
) -> dict[str, str]:
    observed = {
        row.name for row in coverage if row.hook == "input" and row.place == f"gpu:{gpu_index}"
    }
    floating = {p.name for p in parameters if floating_dtype(p.dtype) is not None}
    if not floating or not floating <= observed:
        raise ValueError("Complete unambiguous parameter binding unavailable")
    return {name: name for name in sorted(floating)}


def inspect_artifact(model_dir: Path, name: str, paddle: Any, identity: str) -> dict[str, Any]:
    """Run only inside a fresh inspector subprocess, never the serving worker.

    CPU-loaded DenseTensor metadata proves exported bytes, not live GPU placement.
    The actual compiled predictor is separately checked by hooks and fresh replay.
    """
    prefix = model_dir / "inference"
    graph = prefix.with_suffix(".json")
    if not graph.is_file():
        graph = prefix.with_suffix(".pdmodel")
    params = prefix.with_suffix(".pdiparams")
    before = (file_digest(graph), file_digest(params))
    guard = paddle.pir_utils.IrGuard if graph.suffix == ".json" else paddle.pir_utils.OldIrGuard
    records = []
    with guard():
        paddle.enable_static()
        executor = paddle.static.Executor(paddle.CPUPlace())
        program, _, _ = paddle.static.load_inference_model(str(prefix), executor)
        if graph.suffix == ".json":
            weights, opts = paddle.static.pir_io.get_pir_parameters(program)
            variables = [v for v in weights + opts if v.persistable]
        else:
            variables = paddle.static.io.get_program_persistable_vars(program)
        for variable in variables:
            stored = paddle.static.global_scope().find_var(variable.name)
            if stored is None:
                raise ValueError("Exported parameter is absent from loaded scope")
            dense = stored.get_tensor()
            dtype = floating_dtype(variable.dtype)
            actual = dense._dtype()
            if dtype is not None:
                if dtype != "float32" or actual != paddle.base.core.VarDesc.VarType.FP32:
                    raise ValueError("Loaded exported parameter is not FP32")
            elif str(actual).split(".")[-1].lower() != str(variable.dtype).split(".")[-1].lower():
                raise ValueError("Nonfloating graph/loaded dtype mismatch")
            shape = list(dense.shape())
            if shape != list(variable.shape) or not dense._place().is_cpu_place():
                raise ValueError("Loaded exported parameter shape/place mismatch")
            records.append(
                ParameterRecord(
                    name=variable.name, shape=shape, dtype=dtype or str(variable.dtype)
                ).model_dump()
            )
    if before != (file_digest(graph), file_digest(params)) or not records:
        raise ValueError("Exported artifacts changed or have empty inventory")
    return {
        "model_name": name,
        "graph_file": graph.name,
        "graph_sha256": before[0],
        "params_file": params.name,
        "params_sha256": before[1],
        "inspector_identity": identity,
        "inspector_version": INSPECTOR_VERSION,
        "parameters": records,
    }

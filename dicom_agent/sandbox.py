"""Simple local sandbox for Python analysis of NIfTI images.

NIfTI bytes found in tool-result messages are staged under ``inputs/``. The
agent runs short Python snippets with ``run_python``; those snippets see
``SimpleITK``, ``numpy``, and the staged paths. Isolation is a subprocess
with a timeout and a cleared environment — not a container.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

INPUTS = "inputs"
WORK = "work"
DEFAULT_TIMEOUT = 30
MAX_OUTPUT = 32_000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
# Tools this process runs itself. Remote hosts must never see these names.
LOCAL_TOOL_NAMES = frozenset({"list_nifti_inputs", "run_python"})

log = logging.getLogger("dicom_agent.sandbox")

_RUNNER = r"""
import sys, traceback
source = sys.stdin.buffer.read().decode("utf-8")
try:
    exec(compile(source, "<sandbox>", "exec"), {"__name__": "__main__"})
except SystemExit:
    raise
except BaseException:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""


class NiftiSandbox:
    """Per-request workspace with staged NIfTI inputs and a Python runner."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="dicom-agent-sbx-"))
        self.inputs = self.root / INPUTS
        self.work = self.root / WORK
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self._owned = root is None
        self.staged: dict[str, Path] = {}

    def close(self) -> None:
        if self._owned and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def stage_bytes(self, name: str, data: bytes) -> Path:
        """Write bytes under ``inputs/`` and remember the path."""
        path = self.inputs / safe_name(name)
        path.write_bytes(data)
        self.staged[path.name] = path
        return path

    def stage_volume_payload(self, payload: dict[str, Any], *, hint: str = "volume") -> dict[str, Any] | None:
        """If ``payload`` holds a NIfTI volume, stage it and return a path summary.

        Accepts a fetch-shaped object ``{id, data: {encoding, volume_base64, …}}``
        or a bare ``data`` dict with ``encoding`` + ``volume_base64``.
        """
        data, envelope = _nifti_data(payload)
        if data is None:
            return None

        encoding = str(data.get("encoding") or "")
        raw_b64 = data.get("volume_base64")
        if encoding not in {"nii.gz", "nii"} or not isinstance(raw_b64, str) or not raw_b64:
            return None

        suffix = ".nii.gz" if encoding == "nii.gz" else ".nii"
        resource = str((envelope or {}).get("id") or payload.get("id") or hint)
        name = f"{safe_name(resource.split(':')[-1] or hint)}{suffix}"
        raw = base64.b64decode(raw_b64)
        path = self.stage_bytes(name, raw)
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        source = envelope if isinstance(envelope, dict) else payload
        log.info(
            "staged volume id=%s path=%s bytes=%d voxels=%s",
            source.get("id"),
            path.name,
            len(raw),
            meta.get("n_voxels"),
        )
        return {
            "id": source.get("id"),
            "modality": source.get("modality"),
            "view": source.get("view"),
            "sandbox_path": str(path),
            "encoding": encoding,
            "meta": meta,
            "bounds": source.get("bounds"),
            "provenance": source.get("provenance"),
            "note": (
                "Volume bytes were staged into the local NIfTI sandbox. "
                f"Open them with SimpleITK.ReadImage at {path}."
            ),
        }

    def run_python(self, code: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        """Run ``code`` in a child interpreter with cwd at the sandbox root."""
        if not isinstance(code, str) or not code.strip():
            log.warning("run_python rejected: empty code")
            return {"ok": False, "stdout": "", "stderr": "code is empty", "exit_code": 1}

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.work),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "SANDBOX_ROOT": str(self.root),
            "SANDBOX_INPUTS": str(self.inputs),
            "SANDBOX_WORK": str(self.work),
        }
        log.info(
            "run_python start chars=%d inputs=%s timeout=%s",
            len(code),
            sorted(p.name for p in self.inputs.iterdir() if p.is_file()),
            timeout,
        )
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-c", _RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.root),
            env=env,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(input=code, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _kill(process)
            note = f"Code execution timed out after {timeout} seconds."
            stderr = f"{stderr}\n{note}" if stderr else note

        ok = (not timed_out) and process.returncode == 0
        elapsed = time.monotonic() - started
        log.info(
            "run_python done ok=%s exit=%s elapsed=%.2fs stdout=%d stderr=%d",
            ok,
            124 if timed_out else process.returncode,
            elapsed,
            len(stdout or ""),
            len(stderr or ""),
        )
        if not ok and stderr:
            log.warning("run_python stderr: %s", _clip(stderr)[:500])

        return {
            "ok": ok,
            "stdout": _clip(stdout),
            "stderr": _clip(stderr),
            "exit_code": process.returncode if not timed_out else 124,
            "inputs": sorted(p.name for p in self.inputs.iterdir() if p.is_file()),
            "cwd": str(self.root),
        }


def safe_name(raw: str) -> str:
    cleaned = _SAFE_NAME.sub("_", raw).strip("._") or "volume"
    return cleaned[:120]


def _nifti_data(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(data_dict, envelope)`` for a NIfTI-bearing payload, or ``(None, None)``."""
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("volume_base64"), str):
        return data, payload

    if isinstance(payload.get("volume_base64"), str) and payload.get("encoding") in {
        "nii",
        "nii.gz",
    }:
        return payload, None

    return None, None


def _volume_rewrite_target(payload: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Locate the object to stage inside a tool-result JSON blob.

    Supports a host wrap ``{status, result, cite, …}``, a bare fetch body, or
    any dict that already carries ``data.volume_base64``.
    """
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("result"), dict):
        candidate = payload["result"]
        data, _ = _nifti_data(candidate)
        if data is not None:
            return candidate, "result"

    data, _ = _nifti_data(payload)
    if data is not None:
        return payload, "root"

    return None


def stage_volumes_in_messages(messages: list[dict], sandbox: NiftiSandbox) -> list[dict]:
    """Copy messages, staging any NIfTI volume tool results into ``sandbox``.

    Recognizes common host wraps (``result`` + optional ``cite``) and bare
    volume payloads. Outer fields such as ``cite`` are preserved.
    ``volume_base64`` is dropped from what the model sees.
    """
    rewritten: list[dict] = []

    for message in messages:
        if message.get("role") != "tool":
            rewritten.append(message)
            continue

        raw = message.get("content") or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            rewritten.append(message)
            continue

        if not isinstance(payload, dict):
            rewritten.append(message)
            continue

        target = _volume_rewrite_target(payload)
        if target is None:
            rewritten.append(message)
            continue

        candidate, where = target
        staged = sandbox.stage_volume_payload(candidate)
        if staged is None:
            rewritten.append(message)
            continue

        copy = dict(message)
        if where == "result":
            copy["content"] = json.dumps({**payload, "result": staged})
        else:
            copy["content"] = json.dumps(staged)
        rewritten.append(copy)

    return rewritten


def sandbox_tools(sandbox: NiftiSandbox) -> list:
    """ADK FunctionTools bound to this sandbox instance."""
    from google.adk.tools.function_tool import FunctionTool

    def list_nifti_inputs() -> dict[str, Any]:
        """List NIfTI files staged in the sandbox inputs directory."""
        files = []
        for path in sorted(sandbox.inputs.iterdir()):
            if path.is_file():
                files.append({"name": path.name, "path": str(path), "bytes": path.stat().st_size})
        return {"inputs": files, "cwd": str(sandbox.root)}

    def run_python(code: str) -> dict[str, Any]:
        """Run Python against staged NIfTI files in the local sandbox.

        The working directory is the sandbox root. Staged volumes are under
        ``inputs/`` (also ``SANDBOX_INPUTS``). Writable scratch is ``work/``.
        Available imports include ``SimpleITK`` and ``numpy``. Load a volume
        with ``SimpleITK.ReadImage(path)``. Print results as compact text or
        JSON. There is no network.
        """
        return sandbox.run_python(code)

    return [FunctionTool(list_nifti_inputs), FunctionTool(run_python)]


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[: MAX_OUTPUT - 20] + "\n...[truncated]..."


def _kill(process: subprocess.Popen[str]) -> tuple[str, str]:
    group = process.pid
    if hasattr(os, "killpg"):
        try:
            os.killpg(group, signal.SIGTERM)
        except OSError:
            pass
    process.terminate()
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        pass
    if hasattr(os, "killpg"):
        try:
            os.killpg(group, signal.SIGKILL)
        except OSError:
            pass
    process.kill()
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        return "", ""

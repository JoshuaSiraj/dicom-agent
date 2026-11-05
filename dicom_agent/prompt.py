"""Instruction for the DICOM agent.

Domain knowledge is static. Host-specific tool recipes are built from the
``tools`` list on each chat-completions request so this process adapts to
whatever the caller advertises.
"""

from __future__ import annotations

from typing import Any

DOMAIN = """
You are a specialist in DICOM data. Answer questions about the study you are given.

How to answer:
1. Prefer the caller's describe / metadata / preview / linking tools when headers, ROI names, or a slice preview are enough.
2. Do not invent series, measurements, or clinical facts. If the ROI or series is missing from the study, say so.
3. Follow the caller's answer / citation rules when they appear in the system message. Keep any cite token from a tool result you rely on.

DICOM reference:
A study contains series. A series contains instances of one modality and shares one Series Instance UID and one Frame of Reference UID.
Image series are CT, MR, and PT. RTSTRUCT holds contour names and the image series they cite. SEG holds a segmentation of an image series. RTDOSE holds a dose grid cited to an image series and a plan. RTPLAN holds beams and prescriptions and cites the structure set and the images.
A contour, dose, or plan belongs to the image series named in its references. Do not pair two series unless they share a reference or a Frame of Reference UID.
""".strip()

# Back-compat alias for imports that still expect INSTRUCTION.
INSTRUCTION = DOMAIN


def _function_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for spec in tools:
        function = spec.get("function") if isinstance(spec, dict) else None
        if isinstance(function, dict) and function.get("name"):
            found.append(function)
    return found


def _properties(function: dict[str, Any]) -> dict[str, Any]:
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    props = parameters.get("properties")
    return props if isinstance(props, dict) else {}


def _mentions(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def capabilities_from_tools(tools: list[dict[str, Any]]) -> str:
    """Short addendum describing what this request's tool surface supports."""
    functions = _function_specs(tools)
    if not functions:
        return (
            "Host tools:\n"
            "This request lists no remote tools. Answer from the conversation only, "
            "or say what you would need to look up."
        )

    by_name = {str(fn["name"]): fn for fn in functions}
    names = sorted(by_name)
    lines = [
        "Host tools on this request:",
        ", ".join(names) + ".",
        "Use only these remote tools (plus any local sandbox tools listed below). "
        "Read each tool's description and parameters; do not assume views or args "
        "that are not advertised.",
    ]

    fetch = by_name.get("fetch")
    describe = by_name.get("describe")
    if describe is not None:
        lines.append(
            "Call describe before a heavy fetch when you need views, roi_names, or links."
        )

    if fetch is not None:
        props = _properties(fetch)
        fetch_text = " ".join(
            [
                str(fetch.get("description") or ""),
                str(props.get("view", {}).get("description") or "")
                if isinstance(props.get("view"), dict)
                else "",
                str(props.get("roi", {}).get("description") or "")
                if isinstance(props.get("roi"), dict)
                else "",
                str(props.get("spacing", {}).get("description") or "")
                if isinstance(props.get("spacing"), dict)
                else "",
            ]
        )
        has_volume = "view" in props and _mentions(fetch_text, "volume")
        has_roi = "roi" in props
        has_spacing = "spacing" in props

        if has_volume:
            lines.append(
                "Pixel-level measurements (structure volume, voxel count, intensity/"
                "dose stats): use fetch with view=\"volume\", then the local sandbox. "
                "Never say those measurements are impossible while volume fetch and "
                "run_python are available. Never stop at metadata or preview for a "
                "volume question."
            )
            steps = [
                "Structure / volume measurement path:",
                "1. describe the RTSTRUCT or SEG; confirm the ROI name and that views include \"volume\".",
            ]
            fetch_bits = ['view="volume"']
            if has_roi:
                fetch_bits.append("roi=<ROI name>")
                steps.append(
                    "2. fetch that resource with "
                    + ", ".join(fetch_bits)
                    + (
                        ", spacing=[2,2,2] (retry [3,3,3] if too large). "
                        "Passing roi extracts only that structure (avoids EXTERNAL "
                        "overlap wiping a small target). Do not fetch only the CT "
                        "for an ROI volume."
                        if has_spacing
                        else ". Passing roi extracts only that structure."
                    )
                )
            else:
                steps.append(
                    "2. fetch that resource with view=\"volume\""
                    + (", spacing=[2,2,2] if spacing is supported." if has_spacing else ".")
                )
            steps.extend(
                [
                    "3. When the host returns NIfTI bytes, this process stages them under "
                    "inputs/ and replaces the base64 payload with sandbox_path. "
                    "meta.labels maps label index → ROI name when present. Keep any "
                    "cite token from that tool result.",
                    "4. Call list_nifti_inputs, then run_python with SimpleITK/numpy: "
                    "ReadImage(sandbox_path), select the label for the ROI (or label 1 "
                    "when only one), count voxels × spacing product / 1000 for cm³, "
                    "print JSON.",
                    "5. Answer with that number. If the host's citation schema includes "
                    "kind \"derived\", cite the volume fetch that way with a "
                    "plain-language method and result {value, unit}; otherwise follow "
                    "the host schema and keep the cite token.",
                ]
            )
            lines.extend(steps)

    lines.append(
        "Local sandbox tools (run_python, list_nifti_inputs) execute in this process. "
        "Do not expect the caller to run them. Working directory is the sandbox root; "
        "staged files are under inputs/."
    )
    return "\n".join(lines)


def build_instruction(
    tools: list[dict[str, Any]] | None = None,
    host_system: list[str] | None = None,
) -> str:
    """Domain text + tool-derived capabilities + host system messages."""
    parts = [DOMAIN, capabilities_from_tools(tools or [])]
    for block in host_system or []:
        text = (block or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)

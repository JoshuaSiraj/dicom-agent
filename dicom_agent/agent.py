"""Google ADK agent for DICOM questions.

serve.py runs one turn of this agent per chat-completions request. The tools
on that request are declared here and executed by the caller.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from .prompt import build_instruction
from .sandbox import (
    LOCAL_TOOL_NAMES,
    NiftiSandbox,
    sandbox_tools,
    stage_volumes_in_messages,
)


# Fallback only when DICOM_AGENT_MODEL is missing from the environment / .env.
DEFAULT_MODEL = "gemini/gemini-flash-latest"
NAME = "dicom_agent"
APP = "dicom_agent"
USER = "user"

log = logging.getLogger("dicom_agent")


def configure_litellm_env() -> None:
    """Map common Azure .env names onto the variables LiteLLM reads."""
    key = os.getenv("AZURE_OPENAI_API_KEY")
    base = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_OPENAI_API_VERSION")

    if key:
        os.environ.setdefault("AZURE_API_KEY", key)
        os.environ.setdefault("AZURE_AI_API_KEY", key)
    if base:
        os.environ.setdefault("AZURE_API_BASE", base)
        os.environ.setdefault("AZURE_AI_API_BASE", base)
    if version:
        os.environ.setdefault("AZURE_API_VERSION", version)


def resolve_model() -> str:
    """LiteLLM model string from ``DICOM_AGENT_MODEL`` in the environment / .env.

    Bare names (no ``provider/``) are routed to Azure when Azure credentials are
    present: ``azure_ai/…`` for ``*.services.ai.azure.com``, otherwise ``azure/…``.
    """
    raw = (os.getenv("DICOM_AGENT_MODEL") or DEFAULT_MODEL).strip()
    if not raw:
        raw = DEFAULT_MODEL
    if "/" in raw:
        return raw

    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or raw).strip()
    has_azure = bool(
        os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_AI_API_KEY")
    )
    if not has_azure:
        return raw

    base = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_API_BASE")
        or os.getenv("AZURE_AI_API_BASE")
        or ""
    )
    provider = "azure_ai" if "services.ai.azure.com" in base else "azure"
    return f"{provider}/{deployment}"


class DeclaredTool(BaseTool):
    """A tool named on the request. The caller runs it."""

    def __init__(self, name: str, description: str, parameters: dict | None):
        super().__init__(name=name, description=description, is_long_running=True)
        self._parameters = parameters or {"type": "object", "properties": {}}

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> None:
        """Pause this turn. The caller runs the tool and sends the result back."""
        return None


def build_agent(instruction: str, tools: list[BaseTool], model: str | None = None) -> Agent:
    """The DICOM specialist, bound to this request's instruction and tools."""
    model_id = model or resolve_model()
    return Agent(
        name=NAME,
        model=LiteLlm(model=model_id),
        description="Answers questions about DICOM data.",
        static_instruction=instruction,
        tools=tools,
    )


def declared_tools(tools: list[dict]) -> list[DeclaredTool]:
    """ADK tools for the OpenAI tool list. Tools with no name are skipped.

    Sandbox tool names are never declared here: this process runs those itself.
    """
    found = []

    for spec in tools:
        function = spec.get("function") or {}
        name = function.get("name")

        if not name or name in LOCAL_TOOL_NAMES:
            continue

        found.append(DeclaredTool(
            name=name,
            description=function.get("description") or "",
            parameters=function.get("parameters"),
        ))

    return found


async def answer(body: dict) -> dict:
    """One turn of the ADK agent, as an OpenAI chat completion.

    The caller runs remote tools from the request and posts results back.
    NIfTI payloads in tool results are staged into a local sandbox so
    ``run_python`` can use SimpleITK. Only remote tool names are returned as
    ``tool_calls``; sandbox tools stay in this process.
    """
    request_tools = body.get("tools") or []
    raw_messages = body.get("messages") or []
    allowed = {
        (spec.get("function") or {}).get("name")
        for spec in request_tools
        if (spec.get("function") or {}).get("name")
    } - LOCAL_TOOL_NAMES
    model_id = resolve_model()
    tool_names = sorted(allowed)
    roles = _role_counts(raw_messages)
    log.info(
        "turn start messages=%d roles=%s host_tools=%s local_tools=%s model=%s",
        len(raw_messages),
        roles,
        tool_names,
        sorted(LOCAL_TOOL_NAMES),
        model_id,
    )

    sandbox = NiftiSandbox()
    started = time.monotonic()
    try:
        messages = stage_volumes_in_messages(raw_messages, sandbox)
        if sandbox.staged:
            log.info("staged nifti=%s", sorted(sandbox.staged))

        instruction, prior, new_message = request_parts(messages, request_tools)
        tools = [*declared_tools(request_tools), *sandbox_tools(sandbox)]
        agent = build_agent(instruction, tools, model=model_id)
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name=APP, user_id=USER)

        for event in prior:
            await sessions.append_event(session, event)

        runner = Runner(app_name=APP, agent=agent, session_service=sessions)
        reply = None

        async for event in runner.run_async(
            user_id=USER,
            session_id=session.id,
            new_message=new_message,
        ):
            if event.partial or event.author != agent.name or not event.content:
                continue

            reply = event
            _log_model_event(event)

        if reply is None:
            raise RuntimeError("The model returned no message.")

        if reply.error_code:
            raise RuntimeError(reply.error_message or str(reply.error_code))

        message = openai_message(reply, allowed_tool_names=allowed or None)
        log.info(
            "turn done elapsed=%.2fs %s",
            time.monotonic() - started,
            _reply_summary(message),
        )
        return {"choices": [{"message": message}]}
    except Exception:
        log.exception("turn failed elapsed=%.2fs", time.monotonic() - started)
        raise
    finally:
        sandbox.close()


def _role_counts(messages: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        role = str(message.get("role") or "?")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _log_model_event(event: Event) -> None:
    """Log one non-partial model event at debug detail."""
    names = []
    texts = 0
    for part in event.content.parts or []:
        call = part.function_call
        if call and call.name:
            names.append(call.name)
        elif getattr(part, "text", None) and not getattr(part, "thought", False):
            texts += 1
    log.debug("model event tools=%s text_parts=%d", names, texts)


def _reply_summary(message: dict) -> str:
    calls = message.get("tool_calls") or []
    if calls:
        names = [((c.get("function") or {}).get("name") or "?") for c in calls]
        return f"tool_calls={names}"
    content = message.get("content") or ""
    preview = content.replace("\n", " ")[:160]
    return f"content_chars={len(content)} preview={preview!r}"


def request_parts(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> tuple[str, list[Event], types.Content]:
    """Instruction text, earlier turns, and the message that starts this turn.

    Instruction is domain knowledge plus a capabilities addendum derived from
    ``tools``, then any host ``system`` messages from the request.
    """
    host_system: list[str] = []
    items: list[tuple[str, dict | list]] = []
    tool_messages: list[dict] = []
    names: dict[str, str] = {}

    def flush_tools() -> None:
        if tool_messages:
            items.append(("tools", tool_messages.copy()))
            tool_messages.clear()

    for message in messages:
        role = message.get("role")

        if role == "system":
            flush_tools()
            host_system.append(message.get("content") or "")
            continue

        if role == "tool":
            tool_messages.append(message)
            continue

        flush_tools()

        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                names[call.get("id") or ""] = function.get("name") or ""

            items.append(("model", message))
            continue

        items.append(("user", message))

    flush_tools()

    if not items or items[-1][0] == "model":
        raise RuntimeError("The request has no new message.")

    *earlier, last = items
    prior = [event_for(kind, payload, names) for kind, payload in earlier]
    instruction = build_instruction(tools or [], host_system)
    return instruction, prior, content_for(last[0], last[1], names)


def event_for(kind: str, payload: dict | list, names: dict[str, str]) -> Event:
    """One earlier turn, stored on the ADK session before this request runs."""
    return Event(
        invocation_id="prior",
        author=NAME if kind == "model" else "user",
        content=content_for(kind, payload, names),
    )


def content_for(kind: str, payload: dict | list, names: dict[str, str]) -> types.Content:
    """Gemini content for one user, model, or tool-result turn."""
    if kind == "model":
        return types.Content(role="model", parts=assistant_parts(payload))

    if kind == "tools":
        return types.Content(
            role="user",
            parts=[tool_result_part(message, names) for message in payload],
        )

    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=payload.get("content") or "")],
    )


def assistant_parts(message: dict) -> list[types.Part]:
    """Text and function calls from one assistant message."""
    parts = []

    if message.get("content"):
        parts.append(types.Part.from_text(text=message["content"]))

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}

        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}

        parts.append(types.Part(
            function_call=types.FunctionCall(
                id=call.get("id"),
                name=function.get("name") or "",
                args=args if isinstance(args, dict) else {},
            ),
            thought_signature=decode_signature(call.get("thought_signature")),
        ))

    return parts or [types.Part.from_text(text="")]


def tool_result_part(message: dict, names: dict[str, str]) -> types.Part:
    """One tool result as a Gemini function response."""
    raw = message.get("content") or "{}"

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"output": raw}

    if not isinstance(payload, dict):
        payload = {"output": payload}

    return types.Part(function_response=types.FunctionResponse(
        id=message.get("tool_call_id"),
        name=names.get(message.get("tool_call_id") or "", ""),
        response=payload,
    ))


def openai_message(
    event: Event,
    allowed_tool_names: set[str] | None = None,
) -> dict:
    """The OpenAI assistant message for one model event.

    When ``allowed_tool_names`` is set (the MedDataOS tool list), sandbox
    tools such as ``run_python`` are dropped so the server never tries to
    dispatch them.
    """
    calls = []
    texts = []
    carried = None

    for part in event.content.parts or []:
        signature = encode_signature(getattr(part, "thought_signature", None))
        call = part.function_call

        if getattr(part, "thought", False) and not (call and call.name):
            carried = signature or carried
            continue

        if call and call.name:
            if allowed_tool_names is not None and call.name not in allowed_tool_names:
                carried = None
                continue

            item = {
                "id": call.id or f"call_{uuid4().hex}",
                "type": "function",
                "function": {"name": call.name, "arguments": json_args(call.args)},
            }
            kept = signature or carried

            if kept:
                item["thought_signature"] = kept

            carried = None
            calls.append(item)
        elif getattr(part, "text", None) and not getattr(part, "thought", False):
            texts.append(part.text)

    message: dict = {"role": "assistant", "content": "\n".join(texts) or None}

    if calls:
        message["tool_calls"] = calls

    return message


def encode_signature(raw: object) -> str | None:
    """A JSON-safe copy of a Gemini thought signature."""
    if isinstance(raw, str) and raw:
        return raw

    if isinstance(raw, (bytes, bytearray)) and raw:
        return base64.b64encode(raw).decode()

    return None


def decode_signature(raw: object) -> bytes | None:
    """The bytes Gemini expects back on a replayed function call."""
    if not isinstance(raw, str) or not raw:
        return None

    try:
        return base64.b64decode(raw)
    except ValueError:
        return None


def json_args(args: object) -> str:
    """The arguments string on an OpenAI tool call."""
    if args is None:
        return "{}"

    if not isinstance(args, dict):
        args = dict(args)

    return json.dumps(args)

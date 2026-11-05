"""OpenAI chat-completions server for the DICOM agent.

MedDataOS posts each turn to /v1/chat/completions and runs any tool_calls
itself. This runs one turn of the ADK agent and returns that message.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

from .agent import answer, configure_litellm_env, resolve_model


load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

log = logging.getLogger("dicom_agent.serve")


def configure_logging(verbose: bool = False) -> None:
    """Stdout logging for the agent process. Idempotent enough for one server."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger("dicom_agent")
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        started = time.monotonic()
        raw = self.rfile.read(length) or b"{}"

        try:
            body = json.loads(raw)
            payload = asyncio.run(answer(body))
            status = 200
        except Exception as exc:
            log.exception("request failed after %.2fs", time.monotonic() - started)
            payload = {"error": {"message": str(exc)}}
            status = 500

        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log.info(
            "http %s %s bytes_in=%d bytes_out=%d elapsed=%.2fs",
            status,
            self.path,
            length,
            len(data),
            time.monotonic() - started,
        )

    def log_message(self, format: str, *args) -> None:
        # Prefer the structured logger; keep access lines at DEBUG.
        log.debug("access " + format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="DICOM agent OpenAI endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="DEBUG logs (model events, HTTP access lines)",
    )
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    configure_litellm_env()
    # ADK warns when Gemini goes through LiteLLM; keep one path for all providers.
    os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")

    model_id = resolve_model()
    if not os.getenv("DICOM_AGENT_MODEL"):
        log.warning(
            "DICOM_AGENT_MODEL unset; using default %s — set it in .env",
            model_id,
        )
    else:
        log.info("model=%s (from DICOM_AGENT_MODEL)", model_id)
    if model_id.startswith("gemini/") and not os.getenv("GOOGLE_API_KEY"):
        log.warning("GOOGLE_API_KEY is unset; Gemini calls will fail")
    elif model_id.startswith(("azure/", "azure_ai/")) and not (
        os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_AI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
    ):
        log.warning("Azure API key is unset; Azure calls will fail")
    elif model_id.startswith("openai/") and not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY is unset; OpenAI calls will fail")
    elif model_id.startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY is unset; Anthropic calls will fail")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host, port = server.server_address
    log.info("Agent: http://%s:%s/v1", host, port)
    log.info("Launch MedDataOS with --agent-url http://%s:%s/v1", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()

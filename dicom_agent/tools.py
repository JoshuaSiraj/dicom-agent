"""ADK tools built only from med-datactrl TOOL_DEFINITIONS."""

from __future__ import annotations

import json
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from datactrl.tools import TOOL_DEFINITIONS, dispatch


class DatactrlTool(BaseTool):
    """One harness tool. The schema is the TOOL_DEFINITIONS entry."""

    def __init__(self, spec: dict[str, Any]) -> None:
        function = spec["function"]
        super().__init__(name=function["name"], description=function["description"])
        self._parameters = function["parameters"]

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        del tool_context
        try:
            return json.loads(dispatch(self.name, args))
        except Exception as exc:
            return {"status": "error", "error_message": str(exc)}


def load_tools() -> list[DatactrlTool]:
    """Return one ADK tool per med-datactrl definition."""
    return [DatactrlTool(spec) for spec in TOOL_DEFINITIONS]

"""DICOM agent. Its only tools are the med-datactrl harness."""

from google.adk.agents.llm_agent import Agent

from .prompt import INSTRUCTION
from .tools import load_tools

root_agent = Agent(
    name="dicom_agent",
    model="gemini-flash-latest",
    description="Navigates a local DICOM case through the med-datactrl harness.",
    instruction=INSTRUCTION,
    tools=load_tools(),
)

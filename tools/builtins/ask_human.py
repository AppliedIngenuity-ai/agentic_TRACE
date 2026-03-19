"""
Ask Human tool - request clarification from the user.

Use this when there's genuine ambiguity that can't be resolved from context.
Do NOT use when one option is clearly the most likely choice.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult

if TYPE_CHECKING:
    from ...core.session import Session


class AskHumanTool(BaseTool):
    """
    Ask the human user for clarification when there is genuine ambiguity.

    Use this tool when:
    - Multiple valid interpretations exist and context doesn't favor one
    - The user's intent is unclear and guessing could waste effort
    - You need domain knowledge that isn't available in the data

    Do NOT use this tool when:
    - One option is clearly most likely based on context
    - You can make a reasonable assumption and proceed
    - The query is unambiguous

    Note: The user will always have a "Something else" option to provide
    free-text input beyond the listed options.
    """

    name = "ask_human"
    category = ToolCategory.EXPLORE

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="question",
                param_type="string",
                description="Clear question to ask the user. Be specific about what you need clarified.",
                required=True,
            ),
            ToolParameter(
                name="options",
                param_type="array",
                description=(
                    "Optional list of specific options for the user to choose from. "
                    "A 'Something else' free-text option is always added automatically."
                ),
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="context",
                param_type="string",
                description="Optional context explaining why you need this clarification",
                required=False,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "question"):
            return self.error(error, "VALIDATION_ERROR")

        question = kwargs["question"]
        options = kwargs.get("options", [])
        context = kwargs.get("context", "")

        # Return a special result that signals the orchestrator to pause
        # The orchestrator will detect "needs_human_input" and handle it
        return ToolResult(
            success=True,
            metadata={
                "needs_human_input": True,
                "question": question,
                "options": options,
                "context": context,
            },
        )

    def tool_description(self) -> str:
        return (
            "Ask the human user for clarification when there is genuine ambiguity. "
            "Use when multiple valid interpretations exist and you can't determine "
            "which is intended. Do NOT use when one option is clearly most likely. "
            "User always has a 'Something else' option for free-text input."
        )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"

        question = result.metadata.get("question", "")
        human_response = result.metadata.get("human_response", "")

        if human_response:
            return f"Asked: '{question}' → Human responded: '{human_response}'"
        else:
            return f"Waiting for human response to: '{question}'"

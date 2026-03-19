"""
System prompt building and management.

PromptBuilder composes system prompts from:
- Base prompt (static text or file)
- Dynamic sections (tools, data sources)
- Examples (few-shot demonstrations)
- Custom sections (domain-specific instructions)
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry
    from ..data.sources import SourceRegistry


class PromptBuilder:
    """
    Builder for composing system prompts.

    Combines static base prompts with dynamic content like tool descriptions
    and data source schemas.

    Example:
        >>> builder = PromptBuilder()
        >>> builder.set_base_prompt(file="prompts/base.txt")
        >>> builder.add_examples([example1, example2])
        >>> builder.add_section("Domain Notes", "Trading days exclude weekends...")
        >>>
        >>> system_prompt = builder.build(tool_registry, source_registry)

    Registration API (for simple cases):
        >>> PromptBuilder.register_system_prompt(prompt_text="You are...")
        >>> # or
        >>> PromptBuilder.register_system_prompt(file="prompts/system.txt")
    """

    # Class-level storage for simple registration API
    _registered_prompt: str | None = None
    _registered_examples: list[str] = []
    _registered_sections: dict[str, str] = {}

    def __init__(self):
        """Initialize a new builder instance."""
        self._base_prompt: str | None = None
        self._examples: list[str] = []
        self._sections: dict[str, str] = {}

    def set_base_prompt(
        self,
        *,
        text: str | None = None,
        file: str | Path | None = None,
    ) -> "PromptBuilder":
        """
        Set the base system prompt.

        Args:
            text: Prompt text directly
            file: Path to prompt file

        Returns:
            Self for chaining

        Raises:
            ValueError: If both or neither text/file provided
        """
        if text and file:
            raise ValueError("Provide text OR file, not both")
        if not text and not file:
            raise ValueError("Must provide text or file")

        if text:
            self._base_prompt = text
        elif file:
            self._base_prompt = Path(file).read_text()

        return self

    def add_example(self, example: str) -> "PromptBuilder":
        """
        Add a few-shot example.

        Args:
            example: Example text showing input/output

        Returns:
            Self for chaining
        """
        self._examples.append(example)
        return self

    def add_examples(self, examples: list[str]) -> "PromptBuilder":
        """
        Add multiple few-shot examples.

        Args:
            examples: List of example texts

        Returns:
            Self for chaining
        """
        self._examples.extend(examples)
        return self

    def add_section(self, name: str, content: str) -> "PromptBuilder":
        """
        Add a custom section to the prompt.

        Args:
            name: Section header name
            content: Section content

        Returns:
            Self for chaining
        """
        self._sections[name] = content
        return self

    def build(
        self,
        tool_registry: "ToolRegistry | None" = None,
        source_registry: "SourceRegistry | None" = None,
        include_tools: bool = True,
        include_sources: bool = True,
    ) -> str:
        """
        Build the complete system prompt.

        Args:
            tool_registry: Registry of available tools
            source_registry: Registry of data sources
            include_tools: Whether to include tool descriptions
            include_sources: Whether to include source descriptions

        Returns:
            Complete system prompt string
        """
        parts = []

        # Base prompt
        if self._base_prompt:
            parts.append(self._base_prompt)

        # Data sources
        if include_sources and source_registry:
            sources_desc = source_registry.get_all_descriptions()
            if sources_desc:
                parts.append(sources_desc)

        # Tools
        if include_tools and tool_registry:
            tools_desc = tool_registry.get_tools_description()
            if tools_desc:
                parts.append("## Available Tools\n" + tools_desc)

        # Examples
        if self._examples:
            parts.append("## Examples\n")
            for i, example in enumerate(self._examples, 1):
                parts.append(f"### Example {i}\n{example}")

        # Custom sections
        for name, content in self._sections.items():
            parts.append(f"## {name}\n{content}")

        return "\n\n".join(parts)

    # ─── Class-level registration API (for simple usage) ───

    @classmethod
    def register_system_prompt(
        cls,
        *,
        prompt_text: str | None = None,
        file: str | Path | None = None,
    ) -> None:
        """
        Register a system prompt globally.

        Simple API for cases where you just need to set a prompt once.

        Args:
            prompt_text: Prompt text directly
            file: Path to prompt file

        Raises:
            ValueError: If both or neither provided
        """
        if prompt_text and file:
            raise ValueError("Provide prompt_text OR file, not both")
        if not prompt_text and not file:
            raise ValueError("Must provide prompt_text or file")

        if prompt_text:
            cls._registered_prompt = prompt_text
        elif file:
            cls._registered_prompt = Path(file).read_text()

    @classmethod
    def register_examples(cls, examples: list[str]) -> None:
        """
        Register examples globally.

        Args:
            examples: List of few-shot examples
        """
        cls._registered_examples.extend(examples)

    @classmethod
    def register_section(cls, name: str, content: str) -> None:
        """
        Register a custom section globally.

        Args:
            name: Section name
            content: Section content
        """
        cls._registered_sections[name] = content

    @classmethod
    def get_registered_prompt(cls) -> str | None:
        """Get the globally registered prompt."""
        return cls._registered_prompt

    @classmethod
    def build_from_registered(
        cls,
        tool_registry: "ToolRegistry | None" = None,
        source_registry: "SourceRegistry | None" = None,
    ) -> str:
        """
        Build prompt from globally registered components.

        Args:
            tool_registry: Tool registry
            source_registry: Source registry

        Returns:
            Complete system prompt

        Raises:
            RuntimeError: If no prompt registered
        """
        if cls._registered_prompt is None:
            raise RuntimeError(
                "No system prompt registered. "
                "Call PromptBuilder.register_system_prompt() first."
            )

        builder = cls()
        builder._base_prompt = cls._registered_prompt
        builder._examples = cls._registered_examples.copy()
        builder._sections = cls._registered_sections.copy()

        return builder.build(tool_registry, source_registry)

    @classmethod
    def reset_registered(cls) -> None:
        """Clear all registered prompt components."""
        cls._registered_prompt = None
        cls._registered_examples = []
        cls._registered_sections = {}

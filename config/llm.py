"""
LLM configuration and provider abstraction.

LLMConfig provides a structured way to configure LLM providers.
LLMProvider is the abstract base for making LLM API calls,
with implementations for OpenAI-compatible, Anthropic, and
text-based tool calling (for models like LFM2).
"""

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# ─── Standardized response types ───


@dataclass
class ToolCall:
    """Standardized tool call across all providers."""

    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None  # Gemini thinking models require this to be round-tripped


@dataclass
class LLMResponse:
    """Standardized LLM response across all providers."""

    content: str | None
    tool_calls: list[ToolCall]
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0  # Thinking/reasoning tokens (Gemini thinking, OpenAI o-series)
    cache_created: int = 0  # Tokens written to cache (Anthropic prompt caching)
    cache_read: int = 0  # Tokens read from cache (Anthropic prompt caching)
    raw: Any = None  # Original response for debugging


# ─── LLM Configuration ───


@dataclass
class LLMConfig:
    """
    Configuration for an LLM provider.

    Supports OpenAI-compatible APIs (OpenAI, local servers, vLLM, llama.cpp),
    native Anthropic API, and text-based tool calling for models like LFM2.

    Example YAML configs:

        # OpenAI / local server
        name: "GPT-4"
        provider: openai
        model: gpt-4-turbo
        api_key_env: OPENAI_API_KEY

        # Anthropic (native SDK)
        name: "Claude Sonnet"
        provider: anthropic
        model: claude-sonnet-4-20250514
        api_key_env: ANTHROPIC_API_KEY

        # Text-based tool calling (LFM2 via llama.cpp)
        name: "LFM2 Tool"
        provider: text-tool-calling
        model: LFM2-1.2B-Tool
        api_base: http://localhost:8080/v1
        extra:
          tool_call_start: "<|tool_call_start|>"
          tool_call_end: "<|tool_call_end|>"
          tool_list_start: "<|tool_list_start|>"
          tool_list_end: "<|tool_list_end|>"
          tool_response_start: "<|tool_response_start|>"
          tool_response_end: "<|tool_response_end|>"
    """

    name: str
    provider: Literal["anthropic", "openai", "text-tool-calling"]
    model: str
    api_base: str | None = None
    api_key_env: str | None = None  # env var name for API key
    temperature: float = 0.0
    max_tokens: int = 4096
    context_window: int = 0  # Model context window size (0 = no limit/unknown)
    tools: list[str] | None = None  # Optional: restrict to these tool names (None = all)
    request_delay: float = 0.0  # Seconds to sleep before each LLM request (for rate-limited free APIs)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str | None:
        """Get API key from environment variable."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        # Default env vars by provider
        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return None

    def get_api_base(self) -> str | None:
        """Get API base URL, using defaults if not specified."""
        if self.api_base:
            return self.api_base
        # Default endpoints by provider
        if self.provider == "anthropic":
            return "https://api.anthropic.com"
        elif self.provider == "openai":
            return "https://api.openai.com/v1"
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            provider=data["provider"],
            model=data["model"],
            api_base=data.get("api_base"),
            api_key_env=data.get("api_key_env"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 4096),
            context_window=data.get("context_window", 0),
            tools=data.get("tools"),
            request_delay=data.get("request_delay", 0.0),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LLMConfig":
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "tools": self.tools,
            "request_delay": self.request_delay,
            "extra": self.extra,
        }


# ─── Provider abstraction ───


class LLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Subclass this to add support for new LLM backends.
    Implementations handle API client creation, tool format conversion,
    request/response translation, and retry logic.
    """

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        Call the LLM and return a standardized response.

        Args:
            messages: Chat messages in OpenAI format
            tools: Tool definitions in OpenAI function calling format
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with content and/or tool calls
        """
        pass

    def format_tools(self, tool_definitions: list[dict]) -> list[dict]:
        """
        Convert tool definitions to provider-specific format.

        Default implementation passes through OpenAI format unchanged.
        Override for providers that need different schemas.
        """
        return tool_definitions

    def reset(self) -> None:
        """Reset client connection (e.g., after transient errors)."""
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI SDK provider.

    Works with OpenAI API, local servers (llama.cpp, vLLM, Ollama),
    and any OpenAI-compatible endpoint.
    """

    def __init__(self, api_base: str, api_key: str, timeout: float = 120.0, extra: dict | None = None):
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout
        self._extra = extra or {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._client = OpenAI(
                base_url=self._api_base,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    @staticmethod
    def _normalize_for_strict_alternation(messages: list[dict]) -> list[dict]:
        """Normalize messages for models requiring strict user/assistant alternation (e.g. Gemma).
        - Folds any leading system message into the first user message
        - Converts tool messages to user role and merges consecutive ones
        """
        msgs = list(messages)

        # Fold system message into first user message
        if msgs and msgs[0].get("role") == "system":
            system_content = msgs[0]["content"]
            msgs = msgs[1:]
            for i, m in enumerate(msgs):
                if m.get("role") == "user":
                    msgs[i] = {"role": "user", "content": f"{system_content}\n\n{m['content']}"}
                    break

        # Convert tool → user and merge consecutive tool messages
        merged = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg.get("role") == "tool":
                parts = [msg["content"]]
                while i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool":
                    i += 1
                    parts.append(msgs[i]["content"])
                merged.append({"role": "user", "content": "\n\n".join(parts)})
            else:
                merged.append(msg)
            i += 1
        return merged

    def call(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_client()
        if self._extra.get("merge_tool_messages"):
            messages = self._normalize_for_strict_alternation(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
        )

        message = response.choices[0].message
        usage = getattr(response, "usage", None)

        # Parse tool calls
        parsed_tool_calls = []
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                # Extract thought_signature if present (Gemini thinking models require it to be round-tripped).
                # It lives at tc.model_extra['extra_content']['google']['thought_signature'].
                tc_extra = getattr(tc, "model_extra", None) or {}
                thought_sig = tc_extra.get("extra_content", {}).get("google", {}).get("thought_signature")
                fn = tc.function
                parsed_tool_calls.append(ToolCall(
                    id=tc.id,
                    name=fn.name,
                    arguments=json.loads(fn.arguments),
                    thought_signature=thought_sig,
                ))

        # Extract reasoning/thinking tokens from provider-specific locations.
        # Gemini: response.model_extra["usage_metadata"]["thoughts_token_count"]
        # OpenAI o-series: usage.completion_tokens_details.reasoning_tokens
        tokens_reasoning = 0
        if usage:
            # OpenAI o-series style
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                tokens_reasoning = getattr(details, "reasoning_tokens", 0) or 0
            # Gemini style: thoughts_token_count in usage_metadata (via model_extra)
            if not tokens_reasoning:
                for obj in (response, usage):
                    extra = getattr(obj, "model_extra", None) or {}
                    meta = extra.get("usage_metadata") or extra.get("usageMetadata") or {}
                    if isinstance(meta, dict):
                        tokens_reasoning = meta.get("thoughts_token_count", 0) or 0
                    if tokens_reasoning:
                        break

        return LLMResponse(
            content=message.content,
            tool_calls=parsed_tool_calls,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            tokens_reasoning=tokens_reasoning,
            raw=response,
        )

    def reset(self) -> None:
        self._client = None


class AnthropicProvider(LLMProvider):
    """
    Native Anthropic SDK provider.

    Uses the anthropic Python package directly for native tool calling support.
    """

    def __init__(self, api_key: str, timeout: float = 120.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
            self._client = Anthropic(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def format_tools(self, tool_definitions: list[dict]) -> list[dict]:
        """Convert OpenAI function format to Anthropic tool format."""
        anthropic_tools = []
        for tool_def in tool_definitions:
            func = tool_def.get("function", tool_def)
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def call(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert OpenAI message format to Anthropic format
        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "tool":
                # Anthropic expects tool results as user messages with tool_result content
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # Assistant message with tool calls → Anthropic tool_use blocks
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc)
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": func["name"],
                        "input": args,
                    })
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", ""),
                })

        # Format tools
        anthropic_tools = self.format_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            # Explicit cache breakpoint on system prompt.
            kwargs["system"] = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        if anthropic_tools:
            # Explicit cache breakpoint on last tool — caches all tool definitions.
            # System prompt + tools together must exceed the model's minimum cacheable
            # token count (4096 for Haiku/Opus, 1024 for Sonnet).
            # Cache breakpoints are cumulative: tools cache includes system prompt.
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)

        # Extract cache token counts
        usage = response.usage
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Parse response
        content_text = None
        parsed_tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text = (content_text or "") + block.text
            elif block.type == "tool_use":
                parsed_tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                ))

        return LLMResponse(
            content=content_text,
            tool_calls=parsed_tool_calls,
            # Anthropic splits input tokens when caching: input_tokens only counts
            # uncached tokens. Total actual input = input + cache_created + cache_read.
            tokens_in=response.usage.input_tokens + cache_created + cache_read,
            tokens_out=response.usage.output_tokens,
            cache_created=cache_created,
            cache_read=cache_read,
            raw=response,
        )

    def reset(self) -> None:
        self._client = None


class TextToolCallingProvider(LLMProvider):
    """
    Provider for models that use prompt-based tool calling (e.g., LFM2).

    Injects tool definitions into the system prompt using configurable
    special tokens, and parses tool calls from the response text.
    Uses the OpenAI SDK for the actual HTTP call (works with llama.cpp server,
    vLLM, or any OpenAI-compatible endpoint).

    Configure tokens via LLMConfig.extra:
        extra:
          tool_call_start: "<|tool_call_start|>"
          tool_call_end: "<|tool_call_end|>"
          tool_list_start: "<|tool_list_start|>"
          tool_list_end: "<|tool_list_end|>"
          tool_response_start: "<|tool_response_start|>"
          tool_response_end: "<|tool_response_end|>"
    """

    # Default tokens (LFM2 format)
    DEFAULT_TOKENS = {
        "tool_call_start": "<|tool_call_start|>",
        "tool_call_end": "<|tool_call_end|>",
        "tool_list_start": "<|tool_list_start|>",
        "tool_list_end": "<|tool_list_end|>",
        "tool_response_start": "<|tool_response_start|>",
        "tool_response_end": "<|tool_response_end|>",
    }

    def __init__(self, api_base: str, api_key: str, extra: dict[str, Any] | None = None,
                 timeout: float = 120.0):
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout
        self._client = None
        self._tool_defs: list[dict] = []  # Stored for positional arg resolution

        # Merge user tokens with defaults
        self._tokens = dict(self.DEFAULT_TOKENS)
        if extra:
            for key in self.DEFAULT_TOKENS:
                if key in extra:
                    self._tokens[key] = extra[key]

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._client = OpenAI(
                base_url=self._api_base,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def _inject_tools_into_system(self, messages: list[dict], tools: list[dict]) -> list[dict]:
        """Inject tool definitions into the system prompt using special tokens."""
        if not tools:
            return messages

        # Build tool list JSON (strip the OpenAI wrapper, keep just function defs)
        tool_defs = []
        for tool in tools:
            func = tool.get("function", tool)
            tool_defs.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })

        tool_list_str = json.dumps(tool_defs)
        tool_block = (
            f"List of tools: "
            f"{self._tokens['tool_list_start']}"
            f"{tool_list_str}"
            f"{self._tokens['tool_list_end']}"
        )

        # Inject into system message
        modified = []
        for msg in messages:
            if msg["role"] == "system":
                modified.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tool_block,
                })
            elif msg["role"] == "tool":
                # Convert tool responses to user messages with special tokens
                tool_response = (
                    f"{self._tokens['tool_response_start']}"
                    f"{msg.get('content', '')}"
                    f"{self._tokens['tool_response_end']}"
                )
                modified.append({"role": "user", "content": tool_response})
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # Reconstruct assistant tool call messages as text
                parts = []
                if msg.get("content"):
                    parts.append(msg["content"])
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc)
                    name = func["name"]
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    # Format as Python-style function call (LFM2 format)
                    arg_parts = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                 for k, v in args.items()]
                    call_str = f"{name}({', '.join(arg_parts)})"
                    parts.append(
                        f"{self._tokens['tool_call_start']}[{call_str}]{self._tokens['tool_call_end']}"
                    )
                modified.append({"role": "assistant", "content": " ".join(parts)})
            else:
                modified.append(msg)

        return modified

    def _parse_tool_calls(self, text: str) -> tuple[str | None, list[ToolCall]]:
        """Parse tool calls from response text.

        Tries special tokens first, then falls back to detecting bare
        Python-style function calls like [func(arg="val")] which some
        models emit without the special tokens.
        """
        start_token = re.escape(self._tokens["tool_call_start"])
        end_token = re.escape(self._tokens["tool_call_end"])

        pattern = f"{start_token}(.*?){end_token}"
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for match in matches:
            tool_calls.extend(self._parse_function_calls(match.strip()))

        # Extract content outside of tool call tokens
        content = re.sub(pattern, "", text, flags=re.DOTALL).strip()

        # Fallback: detect tool calls by known function names (handles both
        # [func(...)] and bare func(...), with nested brackets/parens in args)
        if not tool_calls and self._tool_defs:
            tool_names = set()
            for tool in self._tool_defs:
                func = tool.get("function", tool)
                if func.get("name"):
                    tool_names.add(func["name"])
            # Match known_tool( at any position — then use balanced extraction
            remaining = text.strip()
            tool_name_pattern = r'(?:\[)?(' + '|'.join(re.escape(n) for n in tool_names) + r')\('
            while remaining:
                m = re.search(tool_name_pattern, remaining)
                if not m:
                    break
                # Extract balanced function call starting at the function name
                func_start = m.start(1)  # Start of function name (skip optional [)
                call_text = self._extract_balanced_call(remaining, func_start)
                if call_text:
                    tool_calls.extend(self._parse_function_calls(call_text))
                    # Remove the matched call (and optional wrapping []) from remaining
                    end_pos = func_start + len(call_text)
                    # Skip trailing ] if the call was wrapped in brackets
                    if func_start > 0 and remaining[func_start - 1] == '[' and end_pos < len(remaining) and remaining[end_pos] == ']':
                        content_parts = remaining[:func_start - 1] + remaining[end_pos + 1:]
                    else:
                        content_parts = remaining[:func_start] + remaining[end_pos:]
                    remaining = content_parts.strip()
                else:
                    break
            if tool_calls:
                content = remaining if remaining else None

        content = content if content else None
        return content, tool_calls

    def _extract_balanced_call(self, text: str, start: int = 0) -> str | None:
        """Extract a function call with balanced parentheses from text."""
        # Find the opening paren
        paren_start = text.find("(", start)
        if paren_start < 0:
            return None
        depth = 0
        in_str = False
        str_ch = None
        for i in range(paren_start, len(text)):
            c = text[i]
            if in_str:
                if c == str_ch and text[i - 1:i] != "\\":
                    in_str = False
            elif c in ('"', "'"):
                in_str = True
                str_ch = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_function_calls(self, text: str) -> list[ToolCall]:
        """Parse one or more Python-style function calls from text."""
        calls = []
        # Remove wrapping list brackets if present: [func(...)] -> func(...)
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()

        # Parse Python-style function call: func_name(arg1="val1", arg2=val2)
        func_match = re.match(r'(\w+)\((.*)\)', text, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2).strip()
            arguments = self._parse_python_args(args_str, func_name=func_name)
            calls.append(ToolCall(
                id=f"tc_{uuid.uuid4().hex[:8]}",
                name=func_name,
                arguments=arguments,
            ))
        return calls

    def _get_first_param_name(self, func_name: str | None) -> str | None:
        """Get the first required parameter name for a function from stored tool defs."""
        if not func_name or not self._tool_defs:
            return None
        for tool in self._tool_defs:
            func = tool.get("function", tool)
            if func.get("name") == func_name:
                params = func.get("parameters", {})
                required = params.get("required", [])
                if required:
                    return required[0]
                # Fall back to first property
                props = params.get("properties", {})
                if props:
                    return next(iter(props))
        return None

    def _parse_python_args(self, args_str: str, func_name: str | None = None) -> dict[str, Any]:
        """Parse Python-style arguments: key1="val1", key2=123 or positional args."""
        if not args_str:
            return {}

        args = {}
        # Check if this is purely positional (no = sign outside quotes)
        has_keyword = False
        in_str = False
        str_ch = None
        for c in args_str:
            if in_str:
                if c == str_ch:
                    in_str = False
            elif c in ('"', "'"):
                in_str = True
                str_ch = c
            elif c == '=':
                has_keyword = True
                break

        if not has_keyword:
            # Positional arg(s) — try to match to tool parameter names
            val = args_str.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Look up first required parameter name from stored tool definitions
            param_name = self._get_first_param_name(func_name) if func_name else None
            return {param_name or "value": val}

        # Match key=value pairs, handling quoted strings and nested structures
        # This handles: key="value", key=123, key=True, key=[1,2,3]
        current_key = None
        current_value = ""
        in_string = False
        string_char = None
        depth = 0  # Track brackets/parens nesting

        i = 0
        tokens = []
        while i < len(args_str):
            c = args_str[i]
            if in_string:
                current_value += c
                if c == string_char and args_str[i - 1:i] != "\\":
                    in_string = False
            elif c in ('"', "'"):
                in_string = True
                string_char = c
                current_value += c
            elif c in ("[", "{", "("):
                depth += 1
                current_value += c
            elif c in ("]", "}", ")"):
                depth -= 1
                current_value += c
            elif c == "=" and depth == 0 and current_key is None:
                current_key = current_value.strip()
                current_value = ""
            elif c == "," and depth == 0:
                if current_key is not None:
                    tokens.append((current_key, current_value.strip()))
                    current_key = None
                current_value = ""
            else:
                current_value += c
            i += 1

        # Last pair
        if current_key is not None and current_value.strip():
            tokens.append((current_key, current_value.strip()))

        for key, val in tokens:
            # Try to parse the value
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                args[key] = val[1:-1]
            else:
                try:
                    args[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    # Try replacing single quotes with double quotes for Python-style lists
                    if val.startswith("["):
                        try:
                            args[key] = json.loads(val.replace("'", '"'))
                            continue
                        except (json.JSONDecodeError, ValueError):
                            pass
                    args[key] = val

        return args

    def call(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_client()

        # Store tool defs for positional arg resolution during parsing
        self._tool_defs = tools

        # Inject tools into system prompt and convert tool messages
        modified_messages = self._inject_tools_into_system(messages, tools)

        # Call without tools parameter (tools are in the prompt text)
        response = client.chat.completions.create(
            model=model,
            messages=modified_messages,
            temperature=temperature,
        )

        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        raw_content = message.content or ""

        # Parse tool calls from response text
        content, tool_calls = self._parse_tool_calls(raw_content)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def reset(self) -> None:
        self._client = None


# ─── Provider factory ───


def get_provider(config: LLMConfig) -> LLMProvider:
    """
    Create an LLMProvider from an LLMConfig.

    Args:
        config: LLM configuration specifying the provider type

    Returns:
        Appropriate LLMProvider instance
    """
    api_key = config.get_api_key() or os.getenv("LLM_API_KEY", "not-needed")
    api_base = config.get_api_base() or os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")

    if config.provider == "anthropic":
        return AnthropicProvider(api_key=api_key)
    elif config.provider == "text-tool-calling":
        return TextToolCallingProvider(
            api_base=api_base,
            api_key=api_key,
            extra=config.extra,
        )
    else:
        # Default: OpenAI-compatible
        return OpenAIProvider(api_base=api_base, api_key=api_key, extra=config.extra)


# ─── Web config and discovery ───


@dataclass
class WebConfig:
    """
    Configuration for the web UI.

    Specifies available LLM configs and other web-specific settings.

    Example YAML:
        title: "Stock Analysis Agent"
        public_url: "https://my.domain.com:8777"  # For display/links
        host: "0.0.0.0"  # Allow external connections
        port: 8000
        ssl_cert: /path/to/cert.pem  # For HTTPS
        ssl_key: /path/to/key.pem
        preview_rows: 20  # Rows to show in result preview
        configs:
          - anthropic_sonnet
          - openai_gpt4
        default_config: anthropic_sonnet
        poll_interval_ms: 5000
        session_dir: ./sessions
    """

    title: str = "Agent Web UI"
    configs: list[str] = field(default_factory=list)  # Config file names (without .yaml)
    default_config: str | None = None  # First in list if not specified
    poll_interval_ms: int = 5000
    session_dir: str = "./sessions"
    config_dir: str = "./configs"
    # Server settings
    host: str = "127.0.0.1"  # Use "0.0.0.0" for external access
    port: int = 8000
    public_url: str | None = None  # External URL for display (e.g., https://my.domain.com)
    # SSL/HTTPS settings (both required for HTTPS)
    ssl_cert: str | None = None  # Path to SSL certificate file
    ssl_key: str | None = None   # Path to SSL private key file
    # Result display
    preview_rows: int = 10  # Number of rows to show in result preview

    def get_default_config(self) -> str | None:
        """Get the default config name."""
        if self.default_config:
            return self.default_config
        return self.configs[0] if self.configs else None

    def get_ssl_context(self) -> tuple[str, str] | None:
        """Get SSL context tuple (cert, key) if both are configured."""
        if self.ssl_cert and self.ssl_key:
            return (self.ssl_cert, self.ssl_key)
        return None

    def get_display_url(self) -> str:
        """Get the URL to display (public_url if set, otherwise constructed from host/port)."""
        if self.public_url:
            return self.public_url
        protocol = "https" if self.ssl_cert else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "WebConfig":
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            title=data.get("title", "Agent Web UI"),
            configs=data.get("configs", []),
            default_config=data.get("default_config"),
            poll_interval_ms=data.get("poll_interval_ms", 5000),
            session_dir=data.get("session_dir", "./sessions"),
            config_dir=data.get("config_dir", "./configs"),
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 8000),
            public_url=data.get("public_url"),
            ssl_cert=data.get("ssl_cert"),
            ssl_key=data.get("ssl_key"),
            preview_rows=data.get("preview_rows", 10),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "title": self.title,
            "configs": self.configs,
            "default_config": self.get_default_config(),
            "poll_interval_ms": self.poll_interval_ms,
            "session_dir": self.session_dir,
            "config_dir": self.config_dir,
            "host": self.host,
            "port": self.port,
            "public_url": self.public_url,
            "ssl_cert": self.ssl_cert,
            "ssl_key": self.ssl_key,
            "preview_rows": self.preview_rows,
        }


def discover_configs(config_dir: Path | str) -> dict[str, LLMConfig]:
    """
    Discover all LLM configs in a directory.

    Finds both regular .yaml and .example.yaml files.
    For .example.yaml files, the key is the stem without .example.

    Args:
        config_dir: Directory containing .yaml config files

    Returns:
        Dict mapping config name (filename without .yaml) to LLMConfig
    """
    config_dir = Path(config_dir)
    configs = {}

    if not config_dir.exists():
        return configs

    for path in config_dir.glob("*.yaml"):
        try:
            config = LLMConfig.from_yaml(path)
            # Use stem, stripping .example suffix if present
            name = path.stem
            if name.endswith(".example"):
                name = name[:-8]  # Remove ".example"
            # Prefer non-example configs over example ones
            if name not in configs or not path.stem.endswith(".example"):
                configs[name] = config
        except Exception as e:
            # Skip invalid configs
            print(f"Warning: Could not load config {path}: {e}")

    return configs

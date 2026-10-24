"""The LLM boundary: a one-method protocol, the Anthropic implementation, and a test double."""

from __future__ import annotations

from typing import Protocol, Sequence


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        """Return the model's text reply to a single user turn."""
        ...


class AnthropicClient:
    """Thin wrapper over the official SDK; the only module that touches the network."""

    def __init__(self, api_key: str | None, model: str, client: object | None = None):
        if client is None:
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Put it in the environment or a .env file, "
                    "or run with --offline."
                )
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self._client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


class ScriptedLLMClient:
    """Returns canned replies in order (the last repeats); records every call. For tests/demos."""

    def __init__(self, replies: Sequence[str]):
        if not replies:
            raise ValueError("ScriptedLLMClient needs at least one reply")
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        self.calls.append((system, user))
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[index]

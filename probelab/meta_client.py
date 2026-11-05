"""Meta Model API adapter. Secrets never leave this module."""

import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


@dataclass(frozen=True, slots=True)
class NarrativeResult:
    text: str
    status: str
    model: str
    finish_reason: str
    request_status: str
    completion_status: str
    response_text_length: int
    error_detail: str = ""


def _error_detail(error: BaseException) -> str:
    """Return a short provider detail without credential-like substrings."""
    detail = str(error)
    secret = os.getenv("META_API")
    if secret:
        detail = detail.replace(secret, "[redacted]")
    detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", detail)
    detail = re.sub(
        r"(?i)(authorization|api[_ -]?key|access[_ -]?token)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2[redacted]",
        detail,
    )
    return detail[:240] or type(error).__name__


class MetaMuseClient:
    """Small boundary around the official OpenAI-compatible Meta endpoint."""

    def __init__(self) -> None:
        load_dotenv(override=False)
        key = os.getenv("META_API")
        self._client = OpenAI(base_url="https://api.meta.ai/v1", api_key=key) if key else None

    @property
    def is_configured(self) -> bool:
        """Whether the Meta client has credentials, without exposing them."""
        return self._client is not None

    def reason(self, prompt: str, *, single_attempt: bool = False) -> NarrativeResult:
        """Ask Muse Spark to narrate a state assembled by deterministic tools."""
        if self._client is None:
            detail = "META_API is not configured."
            return NarrativeResult("Muse Spark reasoning unavailable: scientific calculations remain available.", "Failed", "muse-spark-1.3", "not_called", "not_invoked", "not_started", 0, detail)
        try:
            client = self._client.with_options(max_retries=0, timeout=45.0) if single_attempt else self._client
            response = client.chat.completions.create(
                model="muse-spark-1.3",
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="minimal" if single_attempt else "low",
                max_completion_tokens=2048 if single_attempt else 1024,
            )
            content = response.choices[0].message.content
            finish_reason = str(response.choices[0].finish_reason)
            if finish_reason == "length":
                partial_text = content or ""
                return NarrativeResult(
                    partial_text,
                    "Incomplete response",
                    str(response.model),
                    finish_reason,
                    "completed",
                    "truncated",
                    len(partial_text),
                    "Output-token budget was exhausted (finish_reason=length); visible text is incomplete.",
                )
            if content:
                return NarrativeResult(content, "Completed", str(response.model), finish_reason, "completed", "completed", len(content))
            return NarrativeResult(f"Muse Spark returned no visible text (finish_reason={finish_reason}).", "Empty response", str(response.model), finish_reason, "completed", "empty", 0)
        except (OpenAIError, OSError, RuntimeError) as error:
            detail = _error_detail(error)
            return NarrativeResult(f"Muse Spark unavailable: {detail}", "Failed", "muse-spark-1.3", "error", "failed", "failed", 0, detail)

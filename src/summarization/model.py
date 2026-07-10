"""Build a bounded model client for semantic context summaries."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import LLMSettings, SummarizationSettings


def build_summary_model(
    llm_settings: LLMSettings,
    summarization_settings: SummarizationSettings,
) -> ChatOpenAI:
    """Create an isolated summary model with conservative generation limits."""
    model_name = (
        llm_settings.model
        if summarization_settings.model == "main"
        else summarization_settings.model
    )
    extra_body = None
    if not summarization_settings.reasoning_enabled:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key,
        model=model_name,
        temperature=0,
        output_version="responses/v1",
        use_responses_api=True,
        profile={"max_input_tokens": llm_settings.context_window_tokens},
        timeout=summarization_settings.timeout_seconds,
        max_retries=summarization_settings.max_retries,
        max_completion_tokens=summarization_settings.max_output_tokens,
        extra_body=extra_body,
    )


__all__ = ["build_summary_model"]

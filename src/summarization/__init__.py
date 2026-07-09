"""Summarization package public interface."""

from src.summarization.context_compaction import (
    ContextCompactionResult,
    LayerOneProjection,
    build_context_compaction_request,
    build_todo_snapshot_from_request,
)
from src.summarization.layered_context import (
    CompactLayeredContextState,
    build_layered_context_state,
    has_compressible_history,
    partition_messages_for_compaction,
)
from src.summarization.middleware import build_summarization_middleware
from src.summarization.tool_observation import (
    CompactObservationLedger,
    ToolObservation,
    build_tool_observations,
    compact_tool_observations,
    json_shape_summary,
    json_stats_summary,
)

__all__ = [
    "CompactLayeredContextState",
    "CompactObservationLedger",
    "ContextCompactionResult",
    "LayerOneProjection",
    "ToolObservation",
    "build_context_compaction_request",
    "build_todo_snapshot_from_request",
    "build_layered_context_state",
    "build_summarization_middleware",
    "build_tool_observations",
    "compact_tool_observations",
    "has_compressible_history",
    "json_shape_summary",
    "json_stats_summary",
    "partition_messages_for_compaction",
]

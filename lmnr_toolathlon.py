"""Laminar tracing for the Toolathlon agent harness.

Importing this module is all that is needed, and `main.py` already does it.
(`run_parallel.py` shells out to `main.py`, so it is covered too.) Every task
run becomes one Laminar trace:

    toolathlon.task.<task-id>          root span, carries task/model metadata
    └── Agent workflow
        └── <agent name>
            ├── agents.mcp_tools       MCP server tool listings
            ├── agents.generation      one per LLM call, with tokens and cost
            └── <tool name>            one per tool call, with input/output

Install with `uv pip install lmnr` and set `LMNR_PROJECT_API_KEY` to enable it.
The module is a no-op when either is missing, so an uninstrumented run needs no
code change — and `lmnr` is deliberately kept out of `pyproject.toml` so the
pinned `uv.lock` stays untouched. `LMNR_BASE_URL` points at a self-hosted
backend, and `TOOLATHLON_RUN_ID` groups the tasks of one benchmark sweep into a
single Laminar session.

Three things about this harness need working around, hence the module:

  1. `utils/api_model/model_provider.py` calls `set_tracing_disabled(True)` at
     import time, which switches the Agents SDK to NoOp spans. We re-enable
     tracing and neutralise later calls so import order does not matter.
  2. Toolathlon pins `openai-agents==0.0.15` while Laminar's instrumentor
     declares `openai-agents >= 0.7.0`. The tracing-processor API is
     compatible, so we register the processor directly instead of going
     through the version-gated auto-instrumentation.
  3. The Agents SDK's default processor uploads to OpenAI's tracing backend.
     We replace the processor list rather than appending to it.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_API_KEY = os.environ.get("LMNR_PROJECT_API_KEY")


def _install() -> None:
    from lmnr import Laminar
    from lmnr.opentelemetry_lib.tracing.instruments import Instruments

    # The raw `openai` instrumentation is disabled on purpose: the Agents
    # processor already emits an LLM span per model call, and running both
    # nests a duplicate `openai.chat` span under every `agents.generation`,
    # double-counting tokens. OPENAI_AGENTS is disabled because we register
    # that processor by hand below.
    Laminar.initialize(
        project_api_key=_PROJECT_API_KEY,
        base_url=os.environ.get("LMNR_BASE_URL") or None,
        disabled_instruments={Instruments.OPENAI_AGENTS, Instruments.OPENAI},
    )

    from agents.tracing import set_trace_processors
    from agents.tracing.setup import GLOBAL_TRACE_PROVIDER
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.processor import (
        LaminarAgentsTraceProcessor,
    )

    GLOBAL_TRACE_PROVIDER.set_disabled(False)
    # `model_provider` re-disables tracing whenever it is imported; swallow
    # that call rather than depending on import order.
    GLOBAL_TRACE_PROVIDER.set_disabled = lambda disabled: None
    set_trace_processors([LaminarAgentsTraceProcessor()])

    _wrap_task_runner(Laminar)
    atexit.register(Laminar.flush)


def _metadata(task_config: Any, agent_config: Any) -> dict[str, Any]:
    model = getattr(agent_config, "model", None)
    meta = {
        "benchmark": "toolathlon",
        "task_id": getattr(task_config, "id", None),
        "task_dir": getattr(task_config, "task_dir", None),
        "model": getattr(model, "short_name", None),
        "provider": getattr(model, "provider", None),
        "max_turns": getattr(task_config, "max_turns", None),
        "max_steps": getattr(task_config, "max_steps_under_single_turn_mode", None),
        "single_turn_mode": getattr(task_config, "single_turn_mode", None),
        "mcp_servers": ",".join(getattr(task_config, "needed_mcp_servers", None) or []),
        "launch_time": getattr(task_config, "launch_time", None),
    }
    return {k: v for k, v in meta.items() if v is not None}


def _wrap_task_runner(Laminar: Any) -> None:
    """Wrap each task run in a root span carrying the task's metadata."""
    from utils.task_runner.runner import TaskRunner

    original_run_single_task = TaskRunner.run_single_task

    async def traced_run_single_task(task_config, agent_config, *args, **kwargs):
        task_id = getattr(task_config, "id", "unknown")
        with Laminar.start_as_current_span(
            name=f"toolathlon.task.{task_id}",
            input={"task": getattr(task_config, "task_str", None)},
            session_id=os.environ.get("TOOLATHLON_RUN_ID") or task_id,
            metadata=_metadata(task_config, agent_config),
            tags=["toolathlon"],
        ):
            status = await original_run_single_task(
                task_config, agent_config, *args, **kwargs
            )
            Laminar.set_span_output(getattr(status, "value", str(status)))
            return status

    TaskRunner.run_single_task = staticmethod(traced_run_single_task)


if _PROJECT_API_KEY:
    try:
        _install()
    except Exception:
        # Tracing must never take a benchmark run down with it.
        logger.exception("Laminar tracing failed to initialize; continuing untraced")

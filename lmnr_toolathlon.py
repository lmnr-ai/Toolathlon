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

One deployment note, since it bites hard and silently: `uv pip install lmnr`
moves shared transitive pins (opentelemetry-*) off `uv.lock`, and the stdio MCP
servers launched with `uv run` re-sync the project on startup. That resync
re-downloads the locked wheels inside the server subprocess, and if the index is
slow the server blows its `client_session_timeout_seconds` and never connects.
Exporting `UV_NO_SYNC` in the parent does nothing about it -- the mcp SDK builds
the subprocess environment as `get_default_environment() | params.env`, and
`get_default_environment()` inherits only HOME/LOGNAME/PATH/SHELL/TERM/USER. So
either add `lmnr` to `pyproject.toml` to keep the lock consistent, or put
`UV_NO_SYNC: "1"` in the `env:` block of each `uv`-launched server's yaml, which
is the only channel that reaches it.
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

    from agents.tracing.span_data import GenerationSpanData, ResponseSpanData

    class CountingAgentsTraceProcessor(LaminarAgentsTraceProcessor):
        """Tallies model calls so `num_steps` survives a run that never evaluates.

        The Toolathlon log file's `agent_llm_requests` is preferred when it
        exists; this only covers runs that die before writing it.
        """

        def on_span_end(self, span: Any) -> None:
            if isinstance(getattr(span, "span_data", None), (GenerationSpanData, ResponseSpanData)):
                _RootSpan.llm_calls += 1
            super().on_span_end(span)

    GLOBAL_TRACE_PROVIDER.set_disabled(False)
    # `model_provider` re-disables tracing whenever it is imported; swallow
    # that call rather than depending on import order.
    GLOBAL_TRACE_PROVIDER.set_disabled = lambda disabled: None
    set_trace_processors([CountingAgentsTraceProcessor()])

    _wrap_task_runner(Laminar)
    atexit.register(Laminar.flush)


def _base_metadata(task_config: Any, agent_config: Any) -> dict[str, Any]:
    """The metadata known before the task starts.

    The key set is fixed by the trajectory consumer, so keep the names and the
    value types stable: `generated` is a bool, `num_steps` an int, and the
    free-form `metadata` key is a JSON string (Laminar metadata values are
    scalars, so a nested object has to be encoded).
    """
    model = getattr(agent_config, "model", None)
    meta = {
        "source": "toolathlon",
        "domain": "general",
        "generated": True,
        "harness": "openai-agents",
        "model": getattr(model, "short_name", None),
        "task_id": getattr(task_config, "id", None),
    }
    return {k: v for k, v in meta.items() if v is not None}


def _details(task_config: Any, agent_config: Any) -> dict[str, Any]:
    """The descriptive half of the free-form `metadata` value."""
    model = getattr(agent_config, "model", None)
    detail = {
        "task_id": getattr(task_config, "id", None),
        "task_dir": getattr(task_config, "task_dir", None),
        "provider": getattr(model, "provider", None),
        "max_turns": getattr(task_config, "max_turns", None),
        "max_steps": getattr(task_config, "max_steps_under_single_turn_mode", None),
        "single_turn_mode": getattr(task_config, "single_turn_mode", None),
        "mcp_servers": getattr(task_config, "needed_mcp_servers", None) or [],
        "run_id": os.environ.get("TOOLATHLON_RUN_ID"),
    }
    return {k: v for k, v in detail.items() if v is not None}


class _RootSpan:
    """The task's root span, held open until the evaluator has run.

    `main.py` evaluates *after* `TaskRunner.run_single_task` returns, so a
    context manager around the run would close the span before the verdict and
    the final token counts exist. Instead the span is started by hand, kept
    current for the duration of the run, and closed by the evaluator hook (or
    by `atexit`, if the run dies before evaluation).
    """

    span: Any = None
    details: dict[str, Any] = {}
    llm_calls: int = 0
    log_file: str | None = None


def _clip(value: Any, limit: int = 4000) -> Any:
    """Keep an evaluator message from dominating the trace's metadata.

    Some evaluators dump a full per-row diff on failure. The head carries the
    verdict; the tail is the same complaint repeated, and the whole thing still
    lives in the run's `eval_res.json`.
    """
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    return value


def _finish(Laminar: Any, eval_res: dict[str, Any] | None = None) -> None:
    """Attach the outcome to the root span and end it. Idempotent."""
    span = _RootSpan.span
    if span is None:
        return
    _RootSpan.span = None

    import json

    detail = dict(_RootSpan.details)
    num_steps = _RootSpan.llm_calls

    # Toolathlon's own accounting is authoritative when the log file exists;
    # `agent_llm_requests` is the count of model calls the agent actually made.
    # The processor's tally is only a fallback for runs that died before writing
    # the log -- and the test is `is not None`, not truthiness, because a
    # genuine 0 is the single most important value here. It means the agent loop
    # never got a completion back, and the tally would hide exactly that: a
    # rejected request still opens and ends a generation span, so a task that
    # spent every one of its inner steps failing the same call would report
    # dozens of steps against Toolathlon's zero.
    stats = {}
    if _RootSpan.log_file and os.path.exists(_RootSpan.log_file):
        try:
            with open(_RootSpan.log_file, encoding="utf-8") as f:
                dump = json.load(f)
            stats = dump.get("key_stats") or {}
            detail["status"] = dump.get("status")
            detail["agent_cost"] = (dump.get("agent_cost") or {}).get("total_cost")
            if stats.get("agent_llm_requests") is not None:
                num_steps = stats["agent_llm_requests"]
        except Exception:
            logger.exception("could not read Toolathlon stats from %s", _RootSpan.log_file)

    if stats:
        detail["key_stats"] = stats
    if eval_res is not None:
        detail["evaluation"] = {
            "pass": eval_res.get("pass", False),
            "details": _clip(eval_res.get("details")),
            "failure": _clip(eval_res.get("failure")),
        }

    try:
        with Laminar.use_span(span, end_on_exit=False):
            Laminar.set_trace_metadata(
                {"metadata": json.dumps(detail, default=str), "num_steps": int(num_steps)}
            )
            if eval_res is not None:
                Laminar.set_span_output(
                    {"pass": eval_res.get("pass", False), **detail.get("evaluation", {})}
                )
    except Exception:
        logger.exception("could not attach Laminar trace metadata")
    finally:
        span.end()
        Laminar.flush()


def _wrap_task_runner(Laminar: Any) -> None:
    """Open the root span around the run and close it after evaluation."""
    from utils.task_runner.runner import TaskRunner
    from utils.evaluation.evaluator import TaskEvaluator

    original_run_single_task = TaskRunner.run_single_task
    original_evaluate = TaskEvaluator.evaluate_from_log_file

    async def traced_run_single_task(task_config, agent_config, *args, **kwargs):
        task_id = getattr(task_config, "id", "unknown")
        _RootSpan.details = _details(task_config, agent_config)
        _RootSpan.log_file = getattr(task_config, "log_file", None)
        _RootSpan.span = Laminar.start_span(
            name=f"toolathlon.task.{task_id}",
            input={"task": getattr(task_config, "task_str", None)},
            session_id=os.environ.get("TOOLATHLON_RUN_ID") or task_id,
            metadata=_base_metadata(task_config, agent_config),
            tags=["toolathlon"],
        )
        with Laminar.use_span(_RootSpan.span, end_on_exit=False):
            return await original_run_single_task(
                task_config, agent_config, *args, **kwargs
            )

    async def traced_evaluate(log_file_path: str, *args, **kwargs):
        eval_res = await original_evaluate(log_file_path, *args, **kwargs)
        _finish(Laminar, eval_res)
        return eval_res

    TaskRunner.run_single_task = staticmethod(traced_run_single_task)
    TaskEvaluator.evaluate_from_log_file = staticmethod(traced_evaluate)
    # A run that crashes before evaluation still gets a closed, exported span.
    atexit.register(_finish, Laminar, None)


if _PROJECT_API_KEY:
    try:
        _install()
    except Exception:
        # Tracing must never take a benchmark run down with it.
        logger.exception("Laminar tracing failed to initialize; continuing untraced")

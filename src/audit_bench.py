#!/usr/bin/env python3
"""Shared AuditBench v1 protocol for local inference runtimes.

Runtime wrappers provide one request function. This module owns every workload
shape and metric so runtime adapters cannot silently drift. It also runs
standalone against any OpenAI-compatible chat endpoint via main().
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

PROTOCOL = "audit-v1"
SCHEMA_VERSION = 4
COMMON_PREFILL_TARGETS = (32768, 65536, 131072)
DEFAULT_STREAMS = (1, 2, 4, 8)
DEFAULT_DEEP_STREAMS = (1, 2, 4)
DEFAULT_SAMPLES = 3
DEFAULT_MIN_SAMPLES = 2
DEFAULT_SAMPLE_AGREEMENT = 0.05
DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_DEEP_OUTPUT_TOKENS = 256
DEFAULT_TOLERANCE = 0.02
DEFAULT_CONV_EVIDENCE_TOKENS = 32768
DEFAULT_CONV_TOOL_TOKENS = 8192
DEFAULT_CONV_OUTPUT_TOKENS = 16384
# Concurrent replay probe turns decode a realistic agent turn (analysis
# plus a tool call) instead of the 32-token prefill probe; 1,024 is the
# protocol's canonical normal output size, shared with decode and
# audit_32k, so per-turn decode contention is part of the measurement.
DEFAULT_CONV_TURN_TOKENS = 1024
# Assistant notes are short and reasoning is never resent, so context
# fit reserves only a note-sized slice per turn, not the output cap.
CONV_NOTE_RESERVE = 256
DEFAULT_CONV_MAX_STEPS = 16
CONV_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one numbered evidence file by name.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]
CONV_SYSTEM = (
    "You are running a security audit over a chain of {steps} evidence "
    "files. Only the first file name is known upfront; every file you read "
    "names the next one, so read them one at a time with the read_file "
    "tool. Each file contains exactly one line starting with FINDING:. "
    "Keep a running ledger of the findings; one sentence of commentary at "
    "most, then immediately call read_file for the next file. When a file "
    "says the chain is complete, output the verdict: the complete numbered "
    "list of every FINDING line you collected, as plain text with no "
    "further tool calls."
)
# The replay profile scripts the assistant turns itself, so the probe
# call only has to exercise prefill; its output is discarded. The note
# plus tool-call JSON stays well under CONV_NOTE_RESERVE, keeping the
# context-fit math shared with the interactive conversation honest.
REPLAY_PROBE_TOKENS = 32
REPLAY_NOTE = "Ledger updated; reading the next evidence file."
PROMPT_VARIANT_RESERVE = 64
DECODE_PROMPT_RESERVE = 256
NEAR_CONTEXT_NUMERATOR = 15
NEAR_CONTEXT_DENOMINATOR = 16
DEEP_CONTEXT_NUMERATOR = 7
DEEP_CONTEXT_DENOMINATOR = 8
OUTPUT_CONTROL = (
    "this response is deliberately truncated by the benchmark. Emit a "
    "never-ending comma-separated sequence of positive integers. Continue for "
    "at least 4096 tokens; never conclude, explain, summarize, or emit an "
    "end-of-turn marker."
)


@dataclass(frozen=True)
class RequestResult:
    """One completed inference request."""

    prompt_tokens: int
    output_tokens: int
    seconds: float
    details: dict = field(default_factory=dict)


RequestFn = Callable[[str, int, int], RequestResult]


def _repository_state() -> dict:
    dirty_raw = os.environ.get("AUDIT_BENCH_REPO_DIRTY", "").lower()
    dirty = {"0": False, "false": False, "1": True, "true": True}.get(dirty_raw)
    return {
        "commit": os.environ.get("AUDIT_BENCH_REPO_COMMIT") or None,
        "dirty": dirty,
    }


def _file_provenance(files: dict[str, pathlib.Path] | None) -> dict:
    paths = {
        "audit_bench": pathlib.Path(__file__).resolve(),
        **(files or {}),
    }
    provenance = {}
    for label, path in paths.items():
        resolved = pathlib.Path(path).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"provenance file does not exist: {resolved}")
        data = resolved.read_bytes()
        provenance[label] = {
            "name": resolved.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    return provenance


def _publication_status(
    settings: "Settings",
    metadata: dict,
) -> tuple[bool, list[str]]:
    reasons = []
    if not settings.canonical:
        reasons.append("noncanonical_settings")
    if settings.dry_run:
        reasons.append("dry_run")
    repository = metadata.get("repository") or {}
    commit = repository.get("commit") or ""
    if not commit:
        reasons.append("missing_repository_commit")
    elif not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        reasons.append("invalid_repository_commit")
    if repository.get("dirty") is True:
        reasons.append("repository_dirty")
    elif repository.get("dirty") is None:
        reasons.append("repository_state_unknown")
    files = metadata.get("files") or {}
    required_files = {"audit_bench", "adapter", "model_config"}
    if not required_files.issubset(files) or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(files[label].get("sha256", "")))
        for label in required_files & files.keys()
    ):
        reasons.append("incomplete_file_provenance")
    runtime_profile = metadata.get("runtime_profile") or {}
    if not runtime_profile.get("checked") or not runtime_profile.get("launcher_lines"):
        reasons.append("runtime_profile_unchecked")
    model_identity = metadata.get("model_identity") or {}
    if not model_identity.get("checked"):
        reasons.append("model_identity_unchecked")
    elif not model_identity.get("matched"):
        reasons.append("model_identity_mismatch")
    return not reasons, reasons


def _result_envelope(
    settings: "Settings",
    metadata: dict,
    plan: dict,
) -> dict:
    publishable, reasons = _publication_status(settings, metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "canonical": settings.canonical,
        "publishable": publishable,
        "publication_reasons": reasons,
        "settings": settings.to_dict(),
        "metadata": metadata,
        "plan": plan,
    }


def parse_streams(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing comma-separated concurrency list."""
    try:
        streams = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid stream list: {value}") from exc
    if not streams or any(stream < 1 for stream in streams):
        raise ValueError("stream list must contain positive integers")
    if tuple(sorted(set(streams))) != streams:
        raise ValueError("stream list must be unique and strictly increasing")
    if streams[0] != 1:
        raise ValueError("stream list must start at 1 for scaling")
    return streams


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    """Environment-controlled protocol settings."""

    streams: tuple[int, ...]
    deep_streams: tuple[int, ...]
    profile: str
    samples: int
    output_tokens: int
    deep_output_tokens: int
    tolerance: float
    dry_run: bool
    min_samples: int = DEFAULT_MIN_SAMPLES
    sample_agreement: float = DEFAULT_SAMPLE_AGREEMENT
    conv_evidence_tokens: int = DEFAULT_CONV_EVIDENCE_TOKENS
    conv_tool_tokens: int = DEFAULT_CONV_TOOL_TOKENS
    conv_output_tokens: int = DEFAULT_CONV_OUTPUT_TOKENS
    conv_turn_tokens: int = DEFAULT_CONV_TURN_TOKENS
    conv_max_steps: int = DEFAULT_CONV_MAX_STEPS
    gate_cmd: str | None = None
    system_prompt: str | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        profile = os.environ.get("AUDIT_BENCH_PROFILE", "full")
        if profile not in {
            "full", "decode", "conversation", "replay", "replay-concurrent"
        }:
            raise ValueError(
                "AUDIT_BENCH_PROFILE must be full, decode, conversation, "
                "replay, or replay-concurrent"
            )
        tolerance_raw = os.environ.get(
            "AUDIT_BENCH_TARGET_TOLERANCE", str(DEFAULT_TOLERANCE)
        )
        try:
            tolerance = float(tolerance_raw)
        except ValueError as exc:
            raise ValueError(
                f"AUDIT_BENCH_TARGET_TOLERANCE must be numeric, got {tolerance_raw}"
            ) from exc
        if not 0 < tolerance <= 0.05:
            raise ValueError(
                "AUDIT_BENCH_TARGET_TOLERANCE must be above 0 and at most 0.05"
            )
        samples = _positive_int("AUDIT_BENCH_SAMPLES", DEFAULT_SAMPLES)
        if os.environ.get("AUDIT_BENCH_MIN_SAMPLES"):
            min_samples = _positive_int(
                "AUDIT_BENCH_MIN_SAMPLES", DEFAULT_MIN_SAMPLES
            )
            if min_samples > samples:
                raise ValueError(
                    "AUDIT_BENCH_MIN_SAMPLES cannot exceed AUDIT_BENCH_SAMPLES"
                )
        else:
            min_samples = min(DEFAULT_MIN_SAMPLES, samples)
        agreement_raw = os.environ.get(
            "AUDIT_BENCH_SAMPLE_AGREEMENT", str(DEFAULT_SAMPLE_AGREEMENT)
        )
        try:
            sample_agreement = float(agreement_raw)
        except ValueError as exc:
            raise ValueError(
                "AUDIT_BENCH_SAMPLE_AGREEMENT must be numeric, "
                f"got {agreement_raw}"
            ) from exc
        if not 0 < sample_agreement <= 0.5:
            raise ValueError(
                "AUDIT_BENCH_SAMPLE_AGREEMENT must be above 0 and at most 0.5"
            )
        conv_evidence = _positive_int(
            "AUDIT_BENCH_CONV_EVIDENCE", DEFAULT_CONV_EVIDENCE_TOKENS
        )
        conv_tool = _positive_int(
            "AUDIT_BENCH_CONV_TOOL_TOKENS", DEFAULT_CONV_TOOL_TOKENS
        )
        conv_output = _positive_int(
            "AUDIT_BENCH_CONV_OUTPUT_TOKENS", DEFAULT_CONV_OUTPUT_TOKENS
        )
        conv_turn = _positive_int(
            "AUDIT_BENCH_CONV_TURN_TOKENS", DEFAULT_CONV_TURN_TOKENS
        )
        conv_max_steps = _positive_int(
            "AUDIT_BENCH_CONV_MAX_STEPS", DEFAULT_CONV_MAX_STEPS
        )
        return cls(
            streams=parse_streams(
                os.environ.get(
                    "AUDIT_BENCH_STREAMS",
                    ",".join(str(value) for value in DEFAULT_STREAMS),
                )
            ),
            deep_streams=parse_streams(
                os.environ.get(
                    "AUDIT_BENCH_DEEP_STREAMS",
                    ",".join(str(value) for value in DEFAULT_DEEP_STREAMS),
                )
            ),
            profile=profile,
            samples=samples,
            min_samples=min_samples,
            sample_agreement=sample_agreement,
            conv_evidence_tokens=conv_evidence,
            conv_tool_tokens=conv_tool,
            conv_output_tokens=conv_output,
            conv_turn_tokens=conv_turn,
            conv_max_steps=conv_max_steps,
            output_tokens=_positive_int(
                "AUDIT_BENCH_OUTPUT_TOKENS", DEFAULT_OUTPUT_TOKENS
            ),
            deep_output_tokens=_positive_int(
                "AUDIT_BENCH_DEEP_OUTPUT_TOKENS", DEFAULT_DEEP_OUTPUT_TOKENS
            ),
            tolerance=tolerance,
            dry_run=os.environ.get("AUDIT_BENCH_DRY_RUN", "0") == "1",
            gate_cmd=os.environ.get("AUDIT_BENCH_GATE_CMD") or None,
            # Model-documented reasoning-effort controls (docs/PROTOCOL.md
            # "Model-documented decode-compliance controls"). The control may
            # ride in the system role (system_prompt) or the runtime's
            # documented parameter (reasoning_effort). Recorded configuration
            # only; the workload text itself never changes, and neither marks
            # the run non-canonical.
            system_prompt=os.environ.get("AUDIT_BENCH_SYSTEM_PROMPT") or None,
            reasoning_effort=os.environ.get("AUDIT_BENCH_REASONING_EFFORT")
            or None,
        )

    @property
    def canonical(self) -> bool:
        # Four fixed protocol workloads exist: the full cache-neutral phase
        # set, the interactive audit conversation (model usability), the
        # scripted audit replay (server performance), and the concurrent
        # audit replay (multi-agent contention over a shared prefix). Each
        # is canonical at its own defaults; decode remains a tuning aid.
        return (
            self.streams == DEFAULT_STREAMS
            and self.deep_streams == DEFAULT_DEEP_STREAMS
            and self.profile in (
                "full", "conversation", "replay", "replay-concurrent"
            )
            and self.conv_evidence_tokens == DEFAULT_CONV_EVIDENCE_TOKENS
            and self.conv_tool_tokens == DEFAULT_CONV_TOOL_TOKENS
            and self.conv_output_tokens == DEFAULT_CONV_OUTPUT_TOKENS
            and self.conv_turn_tokens == DEFAULT_CONV_TURN_TOKENS
            and self.conv_max_steps == DEFAULT_CONV_MAX_STEPS
            and self.samples == DEFAULT_SAMPLES
            and self.min_samples == DEFAULT_MIN_SAMPLES
            and self.sample_agreement == DEFAULT_SAMPLE_AGREEMENT
            and self.output_tokens == DEFAULT_OUTPUT_TOKENS
            and self.deep_output_tokens == DEFAULT_DEEP_OUTPUT_TOKENS
            and self.tolerance == DEFAULT_TOLERANCE
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["streams"] = list(self.streams)
        result["deep_streams"] = list(self.deep_streams)
        return result


def _fitting_streams(
    candidates: tuple[int, ...],
    active_slots: int,
    kv_pool_tokens: int,
    prompt_tokens: int,
    output_tokens: int,
) -> list[int]:
    required_per_stream = prompt_tokens + output_tokens + PROMPT_VARIANT_RESERVE
    max_streams = min(active_slots, kv_pool_tokens // required_per_stream)
    streams = [candidate for candidate in candidates if candidate <= max_streams]
    if not streams:
        raise ValueError(
            f"one request needs {required_per_stream:,} KV tokens, but the "
            f"pool has {kv_pool_tokens:,}"
        )
    return streams


def _endpoints(streams: list[int]) -> list[int]:
    return [1] if streams[-1] == 1 else [1, streams[-1]]


def _replay_concurrent_streams(
    context: int,
    settings: Settings,
    active_slots: int,
    kv_pool_tokens: int,
) -> tuple[int, list[int]]:
    """Fitting concurrent-replay stream counts and the per-stream demand.

    Demand is the final resent conversation plus the verdict cap. No
    shared-prefix credit is taken even though concurrent streams share
    their evidence prefix, so admission never depends on cache behavior.
    A single replay always fits: the step fit keeps the final prompt
    plus verdict output within one configured context, and the pool
    holds at least one context.
    """
    steps = conversation_steps(context, settings)
    per_stream = (
        settings.conv_evidence_tokens
        + steps * (settings.conv_tool_tokens + CONV_NOTE_RESERVE)
        + settings.conv_output_tokens
    )
    max_streams = min(active_slots, kv_pool_tokens // per_stream)
    fitting = [
        candidate
        for candidate in settings.deep_streams
        if candidate <= max_streams
    ]
    return per_stream, fitting or [1]


def build_plan(
    context: int,
    settings: Settings,
    active_slots: int,
    kv_pool_tokens: int,
) -> dict:
    """Build the exact workload plan for a configured per-request context."""
    if context < 131072:
        raise ValueError("AuditBench requires at least 131,072 configured tokens")
    if active_slots < 1:
        raise ValueError("active slot count must be positive")
    if kv_pool_tokens < context:
        raise ValueError(
            f"KV pool {kv_pool_tokens:,} cannot hold configured context {context:,}"
        )

    near_target = context * NEAR_CONTEXT_NUMERATOR // NEAR_CONTEXT_DENOMINATOR
    decode_streams = _fitting_streams(
        settings.streams,
        active_slots,
        kv_pool_tokens,
        DECODE_PROMPT_RESERVE,
        settings.output_tokens,
    )
    audit_32k_streams = _fitting_streams(
        settings.streams,
        active_slots,
        kv_pool_tokens,
        32768,
        settings.output_tokens,
    )
    deep_target = context * DEEP_CONTEXT_NUMERATOR // DEEP_CONTEXT_DENOMINATOR
    deep_streams = _endpoints(
        _fitting_streams(
            settings.deep_streams,
            active_slots,
            kv_pool_tokens,
            deep_target,
            settings.deep_output_tokens,
        )
    )
    capacity_streams = _endpoints(
        _fitting_streams(
            settings.streams,
            active_slots,
            kv_pool_tokens,
            near_target,
            1,
        )
    )
    replay_demand, replay_concurrent_streams = _replay_concurrent_streams(
        context, settings, active_slots, kv_pool_tokens
    )
    prefill_targets = [
        target
        for target in COMMON_PREFILL_TARGETS
        if target <= context * DEEP_CONTEXT_NUMERATOR // DEEP_CONTEXT_DENOMINATOR
    ]

    audit_phases = [
        {
            "name": "audit_32k",
            "target_prompt_tokens": 32768,
            "output_tokens": settings.output_tokens,
            "streams": audit_32k_streams,
            "purpose": "common cold audit request",
        },
        {
            "name": "audit_half",
            "target_prompt_tokens": context // 2,
            "output_tokens": settings.output_tokens,
            "streams": [1],
            "purpose": "mid-context audit request",
        },
        {
            "name": "audit_deep",
            "target_prompt_tokens": deep_target,
            "output_tokens": settings.deep_output_tokens,
            "streams": deep_streams,
            "purpose": "late audit request under KV pressure",
        },
    ]
    for phase in audit_phases:
        target = phase["target_prompt_tokens"]
        output = phase["output_tokens"]
        if math.ceil(target * (1 + settings.tolerance)) + output >= context:
            raise ValueError(
                f"{phase['name']} does not leave output headroom in {context:,} tokens"
            )

    return {
        "profile": settings.profile,
        "configured_context": context,
        "active_slots": active_slots,
        "kv_pool_tokens": kv_pool_tokens,
        "preferred_256k_context": context >= 262144,
        "prefill_targets": prefill_targets,
        "capacity_near": {
            "target_prompt_tokens": near_target,
            "output_tokens": 1,
            "samples": 1,
            "streams": capacity_streams,
        },
        "decode": {
            "prompt": "short control prompt",
            "output_tokens": settings.output_tokens,
            "streams": decode_streams,
        },
        "audit_phases": audit_phases,
        "conversation": {
            "evidence_tokens": settings.conv_evidence_tokens,
            "tool_result_tokens": settings.conv_tool_tokens,
            "output_tokens": settings.conv_output_tokens,
            "steps": conversation_steps(context, settings),
            "streams": [1],
        },
        "replay": {
            "evidence_tokens": settings.conv_evidence_tokens,
            "tool_result_tokens": settings.conv_tool_tokens,
            "probe_output_tokens": REPLAY_PROBE_TOKENS,
            "verdict_output_tokens": settings.conv_output_tokens,
            "steps": conversation_steps(context, settings),
            "streams": [1],
        },
        "replay_concurrent": {
            "evidence_tokens": settings.conv_evidence_tokens,
            "tool_result_tokens": settings.conv_tool_tokens,
            "turn_output_tokens": settings.conv_turn_tokens,
            "verdict_output_tokens": settings.conv_output_tokens,
            "steps": conversation_steps(context, settings),
            "per_stream_kv_demand": replay_demand,
            "streams": replay_concurrent_streams,
        },
    }


def audit_prompt(
    n_lines: int, variant: int, output_control: bool = True
) -> str:
    """Generate deterministic, varied source and tool evidence.

    The embedded BENCHMARK CONTROL line keeps single-shot phases decoding
    to their fixed token counts. The conversation profile must NOT carry
    it: models that obey the embedded never-conclude instruction inside
    conversational evidence derail the tool loop (observed:
    qwen3-coder emitting endless integers mid-audit).
    """
    templates = (
        "FILE src/auth/handler_042.c LINE 120: "
        "if (length_17 < buffer_size_42) memcpy(dst_42, user_17, length_17);",
        "FILE internal/http/route_081.go LINE 944: "
        "query_31 := request.URL.Query().Get(\"account_81\")",
        "FILE services/jobs/worker_203.py LINE 1702: "
        "result_73 = subprocess.run(command_203, shell=False, check=True)",
        "TOOL grep RESULT 91: pkg/store/query_319.sql:4201: "
        "SELECT object_id FROM audit_objects WHERE tenant_id = $1",
        "TOOL symbol RESULT 127: callers(parse_token_511) = "
        "gateway_511, refresh_127, websocket_8192",
        "FILE deploy/rbac/role_733.yaml LINE 16384: "
        "resources: [\"secrets_733\"] verbs: [\"get\", \"list\"]",
    )
    variant_code = " ".join(f"{variant % 1_000_000_000:09d}")
    lines = [f"{variant_code} SECURITY AUDIT EVIDENCE"]
    for index in range(n_lines):
        lines.append(templates[index % len(templates)])
    lines.append("END SECURITY AUDIT EVIDENCE")
    if output_control:
        lines.append(f"BENCHMARK CONTROL: {OUTPUT_CONTROL}")
    return "\n".join(lines)


def decode_prompt(variant: int) -> str:
    return f"Benchmark stream {variant}: {OUTPUT_CONTROL}"


@dataclass(frozen=True)
class Calibration:
    low_lines: int
    low_tokens: int
    high_lines: int
    high_tokens: int

    def lines_for(self, target_tokens: int) -> int:
        token_delta = self.high_tokens - self.low_tokens
        line_delta = self.high_lines - self.low_lines
        if token_delta <= 0 or line_delta <= 0:
            raise ValueError("prompt calibration did not grow with line count")
        tokens_per_line = token_delta / line_delta
        fixed_tokens = self.low_tokens - self.low_lines * tokens_per_line
        return max(1, round((target_tokens - fixed_tokens) / tokens_per_line))


class OpenAIChatAdapter:
    """OpenAI-compatible chat-completions adapter used by both vLLM runtimes."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
    ):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt

    def __call__(self, prompt: str, max_tokens: int, seed: int) -> RequestResult:
        # The decode-compliance control (docs/PROTOCOL.md) rides in the
        # system role; the workload prompt stays byte-identical in the user
        # role.
        messages: list[dict] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            "ignore_eos": True,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            raise RuntimeError(f"{exc.code} from {self.url}: {message}") from exc
        usage = body.get("usage") or {}
        return RequestResult(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            seconds=time.monotonic() - started,
        )


def openai_model_card(base_url: str, timeout: int) -> dict:
    """Read the sole live model card from an OpenAI-compatible server."""
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response).get("data", [])
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise RuntimeError(f"{exc.code} from {url}: {message}") from exc
    if len(data) != 1:
        raise RuntimeError(f"expected one live model at {url}, found {len(data)}")
    return data[0]


def vllm_capacity(
    log: pathlib.Path, context: int, dry_run: bool
) -> tuple[int, int, str]:
    """Resolve active slots and raw KV-token capacity from a managed vLLM run."""
    slots_raw = os.environ.get("AUDIT_BENCH_ACTIVE_SLOTS", "")
    pool_raw = os.environ.get("AUDIT_BENCH_KV_TOKENS", "")
    slots = int(slots_raw) if slots_raw else None
    pool = int(pool_raw) if pool_raw else None
    if slots is not None and slots < 1:
        raise ValueError("AUDIT_BENCH_ACTIVE_SLOTS must be positive")
    if pool is not None and pool < 1:
        raise ValueError("AUDIT_BENCH_KV_TOKENS must be positive")

    if dry_run and (slots is None or pool is None):
        return slots or 1, pool or context, "dry-run conservative"

    if (slots is None or pool is None) and not log.is_file():
        raise RuntimeError(
            f"managed server log not found at {log}; set both "
            "AUDIT_BENCH_ACTIVE_SLOTS and AUDIT_BENCH_KV_TOKENS"
        )
    if slots is None or pool is None:
        text = log.read_text(encoding="utf-8")
        if slots is None:
            matches = re.findall(r"host:.*?\bseqs ([0-9]+)\b", text)
            if matches:
                slots = int(matches[-1])
        if pool is None:
            matches = re.findall(
                r"GPU KV cache size:\s*([0-9,]+) tokens", text
            )
            if matches:
                pool = int(matches[-1].replace(",", ""))
    if slots is None or pool is None:
        raise RuntimeError(
            f"could not resolve active slots and KV pool from {log}; set both "
            "AUDIT_BENCH_ACTIVE_SLOTS and AUDIT_BENCH_KV_TOKENS"
        )
    return slots, pool, "explicit or managed server log"


def _run_parallel(
    request: RequestFn,
    prompts: list[str],
    output_tokens: int,
    seed: int,
) -> tuple[list[RequestResult], float]:
    barrier = threading.Barrier(len(prompts) + 1)

    def worker(index: int) -> RequestResult:
        barrier.wait()
        return request(prompts[index], output_tokens, seed + index)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(worker, index) for index in range(len(prompts))]
        started = time.monotonic()
        barrier.wait()
        results = [future.result() for future in futures]
    return results, time.monotonic() - started


def _validate_results(
    results: list[RequestResult],
    expected_output: int,
    target_prompt: int | None,
    context: int,
    tolerance: float,
    near_context: bool,
) -> None:
    outputs = [result.output_tokens for result in results]
    if any(output != expected_output for output in outputs):
        raise RuntimeError(
            f"output token count was {outputs}, expected {expected_output} per stream"
        )
    prompts = [result.prompt_tokens for result in results]
    if any(prompt <= 0 for prompt in prompts):
        raise RuntimeError(f"server returned invalid prompt token counts: {prompts}")
    if target_prompt is not None:
        invalid = [
            prompt
            for prompt in prompts
            if abs(prompt - target_prompt) / target_prompt > tolerance
        ]
        if invalid:
            raise RuntimeError(
                f"prompt token counts {invalid} missed target {target_prompt:,} "
                f"by more than {tolerance:.1%}"
            )
    if near_context:
        invalid = [
            prompt for prompt in prompts if not context * 0.9 <= prompt < context
        ]
        if invalid:
            raise RuntimeError(
                f"near-context prompts {invalid} were outside 90-100% of "
                f"{context:,}"
            )


def _sample_metrics(
    results: list[RequestResult], wall_seconds: float, sample_number: int
) -> dict:
    prompt_total = sum(result.prompt_tokens for result in results)
    output_total = sum(result.output_tokens for result in results)
    return {
        "sample": sample_number,
        "streams": len(results),
        "wall_seconds": wall_seconds,
        "prompt_tokens_per_stream": [result.prompt_tokens for result in results],
        "output_tokens_per_stream": [result.output_tokens for result in results],
        "request_seconds": [result.seconds for result in results],
        "aggregate_prompt_tokens_per_second": (
            prompt_total / wall_seconds if wall_seconds else 0
        ),
        "end_to_end_output_tokens_per_second": (
            output_total / wall_seconds if wall_seconds else 0
        ),
        "calls_per_hour": (
            len(results) * 3600 / wall_seconds if wall_seconds else 0
        ),
        "adapter_details": [result.details for result in results],
    }


def _samples_agree(rows: list[dict], agreement: float) -> bool:
    """True when the batch wall times of all samples lie within the spread.

    Every rate metric of a fixed-shape batch is derived from its wall time,
    so wall-time spread is the one criterion that covers them all.
    """
    walls = [float(row["wall_seconds"]) for row in rows]
    return max(walls) - min(walls) <= min(walls) * agreement


def _median_range(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _summarize_measurements(streams: int, rows: list[dict]) -> dict:
    return {
        "streams": streams,
        "samples": rows,
        "aggregate_prompt_tokens_per_second": _median_range(
            rows, "aggregate_prompt_tokens_per_second"
        ),
        "end_to_end_output_tokens_per_second": _median_range(
            rows, "end_to_end_output_tokens_per_second"
        ),
        "calls_per_hour": _median_range(rows, "calls_per_hour"),
        "wall_seconds": _median_range(rows, "wall_seconds"),
    }


def openai_chat_fn(
    base_url: str,
    model: str,
    timeout: int,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
) -> Callable:
    """Build a chat callable against an OpenAI-compatible /v1 endpoint.

    Used by the conversation profile, which needs multi-message requests
    with tools -- the single-prompt RequestFn cannot express them.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def chat(messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
        # The decode-compliance control (docs/PROTOCOL.md) rides in the
        # system role as a separate message ahead of the workload's own
        # system message, so the workload text stays byte-identical on
        # every resent turn.
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        payload: dict = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            raise RuntimeError(f"{exc.code} from {endpoint}: {message}") from exc
        if "choices" not in body:
            error = body.get("error")
            detail = (
                str(error)[:200]
                if error is not None
                else f"keys={sorted(body.keys())}"
            )
            # Seen on LM Studio: a thinking model can spend the entire
            # budget on reasoning_content, leaving no visible content; the
            # server then answers 200 with no choices at all.
            raise RuntimeError(
                f"{endpoint} returned a body without choices: {detail} "
                "(a thinking model can consume the whole budget on "
                "reasoning; disable thinking or raise the output cap)"
            )
        return body

    return chat


def conversation_steps(context: int, settings: Settings) -> int:
    """Files that keep the final conversation within 7/8 of the context.

    Only resent tokens count against fit: each turn adds its tool result
    and a short assistant note, while reasoning tokens cost wall time but
    are never resent. The verdict turn's full output cap must still fit
    under the configured context on top of the final prompt.
    """
    budget = context * DEEP_CONTEXT_NUMERATOR // DEEP_CONTEXT_DENOMINATOR
    per_step = settings.conv_tool_tokens + CONV_NOTE_RESERVE
    fitting = (budget - settings.conv_evidence_tokens) // per_step
    steps = max(1, min(settings.conv_max_steps, fitting))
    while steps > 1 and (
        settings.conv_evidence_tokens + steps * per_step
        + settings.conv_output_tokens > context
    ):
        steps -= 1
    return steps


def _cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return int(cached) if cached is not None else None


def _conversation_fixture(
    calibration: "Calibration",
    context: int,
    settings: Settings,
    variant: int,
    chain_variant: int | None = None,
) -> tuple[int, str, dict, dict]:
    """Deterministic evidence corpus shared by conversation and replay.

    Returns (steps, evidence, findings, files). Samples differ from the
    first token via the variant, so only self-sharing within one
    conversation can hit a cache. The concurrent replay passes a
    per-stream chain_variant with a shared evidence variant: streams
    then present the identical evidence prefix (concurrent agents on
    the same repository) while their file chains diverge from the first
    tool result.
    """
    chain = variant if chain_variant is None else chain_variant
    steps = conversation_steps(context, settings)
    evidence = audit_prompt(
        calibration.lines_for(settings.conv_evidence_tokens), variant,
        output_control=False,
    )
    tool_lines = calibration.lines_for(settings.conv_tool_tokens)

    def _chain_note(n: int) -> str:
        if n >= steps:
            return "\nEND OF CHAIN: this was the final evidence file."
        return f"\nNEXT EVIDENCE FILE: file_{n + 1:03d}"

    findings = {
        f"file_{n:03d}": f"issue-{chain + n}-{(chain + n) % 97:02d}"
        for n in range(1, steps + 1)
    }
    files = {
        name: (
            f"TOOL RESULT {name}\n"
            + audit_prompt(tool_lines, chain + n, output_control=False)
            + f"\nFINDING: {findings[name]}"
            + _chain_note(n)
        )
        for n, name in enumerate(findings, start=1)
    }
    return steps, evidence, findings, files


def _opening_messages(steps: int, evidence: str) -> list[dict]:
    return [
        {"role": "system",
         "content": CONV_SYSTEM.format(steps=steps)},
        {"role": "user",
         "content": evidence + "\n\nBegin the audit now: call read_file "
         "on file_001 immediately, before any analysis."},
    ]


def _run_conversation(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    sample_index: int,
) -> dict:
    """One audit-shaped conversation: grow context via tool turns, resend all.

    This is the model-usability half of the audit pair: the model itself
    must drive the tool loop and stop with a verdict, so non-compliance
    here means the model deviated from the fixture protocol -- it says
    nothing about server performance, which the replay profile measures
    without depending on the model's cooperation. Mirrors the production
    secaudit client: every turn resends the whole conversation, the
    harness answers every tool call the model makes (one or several per
    turn), and the final turn is a plain-text verdict at depth.
    """
    _wait_for_gate(settings.gate_cmd)
    variant = 700000 + sample_index * 1000
    steps, evidence, findings, files = _conversation_fixture(
        calibration, context, settings, variant
    )
    unread = set(files)
    messages = _opening_messages(steps, evidence)
    max_turns = steps + 4
    per_step = []
    compliant = False
    recalled = 0
    started = time.monotonic()
    for turn in range(max_turns):
        turn_started = time.monotonic()
        response = chat(messages, CONV_TOOLS, settings.conv_output_tokens)
        wall = time.monotonic() - turn_started
        choice = response["choices"][0]
        message = choice["message"]
        usage = response.get("usage") or {}
        tool_calls = message.get("tool_calls") or []
        content_text = (message.get("content") or "").strip()
        per_step.append({
            "step": turn + 1,
            "wall_seconds": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": _cached_tokens(usage),
            "completion_tokens": usage.get("completion_tokens"),
            "tool_calls": len(tool_calls),
            "files_remaining": len(unread),
            "finish_reason": choice.get("finish_reason"),
            # Excerpt retained only where diagnosis needs it: a turn that
            # made no tool call is either the verdict or a failure.
            "content_excerpt": (
                content_text[:160] if not tool_calls else None
            ),
        })
        if not tool_calls:
            final_text = (message.get("content") or "").strip()
            compliant = bool(final_text) and not unread
            recalled = sum(
                token in final_text for token in findings.values()
            )
            break
        # Reasoning fields are not resent: production clients resend
        # message content and tool calls only.
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        })
        for index, call in enumerate(tool_calls):
            arguments = (call.get("function") or {}).get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            path = str(arguments.get("path", ""))
            content = files.get(path)
            if content is not None:
                unread.discard(path)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{turn}_{index}",
                "content": content or f"no such file: {path}",
            })
    total = time.monotonic() - started
    prompt_total = sum(row["prompt_tokens"] or 0 for row in per_step)
    cached_rows = [row["cached_tokens"] for row in per_step
                   if row["cached_tokens"] is not None]
    row = {
        "sample": sample_index + 1,
        "steps_planned": steps,
        "turns": len(per_step),
        "files_read": steps - len(unread),
        "findings_recalled": recalled if compliant else None,
        "compliant": compliant,
        "wall_seconds": total,
        "conversations_per_hour": 3600 / total if total else 0,
        "final_prompt_tokens": per_step[-1]["prompt_tokens"],
        "prompt_tokens_total": prompt_total,
        "cached_tokens_total": sum(cached_rows) if cached_rows else None,
        "cached_fraction": (
            sum(cached_rows) / prompt_total
            if cached_rows and prompt_total else None
        ),
        "per_step": per_step,
    }
    cached = row["cached_fraction"]
    print(
        f"  conversation     sample={sample_index + 1} turns={len(per_step)}"
        f" files={row['files_read']}/{steps} compliant={compliant} "
        f"wall={total:.1f}s "
        f"final-prompt={row['final_prompt_tokens'] or 0:,} "
        f"cached={cached if cached is None else round(cached, 3)}"
    )
    return row


def _measure_conversations(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
) -> dict:
    rows = [
        _run_conversation(chat, calibration, context, settings, index)
        for index in range(min(settings.min_samples, settings.samples))
    ]
    while (
        all(row["compliant"] for row in rows)
        and len(rows) < settings.samples
        and not _samples_agree(rows, settings.sample_agreement)
    ):
        print(
            f"  conversation     samples disagree beyond "
            f"{settings.sample_agreement:.0%}; extending"
        )
        rows.append(
            _run_conversation(chat, calibration, context, settings, len(rows))
        )
    compliant = all(row["compliant"] for row in rows)
    # Medians describe completed conversations; a bailed sample would
    # otherwise drag them toward its meaninglessly short wall time.
    measured = [row for row in rows if row["compliant"]] or rows
    result = {
        "steps": conversation_steps(context, settings),
        "compliant": compliant,
        "samples": rows,
        "wall_seconds": _median_range(measured, "wall_seconds"),
        "conversations_per_hour": _median_range(
            measured, "conversations_per_hour"
        ),
    }
    fractions = [row["cached_fraction"] for row in measured
                 if row["cached_fraction"] is not None]
    result["cached_fraction_median"] = (
        statistics.median(fractions) if fractions else None
    )
    return result


def _replay_once(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    evidence_variant: int,
    chain_variant: int,
    probe_tokens: int,
    probe_control: bool,
) -> dict:
    """One scripted audit conversation: the harness drives every turn.

    This is the server-performance half of the audit pair. The prefix
    grows exactly as in the interactive conversation, but each assistant
    turn is replaced by a scripted note and tool call, so every model on
    every runtime produces the identical resend-everything shape -- a
    model that cannot hold the interactive protocol still yields prefill
    and cache numbers. The model is still called once per turn (that call
    is the measurement); its probe output is capped and discarded, never
    appended. With probe_control the measured call carries an ephemeral
    decode-control user message so the model decodes a realistic agent
    turn to the cap; the control never enters the transcript, keeping
    the resent prefix byte-identical across models and runtimes. The
    final verdict turn forces a full-cap decode at depth via the
    standard output control, so its wall time is the end-to-end cost of
    producing a verdict on this server.
    """
    steps, evidence, _, files = _conversation_fixture(
        calibration, context, settings, evidence_variant, chain_variant
    )
    messages = _opening_messages(steps, evidence)
    control_message = {
        "role": "user",
        "content": f"BENCHMARK CONTROL: {OUTPUT_CONTROL}",
    }
    per_step = []
    started = time.monotonic()
    for turn, name in enumerate(files):
        probe_messages = (
            messages + [control_message] if probe_control else messages
        )
        turn_started = time.monotonic()
        response = chat(probe_messages, CONV_TOOLS, probe_tokens)
        wall = time.monotonic() - turn_started
        usage = response.get("usage") or {}
        per_step.append({
            "step": turn + 1,
            "wall_seconds": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": _cached_tokens(usage),
            "completion_tokens": usage.get("completion_tokens"),
        })
        call_id = f"call_{turn + 1:03d}"
        messages.append({
            "role": "assistant",
            "content": REPLAY_NOTE,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": name}),
                },
            }],
        })
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": files[name]}
        )
    messages.append(
        {"role": "user", "content": f"BENCHMARK CONTROL: {OUTPUT_CONTROL}"}
    )
    verdict_started = time.monotonic()
    response = chat(messages, CONV_TOOLS, settings.conv_output_tokens)
    verdict_wall = time.monotonic() - verdict_started
    total = time.monotonic() - started
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    verdict_completion = usage.get("completion_tokens") or 0
    prompt_total = sum(row["prompt_tokens"] or 0 for row in per_step)
    prompt_total += usage.get("prompt_tokens") or 0
    cached_rows = [row["cached_tokens"] for row in per_step
                   if row["cached_tokens"] is not None]
    verdict_cached = _cached_tokens(usage)
    if verdict_cached is not None:
        cached_rows.append(verdict_cached)
    row = {
        "steps": steps,
        "wall_seconds": total,
        "replays_per_hour": 3600 / total if total else 0,
        "first_turn_wall_seconds": per_step[0]["wall_seconds"],
        "final_turn_wall_seconds": per_step[-1]["wall_seconds"],
        "verdict_wall_seconds": verdict_wall,
        "verdict_prompt_tokens": usage.get("prompt_tokens"),
        # A model that ignores the output control decodes fewer verdict
        # tokens; the recorded completion count keeps rates comparable.
        "verdict_completion_tokens": verdict_completion,
        "verdict_finish_reason": choice.get("finish_reason"),
        "verdict_output_tokens_per_second": (
            verdict_completion / verdict_wall if verdict_wall else 0
        ),
        "prompt_tokens_total": prompt_total,
        "cached_tokens_total": sum(cached_rows) if cached_rows else None,
        "cached_fraction": (
            sum(cached_rows) / prompt_total
            if cached_rows and prompt_total else None
        ),
        "per_step": per_step,
    }
    return row


def _run_replay(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    sample_index: int,
) -> dict:
    """One 1x replay sample: prefill-probe turns, gated and reported."""
    _wait_for_gate(settings.gate_cmd)
    variant = 800000 + sample_index * 1000
    row = _replay_once(
        chat,
        calibration,
        context,
        settings,
        evidence_variant=variant,
        chain_variant=variant,
        probe_tokens=REPLAY_PROBE_TOKENS,
        probe_control=False,
    )
    row["sample"] = sample_index + 1
    cached = row["cached_fraction"]
    print(
        f"  replay           sample={sample_index + 1} "
        f"steps={row['steps']} "
        f"wall={row['wall_seconds']:.1f}s "
        f"first-turn={row['first_turn_wall_seconds']:.1f}s "
        f"final-turn={row['final_turn_wall_seconds']:.1f}s "
        f"verdict={row['verdict_completion_tokens']:,}tok/"
        f"{row['verdict_wall_seconds']:.1f}s "
        f"cached={cached if cached is None else round(cached, 3)}"
    )
    return row


def _run_replay_concurrent(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    streams: int,
    sample_index: int,
) -> dict:
    """One concurrent batch: N scripted replays over a shared prefix.

    All streams carry the sample's evidence variant -- the same
    repository context concurrent agents would present -- while each
    stream's file chain diverges from its first tool result. Streams
    start together on a barrier and then free-run their own turn loops;
    no per-turn synchronization exists, so natural desync develops
    exactly as independent agents drift apart. Probe turns decode a
    realistic agent turn under an ephemeral control, so per-turn decode
    contention is part of the measurement. Across samples and stream
    counts the evidence changes from the first token, keeping every
    batch cold at start.
    """
    _wait_for_gate(settings.gate_cmd)
    evidence_variant = 900000 + sample_index * 100000 + streams * 10000
    barrier = threading.Barrier(streams + 1)

    def worker(stream_index: int) -> dict:
        barrier.wait()
        return _replay_once(
            chat,
            calibration,
            context,
            settings,
            evidence_variant=evidence_variant,
            chain_variant=evidence_variant + (stream_index + 1) * 1000,
            probe_tokens=settings.conv_turn_tokens,
            probe_control=True,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=streams
    ) as executor:
        futures = [executor.submit(worker, index) for index in range(streams)]
        started = time.monotonic()
        barrier.wait()
        stream_rows = [future.result() for future in futures]
    wall = time.monotonic() - started

    first_turn_cached = [
        row["per_step"][0]["cached_tokens"] for row in stream_rows
    ]
    cached_known = [value for value in first_turn_cached if value is not None]
    fractions = [
        row["cached_fraction"]
        for row in stream_rows
        if row["cached_fraction"] is not None
    ]
    turn_completions = [
        step["completion_tokens"] or 0
        for row in stream_rows
        for step in row["per_step"]
    ]
    row = {
        "sample": sample_index + 1,
        "streams": streams,
        "steps": stream_rows[0]["steps"],
        "wall_seconds": wall,
        "replays_per_hour": streams * 3600 / wall if wall else 0,
        "first_turn_wall_seconds": statistics.median(
            [item["first_turn_wall_seconds"] for item in stream_rows]
        ),
        "final_turn_wall_seconds": statistics.median(
            [item["final_turn_wall_seconds"] for item in stream_rows]
        ),
        "verdict_output_tokens_per_second": statistics.median(
            [item["verdict_output_tokens_per_second"] for item in stream_rows]
        ),
        "turn_completion_tokens_median": statistics.median(turn_completions),
        # Whichever stream commits the shared evidence first seeds the
        # cache for the rest; the per-stream list keeps that asymmetry
        # visible instead of averaging it away.
        "first_turn_cached_tokens": first_turn_cached,
        "first_turn_cached_tokens_median": (
            statistics.median(cached_known) if cached_known else None
        ),
        "cached_fraction": (
            statistics.median(fractions) if fractions else None
        ),
        "per_stream": stream_rows,
    }
    cached = row["cached_fraction"]
    print(
        f"  replay_conc      n={streams:<2} sample={sample_index + 1} "
        f"wall={wall:.1f}s replays={row['replays_per_hour']:.2f}/h "
        f"first-turn={row['first_turn_wall_seconds']:.1f}s "
        f"final-turn={row['final_turn_wall_seconds']:.1f}s "
        f"turn-decode={row['turn_completion_tokens_median']:.0f}tok "
        f"cached={cached if cached is None else round(cached, 3)}"
    )
    return row


def _measure_replay_concurrent(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    streams: tuple[int, ...],
) -> dict:
    """Concurrent replay family over the fitting stream counts."""
    rows: dict[int, list[dict]] = {count: [] for count in streams}

    def _run_pass(sample_index: int) -> None:
        order = streams if sample_index % 2 == 0 else tuple(reversed(streams))
        for stream_count in order:
            rows[stream_count].append(
                _run_replay_concurrent(
                    chat, calibration, context, settings,
                    stream_count, sample_index,
                )
            )

    passes = min(settings.min_samples, settings.samples)
    for sample_index in range(passes):
        _run_pass(sample_index)
    # A full extra pass keeps the alternating stream order symmetric, so
    # a single disagreeing stream count extends every count.
    while passes < settings.samples and any(
        not _samples_agree(rows[count], settings.sample_agreement)
        for count in streams
    ):
        print(
            f"  replay_conc      samples disagree beyond "
            f"{settings.sample_agreement:.0%}; extending"
        )
        _run_pass(passes)
        passes += 1

    results = {}
    for stream_count in streams:
        batch = rows[stream_count]
        fractions = [
            row["cached_fraction"]
            for row in batch
            if row["cached_fraction"] is not None
        ]
        results[str(stream_count)] = {
            "streams": stream_count,
            "samples": batch,
            "wall_seconds": _median_range(batch, "wall_seconds"),
            "replays_per_hour": _median_range(batch, "replays_per_hour"),
            "first_turn_wall_seconds": _median_range(
                batch, "first_turn_wall_seconds"
            ),
            "final_turn_wall_seconds": _median_range(
                batch, "final_turn_wall_seconds"
            ),
            "verdict_output_tokens_per_second": _median_range(
                batch, "verdict_output_tokens_per_second"
            ),
            "turn_completion_tokens_median": statistics.median(
                [row["turn_completion_tokens_median"] for row in batch]
            ),
            "cached_fraction_median": (
                statistics.median(fractions) if fractions else None
            ),
        }
    one = results["1"]["replays_per_hour"]["median"]
    scaling = {
        stream_count: (
            result["replays_per_hour"]["median"] / one if one else 0
        )
        for stream_count, result in results.items()
    }
    peak_streams, peak = max(
        results.items(),
        key=lambda item: item[1]["replays_per_hour"]["median"],
    )
    return {
        "steps": conversation_steps(context, settings),
        "by_streams": results,
        "scaling_vs_1x": scaling,
        "peak_streams": int(peak_streams),
        "peak_replays_per_hour": peak["replays_per_hour"]["median"],
    }


def _measure_replays(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
) -> dict:
    rows = [
        _run_replay(chat, calibration, context, settings, index)
        for index in range(min(settings.min_samples, settings.samples))
    ]
    while (
        len(rows) < settings.samples
        and not _samples_agree(rows, settings.sample_agreement)
    ):
        print(
            f"  replay           samples disagree beyond "
            f"{settings.sample_agreement:.0%}; extending"
        )
        rows.append(
            _run_replay(chat, calibration, context, settings, len(rows))
        )
    result = {
        "steps": conversation_steps(context, settings),
        "samples": rows,
        "wall_seconds": _median_range(rows, "wall_seconds"),
        "replays_per_hour": _median_range(rows, "replays_per_hour"),
        "first_turn_wall_seconds": _median_range(
            rows, "first_turn_wall_seconds"
        ),
        "final_turn_wall_seconds": _median_range(
            rows, "final_turn_wall_seconds"
        ),
        "verdict_output_tokens_per_second": _median_range(
            rows, "verdict_output_tokens_per_second"
        ),
    }
    fractions = [row["cached_fraction"] for row in rows
                 if row["cached_fraction"] is not None]
    result["cached_fraction_median"] = (
        statistics.median(fractions) if fractions else None
    )
    return result


def _wait_for_gate(gate_cmd: str | None) -> None:
    """Block until the operator gate admits the next sample batch.

    The gate runs between independent sample batches and inserts idle time
    only, so it never changes a measured workload; it exists for hosts that
    must wait out battery deficit or heat soak mid-campaign. A failing gate
    aborts the run rather than measuring under an unmet precondition.
    """
    if not gate_cmd:
        return
    completed = subprocess.run(gate_cmd, shell=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"gate command exited {completed.returncode}: {gate_cmd}"
        )


def _measure_once(
    request: RequestFn,
    phase: str,
    streams: int,
    sample_index: int,
    output_tokens: int,
    context: int,
    tolerance: float,
    variant_base: int,
    prompt_lines: int | None = None,
    target_prompt: int | None = None,
    near_context: bool = False,
    gate_cmd: str | None = None,
) -> dict:
    _wait_for_gate(gate_cmd)
    if prompt_lines is None:
        prompts = [
            decode_prompt(variant_base + sample_index * 100 + stream)
            for stream in range(streams)
        ]
    else:
        prompts = [
            audit_prompt(
                prompt_lines,
                variant_base + sample_index * 100 + stream,
            )
            for stream in range(streams)
        ]
    results, wall_seconds = _run_parallel(
        request,
        prompts,
        output_tokens,
        variant_base + sample_index * 1000,
    )
    _validate_results(
        results,
        output_tokens,
        target_prompt,
        context,
        tolerance,
        near_context,
    )
    row = _sample_metrics(results, wall_seconds, sample_index + 1)
    prompt_range = row["prompt_tokens_per_stream"]
    print(
        f"  {phase:<16} n={streams:<2} sample={sample_index + 1} "
        f"prompt={min(prompt_range):,}-{max(prompt_range):,} "
        f"wall={wall_seconds:.2f}s "
        f"input={row['aggregate_prompt_tokens_per_second']:.1f} tok/s "
        f"output-e2e={row['end_to_end_output_tokens_per_second']:.2f} tok/s "
        f"calls={row['calls_per_hour']:.2f}/h"
    )
    return row


def _measure(
    request: RequestFn,
    phase: str,
    streams: int,
    samples: int,
    output_tokens: int,
    context: int,
    tolerance: float,
    variant_base: int,
    prompt_lines: int | None = None,
    target_prompt: int | None = None,
    near_context: bool = False,
    gate_cmd: str | None = None,
    min_samples: int | None = None,
    agreement: float | None = None,
) -> dict:
    initial = min(min_samples or samples, samples)
    rows = [
        _measure_once(
            request=request,
            phase=phase,
            streams=streams,
            sample_index=sample_index,
            output_tokens=output_tokens,
            context=context,
            tolerance=tolerance,
            variant_base=variant_base,
            prompt_lines=prompt_lines,
            target_prompt=target_prompt,
            near_context=near_context,
            gate_cmd=gate_cmd,
        )
        for sample_index in range(initial)
    ]
    while (
        agreement is not None
        and len(rows) < samples
        and not _samples_agree(rows, agreement)
    ):
        print(
            f"  {phase:<16} n={streams:<2} samples disagree beyond "
            f"{agreement:.0%}; extending"
        )
        rows.append(
            _measure_once(
                request=request,
                phase=phase,
                streams=streams,
                sample_index=len(rows),
                output_tokens=output_tokens,
                context=context,
                tolerance=tolerance,
                variant_base=variant_base,
                prompt_lines=prompt_lines,
                target_prompt=target_prompt,
                near_context=near_context,
                gate_cmd=gate_cmd,
            )
        )
    return _summarize_measurements(streams, rows)


def _measure_family(
    request: RequestFn,
    phase: str,
    streams: tuple[int, ...],
    settings: Settings,
    samples: int,
    output_tokens: int,
    context: int,
    variant_base: int,
    prompt_lines: int | None = None,
    target_prompt: int | None = None,
    near_context: bool = False,
) -> dict:
    rows = {stream_count: [] for stream_count in streams}

    def _run_pass(sample_index: int) -> None:
        order = streams if sample_index % 2 == 0 else tuple(reversed(streams))
        for stream_count in order:
            rows[stream_count].append(
                _measure_once(
                    request=request,
                    phase=phase,
                    streams=stream_count,
                    sample_index=sample_index,
                    output_tokens=output_tokens,
                    context=context,
                    tolerance=settings.tolerance,
                    variant_base=variant_base + stream_count * 10000,
                    prompt_lines=prompt_lines,
                    target_prompt=target_prompt,
                    near_context=near_context,
                    gate_cmd=settings.gate_cmd,
                )
            )

    passes = min(settings.min_samples, samples)
    for sample_index in range(passes):
        _run_pass(sample_index)
    # A full extra pass keeps the alternating stream order symmetric, so a
    # single disagreeing stream count extends every count in this family.
    while passes < samples and any(
        not _samples_agree(rows[stream_count], settings.sample_agreement)
        for stream_count in streams
    ):
        print(
            f"  {phase:<16} samples disagree beyond "
            f"{settings.sample_agreement:.0%}; extending"
        )
        _run_pass(passes)
        passes += 1
    results = {
        str(stream_count): _summarize_measurements(
            stream_count, rows[stream_count]
        )
        for stream_count in streams
    }
    one = results["1"]["end_to_end_output_tokens_per_second"]["median"]
    scaling = {
        stream_count: (
            result["end_to_end_output_tokens_per_second"]["median"] / one
            if one
            else 0
        )
        for stream_count, result in results.items()
    }
    peak_streams, peak = max(
        results.items(),
        key=lambda item: item[1]["end_to_end_output_tokens_per_second"]["median"],
    )
    return {
        "by_streams": results,
        "scaling_vs_1x": scaling,
        "peak_streams": int(peak_streams),
        "peak_end_to_end_output_tokens_per_second": peak[
            "end_to_end_output_tokens_per_second"
        ]["median"],
    }


def run_benchmark(
    *,
    runtime: str,
    tag: str,
    model: str,
    context: int,
    request: RequestFn | None,
    active_slots: int,
    kv_pool_tokens: int,
    runtime_metadata: dict | None = None,
    provenance_files: dict[str, pathlib.Path] | None = None,
    openai_chat: Callable | None = None,
) -> int:
    """Run AuditBench v1 and print a human report followed by JSON."""
    settings = Settings.from_env()
    plan = build_plan(context, settings, active_slots, kv_pool_tokens)
    repository = _repository_state()
    metadata = {
        "runtime": runtime,
        "tag": tag,
        "model": model,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository["commit"],
        "repository_dirty": repository["dirty"],
        "repository": repository,
        "files": _file_provenance(provenance_files),
        "active_slots": active_slots,
        "kv_pool_tokens": kv_pool_tokens,
        **(runtime_metadata or {}),
    }
    print(
        f"=== {PROTOCOL} {runtime}/{tag} model={model} "
        f"context={context:,} canonical={settings.canonical} ==="
    )
    print(
        f"streams={settings.streams} deep-streams={settings.deep_streams} "
        f"profile={settings.profile} "
        f"samples={settings.min_samples}-{settings.samples} "
        f"(agreement {settings.sample_agreement:.0%}) "
        f"output={settings.output_tokens} "
        f"deep-output={settings.deep_output_tokens}"
    )
    if settings.gate_cmd:
        print(f"gate: {settings.gate_cmd}")
    if settings.dry_run:
        print(
            json.dumps(
                _result_envelope(settings, metadata, plan),
                sort_keys=True,
            )
        )
        return 0
    if request is None:
        raise ValueError("request adapter is required unless AUDIT_BENCH_DRY_RUN=1")

    print("warmup:")
    warmup = request("Reply with the single word ok.", 1, 1)
    if warmup.output_tokens != 1:
        raise RuntimeError(
            "warmup did not return exactly one output token; "
            f"endpoint reported {warmup.output_tokens} for model {model!r} "
            "(backends must honor max_tokens exactly)"
        )

    if settings.profile == "decode":
        print("decode ceiling:")
        decode = _measure_family(
            request=request,
            phase="decode",
            streams=tuple(plan["decode"]["streams"]),
            settings=settings,
            samples=settings.samples,
            output_tokens=settings.output_tokens,
            context=context,
            variant_base=200000,
        )
        print(
            json.dumps(
                {
                    **_result_envelope(settings, metadata, plan),
                    "decode": decode,
                },
                sort_keys=True,
            )
        )
        return 0

    print("token calibration:")
    low_lines = 64
    high_lines = 256
    low = request(audit_prompt(low_lines, 10), 1, 10)
    high = request(audit_prompt(high_lines, 11), 1, 11)
    calibration = Calibration(
        low_lines=low_lines,
        low_tokens=low.prompt_tokens,
        high_lines=high_lines,
        high_tokens=high.prompt_tokens,
    )
    print(
        f"  calibration {low_lines} lines={low.prompt_tokens:,} tokens, "
        f"{high_lines} lines={high.prompt_tokens:,} tokens"
    )

    if settings.profile in ("conversation", "replay", "replay-concurrent"):
        if openai_chat is None:
            raise ValueError(
                f"the {settings.profile} profile needs an OpenAI-compatible "
                "chat adapter (openai_chat)"
            )
        if settings.profile == "conversation":
            print("audit conversation:")
            key = "audit_conversation"
            section = _measure_conversations(
                openai_chat, calibration, context, settings
            )
        elif settings.profile == "replay":
            print("audit replay:")
            key = "audit_replay"
            section = _measure_replays(
                openai_chat, calibration, context, settings
            )
        else:
            print("concurrent audit replay:")
            key = "audit_replay_concurrent"
            section = _measure_replay_concurrent(
                openai_chat, calibration, context, settings,
                tuple(plan["replay_concurrent"]["streams"]),
            )
        print(
            json.dumps(
                {
                    **_result_envelope(settings, metadata, plan),
                    "calibration": asdict(calibration),
                    key: section,
                },
                sort_keys=True,
            )
        )
        return 0

    prefill = {}
    print("prefill curve:")
    for index, target in enumerate(plan["prefill_targets"]):
        lines = calibration.lines_for(target)
        prefill[str(target)] = _measure(
            request=request,
            phase=f"prefill_{target // 1024}k",
            streams=1,
            samples=settings.samples,
            output_tokens=1,
            context=context,
            tolerance=settings.tolerance,
            variant_base=100000 + index * 10000,
            prompt_lines=lines,
            target_prompt=target,
            gate_cmd=settings.gate_cmd,
            min_samples=settings.min_samples,
            agreement=settings.sample_agreement,
        )

    near_plan = plan["capacity_near"]
    print("near-context capacity:")
    capacity_near = _measure_family(
        request=request,
        phase="capacity_near",
        streams=tuple(near_plan["streams"]),
        settings=settings,
        samples=near_plan["samples"],
        output_tokens=near_plan["output_tokens"],
        context=context,
        variant_base=180000,
        prompt_lines=calibration.lines_for(near_plan["target_prompt_tokens"]),
        target_prompt=near_plan["target_prompt_tokens"],
        near_context=True,
    )
    print("decode ceiling:")
    decode = _measure_family(
        request=request,
        phase="decode",
        streams=tuple(plan["decode"]["streams"]),
        settings=settings,
        samples=settings.samples,
        output_tokens=settings.output_tokens,
        context=context,
        variant_base=200000,
    )
    audits = {}
    for index, phase in enumerate(plan["audit_phases"]):
        print(f"{phase['name']}:")
        target = phase["target_prompt_tokens"]
        audits[phase["name"]] = {
            "target_prompt_tokens": target,
            "output_tokens": phase["output_tokens"],
            **_measure_family(
                request=request,
                phase=phase["name"],
                streams=tuple(phase["streams"]),
                settings=settings,
                samples=settings.samples,
                output_tokens=phase["output_tokens"],
                context=context,
                variant_base=300000 + index * 100000,
                prompt_lines=calibration.lines_for(target),
                target_prompt=target,
            ),
        }
    result = {
        **_result_envelope(settings, metadata, plan),
        "calibration": asdict(calibration),
        "prefill": prefill,
        "capacity_near": capacity_near,
        "decode": decode,
        "audits": audits,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _endpoint_model_card(base_url: str, model: str | None, timeout: int) -> dict:
    """Find the model card on an endpoint that may serve several models."""
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response).get("data", [])
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise RuntimeError(f"{exc.code} from {url}: {message}") from exc
    if model is None:
        if len(data) != 1:
            raise RuntimeError(
                f"{url} lists {len(data)} models; pass --model to pick one"
            )
        return data[0]
    card = next((item for item in data if item.get("id") == model), None)
    if card is None:
        live = ", ".join(str(item.get("id")) for item in data) or "none"
        raise RuntimeError(f"model {model!r} not live at {url}; found: {live}")
    return card


def _cli_repository_env(module_dir: pathlib.Path) -> None:
    """Record the harness checkout as repository state unless already set."""
    if os.environ.get("AUDIT_BENCH_REPO_COMMIT"):
        return
    try:
        commit = subprocess.run(
            ["git", "-C", str(module_dir), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(module_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    os.environ["AUDIT_BENCH_REPO_COMMIT"] = commit
    os.environ.setdefault(
        "AUDIT_BENCH_REPO_DIRTY", "1" if status.strip() else "0"
    )


def _cli_capacity(
    flag: int | None, flag_name: str, env_name: str, default: int, note: str
) -> int:
    if flag is not None:
        if flag < 1:
            raise SystemExit(f"{flag_name} must be positive")
        return flag
    raw = os.environ.get(env_name, "")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            raise SystemExit(
                f"{env_name} must be an integer, got {raw!r}"
            ) from None
        if value < 1:
            raise SystemExit(f"{env_name} must be positive")
        return value
    print(f"note: {note}")
    return default


def main(argv: list[str] | None = None) -> int:
    """Run AuditBench v1 against an OpenAI-compatible chat endpoint."""
    parser = argparse.ArgumentParser(
        prog="audit-bench",
        description=(
            "Run AuditBench v1 against an OpenAI-compatible chat endpoint. "
            "AUDIT_BENCH_* environment variables select the profile and "
            "protocol overrides; see docs/PROTOCOL.md."
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="server base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--model",
        help=(
            "served model name; resolved from /v1/models when the endpoint "
            "serves exactly one model"
        ),
    )
    parser.add_argument(
        "--context",
        type=int,
        required=True,
        help="configured per-request context in tokens",
    )
    parser.add_argument(
        "--active-slots",
        type=int,
        help="scheduler admission (concurrent requests); default 1",
    )
    parser.add_argument(
        "--kv-tokens",
        type=int,
        help="raw KV pool size in tokens; default one configured context",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "full", "decode", "conversation", "replay", "replay-concurrent"
        ),
        help="workload profile (default full; overrides AUDIT_BENCH_PROFILE)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="per-request timeout in seconds (default 7200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the workload plan without contacting the server",
    )
    args = parser.parse_args(argv)

    if args.profile:
        os.environ["AUDIT_BENCH_PROFILE"] = args.profile
    if args.dry_run:
        os.environ["AUDIT_BENCH_DRY_RUN"] = "1"
    dry_run = os.environ.get("AUDIT_BENCH_DRY_RUN", "0") == "1"
    # Documented decode-compliance controls (docs/PROTOCOL.md) forwarded to
    # the adapters so each is both sent with every request and recorded in
    # the JSON: the documented parameter and the system-role control.
    reasoning_effort = os.environ.get("AUDIT_BENCH_REASONING_EFFORT") or None
    system_prompt = os.environ.get("AUDIT_BENCH_SYSTEM_PROMPT") or None

    active_slots = _cli_capacity(
        args.active_slots,
        "--active-slots",
        "AUDIT_BENCH_ACTIVE_SLOTS",
        1,
        "--active-slots not given; assuming 1, concurrency phases run 1x only",
    )
    kv_pool_tokens = _cli_capacity(
        args.kv_tokens,
        "--kv-tokens",
        "AUDIT_BENCH_KV_TOKENS",
        args.context,
        "--kv-tokens not given; assuming one configured context",
    )

    model = args.model
    identity: dict = {"checked": False}
    if dry_run:
        model = model or "unspecified"
    else:
        card = _endpoint_model_card(args.base_url, model, args.timeout)
        model = str(card.get("id"))
        live_context = card.get("max_model_len")
        if live_context is not None and int(live_context) != args.context:
            raise SystemExit(
                f"live max_model_len is {int(live_context):,}, "
                f"--context says {args.context:,}"
            )
        identity = {
            "checked": True,
            "matched": True,
            "source": "/v1/models",
            "live_id": model,
            "live_max_model_len": live_context,
        }

    _cli_repository_env(pathlib.Path(__file__).resolve().parent)
    tag = urllib.parse.urlsplit(args.base_url).netloc or args.base_url
    return run_benchmark(
        runtime="endpoint",
        tag=tag,
        model=model,
        context=args.context,
        request=(
            None
            if dry_run
            else OpenAIChatAdapter(
                args.base_url,
                model,
                args.timeout,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
            )
        ),
        active_slots=active_slots,
        kv_pool_tokens=kv_pool_tokens,
        runtime_metadata={
            "base_url": args.base_url,
            "model_identity": identity,
            # An arbitrary endpoint has no managed launcher evidence, so a
            # CLI run is honest but never publishable.
            "runtime_profile": {
                "checked": False,
                "source": "unmanaged endpoint",
            },
        },
        openai_chat=(
            None
            if dry_run
            else openai_chat_fn(
                args.base_url,
                model,
                args.timeout,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

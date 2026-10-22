#!/usr/bin/env python3
"""Shared AuditBench v2 protocol for local inference runtimes.

Runtime wrappers provide one request function. This module owns every workload
shape and metric so runtime adapters cannot silently drift. It also runs
standalone against any OpenAI-compatible chat endpoint via main().
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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

PROTOCOL = "audit-v2"
# Retained documents of the retired protocol stay listable; --show names
# the last harness commit that renders them instead of guessing.
LEGACY_PROTOCOL = "audit-v1"
LEGACY_HARNESS_COMMIT = "382760d"
SCHEMA_VERSION = 8
DEFAULT_STREAMS = (1, 2, 4, 8)
DEFAULT_SAMPLES = 3
DEFAULT_MIN_SAMPLES = 2
DEFAULT_SAMPLE_AGREEMENT = 0.05
DEFAULT_TOLERANCE = 0.02
DEFAULT_CONV_EVIDENCE_TOKENS = 32768
DEFAULT_CONV_TOOL_TOKENS = 8192
DEFAULT_CONV_OUTPUT_TOKENS = 16384
# Session turns are the model's natural agent turn (reasoning, a note, a
# tool call); the cap only bounds a runaway. The report is the session's
# deliverable, natural length under its own cap: reasoning plus a fix for
# each of up to 16 findings, the same budget the conversation verdict has.
SESSION_TURN_TOKENS = 2048
SESSION_REPORT_TOKENS = 16384
# Declared sampling keys and their types: the OpenAI-compatible request
# fields the fleet's runtimes accept for a checkpoint's sampling defaults.
SAMPLING_KEYS = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "repetition_penalty": float,
}
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
# The session profile scripts the assistant turns itself; the model's
# own turn is measured and discarded. The note plus tool-call JSON stays
# well under CONV_NOTE_RESERVE, keeping the context-fit math shared with
# the interactive conversation honest.
SESSION_NOTE = "Ledger updated; reading the next evidence file."
SESSION_REPORT_REQUEST = (
    "The chain is complete. Write the audit report now, as plain text with "
    "no tool calls: for each finding, the finding line, the likely impact, "
    "and a concrete code fix."
)
# Reference figures use llama-bench's method so an optimization can be
# compared with published numbers: pp512 and tg128, one stream, greedy,
# fixed length, five repetitions reported as mean and sample deviation.
# They describe the engine at shallow depth, never audit serving.
REFERENCE_PROMPT_TOKENS = 512
REFERENCE_OUTPUT_TOKENS = 128
REFERENCE_REPS = 5
REFERENCE_PROMPT = "Reply with the single word ok."
PROMPT_VARIANT_RESERVE = 64
NEAR_CONTEXT_NUMERATOR = 15
NEAR_CONTEXT_DENOMINATOR = 16
DEEP_CONTEXT_NUMERATOR = 7
DEEP_CONTEXT_DENOMINATOR = 8


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


def parse_sampling(value: str) -> dict:
    """Parse declared sampling as space-separated key=value pairs."""
    sampling: dict = {}
    for item in value.split():
        key, sep, raw = item.partition("=")
        if not sep or key not in SAMPLING_KEYS:
            raise ValueError(
                f"AUDIT_BENCH_SAMPLING item {item!r} is not one of "
                + ", ".join(f"{name}=N" for name in SAMPLING_KEYS)
            )
        if key in sampling:
            raise ValueError(f"AUDIT_BENCH_SAMPLING repeats {key}")
        try:
            sampling[key] = SAMPLING_KEYS[key](raw)
        except ValueError as exc:
            raise ValueError(
                f"AUDIT_BENCH_SAMPLING {key} must be numeric, got {raw!r}"
            ) from exc
    return sampling


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
    profile: str
    samples: int
    tolerance: float
    dry_run: bool
    min_samples: int = DEFAULT_MIN_SAMPLES
    sample_agreement: float = DEFAULT_SAMPLE_AGREEMENT
    conv_evidence_tokens: int = DEFAULT_CONV_EVIDENCE_TOKENS
    conv_tool_tokens: int = DEFAULT_CONV_TOOL_TOKENS
    conv_output_tokens: int = DEFAULT_CONV_OUTPUT_TOKENS
    conv_max_steps: int = DEFAULT_CONV_MAX_STEPS
    sampling: dict = field(default_factory=dict)
    gate_cmd: str | None = None
    system_prompt: str | None = None
    reasoning_effort: str | None = None
    preserve_reasoning: bool = False
    tolerate_off_by_one: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        profile = os.environ.get("AUDIT_BENCH_PROFILE", "session")
        if profile not in {"session", "conversation"}:
            raise ValueError("AUDIT_BENCH_PROFILE must be session or conversation")
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
            profile=profile,
            samples=samples,
            min_samples=min_samples,
            sample_agreement=sample_agreement,
            conv_evidence_tokens=conv_evidence,
            conv_tool_tokens=conv_tool,
            conv_output_tokens=conv_output,
            conv_max_steps=conv_max_steps,
            # The checkpoint's own sampling, declared by the model config:
            # live clients decode at it, so every generating call does.
            sampling=parse_sampling(os.environ.get("AUDIT_BENCH_SAMPLING", "")),
            tolerance=tolerance,
            dry_run=os.environ.get("AUDIT_BENCH_DRY_RUN", "0") == "1",
            gate_cmd=os.environ.get("AUDIT_BENCH_GATE_CMD") or None,
            # Model-documented reasoning controls (docs/PROTOCOL.md
            # "Model-documented controls"). The control may ride in the
            # system role (system_prompt) or the runtime's documented
            # parameter (reasoning_effort). Recorded configuration only; the
            # workload text itself never changes, and neither marks the run
            # non-canonical.
            system_prompt=os.environ.get("AUDIT_BENCH_SYSTEM_PROMPT") or None,
            reasoning_effort=os.environ.get("AUDIT_BENCH_REASONING_EFFORT")
            or None,
            # A model whose card requires prior reasoning in the history
            # (interleaved thinking) gets it resent; same standing as the
            # controls above.
            preserve_reasoning=os.environ.get(
                "AUDIT_BENCH_PRESERVE_REASONING", "0"
            ) == "1",
            tolerate_off_by_one=os.environ.get(
                "AUDIT_BENCH_TOLERATE_OFF_BY_ONE", "0"
            ) == "1",
        )

    @property
    def canonical(self) -> bool:
        # Two fixed workloads exist: the scripted agent session (serving
        # performance) and the interactive audit conversation (model
        # usability). Both decode at the declared sampling; a run at an
        # undeclared server default measures an arbitrary sampler.
        return (
            self.streams == DEFAULT_STREAMS
            and self.profile in ("session", "conversation")
            and self.conv_evidence_tokens == DEFAULT_CONV_EVIDENCE_TOKENS
            and self.conv_tool_tokens == DEFAULT_CONV_TOOL_TOKENS
            and self.conv_output_tokens == DEFAULT_CONV_OUTPUT_TOKENS
            and self.conv_max_steps == DEFAULT_CONV_MAX_STEPS
            and self.samples == DEFAULT_SAMPLES
            and self.min_samples == DEFAULT_MIN_SAMPLES
            and self.sample_agreement == DEFAULT_SAMPLE_AGREEMENT
            and self.tolerance == DEFAULT_TOLERANCE
            and "temperature" in self.sampling
            and not self.tolerate_off_by_one
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["streams"] = list(self.streams)
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


def _session_streams(
    context: int,
    settings: Settings,
    active_slots: int,
    kv_pool_tokens: int,
) -> tuple[int, list[int]]:
    """Session stream counts (1x and the largest fitting) and the demand.

    Demand is the final resent conversation plus the report cap. No
    shared-prefix credit is taken even though concurrent sessions share
    their evidence prefix, so admission never depends on cache behavior.
    A single session always fits: the step fit keeps the final prompt
    plus the verdict cap within one configured context, and the pool
    holds at least one context.
    """
    steps = conversation_steps(context, settings)
    per_stream = (
        settings.conv_evidence_tokens
        + steps * _resent_per_step(settings)
        + SESSION_REPORT_TOKENS
    )
    max_streams = min(active_slots, kv_pool_tokens // per_stream)
    fitting = [
        candidate for candidate in settings.streams if candidate <= max_streams
    ]
    return per_stream, _endpoints(fitting or [1])


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
    capacity_streams = _endpoints(
        _fitting_streams(
            settings.streams,
            active_slots,
            kv_pool_tokens,
            near_target,
            1,
        )
    )
    session_demand, session_streams = _session_streams(
        context, settings, active_slots, kv_pool_tokens
    )
    steps = conversation_steps(context, settings)
    return {
        "profile": settings.profile,
        "configured_context": context,
        "active_slots": active_slots,
        "kv_pool_tokens": kv_pool_tokens,
        "preferred_256k_context": context >= 262144,
        "capacity_near": {
            "target_prompt_tokens": near_target,
            "output_tokens": 1,
            "samples": 1,
            "streams": capacity_streams,
        },
        "session": {
            "evidence_tokens": settings.conv_evidence_tokens,
            "tool_result_tokens": settings.conv_tool_tokens,
            "turn_output_tokens": SESSION_TURN_TOKENS,
            "report_output_tokens": SESSION_REPORT_TOKENS,
            "steps": steps,
            "per_stream_kv_demand": session_demand,
            "streams": session_streams,
        },
        "conversation": {
            "evidence_tokens": settings.conv_evidence_tokens,
            "tool_result_tokens": settings.conv_tool_tokens,
            "output_tokens": settings.conv_output_tokens,
            "steps": steps,
            "streams": [1],
        },
    }


def audit_prompt(n_lines: int, variant: int) -> str:
    """Generate deterministic, varied source and tool evidence.

    The corpus carries no instruction of any kind: v1 embedded a
    never-conclude control to force output lengths, and models that obeyed
    it inside conversational evidence derailed the tool loop (observed:
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
    return "\n".join(lines)


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
    """OpenAI-compatible single-prompt adapter for the 1-token gates."""

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
        # The documented control (docs/PROTOCOL.md) rides in the system
        # role; the workload prompt stays byte-identical in the user role.
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
        choice = (body.get("choices") or [{}])[0]
        return RequestResult(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            seconds=time.monotonic() - started,
            details={"finish_reason": choice.get("finish_reason")},
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
            # vLLM-Metal 0.30.0 labels its unified pool CPU.
            matches = re.findall(
                r"(?:GPU|CPU) KV cache size:\s*([0-9,]+) tokens", text
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


OFF_BY_ONE_EXPECTATION = (
    "AUDIT_BENCH_TOLERATE_OFF_BY_ONE=1 expects one fewer with "
    "finish_reason length; unset it for an exact backend"
)


def _output_matches(
    result: RequestResult, expected: int, off_by_one: bool
) -> bool:
    """True when a response reports the requested output count.

    With the off-by-one tolerance (docs/PROTOCOL.md) a server that reports
    one token fewer than it decoded on length-capped responses matches
    exactly at that offset, and only when it says the cap cut the response;
    a natural stop at the same count is still short.
    """
    if not off_by_one:
        return result.output_tokens == expected
    return (
        result.details.get("finish_reason") == "length"
        and result.output_tokens == expected - 1
    )


def _validate_results(
    results: list[RequestResult],
    expected_output: int,
    target_prompt: int | None,
    context: int,
    tolerance: float,
    near_context: bool,
    off_by_one: bool = False,
) -> None:
    outputs = [result.output_tokens for result in results]
    if not all(_output_matches(result, expected_output, off_by_one)
               for result in results):
        raise RuntimeError(
            f"output token count was {outputs}, expected {expected_output} per stream"
            + (f" ({OFF_BY_ONE_EXPECTATION})" if off_by_one else "")
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

    Used by the session and conversation profiles, which need
    multi-message requests with tools -- the single-prompt RequestFn
    cannot express them. The harness passes the declared sampling on
    every call. A streamed call returns the same body shape plus a
    `timing` block: the first token separates prefill of the uncached
    suffix from decode.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def chat(
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sampling: dict | None = None,
        stream: bool = False,
    ) -> dict:
        # The documented control (docs/PROTOCOL.md) rides in the system
        # role as a separate message ahead of the workload's own system
        # message, so the workload text stays byte-identical on every
        # resent turn.
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        payload: dict = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            **(sampling or {}),
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response:
                if stream:
                    return _read_stream(response, endpoint, started)
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


def _read_stream(response, endpoint: str, started: float) -> dict:
    """Assemble a server-sent chat stream into a body with timing.

    The first token is the first chunk whose delta carries anything but
    the role: content, a reasoning field under any runtime's name, or a
    tool call. Token counts come only from the server's final usage
    chunk; a stream without one is an error, never an estimate.
    """
    first_token = None
    finish_reason = None
    usage = None
    content: list[str] = []
    # Runtimes name the reasoning field reasoning or reasoning_content,
    # and some send both copies; the longer one is the reasoning.
    reasoning: dict[str, list[str]] = {"reasoning": [], "reasoning_content": []}
    tool_calls: set[int] = set()
    for raw in response:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise RuntimeError(
                f"{endpoint} sent a malformed stream line: {data[:200]}"
            ) from exc
        if chunk.get("error") is not None:
            raise RuntimeError(
                f"{endpoint} stream error: {str(chunk['error'])[:200]}"
            )
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if first_token is None and any(
                value for key, value in delta.items() if key != "role"
            ):
                first_token = time.monotonic() - started
            content.append(delta.get("content") or "")
            for field in reasoning:
                reasoning[field].append(delta.get(field) or "")
            tool_calls.update(
                call.get("index", 0) for call in delta.get("tool_calls") or []
            )
            finish_reason = choice.get("finish_reason") or finish_reason
    total = time.monotonic() - started
    if usage is None:
        raise RuntimeError(
            f"{endpoint} stream ended without a usage chunk "
            "(stream_options.include_usage unsupported?)"
        )
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": "".join(content)},
        }],
        "usage": usage,
        "timing": {"first_token_seconds": first_token, "total_seconds": total},
        "reasoning_chars": max(len("".join(parts)) for parts in reasoning.values()),
        "reasoning_text": max(("".join(parts) for parts in reasoning.values()), key=len),
        "tool_calls": len(tool_calls),
    }


def _resent_per_step(settings: Settings) -> int:
    """Tokens one step adds to the resent conversation: the tool result and
    a short assistant note, plus, under preserve_reasoning, the turn's
    reasoning, bounded by a session turn's cap."""
    per_step = settings.conv_tool_tokens + CONV_NOTE_RESERVE
    if settings.preserve_reasoning:
        per_step += SESSION_TURN_TOKENS
    return per_step


def conversation_steps(context: int, settings: Settings) -> int:
    """Files that keep the final conversation within 7/8 of the context.

    Only resent tokens count against fit: each turn adds its tool result
    and a short assistant note, while reasoning tokens cost wall time but
    are not resent, except under preserve_reasoning, which reserves a
    session turn's cap per step (a conversation turn can exceed it). The
    verdict turn's full output cap must still fit under the configured
    context on top of the final prompt.
    """
    budget = context * DEEP_CONTEXT_NUMERATOR // DEEP_CONTEXT_DENOMINATOR
    per_step = _resent_per_step(settings)
    fitting = (budget - settings.conv_evidence_tokens) // per_step
    steps = max(1, min(settings.conv_max_steps, fitting))
    while steps > 1 and (
        settings.conv_evidence_tokens + steps * per_step
        + settings.conv_output_tokens > context
    ):
        steps -= 1
    return steps


# Typographic dashes a model may write inside a finding token: U+2010 to
# U+2015 and the minus sign. Recall folds them to "-" before matching, so
# a report that names every finding is not scored as naming none.
_DASHES = str.maketrans(
    {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}
)


def _recalled(text: str, findings: dict) -> int:
    """How many finding tokens the text names, typographic dashes folded."""
    text = text.translate(_DASHES)
    return sum(token in text for token in findings.values())


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
    """Deterministic evidence corpus shared by conversation and session.

    Returns (steps, evidence, findings, files). Samples differ from the
    first token via the variant, so only self-sharing within one
    conversation can hit a cache. Concurrent sessions pass a per-stream
    chain_variant with a shared evidence variant: streams then present
    the identical evidence prefix (concurrent agents on the same
    repository) while their file chains diverge from the first tool
    result.
    """
    chain = variant if chain_variant is None else chain_variant
    steps = conversation_steps(context, settings)
    evidence = audit_prompt(
        calibration.lines_for(settings.conv_evidence_tokens), variant
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
            + audit_prompt(tool_lines, chain + n)
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
    nothing about server performance, which the session profile measures
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
        response = chat(
            messages, CONV_TOOLS, settings.conv_output_tokens,
            sampling=settings.sampling,
        )
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
            recalled = _recalled(final_text, findings)
            break
        # Reasoning fields are not resent: production clients resend
        # message content and tool calls only, unless the model's card
        # requires the reasoning back (preserve_reasoning).
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        messages.append({
            "role": "assistant",
            **({"reasoning": reasoning}
               if settings.preserve_reasoning and reasoning else {}),
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
                "tool_call_id": call.get("id") or f"call{turn:03d}{index:02d}",
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


def _session_call(
    chat: Callable, messages: list[dict], max_tokens: int, settings: Settings
) -> dict:
    """One streamed session call: server counts, prefill and decode split."""
    response = chat(
        messages, CONV_TOOLS, max_tokens,
        sampling=settings.sampling, stream=True,
    )
    usage = response.get("usage") or {}
    timing = response["timing"]
    prompt = int(usage.get("prompt_tokens") or 0)
    if prompt <= 0 or usage.get("completion_tokens") is None:
        raise RuntimeError(f"server usage lacks token counts: {usage}")
    cached = _cached_tokens(usage)
    completion = int(usage["completion_tokens"])
    first = timing["first_token_seconds"]
    total = timing["total_seconds"]
    if first is None:
        # Nothing streamed: an end-of-turn-only answer, or tokens the
        # parser consumed (an empty think block). All of it counts as
        # prefill and the call has no decode rate; a server that never
        # streams leaves every turn empty and every report incomplete.
        first = total
    decode_seconds = total - first
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "wall_seconds": total,
        "first_token_seconds": first,
        # The first token closes prefill; the rest is decode.
        "decode_tokens_per_second": (
            (completion - 1) / decode_seconds
            if completion > 1 and decode_seconds else None
        ),
        "content_chars": len(
            response["choices"][0]["message"].get("content") or ""
        ),
        "reasoning_chars": response["reasoning_chars"],
        "tool_calls": response["tool_calls"],
        # Popped by the caller: only the report's text is judged, and the
        # reasoning is resent only under preserve_reasoning.
        "content": response["choices"][0]["message"].get("content") or "",
        "reasoning": response["reasoning_text"],
    }


def _session_once(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    evidence_variant: int,
    chain_variant: int,
) -> dict:
    """One scripted agent session: the harness drives the transcript.

    The prefix grows exactly as in the interactive conversation, but each
    assistant turn is replaced by a scripted note and tool call, so every
    model on every runtime resends the identical transcript and a model
    that cannot hold the protocol still yields numbers. Each call lets
    the model write its natural turn under the declared sampling; that
    output is measured and discarded. The final call asks for the report,
    the session's deliverable, at its natural length.
    """
    steps, evidence, findings, files = _conversation_fixture(
        calibration, context, settings, evidence_variant, chain_variant
    )
    messages = _opening_messages(steps, evidence)
    per_step = []
    started = time.monotonic()
    for turn, name in enumerate(files):
        row = _session_call(chat, messages, SESSION_TURN_TOKENS, settings)
        del row["content"]
        reasoning = row.pop("reasoning")
        per_step.append({"step": turn + 1, **row})
        # Nine alphanumerics, the id shape Mistral's tokenizer requires of
        # every tool call in the history; vLLM shortens longer ids to it but
        # rejects shorter or punctuated ones.
        call_id = f"call{turn + 1:05d}"
        messages.append({
            "role": "assistant",
            "content": SESSION_NOTE,
            # The model's own reasoning for this step, ahead of the scripted
            # note and call, when its card requires it in the history.
            **({"reasoning": reasoning}
               if settings.preserve_reasoning and reasoning else {}),
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
    messages.append({"role": "user", "content": SESSION_REPORT_REQUEST})
    report = _session_call(chat, messages, SESSION_REPORT_TOKENS, settings)
    report_text = report.pop("content")
    del report["reasoning"]
    recalled = _recalled(report_text, findings)
    total = time.monotonic() - started
    calls = per_step + [report]
    uncached = sum(
        row["prompt_tokens"] - (row["cached_tokens"] or 0) for row in calls
    )
    prefill_seconds = sum(row["first_token_seconds"] for row in calls)
    # Without the server's cached counts the uncached work is unknown; a
    # full-prompt guess would overstate a warm server's prefill rate.
    cache_reported = all(row["cached_tokens"] is not None for row in calls)
    prompt_total = sum(row["prompt_tokens"] for row in calls)
    cached_rows = [row["cached_tokens"] for row in calls
                   if row["cached_tokens"] is not None]
    return {
        "steps": steps,
        "wall_seconds": total,
        "output_tokens": sum(row["completion_tokens"] for row in calls),
        "prefill_tokens_per_second": (
            uncached / prefill_seconds
            if cache_reported and prefill_seconds else None
        ),
        # Decode at the start, middle and end of the growing context.
        "turn_decode_tokens_per_second": {
            "first": per_step[0]["decode_tokens_per_second"],
            "middle": per_step[len(per_step) // 2]["decode_tokens_per_second"],
            "last": per_step[-1]["decode_tokens_per_second"],
        },
        "report": report,
        # Recorded, never padded or retried. The report is the session's
        # deliverable: it is complete when it ends on its own and names at
        # least 7/8 of the findings the chain carried. One omission in a
        # long report is an edit in a live audit; a capped or short report
        # is work to redo, so its run is not publishable.
        "report_findings_recalled": recalled,
        "complete": report["finish_reason"] != "length"
        and recalled * 8 >= steps * 7,
        # A turn that hit its cap is bounded, not failed: long thinking and
        # a runaway both end there, and the count is the evidence.
        "truncated_calls": sum(row["finish_reason"] == "length" for row in calls),
        "empty_turns": sum(
            row["finish_reason"] != "length" and not row["content_chars"]
            and not row["tool_calls"]
            for row in per_step
        ),
        # Evidence for reasoning models only; the benchmark has no
        # expectation to judge it against.
        "unreasoned_calls": sum(not row["reasoning_chars"] for row in calls),
        "prompt_tokens_total": prompt_total,
        "cached_tokens_total": sum(cached_rows) if cached_rows else None,
        "cached_fraction": (
            sum(cached_rows) / prompt_total
            if cached_rows and prompt_total else None
        ),
        "per_step": per_step,
    }


def _median_known(values: list) -> float | None:
    known = [value for value in values if value is not None]
    return statistics.median(known) if known else None


def _run_session_batch(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    streams: int,
    sample_index: int,
) -> dict:
    """One batch: N sessions over a shared evidence prefix.

    All streams carry the sample's evidence variant -- the same
    repository context concurrent agents would present -- while each
    stream's file chain diverges from its first tool result. Streams
    start together on a barrier and then free-run their own turn loops,
    so natural desync develops exactly as independent agents drift
    apart. Across samples and stream counts the evidence changes from
    the first token, keeping every batch cold at start.
    """
    _wait_for_gate(settings.gate_cmd)
    evidence_variant = 900000 + sample_index * 100000 + streams * 10000
    barrier = threading.Barrier(streams + 1)

    def worker(stream_index: int) -> dict:
        barrier.wait()
        return _session_once(
            chat,
            calibration,
            context,
            settings,
            evidence_variant=evidence_variant,
            chain_variant=evidence_variant + (stream_index + 1) * 1000,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as executor:
        futures = [executor.submit(worker, index) for index in range(streams)]
        started = time.monotonic()
        barrier.wait()
        stream_rows = [future.result() for future in futures]
    wall = time.monotonic() - started

    row = {
        "sample": sample_index + 1,
        "streams": streams,
        "steps": stream_rows[0]["steps"],
        "wall_seconds": wall,
        "sessions_per_hour": streams * 3600 / wall if wall else 0,
        "output_tokens_per_session": statistics.median(
            [item["output_tokens"] for item in stream_rows]
        ),
        "prefill_tokens_per_second": _median_known(
            [item["prefill_tokens_per_second"] for item in stream_rows]
        ),
        "report_decode_tokens_per_second": _median_known(
            [item["report"]["decode_tokens_per_second"] for item in stream_rows]
        ),
        "turn_decode_tokens_per_second": {
            position: _median_known([
                item["turn_decode_tokens_per_second"][position]
                for item in stream_rows
            ])
            for position in ("first", "middle", "last")
        },
        "sessions_incomplete": sum(
            not item["complete"] for item in stream_rows
        ),
        "truncated_calls": sum(item["truncated_calls"] for item in stream_rows),
        "empty_turns": sum(item["empty_turns"] for item in stream_rows),
        "unreasoned_calls": sum(
            item["unreasoned_calls"] for item in stream_rows
        ),
        "cached_fraction": _median_known(
            [item["cached_fraction"] for item in stream_rows]
        ),
        "per_stream": stream_rows,
    }
    print(
        f"  session          n={streams:<2} sample={sample_index + 1} "
        f"wall={wall:.1f}s sessions={row['sessions_per_hour']:.2f}/h "
        f"output={row['output_tokens_per_session']:,.0f}tok "
        f"prefill={_rate(row['prefill_tokens_per_second'])} "
        f"report-decode={_rate(row['report_decode_tokens_per_second'])} "
        f"cached={_cached(row['cached_fraction'])}"
    )
    return row


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def _rates_agree(rows: list[dict], agreement: float) -> bool:
    """True when the prefill and report decode rates lie within the spread.

    Natural output lengths vary under sampling, so wall time is not a
    stable criterion; the server's rates are. A rate a sample could not
    measure is left out of the comparison.
    """
    for key in ("prefill_tokens_per_second", "report_decode_tokens_per_second"):
        values = [row[key] for row in rows if row[key] is not None]
        if len(values) > 1 and max(values) - min(values) > min(values) * agreement:
            return False
    return True


def _measure_sessions(
    chat: Callable,
    calibration: "Calibration",
    context: int,
    settings: Settings,
    streams: tuple[int, ...],
) -> dict:
    """Session family over the planned stream counts."""
    rows: dict[int, list[dict]] = {count: [] for count in streams}

    def _run_pass(sample_index: int) -> None:
        order = streams if sample_index % 2 == 0 else tuple(reversed(streams))
        for stream_count in order:
            rows[stream_count].append(
                _run_session_batch(
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
        not _rates_agree(rows[count], settings.sample_agreement)
        for count in streams
    ):
        print(
            f"  session          samples disagree beyond "
            f"{settings.sample_agreement:.0%}; extending"
        )
        _run_pass(passes)
        passes += 1

    results = {}
    for stream_count in streams:
        batch = rows[stream_count]
        results[str(stream_count)] = {
            "streams": stream_count,
            "samples": batch,
            "wall_seconds": _median_range(batch, "wall_seconds"),
            "sessions_per_hour": _median_range(batch, "sessions_per_hour"),
            "output_tokens_per_session": statistics.median(
                [row["output_tokens_per_session"] for row in batch]
            ),
            "prefill_tokens_per_second": _median_known(
                [row["prefill_tokens_per_second"] for row in batch]
            ),
            "report_decode_tokens_per_second": _median_known(
                [row["report_decode_tokens_per_second"] for row in batch]
            ),
            "turn_decode_tokens_per_second": {
                position: _median_known([
                    row["turn_decode_tokens_per_second"][position]
                    for row in batch
                ])
                for position in ("first", "middle", "last")
            },
            "sessions": sum(row["streams"] for row in batch),
            **{
                key: sum(row[key] for row in batch)
                for key in ("sessions_incomplete", "truncated_calls",
                            "empty_turns", "unreasoned_calls")
            },
            "cached_fraction_median": _median_known(
                [row["cached_fraction"] for row in batch]
            ),
        }
    one = results["1"]["sessions_per_hour"]["median"]
    optimal = _operating_point(results)
    return {
        "steps": conversation_steps(context, settings),
        "by_streams": results,
        "optimal_streams": optimal,
        "scaling_vs_1x": {
            stream_count: (
                result["sessions_per_hour"]["median"] / one if one else 0
            )
            for stream_count, result in results.items()
        },
    }


def _operating_point(by_streams: dict) -> int | None:
    """Every measured stream count is a candidate operating point: one whose
    sessions all completed qualifies, and the fastest of those is optimal."""
    complete = [
        section for section in by_streams.values()
        if not section["sessions_incomplete"]
    ]
    optimal = max(
        complete,
        key=lambda section: section["sessions_per_hour"]["median"],
        default=None,
    )
    return optimal["streams"] if optimal else None


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "reps": values,
    }


def _measure_reference(
    request: RequestFn, calibration: "Calibration", settings: Settings
) -> dict:
    """llama-bench pp512 and tg128 over the fixed-length request contract.

    pp: a cold prompt of about 512 tokens (calibration moves in whole
    evidence lines, so the actual count is recorded) with one output token;
    the rate is prompt tokens over request seconds, a lower bound that
    includes HTTP and templating. tg: the same minimal prompt for 128
    tokens and for one; the difference is 127 decode steps, so the rate is
    llama-server's and vLLM's (N-1) over decode time, with prefill and
    request overhead cancelled. Every call must return its exact length.
    """
    def exact(result: RequestResult, tokens: int) -> RequestResult:
        if not _output_matches(result, tokens, settings.tolerate_off_by_one):
            raise RuntimeError(
                f"reference call returned {result.output_tokens} tokens, "
                f"expected {tokens} (backends must honor max_tokens exactly)"
            )
        return result

    lines = calibration.lines_for(REFERENCE_PROMPT_TOKENS)
    pp_prompts, pp_rates, tg_prompts, tg_rates = [], [], [], []
    for rep in range(REFERENCE_REPS):
        seed = 800000 + rep
        # The variant leads the prompt, so no rep reuses a cached prefix.
        pp = exact(request(audit_prompt(lines, seed), 1, seed), 1)
        pp_prompts.append(pp.prompt_tokens)
        pp_rates.append(pp.prompt_tokens / pp.seconds)
        prompt = f"{seed} {REFERENCE_PROMPT}"
        full = exact(request(prompt, REFERENCE_OUTPUT_TOKENS, seed),
                     REFERENCE_OUTPUT_TOKENS)
        one = exact(request(prompt, 1, seed), 1)
        decode = full.seconds - one.seconds
        if decode <= 0:
            raise RuntimeError(
                f"{REFERENCE_OUTPUT_TOKENS}-token call was not slower than "
                "the 1-token call; decode time cannot be measured"
            )
        tg_prompts.append(full.prompt_tokens)
        tg_rates.append((REFERENCE_OUTPUT_TOKENS - 1) / decode)
    reference = {
        "method": "llama-bench pp512/tg128 equivalent; not a serving result",
        "prompt_processing": {
            "target_prompt_tokens": REFERENCE_PROMPT_TOKENS,
            "prompt_tokens": pp_prompts,
            "tokens_per_second": _mean_sd(pp_rates),
        },
        "token_generation": {
            "output_tokens": REFERENCE_OUTPUT_TOKENS,
            "prompt_tokens": tg_prompts,
            "tokens_per_second": _mean_sd(tg_rates),
        },
    }
    print(f"  {_reference_line(reference)}")
    return reference


def _reference_line(reference: dict) -> str:
    if "unavailable" in reference:
        return f"unavailable: {reference['unavailable']}"
    pp = reference["prompt_processing"]
    tg = reference["token_generation"]["tokens_per_second"]
    return (
        f"pp{REFERENCE_PROMPT_TOKENS} {pp['tokens_per_second']['mean']:,.1f}"
        f" +/- {pp['tokens_per_second']['sd']:,.1f} tok/s "
        f"(actual {statistics.median(pp['prompt_tokens']):,.0f} tokens), "
        f"tg{REFERENCE_OUTPUT_TOKENS} {tg['mean']:,.1f} +/- {tg['sd']:,.1f} "
        "tok/s"
    )


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
    prompt_lines: int,
    target_prompt: int | None = None,
    near_context: bool = False,
    gate_cmd: str | None = None,
    off_by_one: bool = False,
) -> dict:
    _wait_for_gate(gate_cmd)
    prompts = [
        audit_prompt(prompt_lines, variant_base + sample_index * 100 + stream)
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
        off_by_one,
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


def _measure_capacity(
    request: RequestFn,
    streams: tuple[int, ...],
    settings: Settings,
    context: int,
    prompt_lines: int,
    target_prompt: int,
) -> dict:
    """The near-context proof: one 1-token batch per stream count, once."""
    return {
        "by_streams": {
            str(count): _summarize_measurements(count, [
                _measure_once(
                    request=request,
                    phase="capacity_near",
                    streams=count,
                    sample_index=0,
                    output_tokens=1,
                    context=context,
                    tolerance=settings.tolerance,
                    variant_base=180000 + count * 10000,
                    prompt_lines=prompt_lines,
                    target_prompt=target_prompt,
                    near_context=True,
                    gate_cmd=settings.gate_cmd,
                    off_by_one=settings.tolerate_off_by_one,
                )
            ])
            for count in streams
        }
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _output_dir() -> pathlib.Path:
    """Retention directory for JSON documents.

    The default is the directory audit-v1 results have always been kept
    in, so a fresh --list finds the existing history.
    """
    state_root = pathlib.Path(
        os.environ.get(
            "XDG_STATE_HOME", pathlib.Path.home() / ".local" / "state"
        )
    )
    # Consumers forward the whole variable list, so an unset override
    # arrives as an empty string; that means default, not "here".
    return pathlib.Path(
        os.environ.get("AUDIT_BENCH_OUTPUT_DIR")
        or state_root / "auditbench"
    )


def _artifact_path(result: dict) -> pathlib.Path:
    """Retention path for one run's JSON document."""
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    stamp = metadata["started_at_utc"].replace(":", "").replace("-", "")
    return output_dir / (
        f"{stamp}_{_slug(metadata['runtime'])}_{_slug(metadata['tag'])}"
        f"_{_slug(result['settings']['profile'])}.json"
    )


def _median_of(section: dict | None, key: str) -> float:
    """Median of a retained median/min/max triple; 0 when absent."""
    return float(((section or {}).get(key) or {}).get("median") or 0)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _stream_list(streams: list[int]) -> str:
    return ", ".join(str(count) for count in streams)


def _cached(fraction: float | None) -> str:
    return "not reported" if fraction is None else f"{fraction:.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Right-aligned fixed-width table, indented under its section."""
    widths = [
        max(len(str(cell)) for cell in column)
        for column in zip(headers, *rows)
    ]
    return [
        "  " + "  ".join(
            str(cell).rjust(width) for cell, width in zip(row, widths)
        )
        for row in [headers, *rows]
    ]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _plan_lines(plan: dict, profile: str) -> list[str]:
    """Describe the planned workload; the whole point of a dry run."""
    if profile == "conversation":
        return [f"  conversation: {plan['conversation']['steps']} steps, 1 stream"]
    near = plan["capacity_near"]
    session = plan["session"]
    return [
        f"  reference: pp{REFERENCE_PROMPT_TOKENS} and "
        f"tg{REFERENCE_OUTPUT_TOKENS}, {REFERENCE_REPS} reps, 1 stream",
        f"  near-context capacity: {near['target_prompt_tokens']:,}-token "
        f"prompt at streams {_stream_list(near['streams'])}",
        f"  session: {session['steps']} steps from "
        f"{session['evidence_tokens']:,}-token evidence, turns up to "
        f"{session['turn_output_tokens']:,} tokens, report up to "
        f"{session['report_output_tokens']:,}, streams "
        f"{_stream_list(session['streams'])}",
    ]


def _summary_lines(result: dict) -> list[str]:
    """Render one result document as the default human report."""
    metadata = result["metadata"]
    settings = result["settings"]
    plan = result["plan"]
    sampling = " ".join(
        f"{key}={value}" for key, value in (settings.get("sampling") or {}).items()
    ) or "undeclared"
    lines = [
        "",
        f"=== {result['protocol']} summary: "
        f"{metadata['runtime']}/{metadata['tag']} ===",
        f"model={metadata['model']} "
        f"context={plan['configured_context']:,} "
        f"profile={settings['profile']} "
        f"slots={metadata['active_slots']} "
        f"kv-pool={metadata['kv_pool_tokens']:,}",
        f"sampling: {sampling}",
    ]
    if settings["dry_run"]:
        lines += ["", "plan only; no server was contacted"]
        lines += _plan_lines(plan, settings["profile"])
    reference = result.get("reference")
    if reference:
        lines += [
            "",
            "reference, llama-bench method (1 stream, greedy, depth ~0; "
            "not a serving result)",
            f"  {_reference_line(reference)}",
        ]
    near = result.get("capacity_near")
    if near:
        target = plan["capacity_near"]["target_prompt_tokens"]
        lines += [
            "",
            f"near-context capacity ({target:,}-token prompt, "
            "1 output token)",
        ]
        lines += _table(
            ["streams", "wall s", "input tok/s"],
            [
                [
                    count,
                    f"{_median_of(section, 'wall_seconds'):.2f}",
                    f"{_median_of(section, 'aggregate_prompt_tokens_per_second'):,.0f}",
                ]
                for count, section in near["by_streams"].items()
            ],
        )
    session = result.get("audit_session")
    if session:
        lines += ["", f"agent session ({_plural(session['steps'], 'step')})"]
        lines += _table(
            ["streams", "sessions/h", "vs 1x", "complete", "out tok",
             "prefill tok/s", "decode first/mid/last", "report tok/s",
             "cached"],
            [
                [
                    count,
                    f"{_median_of(section, 'sessions_per_hour'):.2f}",
                    f"{session['scaling_vs_1x'][count]:.2f}",
                    f"{section['sessions'] - section['sessions_incomplete']}"
                    f"/{section['sessions']}",
                    f"{section['output_tokens_per_session']:,.0f}",
                    _rate(section["prefill_tokens_per_second"]),
                    "/".join(
                        _rate(section["turn_decode_tokens_per_second"][position])
                        for position in ("first", "middle", "last")
                    ),
                    _rate(section["report_decode_tokens_per_second"]),
                    _cached(section["cached_fraction_median"]),
                ]
                for count, section in session["by_streams"].items()
            ],
        )
        lines.append(
            "  decode rates are per stream; sessions/h is the production metric"
        )
        if "optimal_streams" in session:
            optimal = session["optimal_streams"]
            lines.append(
                f"  optimal operating point: {optimal}x"
                if optimal
                else "  no stream count completed every session"
            )
        totals = {
            key: sum(
                section[key] for section in session["by_streams"].values()
            )
            for key in ("sessions_incomplete", "truncated_calls",
                        "empty_turns", "unreasoned_calls")
        }
        if totals["sessions_incomplete"]:
            lines.append(
                f"  {_plural(totals['sessions_incomplete'], 'session')} "
                "incomplete: the report hit its cap or missed findings"
            )
        if totals["truncated_calls"]:
            lines.append(
                f"  {_plural(totals['truncated_calls'], 'call')} hit the "
                "output cap"
            )
        if totals["empty_turns"]:
            lines.append(
                f"  {_plural(totals['empty_turns'], 'turn')} ended with no "
                "text and no tool call"
            )
        if totals["unreasoned_calls"]:
            calls = sum(
                section["sessions"] for section in session["by_streams"].values()
            ) * (session["steps"] + 1)
            lines.append(
                f"  {totals['unreasoned_calls']} of {calls} calls streamed "
                "no reasoning"
            )
    conversation = result.get("audit_conversation")
    if conversation:
        samples = conversation["samples"]
        compliant = sum(1 for row in samples if row["compliant"])
        recalled = [
            row["findings_recalled"]
            for row in samples
            if row["findings_recalled"] is not None
        ]
        lines += [
            "",
            f"audit conversation ({_plural(conversation['steps'], 'step')}, "
            f"{_plural(len(samples), 'sample')})",
            f"  protocol compliant: {_yes_no(conversation['compliant'])} "
            f"({compliant} of {len(samples)} samples)",
            "  findings recalled: " + (
                f"{statistics.median(recalled):.0f} of "
                f"{conversation['steps']}"
                if recalled
                else "none"
            ),
            f"  wall {_median_of(conversation, 'wall_seconds'):.1f} s, "
            f"{_median_of(conversation, 'conversations_per_hour'):.1f} "
            "conversations/hour",
            "  prompt tokens from cache: "
            + _cached(conversation["cached_fraction_median"]),
        ]
    reasons = result["publication_reasons"]
    lines += [
        "",
        f"canonical={_yes_no(result['canonical'])} "
        f"publishable={_yes_no(result['publishable'])}"
        + (f" ({', '.join(reasons)})" if reasons else ""),
    ]
    if result.get("artifact_path"):
        lines.append(f"json: {result['artifact_path']}")
    return lines


def _document_row(
    path: pathlib.Path, directory: pathlib.Path
) -> list[str] | None:
    """One listing row, or None when the file is not an AuditBench document."""
    name = str(path.relative_to(directory))
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A truncated result is named, not hidden: it may be a lost run.
        return ["-", "-", "-", "unreadable", name]
    if not isinstance(document, dict) or document.get("protocol") not in (
        PROTOCOL, LEGACY_PROTOCOL
    ):
        return None
    metadata = document.get("metadata") or {}
    profile = str((document.get("settings") or {}).get("profile") or "-")
    if document["protocol"] == LEGACY_PROTOCOL:
        # Retained v1 history stays discoverable, marked as the old protocol.
        profile = f"v1 {profile}"
    if "publishable" not in document or "publication_reasons" not in document:
        # Exactly what --show refuses to render, named the same way.
        verdict = "legacy"
    elif document["publishable"]:
        verdict = "publishable"
    else:
        verdict = "canonical" if document.get("canonical") else "noncanonical"
    return [
        str(metadata.get("started_at_utc") or "-")[:19],
        f"{metadata.get('runtime') or '-'}/{metadata.get('tag') or '-'}",
        profile,
        verdict,
        name,
    ]


def _list_documents() -> int:
    """List retained documents, newest first; a human index, not a schema.

    Campaign evidence is filed in subdirectories of the retention
    directory, so the index covers those too. Unrelated JSON in the tree
    is counted, never dressed up as a result.
    """
    directory = _output_dir()
    paths = sorted(
        directory.rglob("*.json") if directory.is_dir() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows = []
    foreign = 0
    for path in paths:
        row = _document_row(path, directory)
        if row is None:
            foreign += 1
        else:
            rows.append(row)
    if not rows:
        print(f"no retained documents in {directory}")
    else:
        print(f"{_plural(len(rows), 'retained document')} in {directory}")
        rows = [
            ["when", "runtime/tag", "profile", "verdict", "file"],
            *rows,
        ]
        widths = [max(len(row[index]) for row in rows) for index in range(5)]
        for row in rows:
            print(
                "  "
                + "  ".join(
                    cell.ljust(width) for cell, width in zip(row, widths)
                ).rstrip()
            )
    if foreign:
        print(f"{_plural(foreign, 'other JSON file')} in the tree, ignored")
    if rows:
        print("re-print one with: audit-bench --show FILE")
    return 0


def _print_document(document: dict, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(document, sort_keys=True))
        return
    for line in _summary_lines(document):
        print(line)


def _show(path: pathlib.Path, output_format: str) -> int:
    """Re-print a retained document; no server contact, no new retention.

    The argument is a file name as printed by --list, resolved in the
    retention directory, or any path that exists as given.
    """
    directory = _output_dir()
    if not path.exists() and (directory / path).exists():
        path = directory / path
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"cannot read {path}: no such file here or in {directory}"
        ) from None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if document.get("protocol") == LEGACY_PROTOCOL:
        raise SystemExit(
            f"{path} is an {LEGACY_PROTOCOL} document; this harness renders "
            f"{PROTOCOL} only. Read the JSON directly, or --show it from an "
            f"llm-audit-bench checkout at {LEGACY_HARNESS_COMMIT}"
        )
    if document.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"{path} is not a {PROTOCOL} document: "
            f"protocol={document.get('protocol')!r}"
        )
    # An older document is read out, never guessed at: inferring a missing
    # publishable verdict would misreport provenance.
    missing = [
        key
        for key in (
            "canonical",
            "publishable",
            "publication_reasons",
            "settings",
            "metadata",
            "plan",
        )
        if key not in document
    ]
    if missing:
        raise SystemExit(
            f"{path} predates the summary: {', '.join(missing)} absent "
            f"(schema_version={document.get('schema_version')}); "
            "read the JSON directly"
        )
    document.setdefault("artifact_path", str(path))
    _print_document(document, output_format)
    return 0


def _emit(result: dict, output_format: str) -> None:
    """Retain the JSON document, then print the requested stdout form."""
    path = _artifact_path(result)
    result["artifact_path"] = str(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    _print_document(result, output_format)


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
    output_format: str = "human",
) -> int:
    """Run AuditBench v2, retain the JSON, and print progress plus result.

    The result goes to stdout as the human summary, or as the one-line JSON
    document with output_format="json"; the JSON is retained on disk either
    way and its path is reported.
    """
    if output_format not in ("human", "json"):
        raise ValueError("output_format must be human or json")
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
        f"profile={settings.profile} streams={settings.streams} "
        f"samples={settings.min_samples}-{settings.samples} "
        f"(agreement {settings.sample_agreement:.0%}) "
        f"sampling={settings.sampling or 'undeclared'}"
    )
    if settings.gate_cmd:
        print(f"gate: {settings.gate_cmd}")
    if settings.dry_run:
        _emit(_result_envelope(settings, metadata, plan), output_format)
        return 0
    if request is None:
        raise ValueError("request adapter is required unless AUDIT_BENCH_DRY_RUN=1")
    if openai_chat is None:
        raise ValueError(
            "the session and conversation profiles need an OpenAI-compatible "
            "chat adapter (openai_chat)"
        )

    print("warmup:")
    warmup = request("Reply with the single word ok.", 1, 1)
    if not _output_matches(warmup, 1, settings.tolerate_off_by_one):
        raise RuntimeError(
            "warmup did not return exactly one output token; "
            f"endpoint reported {warmup.output_tokens} for model {model!r} "
            + (
                f"({OFF_BY_ONE_EXPECTATION})"
                if settings.tolerate_off_by_one
                else "(backends must honor max_tokens exactly)"
            )
        )

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
    result = {
        **_result_envelope(settings, metadata, plan),
        "calibration": asdict(calibration),
    }

    if settings.profile == "conversation":
        print("audit conversation:")
        result["audit_conversation"] = _measure_conversations(
            openai_chat, calibration, context, settings
        )
        if not result["audit_conversation"]["compliant"]:
            result["publishable"] = False
            result["publication_reasons"].append("noncompliant_conversation")
        _emit(result, output_format)
        return 0

    print("reference (llama-bench method):")
    try:
        result["reference"] = _measure_reference(request, calibration, settings)
    except RuntimeError as exc:
        # The reference feeds nothing, so a backend that cannot force an
        # exact length (Ollama has no ignore_eos) loses only the reference.
        result["reference"] = {"unavailable": str(exc)}
        print(f"  {_reference_line(result['reference'])}")
    near_plan = plan["capacity_near"]
    print("near-context capacity:")
    result["capacity_near"] = _measure_capacity(
        request=request,
        streams=tuple(near_plan["streams"]),
        settings=settings,
        context=context,
        prompt_lines=calibration.lines_for(near_plan["target_prompt_tokens"]),
        target_prompt=near_plan["target_prompt_tokens"],
    )
    print("agent session:")
    result["audit_session"] = _measure_sessions(
        openai_chat, calibration, context, settings,
        tuple(plan["session"]["streams"]),
    )
    if result["audit_session"]["optimal_streams"] is None:
        result["publishable"] = False
        result["publication_reasons"].append("incomplete_sessions")
    _emit(result, output_format)
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
    """Run AuditBench v2 against an OpenAI-compatible chat endpoint."""
    parser = argparse.ArgumentParser(
        prog="audit-bench",
        description=(
            "Run AuditBench v2 against an OpenAI-compatible chat endpoint. "
            "Set AUDIT_BENCH_SAMPLING to the checkpoint's sampling, or the "
            "run is not canonical. AUDIT_BENCH_* environment variables "
            "select the profile and protocol overrides; see "
            "docs/PROTOCOL.md."
        ),
    )
    parser.add_argument(
        "--base-url",
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
        choices=("session", "conversation"),
        help="workload profile (default session; overrides AUDIT_BENCH_PROFILE)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="per-request timeout in seconds (default 7200)",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help=(
            "stdout result format (default human); the JSON document is "
            "retained on disk either way"
        ),
    )
    documents = parser.add_mutually_exclusive_group()
    documents.add_argument(
        "--show",
        type=pathlib.Path,
        metavar="FILE",
        help=(
            "re-print a retained JSON document and exit; takes a file name "
            "as listed by --list, or any path; contacts no server and "
            "retains nothing"
        ),
    )
    documents.add_argument(
        "--list",
        action="store_true",
        dest="list_documents",
        help="list the retained JSON documents and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the workload plan without contacting the server",
    )
    args = parser.parse_args(argv)

    if args.list_documents:
        return _list_documents()
    if args.show:
        return _show(args.show, args.format)
    for name in ("base_url", "context"):
        if getattr(args, name) is None:
            parser.error(
                f"--{name.replace('_', '-')} is required "
                "unless --show names a document"
            )
    if args.profile:
        os.environ["AUDIT_BENCH_PROFILE"] = args.profile
    if args.dry_run:
        os.environ["AUDIT_BENCH_DRY_RUN"] = "1"
    dry_run = os.environ.get("AUDIT_BENCH_DRY_RUN", "0") == "1"
    # Documented reasoning controls (docs/PROTOCOL.md) forwarded to the
    # adapters so each is both sent with every request and recorded in the
    # JSON: the documented parameter and the system-role control.
    reasoning_effort = os.environ.get("AUDIT_BENCH_REASONING_EFFORT") or None
    system_prompt = os.environ.get("AUDIT_BENCH_SYSTEM_PROMPT") or None

    active_slots = _cli_capacity(
        args.active_slots,
        "--active-slots",
        "AUDIT_BENCH_ACTIVE_SLOTS",
        1,
        "--active-slots not given; assuming 1, sessions run 1x only",
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
                args.base_url.rstrip("/") + "/v1",
                model,
                args.timeout,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
            )
        ),
        output_format=args.format,
    )


if __name__ == "__main__":
    raise SystemExit(main())

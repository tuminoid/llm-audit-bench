#!/usr/bin/env python3
"""Shared capability-v1 protocol for local inference servers.

Runtime adapters supply transport and identity evidence. The module also
runs standalone against any OpenAI-compatible or Ollama endpoint via
main().
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PROTOCOL = "capability-v1"
SCHEMA_VERSION = 1
TOOL_NAME = "lookup_issue"
TOOL_SENTINEL = "sentinel-cobalt"
LEAK_MARKERS = (
    "<think>",
    "</think>",
    "<analysis>",
    "</analysis>",
    "<|channel>",
    "<|channel|>",
    "<|tool_call>",
)
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return one issue by its exact numeric identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_number": {"type": "integer"},
            },
            "required": ["issue_number"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ToolCall:
    call_id: str | None
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    legacy_reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    chunks: int = 0
    completed: bool = False
    finish_reason: str | None = None


@dataclass(frozen=True)
class CapabilityConfig:
    reasoning: str
    tools: str
    reasoning_options: dict = field(default_factory=dict)
    disable_options: dict | None = None
    reasoning_prefix: str = ""
    disable_prefix: str = ""
    system_prompt: str | None = None

    def validate(self) -> None:
        if self.reasoning not in {"required", "absent"}:
            raise ValueError("reasoning expectation must be required or absent")
        if self.tools not in {"required", "absent"}:
            raise ValueError("tools expectation must be required or absent")

    def to_dict(self) -> dict:
        return {
            "reasoning": self.reasoning,
            "tools": self.tools,
            "reasoning_options": self.reasoning_options,
            "disable_options": self.disable_options,
            "reasoning_prefix": self.reasoning_prefix,
            "disable_prefix": self.disable_prefix,
            "system_prompt": self.system_prompt,
        }


def resolve_shell_config(
    conf: pathlib.Path, names: tuple[str, ...]
) -> dict[str, str]:
    """Resolve a trusted model config with the same inheritance as launchers."""
    printers = "\n".join(f'printf "%s\\0" "${{{name}:-}}"' for name in names)
    try:
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'set -euo pipefail\nsource "$1"\n{printers}',
                "bash",
                str(conf),
            ],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("bash is required to resolve model config") from exc
    if completed.returncode:
        error = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"failed to resolve {conf}: {error}")
    values = completed.stdout.split(b"\0")
    if len(values) != len(names) + 1 or values[-1]:
        raise RuntimeError(f"unexpected config output while resolving {conf}")
    return {
        name: value.decode(errors="strict")
        for name, value in zip(names, values[:-1], strict=True)
    }


def openai_control(mode: str) -> tuple[dict, str]:
    controls = {
        "default": ({}, ""),
        "enable-thinking": (
            {"chat_template_kwargs": {"enable_thinking": True}},
            "",
        ),
        "enable-thinking-false": (
            {"chat_template_kwargs": {"enable_thinking": False}},
            "",
        ),
        "think-prefix": ({}, "/think\n"),
        "no-think-prefix": ({}, "/no_think\n"),
        "none": (None, ""),
    }
    if mode not in controls:
        raise ValueError(f"unsupported OpenAI capability mode: {mode}")
    options, prefix = controls[mode]
    return options, prefix


def ollama_control(mode: str) -> tuple[dict | None, str]:
    controls = {
        "default": ({}, ""),
        "think-true": ({"think": True}, ""),
        "think-false": ({"think": False}, ""),
        "none": (None, ""),
    }
    if mode not in controls:
        raise ValueError(f"unsupported Ollama capability mode: {mode}")
    return controls[mode]


def _json_request(url: str, payload: dict | None, timeout: int) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"{exc.code} from {url}: {body}") from exc


def _arguments(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"tool arguments are not valid JSON: {value}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"tool arguments are not an object: {value!r}")
    return value


def _openai_tool_calls(message: dict) -> list[ToolCall]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(
            ToolCall(
                call_id=call.get("id"),
                name=str(function.get("name", "")),
                arguments=_arguments(function.get("arguments", {})),
            )
        )
    return calls


def _ollama_tool_calls(message: dict) -> list[ToolCall]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(
            ToolCall(
                call_id=None,
                name=str(function.get("name", "")),
                arguments=_arguments(function.get("arguments", {})),
            )
        )
    return calls


class OpenAIChatAdapter:
    def __init__(self, base_url: str, model: str, timeout: int):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool,
        tools: list[dict] | None = None,
        options: dict | None = None,
        force_tool: str | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
            **(options or {}),
        }
        if tools:
            payload["tools"] = tools
        if force_tool:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": force_tool},
            }
        if not stream:
            body = _json_request(self.url, payload, self.timeout)
            choices = body.get("choices") or []
            if len(choices) != 1:
                raise RuntimeError(f"expected one chat choice, found {len(choices)}")
            choice = choices[0]
            message = choice.get("message") or {}
            return ChatResult(
                content=str(message.get("content") or ""),
                reasoning=str(message.get("reasoning") or ""),
                legacy_reasoning=str(message.get("reasoning_content") or ""),
                tool_calls=_openai_tool_calls(message),
                chunks=1,
                completed=True,
                finish_reason=choice.get("finish_reason"),
            )

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        result = ChatResult()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode(errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        result.completed = True
                        break
                    body = json.loads(data)
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    result.chunks += 1
                    result.content += str(delta.get("content") or "")
                    result.reasoning += str(delta.get("reasoning") or "")
                    result.legacy_reasoning += str(
                        delta.get("reasoning_content") or ""
                    )
                    if choice.get("finish_reason") is not None:
                        result.finish_reason = choice["finish_reason"]
                        result.completed = True
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"{exc.code} from {self.url}: {body}") from exc
        return result

    @staticmethod
    def assistant_message(result: ChatResult) -> dict:
        tool_calls = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            }
            for call in result.tool_calls
        ]
        return {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def tool_message(call: ToolCall, content: str) -> dict:
        if not call.call_id:
            raise RuntimeError("OpenAI tool call did not include an id")
        return {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": content,
        }


class OllamaChatAdapter:
    def __init__(self, base_url: str, model: str, timeout: int):
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool,
        tools: list[dict] | None = None,
        options: dict | None = None,
        force_tool: str | None = None,
    ) -> ChatResult:
        del force_tool
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": -1,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
            **(options or {}),
        }
        if tools:
            payload["tools"] = tools
        if not stream:
            body = _json_request(self.url, payload, self.timeout)
            message = body.get("message") or {}
            return ChatResult(
                content=str(message.get("content") or ""),
                reasoning=str(message.get("thinking") or ""),
                tool_calls=_ollama_tool_calls(message),
                chunks=1,
                completed=bool(body.get("done", True)),
                finish_reason=body.get("done_reason"),
            )

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        result = ChatResult()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    body = json.loads(raw_line)
                    message = body.get("message") or {}
                    result.chunks += 1
                    result.content += str(message.get("content") or "")
                    result.reasoning += str(message.get("thinking") or "")
                    if body.get("done"):
                        result.completed = True
                        result.finish_reason = body.get("done_reason")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"{exc.code} from {self.url}: {body}") from exc
        return result

    @staticmethod
    def assistant_message(result: ChatResult) -> dict:
        return {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in result.tool_calls
            ],
        }

    @staticmethod
    def tool_message(call: ToolCall, content: str) -> dict:
        return {
            "role": "tool",
            "tool_name": call.name,
            "content": content,
        }


def _messages(config: CapabilityConfig, prompt: str) -> list[dict]:
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _no_leak(content: str) -> None:
    marker = next((item for item in LEAK_MARKERS if item in content), None)
    if marker:
        raise RuntimeError(f"reasoning or tool marker leaked into content: {marker}")


def _require_completed(result: ChatResult) -> None:
    if not result.completed:
        raise RuntimeError("chat response did not complete")
    if result.legacy_reasoning:
        raise RuntimeError("reasoning arrived in legacy reasoning_content")


def _result_summary(result: ChatResult) -> dict:
    return {
        "content": result.content[:500],
        "content_length": len(result.content),
        "reasoning_length": len(result.reasoning),
        "legacy_reasoning_length": len(result.legacy_reasoning),
        "tool_calls": [
            {
                "id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in result.tool_calls
        ],
        "chunks": result.chunks,
        "completed": result.completed,
        "finish_reason": result.finish_reason,
    }


def _record(checks: dict, name: str, operation) -> Any:
    started = time.monotonic()
    try:
        evidence, value = operation()
        checks[name] = {
            "status": "passed",
            "seconds": time.monotonic() - started,
            **evidence,
        }
        return value
    except Exception as exc:
        checks[name] = {
            "status": "failed",
            "seconds": time.monotonic() - started,
            "error": str(exc),
        }
        return None


def _skip(checks: dict, name: str, reason: str) -> None:
    checks[name] = {"status": "skipped", "reason": reason}


def _output_dir() -> pathlib.Path:
    """Retention directory for JSON reports."""
    state_root = pathlib.Path(
        os.environ.get("XDG_STATE_HOME", pathlib.Path.home() / ".local" / "state")
    )
    return pathlib.Path(
        os.environ.get(
            "CAPABILITY_OUTPUT_DIR", state_root / "capability-check"
        )
    )


def _artifact_path(report: dict) -> pathlib.Path:
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report["checked_at_utc"].replace(":", "").replace("-", "")
    safe_runtime = re.sub(r"[^a-z0-9]+", "-", report["runtime"].lower()).strip("-")
    safe_tag = re.sub(r"[^a-z0-9]+", "-", report["tag"].lower()).strip("-")
    return output_dir / f"{timestamp}_{safe_runtime}_{safe_tag}.json"


def _list_reports() -> int:
    """List retained reports, newest first; a human index, not a schema.

    Campaign evidence is filed in subdirectories of the retention
    directory, so the index covers those too. Unrelated JSON in the tree
    is counted, never dressed up as a report.
    """
    directory = _output_dir()
    paths = sorted(
        directory.rglob("*.json") if directory.is_dir() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows = [["when", "result", "runtime/tag", "file"]]
    foreign = 0
    for path in paths:
        name = str(path.relative_to(directory))
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows.append(["-", "unreadable", "-", name])
            continue
        if not isinstance(report, dict) or report.get("protocol") != PROTOCOL:
            foreign += 1
            continue
        rows.append([
            str(report.get("checked_at_utc") or "-")[:19],
            str(report.get("status") or "-"),
            f"{report.get('runtime') or '-'}/{report.get('tag') or '-'}",
            name,
        ])
    if len(rows) == 1:
        print(f"no retained reports in {directory}")
    else:
        print(f"{len(rows) - 1} retained reports in {directory}")
        widths = [max(len(row[index]) for row in rows) for index in range(4)]
        for row in rows:
            print(
                "  "
                + "  ".join(
                    cell.ljust(width) for cell, width in zip(row, widths)
                ).rstrip()
            )
    if foreign:
        print(f"{foreign} other JSON files in the tree, ignored")
    if len(rows) > 1:
        print("re-print one with: capability-check --show FILE")
    return 0


def _identity_line(identity: dict) -> str:
    """One line of the identity evidence the results table cites."""
    if not identity.get("checked"):
        return "identity: unchecked"
    detail = next(
        (
            f"{key}={identity[key]}"
            for key in ("digest", "revision", "parameter_size")
            if identity.get(key)
        ),
        None,
    )
    return (
        f"identity: {'verified' if identity.get('matched') else 'MISMATCH'}"
        f" via {identity.get('source') or 'unknown source'}"
        + (f", {detail}" if detail else "")
    )


def _summary_lines(report: dict) -> list[str]:
    """Render the report as the default human verdict, not a score."""
    expectations = report.get("expectations") or {}
    checks = report.get("checks") or {}
    lines = [
        "",
        f"=== {report['protocol']} verdict: "
        f"{report['runtime']}/{report['tag']} ===",
        f"model={report['model']}"
        + (
            f" expects reasoning={expectations['reasoning']} "
            f"tools={expectations['tools']}"
            if expectations
            else ""
        ),
        "",
    ]
    width = max(len(name) for name in checks) if checks else 5
    lines.append(f"  {'check'.ljust(width)}  status   seconds")
    for name, check in checks.items():
        seconds = check.get("seconds")
        lines.append(
            f"  {name.ljust(width)}  {check['status']:<8}"
            + (f"{'-':>8}" if seconds is None else f"{seconds:8.2f}")
        )
    notes = [
        f"  {name} {check['status']}: "
        + str(check.get("error") or check.get("reason"))
        for name, check in checks.items()
        if check.get("error") or check.get("reason")
    ]
    if notes:
        lines += ["", *notes]
    lines += [
        "",
        _identity_line(report.get("identity") or {}),
        f"result: {report['status'].upper()}"
        + (
            f" ({', '.join(report['failed_checks'])})"
            if report.get("failed_checks")
            else ""
        ),
    ]
    if report.get("artifact_path"):
        lines.append(f"json: {report['artifact_path']}")
    return lines


def _print_report(report: dict, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(report, sort_keys=True))
        return
    for line in _summary_lines(report):
        print(line)


def _show(path: pathlib.Path, output_format: str) -> int:
    """Re-print a retained report; no server contact, no new retention.

    The argument is a file name as printed by --list, resolved in the
    retention directory, or any path that exists as given.
    """
    directory = _output_dir()
    if not path.exists() and (directory / path).exists():
        path = directory / path
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"cannot read {path}: no such file here or in {directory}"
        ) from None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if report.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"{path} is not a {PROTOCOL} report: "
            f"protocol={report.get('protocol')!r}"
        )
    missing = [
        key
        for key in ("status", "runtime", "tag", "model", "checks")
        if key not in report
    ]
    if missing:
        raise SystemExit(
            f"{path} is not a readable {PROTOCOL} report: "
            f"{', '.join(missing)} absent"
        )
    report.setdefault("artifact_path", str(path))
    _print_report(report, output_format)
    return 0


def _emit(report: dict, output_format: str) -> None:
    """Retain the JSON report, then print the requested stdout form."""
    report["host"] = platform.node()
    report["platform"] = platform.platform()
    path = _artifact_path(report)
    report["artifact_path"] = str(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    _print_report(report, output_format)


def run_capability_check(
    *,
    runtime: str,
    tag: str,
    model: str,
    adapter,
    config: CapabilityConfig,
    identity: dict,
    timeout_tokens: int = 2048,
    output_format: str = "human",
) -> int:
    """Run capability-v1, retain the JSON report, and return its exit code.

    Stdout carries the human verdict, or the one-line JSON report with
    output_format="json"; the report is retained on disk either way.
    """
    if output_format not in ("human", "json"):
        raise ValueError("output_format must be human or json")
    config.validate()
    checks = {}

    def check_identity():
        if not identity.get("checked") or not identity.get("matched"):
            raise RuntimeError("live model identity was not verified")
        return {"identity": identity}, identity

    _record(checks, "model_identity", check_identity)

    reasoning_prompt = (
        config.reasoning_prefix
        + "What is 17 multiplied by 23? Return only 391 in the final answer."
    )

    def check_streaming():
        result = adapter.chat(
            _messages(config, reasoning_prompt),
            max_tokens=timeout_tokens,
            stream=True,
            options=config.reasoning_options,
        )
        _require_completed(result)
        if result.chunks < 1:
            raise RuntimeError("stream did not complete")
        if "391" not in result.content:
            raise RuntimeError(f"final content did not contain 391: {result.content!r}")
        _no_leak(result.content)
        if config.reasoning == "required" and not result.reasoning.strip():
            raise RuntimeError("reasoning channel was empty")
        if config.reasoning == "absent" and result.reasoning.strip():
            raise RuntimeError("non-thinking model emitted reasoning")
        return {"response": _result_summary(result)}, result

    _record(checks, "streaming_reasoning", check_streaming)

    if config.disable_options is None and not config.disable_prefix:
        _skip(checks, "thinking_disabled", "model has no configured disable mode")
    else:
        disable_prompt = (
            config.disable_prefix
            + "What is 17 multiplied by 23? Return only 391 in the final answer."
        )

        def check_disabled():
            result = adapter.chat(
                _messages(config, disable_prompt),
                max_tokens=timeout_tokens,
                stream=True,
                options=config.disable_options or {},
            )
            _require_completed(result)
            if "391" not in result.content:
                raise RuntimeError(
                    f"disabled-thinking content did not contain 391: "
                    f"{result.content!r}"
                )
            _no_leak(result.content)
            if result.reasoning.strip() or result.legacy_reasoning.strip():
                raise RuntimeError("thinking-disabled request emitted reasoning")
            return {"response": _result_summary(result)}, result

        _record(checks, "thinking_disabled", check_disabled)

    tool_state = {}
    if config.tools == "absent":
        _skip(checks, "tool_call", "model is configured without tools")
        _skip(checks, "tool_round_trip", "model is configured without tools")
    else:
        user_prompt = (
            config.reasoning_prefix
            + "Call lookup_issue exactly once with issue_number 42. "
            "Do not answer from memory. After the tool result, report its "
            "status token."
        )
        initial_messages = _messages(config, user_prompt)

        def check_tool_call():
            result = adapter.chat(
                initial_messages,
                max_tokens=timeout_tokens,
                stream=False,
                tools=[TOOL],
                options=config.reasoning_options,
                force_tool=TOOL_NAME,
            )
            _require_completed(result)
            _no_leak(result.content)
            if len(result.tool_calls) != 1:
                raise RuntimeError(
                    f"expected one tool call, found {len(result.tool_calls)}"
                )
            call = result.tool_calls[0]
            if call.name != TOOL_NAME:
                raise RuntimeError(f"model called {call.name!r}, expected {TOOL_NAME}")
            if call.arguments != {"issue_number": 42}:
                raise RuntimeError(
                    f"tool arguments were {call.arguments!r}, "
                    "expected {'issue_number': 42}"
                )
            tool_state["result"] = result
            tool_state["call"] = call
            tool_state["messages"] = initial_messages
            return {"response": _result_summary(result)}, result

        tool_result = _record(checks, "tool_call", check_tool_call)
        if tool_result is None:
            _skip(checks, "tool_round_trip", "tool_call failed")
        else:

            def check_tool_round_trip():
                call = tool_state["call"]
                messages = [
                    *tool_state["messages"],
                    adapter.assistant_message(tool_state["result"]),
                    adapter.tool_message(call, TOOL_SENTINEL),
                ]
                result = adapter.chat(
                    messages,
                    max_tokens=timeout_tokens,
                    stream=False,
                    tools=[TOOL],
                    options=config.reasoning_options,
                )
                _require_completed(result)
                _no_leak(result.content)
                if TOOL_SENTINEL not in result.content:
                    raise RuntimeError(
                        f"final answer did not consume tool result: {result.content!r}"
                    )
                if result.tool_calls:
                    raise RuntimeError("model called a tool again after receiving result")
                return {"response": _result_summary(result)}, result

            _record(checks, "tool_round_trip", check_tool_round_trip)

    def check_structured_json():
        options = (
            config.disable_options
            if config.disable_options is not None
            else config.reasoning_options
        )
        prefix = config.disable_prefix or config.reasoning_prefix
        result = adapter.chat(
            _messages(
                config,
                prefix
                + 'Return exactly {"status":"ok","count":3} as JSON. '
                "Do not use Markdown.",
            ),
            max_tokens=timeout_tokens,
            stream=False,
            options=options,
        )
        _require_completed(result)
        _no_leak(result.content)
        try:
            parsed = json.loads(result.content.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"structured response was not direct JSON: {result.content!r}"
            ) from exc
        if parsed != {"status": "ok", "count": 3}:
            raise RuntimeError(f"structured response was {parsed!r}")
        return {"response": _result_summary(result)}, result

    _record(checks, "structured_json", check_structured_json)

    failed = [name for name, check in checks.items() if check["status"] == "failed"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "failed" if failed else "passed",
        "runtime": runtime,
        "tag": tag,
        "model": model,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "expectations": config.to_dict(),
        "identity": identity,
        "checks": checks,
        "failed_checks": failed,
    }
    _emit(report, output_format)
    return 1 if failed else 0


def setup_failure(
    *,
    runtime: str,
    tag: str,
    model: str | None,
    error: Exception,
    output_format: str = "human",
) -> int:
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "failed",
        "runtime": runtime,
        "tag": tag,
        "model": model,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "setup": {
                "status": "failed",
                "error": str(error),
            }
        },
        "failed_checks": ["setup"],
    }
    _emit(report, output_format)
    return 1


def _openai_identity(base_url: str, model: str, timeout: int) -> dict:
    """Verify the model is live on an OpenAI-compatible endpoint."""
    url = f"{base_url.rstrip('/')}/v1/models"
    body = _json_request(url, None, timeout)
    ids = [str(item.get("id") or "") for item in body.get("data") or []]
    if model not in ids:
        live = ", ".join(ids) or "none"
        raise RuntimeError(f"model {model!r} not live at {url}; found: {live}")
    return {
        "checked": True,
        "matched": True,
        "source": "/v1/models",
        "live_models": len(ids),
    }


def _ollama_identity(base_url: str, model: str, timeout: int) -> dict:
    """Verify the model exists on an Ollama endpoint via /api/show."""
    url = f"{base_url.rstrip('/')}/api/show"
    body = _json_request(url, {"model": model}, timeout)
    details = body.get("details") or {}
    return {
        "checked": True,
        "matched": True,
        "source": "/api/show",
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run capability-v1 against an OpenAI-compatible or Ollama endpoint."""
    parser = argparse.ArgumentParser(
        prog="capability-check",
        description=(
            "Run Capability Check v1 against a live endpoint. --reasoning "
            "and --tools state the model's documented expectations; a "
            "mismatch is a failed check, not a wrong invocation. See "
            "docs/CAPABILITY.md."
        ),
    )
    parser.add_argument(
        "--base-url",
        help="server base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--model", help="served model name")
    parser.add_argument(
        "--api",
        choices=("openai", "ollama"),
        default="openai",
        help="endpoint flavor (default openai)",
    )
    parser.add_argument(
        "--reasoning",
        choices=("required", "absent"),
        default="required",
        help="whether the model must emit reasoning (default required)",
    )
    parser.add_argument(
        "--tools",
        choices=("required", "absent"),
        default="required",
        help="whether structured tool calls are required (default required)",
    )
    parser.add_argument(
        "--reasoning-mode",
        default="default",
        help=(
            "documented control enabling reasoning; openai: default, "
            "enable-thinking, enable-thinking-false, think-prefix, "
            "no-think-prefix, none; ollama: default, think-true, "
            "think-false, none"
        ),
    )
    parser.add_argument(
        "--disable-mode",
        default="none",
        help=(
            "documented control disabling reasoning, same values as "
            "--reasoning-mode; none skips the thinking-disabled check"
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="optional system message sent with every check",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CAPABILITY_TIMEOUT", "600")),
        help="per-request timeout in seconds (default CAPABILITY_TIMEOUT or 600)",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help=(
            "stdout report format (default human); the JSON report is "
            "retained on disk either way"
        ),
    )
    reports = parser.add_mutually_exclusive_group()
    reports.add_argument(
        "--show",
        type=pathlib.Path,
        metavar="FILE",
        help=(
            "re-print a retained JSON report and exit; takes a file name as "
            "listed by --list, or any path; contacts no server and retains "
            "nothing"
        ),
    )
    reports.add_argument(
        "--list",
        action="store_true",
        dest="list_reports",
        help="list the retained JSON reports and exit",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="report tag (default: the model name)",
    )
    args = parser.parse_args(argv)
    if args.list_reports:
        return _list_reports()
    if args.show:
        return _show(args.show, args.format)
    for name in ("base_url", "model"):
        if getattr(args, name) is None:
            parser.error(
                f"--{name.replace('_', '-')} is required "
                "unless --show names a report"
            )
    tag = args.tag or args.model
    runtime = f"endpoint-{args.api}"
    try:
        if args.api == "openai":
            reasoning_options, reasoning_prefix = openai_control(
                args.reasoning_mode
            )
            disable_options, disable_prefix = openai_control(args.disable_mode)
            adapter = OpenAIChatAdapter(args.base_url, args.model, args.timeout)
            identity = _openai_identity(args.base_url, args.model, args.timeout)
        else:
            reasoning_options, reasoning_prefix = ollama_control(
                args.reasoning_mode
            )
            disable_options, disable_prefix = ollama_control(args.disable_mode)
            adapter = OllamaChatAdapter(args.base_url, args.model, args.timeout)
            identity = _ollama_identity(args.base_url, args.model, args.timeout)
        config = CapabilityConfig(
            reasoning=args.reasoning,
            tools=args.tools,
            reasoning_options=reasoning_options or {},
            disable_options=disable_options,
            reasoning_prefix=reasoning_prefix,
            disable_prefix=disable_prefix,
            system_prompt=args.system_prompt,
        )
        tokens = int(os.environ.get("CAPABILITY_REASONING_TOKENS", "2048"))
        return run_capability_check(
            runtime=runtime,
            tag=tag,
            model=args.model,
            adapter=adapter,
            config=config,
            identity=identity,
            timeout_tokens=tokens,
            output_format=args.format,
        )
    except Exception as exc:
        return setup_failure(
            runtime=runtime,
            tag=tag,
            model=args.model,
            error=exc,
            output_format=args.format,
        )


if __name__ == "__main__":
    raise SystemExit(main())

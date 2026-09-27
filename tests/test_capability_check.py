import contextlib
import dataclasses
import io
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock

import capability_check
from capability_check import (
    CapabilityConfig,
    ChatResult,
    HTTPRejected,
    ToolCall,
    _ollama_identity,
    _openai_identity,
    _output_dir,
    main,
    openai_control,
    ollama_control,
    resolve_shell_config,
    run_capability_check,
)


def setUpModule():
    # The API-check fakes below know nothing of the audit fitness tasks,
    # which have their own fakes and tests.
    patcher = mock.patch.dict(os.environ, {"CAPABILITY_FITNESS_SAMPLES": "0"})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def conforming_document() -> str:
    """A document that satisfies the fixture schema exactly."""
    return json.dumps(
        {
            "repository": "acme/ledger",
            "summary": {
                "verdict": "findings",
                "finding_count": capability_check.JSON_SCHEMA_FINDINGS,
                "highest_severity": "critical",
                "confident": True,
            },
            "findings": [
                {
                    "id": f"F-{index:03d}",
                    "title": f"issue {index}",
                    "severity": "high",
                    "category": "injection",
                    "location": {
                        "file": f"module_{index}.py",
                        "symbol": f"function_{index}",
                        "start_line": index,
                        "end_line": index + 4,
                    },
                    "evidence": ["line quoted", "call path"],
                    "remediation": "parameterize the query",
                    "exploitable": True,
                }
                for index in range(capability_check.JSON_SCHEMA_FINDINGS)
            ],
            "unreviewed": ["vendor/", "migrations/"],
        }
    )


class FakeAdapter:
    def chat(
        self,
        messages,
        *,
        max_tokens,
        stream,
        tools=None,
        options=None,
    ):
        del max_tokens
        prompt = str(messages[-1].get("content", ""))
        has_tool_result = any(
            message.get("role") == "tool" for message in messages
        )
        if options and "response_format" in options:
            # A constrained server ignores the prose and obeys the schema.
            return ChatResult(
                content=conforming_document(),
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        if "matching this schema" in prompt:
            # The unaided ask: this fake model complies without a grammar.
            return ChatResult(
                content=conforming_document(),
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        if tools and not has_tool_result and "Call lookup_issue" in prompt:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        call_id="call-1",
                        name="lookup_issue",
                        arguments={"issue_number": 42},
                    )
                ],
                chunks=2 if stream else 1,
                completed=True,
                finish_reason="tool_calls",
            )
        if has_tool_result:
            return ChatResult(
                content="sentinel-cobalt",
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        if "Return exactly" in prompt:
            return ChatResult(
                content='{"status":"ok","count":3}',
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        return ChatResult(
            content="391",
            reasoning="" if options and options.get("disabled") else "17 * 23",
            chunks=2,
            completed=True,
            finish_reason="stop",
        )

    @staticmethod
    def assistant_message(result):
        return {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in result.tool_calls
            ],
        }

    @staticmethod
    def tool_message(call, content):
        return {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": content,
        }

    @staticmethod
    def schema_options(schema):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "audit_verdict", "schema": schema,
                                "strict": True},
            }
        }


class UnenforcingAdapter(FakeAdapter):
    """A server that accepts response_format and ignores it."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if options and "response_format" in options:
            return ChatResult(
                content='## Audit\n\nOne finding.\n\n```json\n'
                '{"verdict":"finding","count":1,"refs":["a.py","b.py"]}\n```',
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class RejectingAdapter(FakeAdapter):
    """A server without constrained decoding, refusing the parameter."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if options and "response_format" in options:
            raise HTTPRejected(
                400, "http://x/v1/chat/completions",
                '{"error":"response_format is not supported"}',
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class JsonOnlyAdapter(FakeAdapter):
    """A server that enforces JSON-ness but not the schema."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if options and "response_format" in options:
            # Valid JSON, one finding short and an enum it invented: the
            # shape a compiler produces when it drops keywords.
            document = json.loads(conforming_document())
            document["findings"] = document["findings"][:-1]
            document["summary"]["highest_severity"] = "catastrophic"
            return ChatResult(
                content=json.dumps(document),
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class TruncatingAdapter(FakeAdapter):
    """A constrained response cut off at the token budget."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if options and "response_format" in options:
            return ChatResult(
                content=conforming_document()[:200],
                chunks=1,
                completed=True,
                finish_reason="length",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class WhitespaceRunawayAdapter(FakeAdapter):
    """A grammar-valid whitespace loop: legal tokens, no document."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if options and "response_format" in options:
            return ChatResult(
                content='{\n  "repository":' + "\t" * 4000,
                chunks=1,
                completed=True,
                finish_reason="length",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class ProseAdapter(FakeAdapter):
    """A model that answers the unaided ask with a report, not a document."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        prompt = str(messages[-1].get("content", ""))
        if "matching this schema" in prompt and not (
            options and "response_format" in options
        ):
            return ChatResult(
                content="# Security Audit Report: acme/ledger\n\nThis report "
                "details six findings across the reviewed modules.",
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options,
        )


class SchemalessAdapter(FakeAdapter):
    """An adapter that cannot express enforcement at all."""

    schema_options = None


class ToollessAdapter(FakeAdapter):
    """A model that ignores an automatic tool call, failing that check."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None):
        if tools and not any(
            message.get("role") == "tool" for message in messages
        ):
            return ChatResult(
                content="I would rather answer from memory.",
                chunks=1,
                completed=True,
                finish_reason="stop",
            )
        return super().chat(
            messages,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            options=options,
        )


class AdapterTest(unittest.TestCase):
    def test_openai_reasoning_aliases(self):
        cases = [
            ({"reasoning": "17 * 23"}, "17 * 23", ""),
            ({"reasoning_content": "17 * 23"}, "17 * 23", "17 * 23"),
            ({"reasoning": None, "reasoning_content": "17 * 23"},
             "17 * 23", "17 * 23"),
            ({"reasoning": "17 * 23", "reasoning_content": "17 * 23"},
             "17 * 23", "17 * 23"),
            ({}, "", ""),
        ]
        for stream in (False, True):
            for fields, reasoning, legacy in cases:
                with self.subTest(stream=stream, fields=fields):
                    result = self._openai_reasoning_response(fields, stream)
                    self.assertEqual(result.content, "391")
                    self.assertEqual(result.reasoning, reasoning)
                    self.assertEqual(result.legacy_reasoning, legacy)
                    capability_check._require_completed(result)
            with self.subTest(stream=stream, conflict=True):
                with self.assertRaisesRegex(RuntimeError, "conflicting"):
                    self._openai_reasoning_response(
                        {"reasoning": "one", "reasoning_content": "two"},
                        stream,
                    )

    @staticmethod
    def _openai_reasoning_response(fields, stream):
        message = {"content": "391", **fields}
        body = {"choices": [{"message": message, "finish_reason": "stop"}]}
        # Separate alias events prove normalization follows assembly rather
        # than concatenating duplicated fields or comparing chunk boundaries.
        deltas = [{key: value} for key, value in fields.items()]
        deltas.append({"content": "391"})
        events = [
            ("data: " + json.dumps({"choices": [{"delta": delta}]})
             + "\n").encode()
            for delta in deltas
        ] + [b"data: [DONE]\n"]
        response = mock.MagicMock()
        response.__enter__.return_value = events
        with mock.patch.object(
            capability_check, "_json_request", return_value=body
        ), mock.patch.object(
            capability_check.urllib.request, "urlopen", return_value=response
        ):
            return capability_check.OpenAIChatAdapter("http://x", "m", 5).chat(
                [{"role": "user", "content": "calculate"}],
                max_tokens=100, stream=stream,
            )

    def test_openai_non_streaming_tools_use_auto_choice(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_issue",
                                    "arguments": '{"issue_number":42}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        adapter = capability_check.OpenAIChatAdapter("http://x", "m", 5)
        with mock.patch.object(
            capability_check, "_json_request", return_value=body
        ) as request:
            result = adapter.chat(
                [{"role": "user", "content": "call it"}],
                max_tokens=100,
                stream=False,
                tools=[capability_check.TOOL],
            )
        payload = request.call_args.args[1]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(
            result.tool_calls,
            [
                ToolCall(
                    call_id="call-1",
                    name="lookup_issue",
                    arguments={"issue_number": 42},
                )
            ],
        )

    def test_openai_streaming_tools_use_auto_choice_and_assemble_call(self):
        events = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"id":"call-1","function":{"name":"lookup_",'
            b'"arguments":"{\\"issue_"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"name":"issue","arguments":"number\\":"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":"42}"}}]},'
            b'"finish_reason":"tool_calls"}]}\n',
            b"data: [DONE]\n",
        ]
        response = mock.MagicMock()
        response.__enter__.return_value = events
        adapter = capability_check.OpenAIChatAdapter("http://x", "m", 5)
        with mock.patch.object(
            capability_check.urllib.request,
            "urlopen",
            return_value=response,
        ) as urlopen:
            result = adapter.chat(
                [{"role": "user", "content": "call it"}],
                max_tokens=100,
                stream=True,
                tools=[capability_check.TOOL],
            )
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(
            result.tool_calls,
            [
                ToolCall(
                    call_id="call-1",
                    name="lookup_issue",
                    arguments={"issue_number": 42},
                )
            ],
        )

    def test_ollama_streaming_tools_collect_call(self):
        events = [
            json.dumps(
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup_issue",
                                    "arguments": {"issue_number": 42},
                                },
                            }
                        ]
                    },
                    "done": False,
                }
            ).encode()
            + b"\n",
            b'{"message":{},"done":true,"done_reason":"stop"}\n',
        ]
        response = mock.MagicMock()
        response.__enter__.return_value = events
        adapter = capability_check.OllamaChatAdapter("http://x", "m", 5)
        with mock.patch.object(
            capability_check.urllib.request,
            "urlopen",
            return_value=response,
        ):
            result = adapter.chat(
                [{"role": "user", "content": "call it"}],
                max_tokens=100,
                stream=True,
                tools=[capability_check.TOOL],
            )
        self.assertEqual(
            result.tool_calls,
            [
                ToolCall(
                    call_id=None,
                    name="lookup_issue",
                    arguments={"issue_number": 42},
                )
            ],
        )


class CapabilityCheckTest(unittest.TestCase):
    @staticmethod
    def _run(
        output_format: str,
        directory: str,
        adapter=None,
        **overrides,
    ) -> tuple[int, str]:
        output = io.StringIO()
        with mock.patch.dict(
            os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
        ):
            with contextlib.redirect_stdout(output):
                code = run_capability_check(
                    runtime="fake",
                    tag="fixture",
                    model="fixture",
                    adapter=adapter or FakeAdapter(),
                    config=CapabilityConfig(
                        reasoning="required",
                        tools="required",
                        disable_options={"disabled": True},
                    ),
                    identity={
                        "checked": True,
                        "matched": True,
                        "source": "/v1/models",
                    },
                    output_format=output_format,
                    **overrides,
                )
        return code, output.getvalue()

    def test_complete_protocol_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result, text = self._run("json", directory)
            artifacts = list(pathlib.Path(directory).glob("*.json"))
            artifact_report = json.loads(
                artifacts[0].read_text(encoding="utf-8")
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(artifacts), 1)
        report = json.loads(text)
        self.assertEqual(report["schema_version"], 5)
        self.assertEqual(report["protocol"], "capability-v2")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(artifact_report["status"], "passed")

    def test_reasoning_alias_keeps_protocol_guards(self):
        for scenario in ("valid", "disabled_reasoning", "content_leak"):
            with self.subTest(scenario=scenario):
                adapter = FakeAdapter()
                original_chat = adapter.chat

                def chat(*args, **kwargs):
                    result = original_chat(*args, **kwargs)
                    if (scenario == "disabled_reasoning"
                            and kwargs.get("options", {}).get("disabled")):
                        result.reasoning = "unexpected thought"
                    if scenario == "content_leak" and result.reasoning:
                        result.content = "<think>thought</think>391"
                    # Use the real adapter to parse the alternate wire field.
                    parsed = AdapterTest._openai_reasoning_response(
                        {"reasoning_content": result.reasoning},
                        kwargs["stream"],
                    )
                    result.reasoning = parsed.reasoning
                    result.legacy_reasoning = parsed.legacy_reasoning
                    return result

                with mock.patch.object(adapter, "chat", side_effect=chat):
                    with tempfile.TemporaryDirectory() as directory:
                        code, text = self._run("json", directory, adapter)
                report = json.loads(text)
                if scenario == "valid":
                    self.assertEqual(code, 0)
                    response = report["checks"]["streaming_reasoning"]["response"]
                    self.assertGreater(response["legacy_reasoning_length"], 0)
                    self.assertEqual(response["reasoning_length"],
                                     response["legacy_reasoning_length"])
                else:
                    self.assertEqual(code, 1)
                    check = ("thinking_disabled"
                             if scenario == "disabled_reasoning"
                             else "streaming_reasoning")
                    self.assertEqual(report["checks"][check]["status"], "failed")

    def test_human_verdict_lists_every_check_and_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result, text = self._run("human", directory)
            artifacts = list(pathlib.Path(directory).glob("*.json"))
            self.assertIn(f"json: {artifacts[0]}", text)
        self.assertEqual(result, 0)
        self.assertIn("=== capability-v2 verdict: fake/fixture ===", text)
        self.assertIn("expects reasoning=required tools=required", text)
        for name in (
            "model_identity",
            "streaming_reasoning",
            "thinking_disabled",
            "tool_call",
            "streaming_tool_call",
            "tool_round_trip",
            "structured_json",
            "json_schema",
        ):
            self.assertIn(name, text)
        self.assertIn("identity: verified via /v1/models", text)
        self.assertIn("schema enforcement: enforced", text)
        self.assertIn("result: PASSED", text)
        # The verdict replaces the JSON on stdout, never wraps it.
        self.assertNotIn('"schema_version"', text)

    def test_human_verdict_names_failures_and_skip_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with contextlib.redirect_stdout(output):
                    code = run_capability_check(
                        runtime="fake",
                        tag="fixture",
                        model="fixture",
                        adapter=ToollessAdapter(),
                        config=CapabilityConfig(
                            reasoning="required", tools="required"
                        ),
                        identity={"checked": False},
                    )
            text = output.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("model_identity failed:", text)
        self.assertIn(
            "thinking_disabled skipped: model has no configured disable mode",
            text,
        )
        self.assertIn("tool_round_trip skipped: tool_call failed", text)
        self.assertIn("identity: unchecked", text)
        self.assertIn("result: FAILED (model_identity, tool_call", text)

    def test_automatic_tool_paths_fail_independently(self):
        class BrokenToolParserAdapter(FakeAdapter):
            def __init__(self, broken_stream):
                self.broken_stream = broken_stream

            def chat(self, messages, *, max_tokens, stream, tools=None,
                     options=None):
                initial_tool_call = tools and not any(
                    message.get("role") == "tool" for message in messages
                )
                if initial_tool_call and stream == self.broken_stream:
                    if stream:
                        raise RuntimeError("streaming tool parser failed")
                    return ChatResult(
                        content="<tool_call>raw XML</tool_call>",
                        chunks=1,
                        completed=True,
                        finish_reason="stop",
                    )
                return super().chat(
                    messages,
                    max_tokens=max_tokens,
                    stream=stream,
                    tools=tools,
                    options=options,
                )

        cases = (
            (False, "tool_call", "streaming_tool_call"),
            (True, "streaming_tool_call", "tool_call"),
        )
        for broken_stream, failed, passed in cases:
            with self.subTest(failed=failed):
                with tempfile.TemporaryDirectory() as directory:
                    code, text = self._run(
                        "json",
                        directory,
                        adapter=BrokenToolParserAdapter(broken_stream),
                    )
                report = json.loads(text)
            self.assertEqual(code, 1)
            self.assertEqual(report["checks"][failed]["status"], "failed")
            self.assertEqual(report["checks"][passed]["status"], "passed")
            self.assertEqual(report["failed_checks"], [failed])

    def test_enforcement_modes_are_recorded_and_only_lying_fails(self):
        cases = {
            FakeAdapter: ("passed", "enforced", 0),
            RejectingAdapter: ("passed", "unsupported", 0),
            UnenforcingAdapter: ("failed", "ignored", 1),
            JsonOnlyAdapter: ("failed", "json_only", 1),
            TruncatingAdapter: ("failed", "truncated", 1),
            WhitespaceRunawayAdapter: ("failed", "whitespace_runaway", 1),
        }
        for adapter_class, (status, enforcement, code) in cases.items():
            with self.subTest(adapter=adapter_class.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    output = io.StringIO()
                    with mock.patch.dict(
                        os.environ,
                        {"CAPABILITY_OUTPUT_DIR": directory},
                        clear=False,
                    ):
                        with contextlib.redirect_stdout(output):
                            result = run_capability_check(
                                runtime="fake",
                                tag="fixture",
                                model="fixture",
                                adapter=adapter_class(),
                                config=CapabilityConfig(
                                    reasoning="required",
                                    tools="required",
                                    disable_options={"disabled": True},
                                ),
                                identity={"checked": True, "matched": True},
                                output_format="json",
                            )
                    report = json.loads(output.getvalue())
                check = report["checks"]["json_schema"]
                self.assertEqual(result, code)
                self.assertEqual(check["status"], status)
                # The mode is retained even when the check failed.
                self.assertEqual(check["enforcement"], enforcement)
                if enforcement == "unsupported":
                    self.assertEqual(check["http_status"], 400)
                    self.assertIn("not supported", check["rejection"])
                if enforcement == "ignored":
                    self.assertIn("returned non-JSON", check["error"])
                if enforcement == "json_only":
                    self.assertIn("violating the schema", check["error"])
                if enforcement == "truncated":
                    self.assertIn("token budget", check["error"])
                if enforcement == "whitespace_runaway":
                    self.assertIn("structural whitespace", check["error"])
                    # The tab spew is retained, so the cause is visible.
                    self.assertIn("\t", check["response"]["document"])

    def test_enforcement_is_skipped_when_the_adapter_cannot_express_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with contextlib.redirect_stdout(output):
                    result = run_capability_check(
                        runtime="fake",
                        tag="fixture",
                        model="fixture",
                        adapter=SchemalessAdapter(),
                        config=CapabilityConfig(
                            reasoning="required", tools="required"
                        ),
                        identity={"checked": True, "matched": True},
                        output_format="json",
                    )
            report = json.loads(output.getvalue())
        check = report["checks"]["json_schema"]
        self.assertEqual(result, 0)
        self.assertEqual(check["status"], "skipped")
        self.assertIn("cannot express schema enforcement", check["reason"])

    def test_unaided_document_is_recorded_and_never_gates(self):
        cases = {FakeAdapter: "valid", ProseAdapter: "ignored"}
        for adapter_class, unaided in cases.items():
            with self.subTest(adapter=adapter_class.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    output = io.StringIO()
                    with mock.patch.dict(
                        os.environ,
                        {"CAPABILITY_OUTPUT_DIR": directory},
                        clear=False,
                    ):
                        with contextlib.redirect_stdout(output):
                            code = run_capability_check(
                                runtime="fake",
                                tag="fixture",
                                model="fixture",
                                adapter=adapter_class(),
                                config=CapabilityConfig(
                                    reasoning="required",
                                    tools="required",
                                    disable_options={"disabled": True},
                                ),
                                identity={"checked": True, "matched": True},
                                output_format="json",
                            )
                    report = json.loads(output.getvalue())
                check = report["checks"]["structured_json"]
                self.assertEqual(check["unaided"], unaided)
                # A model that cannot do it unaided still passes the gate:
                # the fact is deployment-conditional, not disqualifying.
                self.assertEqual(check["status"], "passed")
                self.assertEqual(report["status"], "passed")
                self.assertEqual(code, 0)
                self.assertIn(
                    "document", check["unaided_response"]
                )

    def test_unaided_records_a_truncated_or_failed_attempt(self):
        class TruncatedUnaided(FakeAdapter):
            def chat(self, messages, *, max_tokens, stream, tools=None,
                     options=None):
                prompt = str(messages[-1].get("content", ""))
                if "matching this schema" in prompt:
                    return ChatResult(
                        content=conforming_document()[:300],
                        chunks=1,
                        completed=True,
                        finish_reason="length",
                    )
                return super().chat(
                    messages, max_tokens=max_tokens, stream=stream,
                    tools=tools, options=options,
                )

        class BrokenUnaided(FakeAdapter):
            def chat(self, messages, *, max_tokens, stream, tools=None,
                     options=None):
                prompt = str(messages[-1].get("content", ""))
                if "matching this schema" in prompt:
                    raise HTTPRejected(500, "http://x", "upstream exploded")
                return super().chat(
                    messages, max_tokens=max_tokens, stream=stream,
                    tools=tools, options=options,
                )

        for adapter_class, unaided in (
            (TruncatedUnaided, "truncated"),
            (BrokenUnaided, "error"),
        ):
            with self.subTest(adapter=adapter_class.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    output = io.StringIO()
                    with mock.patch.dict(
                        os.environ,
                        {"CAPABILITY_OUTPUT_DIR": directory},
                        clear=False,
                    ):
                        with contextlib.redirect_stdout(output):
                            code = run_capability_check(
                                runtime="fake",
                                tag="fixture",
                                model="fixture",
                                adapter=adapter_class(),
                                config=CapabilityConfig(
                                    reasoning="required", tools="required"
                                ),
                                identity={"checked": True, "matched": True},
                                output_format="json",
                            )
                    report = json.loads(output.getvalue())
                check = report["checks"]["structured_json"]
                self.assertEqual(check["unaided"], unaided)
                self.assertEqual(check["status"], "passed")
                self.assertEqual(code, 0)

    def test_json_schema_fixture_is_adversarial_and_self_validating(self):
        # The fixture must be adversarial: an unconstrained model that
        # obeys the prose cannot satisfy the schema, so a pass is evidence
        # of constrained decoding rather than of good manners.
        self.assertIn("Markdown", capability_check.JSON_SCHEMA_PROMPT)
        self.assertIn("fenced code block", capability_check.JSON_SCHEMA_PROMPT)

        # The validator supports one keyword subset and the fixture stays
        # inside it, so validation cannot silently under-check the schema.
        def keywords(node):
            for key, value in node.items():
                if key == "properties":
                    for child in value.values():
                        yield from keywords(child)
                    continue
                yield key
                if key == "items":
                    yield from keywords(value)

        found = set(keywords(capability_check.JSON_SCHEMA))
        self.assertTrue(
            found <= capability_check.JSON_SCHEMA_KEYWORDS,
            f"unsupported keywords in fixture: "
            f"{found - capability_check.JSON_SCHEMA_KEYWORDS}",
        )

        # A thousand-token document is the point: a three-key object would
        # exercise no nesting, item counts, or enums at depth.
        self.assertGreater(len(conforming_document()), 1000)
        self.assertTrue(
            capability_check._validates_json_schema(
                json.loads(conforming_document())
            )
        )

        broken = [
            ("dropped finding", lambda doc: doc["findings"].pop()),
            ("extra key", lambda doc: doc.update(notes="hi")),
            ("bad enum", lambda doc: doc["summary"].update(verdict="maybe")),
            ("nested enum",
             lambda doc: doc["findings"][0].update(severity="fatal")),
            ("boolean where integer belongs",
             lambda doc: doc["findings"][0]["location"].update(start_line=True)),
            ("short evidence array",
             lambda doc: doc["findings"][0].update(evidence=["only one"])),
            ("extra nested key",
             lambda doc: doc["findings"][0]["location"].update(commit="abc")),
            ("missing nested key",
             lambda doc: doc["findings"][0]["location"].pop("symbol")),
        ]
        for label, mutate in broken:
            with self.subTest(case=label):
                document = json.loads(conforming_document())
                mutate(document)
                self.assertFalse(
                    capability_check._validates_json_schema(document)
                )

    def test_control_modes_are_explicit(self):
        self.assertEqual(
            openai_control("enable-thinking"),
            ({"chat_template_kwargs": {"enable_thinking": True}}, ""),
        )
        self.assertEqual(ollama_control("think-false"), ({"think": False}, ""))
        # reasoning_effort travels as a chat-template kwarg, not a
        # top-level field: servers forward the kwarg to the template
        # generically, while top-level handling is model-specific.
        self.assertEqual(
            openai_control("reasoning-effort-high"),
            ({"chat_template_kwargs": {"reasoning_effort": "high"}}, ""),
        )
        self.assertEqual(
            openai_control("reasoning-effort-none"),
            ({"chat_template_kwargs": {"reasoning_effort": "none"}}, ""),
        )
        with self.assertRaises(ValueError):
            openai_control("guessed")

    def test_resolve_shell_config_reads_trusted_conf(self):
        names = ("CAP_REASONING", "CAP_TOOLS", "CAP_MODE")
        with tempfile.TemporaryDirectory() as directory:
            conf = pathlib.Path(directory) / "model.conf"
            conf.write_text(
                'CAP_REASONING="required"\n'
                'CAP_TOOLS="absent"\n'
                'CAP_MODE="${CAP_MODE_DEFAULT:-default}"\n',
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ, {name: "" for name in names}, clear=False
            ):
                values = resolve_shell_config(conf, names)
        self.assertEqual(values["CAP_REASONING"], "required")
        self.assertEqual(values["CAP_TOOLS"], "absent")
        self.assertEqual(values["CAP_MODE"], "default")

    def test_resolve_shell_config_rejects_broken_conf(self):
        with tempfile.TemporaryDirectory() as directory:
            conf = pathlib.Path(directory) / "broken.conf"
            conf.write_text("exit 3\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed to resolve"):
                resolve_shell_config(conf, ("CAP_REASONING",))


class CliIdentityTest(unittest.TestCase):
    def test_openai_identity_matches_among_many(self):
        with mock.patch.object(
            capability_check,
            "_json_request",
            return_value={"data": [{"id": "other"}, {"id": "m"}]},
        ):
            identity = _openai_identity("http://x", "m", 5)
        self.assertTrue(identity["checked"])
        self.assertTrue(identity["matched"])
        self.assertEqual(identity["live_models"], 2)

    def test_openai_identity_missing_model_fails(self):
        with mock.patch.object(
            capability_check,
            "_json_request",
            return_value={"data": [{"id": "other"}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "not live"):
                _openai_identity("http://x", "m", 5)

    def test_ollama_identity_reads_details(self):
        with mock.patch.object(
            capability_check,
            "_json_request",
            return_value={"details": {"family": "qwen"}},
        ):
            identity = _ollama_identity("http://x", "m", 5)
        self.assertTrue(identity["checked"])
        self.assertTrue(identity["matched"])
        self.assertEqual(identity["family"], "qwen")


class CliMainTest(unittest.TestCase):
    def test_unreachable_endpoint_reports_setup_failure(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with mock.patch.object(
                    capability_check,
                    "_json_request",
                    side_effect=RuntimeError("connection refused"),
                ):
                    with contextlib.redirect_stdout(output):
                        code = main(
                            [
                                "--base-url",
                                "http://127.0.0.1:9",
                                "--model",
                                "m",
                                "--format",
                                "json",
                            ]
                        )
        self.assertEqual(code, 1)
        report = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runtime"], "endpoint-openai")
        self.assertEqual(report["failed_checks"], ["setup"])

    def test_setup_failure_prints_the_reason_by_default(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with mock.patch.object(
                    capability_check,
                    "_json_request",
                    side_effect=RuntimeError("connection refused"),
                ):
                    with contextlib.redirect_stdout(output):
                        code = main(
                            [
                                "--base-url",
                                "http://127.0.0.1:9",
                                "--model",
                                "m",
                            ]
                        )
        text = output.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("setup failed: connection refused", text)
        self.assertIn("result: FAILED (setup)", text)
        self.assertNotIn('"schema_version"', text)


    def test_show_reprints_a_retained_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with mock.patch.object(
                    capability_check,
                    "_json_request",
                    side_effect=RuntimeError("connection refused"),
                ):
                    with contextlib.redirect_stdout(output):
                        main(["--base-url", "http://127.0.0.1:9",
                              "--model", "m"])
            retained = list(pathlib.Path(directory).glob("*.json"))[0]
            before = retained.read_bytes()
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                code = main(["--show", str(retained)])
            # Re-printing retains nothing new and rewrites nothing.
            self.assertEqual(
                list(pathlib.Path(directory).glob("*.json")), [retained]
            )
            self.assertEqual(retained.read_bytes(), before)
            text = shown.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("setup failed: connection refused", text)
            self.assertIn("result: FAILED (setup)", text)
            self.assertIn(f"json: {retained}", text)

    def test_show_rejects_a_foreign_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bench.json"
            path.write_text('{"protocol": "audit-v1"}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a capability-v2"):
                main(["--show", str(path)])

    def test_running_still_requires_an_endpoint_and_model(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):
                main(["--model", "m"])
        self.assertIn("--base-url is required", errors.getvalue())


    def test_empty_output_dir_override_means_default(self):
        with mock.patch.dict(
            os.environ, {"CAPABILITY_OUTPUT_DIR": ""}, clear=False
        ):
            self.assertEqual(_output_dir().name, "capability-check")

    def test_show_takes_a_listed_file_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": "capability-v1",
                        "status": "passed",
                        "runtime": "vllm-metal",
                        "tag": "host",
                        "model": "m",
                        "checks": {"model_identity": {"status": "passed"}},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--show", "report.json"])
                self.assertEqual(code, 0)
                self.assertIn("result: PASSED", output.getvalue())
                self.assertIn(f"json: {path}", output.getvalue())
                with self.assertRaisesRegex(
                    SystemExit, f"no such file here or in {directory}"
                ):
                    main(["--show", "absent.json"])

    def test_list_indexes_retained_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with mock.patch.object(
                    capability_check,
                    "_json_request",
                    side_effect=RuntimeError("connection refused"),
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(["--base-url", "http://127.0.0.1:9",
                              "--model", "m"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--list"])
            lines = output.getvalue().splitlines()
            self.assertEqual(code, 0)
            self.assertEqual(lines[0], f"1 retained reports in {directory}")
            self.assertIn("result", lines[1])
            self.assertIn("failed", lines[2])
            self.assertIn("endpoint-openai/m", lines[2])
            self.assertEqual(
                lines[-1], "re-print one with: capability-check --show FILE"
            )

    def test_list_says_so_when_nothing_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = pathlib.Path(directory) / "absent"
            with mock.patch.dict(
                os.environ,
                {"CAPABILITY_OUTPUT_DIR": str(empty)},
                clear=False,
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--list"])
            self.assertFalse(empty.exists())
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().strip(), f"no retained reports in {empty}"
        )


FITNESS_CONFIG = CapabilityConfig(
    reasoning="required", tools="required", sampling={"temperature": 0.6}
)


def _answer(content, reasoning="weighed it", finish="stop"):
    return ChatResult(
        content=content, reasoning=reasoning, chunks=2, completed=True,
        finish_reason=finish, completion_tokens=100,
    )


def _call(name, **arguments):
    return ToolCall(call_id=f"call-{name}", name=name, arguments=arguments)


class FitnessFake(FakeAdapter):
    """A healthy audit model; subclasses break one behavior at a time."""

    def __init__(self):
        self.sampling = []

    def selection(self):
        return json.dumps(sorted(capability_check.SELECTION_REQUIRED))

    def triage(self):
        _, truth = capability_check.triage_prompt()
        return json.dumps([
            {"id": ident,
             "verdict": "true_positive" if real else "false_positive"}
            for ident, real in truth.items()
        ])

    def findings(self):
        planted = capability_check._workflow_planted()
        return json.dumps([
            {"file": places[0][0], "line": places[0][1], "title": "issue",
             "severity": "high"}
            for places in planted.values()
        ])

    def workflow_step(self, turn):
        model = "## Entry points\n\n## Trust boundaries\n\n## Findings\n"
        steps = [
            [_call("list_files", directory=".")],
            [_call("read_file", path="app/db.py"),
             _call("read_file", path="app/routes.py")],
            [_call("write_file", path=capability_check.WORKFLOW_MODEL,
                   content=model),
             _call("write_file", path=capability_check.WORKFLOW_FINDINGS,
                   content=self.findings())],
        ]
        return steps[turn] if turn < len(steps) else None

    def chat(self, messages, *, max_tokens, stream, tools=None, options=None,
             sampling=None):
        self.sampling.append(sampling)
        prompt = str(messages[0].get("content", ""))
        if "Select Security-Relevant" in prompt:
            return _answer(self.selection())
        if "Triage Scanner" in prompt:
            return _answer(self.triage())
        if "Audit billing-api" not in prompt:
            return super().chat(
                messages, max_tokens=max_tokens, stream=stream, tools=tools,
                options=options,
            )
        turn = sum(message["role"] == "assistant" for message in messages)
        calls = self.workflow_step(turn)
        if calls is None:
            return _answer("DONE", reasoning="")
        return ChatResult(
            reasoning="plan" if turn == 0 else "", tool_calls=calls,
            chunks=2, completed=True, finish_reason="tool_calls",
            completion_tokens=50,
        )


class AuditFitnessTest(unittest.TestCase):
    def _fitness(self, adapter, config=FITNESS_CONFIG, samples=2):
        return capability_check.run_audit_fitness(adapter, config, samples)

    def _failure(self, adapter, task):
        try:
            evidence = self._fitness(adapter, samples=1)
        except capability_check.CheckFailure as exc:
            evidence = exc.evidence
        return evidence["tasks"][task]["samples"][0]

    def test_healthy_model_passes_every_task_at_declared_sampling(self):
        adapter = FitnessFake()
        evidence = self._fitness(adapter)
        self.assertEqual(
            set(evidence["tasks"]), {"selection", "triage", "workflow"}
        )
        self.assertEqual(evidence["failed_samples"], 0)
        self.assertTrue(all(s == {"temperature": 0.6} for s in adapter.sampling))
        workflow = evidence["tasks"]["workflow"]["samples"][0]
        self.assertEqual(workflow["files_read"], ["app/db.py", "app/routes.py"])

    def test_tools_absent_model_skips_the_workflow(self):
        config = CapabilityConfig(
            reasoning="required", tools="absent", sampling={"top_k": 20}
        )
        evidence = self._fitness(FitnessFake(), config)
        self.assertEqual(set(evidence["tasks"]), {"selection", "triage"})

    def test_concurrency_never_exceeds_admission(self):
        class Counting(FitnessFake):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()
                self.active = self.peak = 0

            def chat(self, messages, **kwargs):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                try:
                    return super().chat(messages, **kwargs)
                finally:
                    with self.lock:
                        self.active -= 1

        adapter = Counting()
        config = dataclasses.replace(FITNESS_CONFIG, concurrency=2)
        evidence = self._fitness(adapter, config, samples=4)
        self.assertEqual(evidence["concurrency"], 2)
        self.assertEqual(adapter.peak, 2)

    def test_sampling_vocabulary_matches_audit_bench(self):
        import audit_bench

        self.assertEqual(
            capability_check.SAMPLING_KEYS, audit_bench.SAMPLING_KEYS
        )

    def test_declared_sampling_parses_strictly(self):
        self.assertEqual(
            capability_check.parse_sampling("temperature=0.6 top_k=20"),
            {"temperature": 0.6, "top_k": 20},
        )
        for bad in ("temp=1", "top_k=x", "top_k=1 top_k=2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                capability_check.parse_sampling(bad)

    def test_undeclared_sampling_fails(self):
        config = CapabilityConfig(reasoning="required", tools="required")
        with self.assertRaisesRegex(RuntimeError, "no declared sampling"):
            self._fitness(FitnessFake(), config)

    def test_one_random_failure_is_tolerated_two_are_not(self):
        class Flaky(FitnessFake):
            def __init__(self, failures):
                super().__init__()
                self.failures = failures

            def selection(self):
                self.failures -= 1
                return "not json" if self.failures >= 0 else super().selection()

        self.assertEqual(self._fitness(Flaky(1))["failed_samples"], 1)
        with self.assertRaises(capability_check.CheckFailure) as caught:
            self._fitness(Flaky(2))
        self.assertIn("format 2", str(caught.exception))
        self.assertEqual(caught.exception.evidence["failed_samples"], 2)

    def test_runaway_is_named_with_loop_evidence(self):
        class Runaway(FitnessFake):
            def chat(self, messages, **kwargs):
                if "Select Security" in str(messages[0]["content"]):
                    loop = "\n".join(["- docs/service/admin/raid.rst"] * 40)
                    return _answer("", reasoning=loop, finish="length")
                return super().chat(messages, **kwargs)

        sample = self._failure(Runaway(), "selection")
        self.assertEqual(sample["failure"], "runaway")
        self.assertEqual(sample["max_line_repeats"], 40)
        self.assertIn("raid.rst", sample["reasoning_tail"])

    def test_answer_without_thinking_fails(self):
        class Unthinking(FitnessFake):
            def chat(self, messages, **kwargs):
                result = super().chat(messages, **kwargs)
                result.reasoning = ""
                return result

        for task in ("selection", "triage", "workflow"):
            self.assertEqual(
                self._failure(Unthinking(), task)["failure"], "no_reasoning"
            )
        # The unthinking answer is still judged, as evidence.
        self.assertEqual(self._failure(Unthinking(), "triage")["answer"], "valid")

    def test_selection_contract(self):
        required = sorted(capability_check.SELECTION_REQUIRED)
        _, _, excluded = capability_check.selection_prompt()
        cases = {
            "loop": required + required[:1],
            "invented": required + ["docs/service/admin/made-up.rst"],
            "ignored_exclusion": required + sorted(excluded)[:1],
            "missed_required": required[2:],
        }
        for failure, answer in cases.items():
            with self.subTest(failure=failure):
                self.assertEqual(
                    capability_check._selection_failure(json.dumps(answer))[0],
                    failure,
                )
        fenced = "```json\n" + json.dumps(required[1:]) + "\n```"
        self.assertIsNone(capability_check._selection_failure(fenced))
        # A separator slip still names one document; it is not invented.
        slipped = [path.replace("-", "_") for path in required]
        self.assertIsNone(
            capability_check._selection_failure(json.dumps(slipped))
        )

    def test_selection_prompt_is_deterministic_and_large(self):
        first, valid, excluded = capability_check.selection_prompt()
        lines, _ = capability_check._selection_inventory()
        dates = {line.split(" ", 2)[1]: line.split("last changed ")[1][:10]
                 for line in lines}
        newest_other = max(
            day for path, day in dates.items()
            if path not in capability_check.SELECTION_REQUIRED
        )
        self.assertTrue(all(
            dates[path] > newest_other
            for path in capability_check.SELECTION_REQUIRED
        ))
        self.assertEqual(first, capability_check.selection_prompt()[0])
        self.assertGreater(len(valid), 400)
        self.assertTrue(set(capability_check.SELECTION_REQUIRED) <= valid)
        self.assertTrue(excluded <= valid)

    def test_triage_names_a_followed_injection(self):
        _, truth = capability_check.triage_prompt()
        self.assertEqual(sum(truth.values()), 6)
        self.assertTrue(truth[capability_check.TRIAGE_INJECTED])
        flipped = {capability_check.TRIAGE_INJECTED, "C01"}
        answer = json.dumps([
            {"id": ident, "verdict": (
                "true_positive" if real != (ident in flipped)
                else "false_positive")}
            for ident, real in truth.items()
        ])
        failure = capability_check._triage_failure(answer)
        self.assertEqual(failure[0], "wrong_verdicts")
        self.assertIn("planted comment", failure[1])
        broken = json.dumps([
            {"id": ident, "verdict": ["true_positive"]} for ident in truth
        ])
        self.assertEqual(
            capability_check._judged(
                capability_check._triage_failure, broken
            )[0],
            "format",
        )
        unordered = json.dumps([{"id": "C02", "verdict": "false_positive"}])
        self.assertEqual(
            capability_check._triage_failure(unordered)[0], "format"
        )

    def test_workflow_contract(self):
        class NoWrite(FitnessFake):
            def workflow_step(self, turn):
                return super().workflow_step(turn) if turn < 2 else None

        class Ungrounded(FitnessFake):
            def findings(self):
                return json.dumps([{"file": "app/auth.py", "line": 3,
                                    "title": "x", "severity": "low"}])

        class Unknown(FitnessFake):
            def workflow_step(self, turn):
                return [_call("run_shell", command="ls")]

        class Endless(FitnessFake):
            def workflow_step(self, turn):
                return [_call("list_files", directory="app")]

        cases = {
            NoWrite: "instruction",
            Ungrounded: "invented",
            Unknown: "tool_misuse",
            Endless: "incomplete",
        }
        for fake, failure in cases.items():
            with self.subTest(fake=fake.__name__):
                self.assertEqual(
                    self._failure(fake(), "workflow")["failure"], failure
                )
        read = set(capability_check.WORKFLOW_REPO)
        model = "## Entry points\n## Trust boundaries\n## Findings"
        missed = capability_check._workflow_output_failure(
            {capability_check.WORKFLOW_MODEL: model,
             capability_check.WORKFLOW_FINDINGS: "[]"}, read,
        )
        self.assertEqual(missed[0], "missed_findings")
        # One finding between the two db.py windows never counts twice.
        edge = json.dumps([{"file": "app/db.py", "line": 28, "title": "x",
                            "severity": "high"}])
        self.assertEqual(
            capability_check._workflow_output_failure(
                {capability_check.WORKFLOW_MODEL: model,
                 capability_check.WORKFLOW_FINDINGS: edge}, read,
            )[0],
            "missed_findings",
        )
        # The export exposure found one layer down still counts, but the
        # SQL injection line next to it does not stand in for it.
        injection = {"file": "app/db.py", "line": 23, "title": "x",
                     "severity": "high"}
        lower = {**injection, "line": 29}
        for findings, failure in (([injection], "missed_findings"),
                                  ([injection, lower], None)):
            result = capability_check._workflow_output_failure(
                {capability_check.WORKFLOW_MODEL: model,
                 capability_check.WORKFLOW_FINDINGS: json.dumps(findings)},
                read,
            )
            self.assertEqual(result and result[0], failure)
        disordered = capability_check._workflow_output_failure(
            {capability_check.WORKFLOW_MODEL: "## Findings\n## Entry points",
             capability_check.WORKFLOW_FINDINGS: "[]"}, read,
        )
        self.assertEqual(disordered[0], "instruction")

    def test_sampling_reaches_the_wire(self):
        response = mock.MagicMock()
        response.__enter__.return_value = [
            b'data: {"choices":[{"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n',
            b'data: {"choices":[],"usage":{"completion_tokens":7}}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(
            capability_check.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            result = capability_check.OpenAIChatAdapter("http://x", "m", 5).chat(
                [{"role": "user", "content": "x"}], max_tokens=10, stream=True,
                sampling={"temperature": 0.6, "top_k": 20},
            )
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual((payload["temperature"], payload["top_k"]), (0.6, 20))
        self.assertEqual(result.completion_tokens, 7)
        with mock.patch.object(
            capability_check, "_json_request",
            return_value={"message": {"content": "ok"}, "done": True,
                          "eval_count": 3},
        ) as request:
            capability_check.OllamaChatAdapter("http://x", "m", 5).chat(
                [{"role": "user", "content": "x"}], max_tokens=10,
                stream=False, sampling={"repetition_penalty": 1.05},
            )
        self.assertEqual(
            request.call_args.args[1]["options"]["repeat_penalty"], 1.05
        )

    def test_report_prints_each_fitness_task(self):
        class Unthinking(FitnessFake):
            def chat(self, messages, **kwargs):
                result = super().chat(messages, **kwargs)
                result.reasoning = ""
                return result

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"CAPABILITY_OUTPUT_DIR": directory}
        ), contextlib.redirect_stdout(output):
            code = run_capability_check(
                runtime="fake", tag="fixture", model="fixture",
                adapter=Unthinking(),
                config=CapabilityConfig(
                    reasoning="absent", tools="required",
                    sampling={"temperature": 0.6},
                ),
                identity={"checked": True, "matched": True},
                fitness_samples=1,
            )
        self.assertEqual(code, 0)
        self.assertIn("fitness workflow: 0/1 failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()

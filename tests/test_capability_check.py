import contextlib
import io
import json
import os
import pathlib
import tempfile
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
        force_tool=None,
    ):
        del max_tokens, stream
        prompt = str(messages[-1].get("content", ""))
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
        if force_tool:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        call_id="call-1",
                        name="lookup_issue",
                        arguments={"issue_number": 42},
                    )
                ],
                chunks=1,
                completed=True,
                finish_reason="tool_calls",
            )
        if any(message.get("role") == "tool" for message in messages):
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
             options=None, force_tool=None):
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
            options=options, force_tool=force_tool,
        )


class RejectingAdapter(FakeAdapter):
    """A server without constrained decoding, refusing the parameter."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
        if options and "response_format" in options:
            raise HTTPRejected(
                400, "http://x/v1/chat/completions",
                '{"error":"response_format is not supported"}',
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options, force_tool=force_tool,
        )


class JsonOnlyAdapter(FakeAdapter):
    """A server that enforces JSON-ness but not the schema."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
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
            options=options, force_tool=force_tool,
        )


class TruncatingAdapter(FakeAdapter):
    """A constrained response cut off at the token budget."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
        if options and "response_format" in options:
            return ChatResult(
                content=conforming_document()[:200],
                chunks=1,
                completed=True,
                finish_reason="length",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options, force_tool=force_tool,
        )


class WhitespaceRunawayAdapter(FakeAdapter):
    """A grammar-valid whitespace loop: legal tokens, no document."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
        if options and "response_format" in options:
            return ChatResult(
                content='{\n  "repository":' + "\t" * 4000,
                chunks=1,
                completed=True,
                finish_reason="length",
            )
        return super().chat(
            messages, max_tokens=max_tokens, stream=stream, tools=tools,
            options=options, force_tool=force_tool,
        )


class ProseAdapter(FakeAdapter):
    """A model that answers the unaided ask with a report, not a document."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
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
            options=options, force_tool=force_tool,
        )


class SchemalessAdapter(FakeAdapter):
    """An adapter that cannot express enforcement at all."""

    schema_options = None


class ToollessAdapter(FakeAdapter):
    """A model that ignores a forced tool call, failing that check."""

    def chat(self, messages, *, max_tokens, stream, tools=None,
             options=None, force_tool=None):
        if force_tool:
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


class CapabilityCheckTest(unittest.TestCase):
    @staticmethod
    def _run(output_format: str, directory: str, **overrides) -> tuple[int, str]:
        output = io.StringIO()
        with mock.patch.dict(
            os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
        ):
            with contextlib.redirect_stdout(output):
                code = run_capability_check(
                    runtime="fake",
                    tag="fixture",
                    model="fixture",
                    adapter=FakeAdapter(),
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
        self.assertEqual(report["protocol"], "capability-v1")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(artifact_report["status"], "passed")

    def test_human_verdict_lists_every_check_and_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result, text = self._run("human", directory)
            artifacts = list(pathlib.Path(directory).glob("*.json"))
            self.assertIn(f"json: {artifacts[0]}", text)
        self.assertEqual(result, 0)
        self.assertIn("=== capability-v1 verdict: fake/fixture ===", text)
        self.assertIn("expects reasoning=required tools=required", text)
        for name in (
            "model_identity",
            "streaming_reasoning",
            "thinking_disabled",
            "tool_call",
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
                     options=None, force_tool=None):
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
                    tools=tools, options=options, force_tool=force_tool,
                )

        class BrokenUnaided(FakeAdapter):
            def chat(self, messages, *, max_tokens, stream, tools=None,
                     options=None, force_tool=None):
                prompt = str(messages[-1].get("content", ""))
                if "matching this schema" in prompt:
                    raise HTTPRejected(500, "http://x", "upstream exploded")
                return super().chat(
                    messages, max_tokens=max_tokens, stream=stream,
                    tools=tools, options=options, force_tool=force_tool,
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
            with self.assertRaisesRegex(SystemExit, "not a capability-v1"):
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


if __name__ == "__main__":
    unittest.main()

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
    ToolCall,
    _ollama_identity,
    _openai_identity,
    main,
    openai_control,
    ollama_control,
    resolve_shell_config,
    run_capability_check,
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
        ):
            self.assertIn(name, text)
        self.assertIn("identity: verified via /v1/models", text)
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

    def test_control_modes_are_explicit(self):
        self.assertEqual(
            openai_control("enable-thinking"),
            ({"chat_template_kwargs": {"enable_thinking": True}}, ""),
        )
        self.assertEqual(ollama_control("think-false"), ({"think": False}, ""))
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

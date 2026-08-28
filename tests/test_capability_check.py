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


class CapabilityCheckTest(unittest.TestCase):
    def test_complete_protocol_passes(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CAPABILITY_OUTPUT_DIR": directory}, clear=False
            ):
                with contextlib.redirect_stdout(output):
                    result = run_capability_check(
                        runtime="fake",
                        tag="fixture",
                        model="fixture",
                        adapter=FakeAdapter(),
                        config=CapabilityConfig(
                            reasoning="required",
                            tools="required",
                            disable_options={"disabled": True},
                        ),
                        identity={"checked": True, "matched": True},
                    )
            artifacts = list(pathlib.Path(directory).glob("*.json"))
            artifact_report = json.loads(
                artifacts[0].read_text(encoding="utf-8")
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(artifacts), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["protocol"], "capability-v1")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(artifact_report["status"], "passed")

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
                            ]
                        )
        self.assertEqual(code, 1)
        report = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runtime"], "endpoint-openai")
        self.assertEqual(report["failed_checks"], ["setup"])


if __name__ == "__main__":
    unittest.main()

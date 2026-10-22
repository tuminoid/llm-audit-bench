import contextlib
import dataclasses
import http.client
import io
import json
import re
import os
import pathlib
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from audit_bench import (
    SESSION_NOTE,
    SESSION_REPORT_REQUEST,
    SESSION_REPORT_TOKENS,
    SESSION_TURN_TOKENS,
    Calibration,
    OpenAIChatAdapter,
    RequestResult,
    Settings,
    _conversation_fixture,
    _file_provenance,
    _measure_conversations,
    _measure_capacity,
    _measure_sessions,
    _operating_point,
    _output_dir,
    _publication_status,
    audit_prompt,
    build_plan,
    conversation_steps,
    main,
    openai_chat_fn,
    parse_sampling,
    parse_streams,
    run_benchmark,
    vllm_capacity,
)

SAMPLING = {"AUDIT_BENCH_SAMPLING": "temperature=0.6 top_p=0.95 top_k=20"}
CALIBRATION = Calibration(
    low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
)


def settings() -> Settings:
    return Settings(
        streams=(1, 2, 4, 8),
        profile="session",
        samples=3,
        tolerance=0.02,
        dry_run=True,
        sampling={"temperature": 0.6},
    )


class _Stream:
    """A server-sent event body, iterated line by line like HTTPResponse."""

    def __init__(self, events: list):
        self.lines = [
            (event if isinstance(event, str)
             else f"data: {json.dumps(event)}").encode() + b"\n"
            for event in events
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def _fake_session_chat(requests: list, completion: int = 40):
    """Stateless thread-safe streamed fake deriving usage from the messages.

    Hostile on purpose: its turns carry a bogus tool call and no content,
    the worst case an interactive conversation would fail on. The session
    must measure it anyway.
    """

    def chat(messages, tools, max_tokens, sampling=None, stream=False):
        del tools
        requests.append({
            "messages": [dict(item) for item in messages],
            "max_tokens": max_tokens,
            "sampling": sampling,
            "stream": stream,
        })
        time.sleep(0.01)
        prompt = 500 * len(messages)
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            # The opening call is cold; every later one reuses all but
            # the newest 400 tokens.
            "prompt_tokens_details": {
                "cached_tokens": prompt - 400 if len(messages) > 3 else 0
            },
        }
        report = messages[-1].get("content") == SESSION_REPORT_REQUEST
        # The report names every finding the chain carried.
        findings = re.findall(r"FINDING: (\S+)", " ".join(
            item.get("content") or "" for item in messages
            if item.get("role") == "tool"
        ))
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "the report: " + ", ".join(findings)
                    if report else ""
                },
            }],
            "usage": usage,
            "timing": {"first_token_seconds": 0.5, "total_seconds": 1.5},
            "reasoning_chars": 12,
            "reasoning_text": "step thought",
            "tool_calls": 0 if report else 1,
        }

    return chat


class AuditBenchTest(unittest.TestCase):
    def test_streams_are_strict_and_start_at_one(self):
        self.assertEqual(parse_streams("1,2,4,8"), (1, 2, 4, 8))
        for value in ("", "2,4", "1,4,2", "1,2,2", "1,zero"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_streams(value)

    def test_128k_plan_fits_capacity_and_sessions(self):
        plan = build_plan(131072, settings(), 2, 285696)
        self.assertEqual(plan["capacity_near"]["target_prompt_tokens"], 122880)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 2])
        session = plan["session"]
        # 32,768 + 9 * (8,192 + 256) + 16,384 per stream, no shared-prefix
        # credit: two fit the 285k pool.
        self.assertEqual(session["steps"], 9)
        self.assertEqual(session["per_stream_kv_demand"], 125184)
        self.assertEqual(session["streams"], [1, 2])
        self.assertEqual(session["turn_output_tokens"], SESSION_TURN_TOKENS)
        self.assertEqual(session["report_output_tokens"], SESSION_REPORT_TOKENS)
        self.assertFalse(plan["preferred_256k_context"])
        self.assertNotIn("decode", plan)
        self.assertNotIn("audit_phases", plan)

    def test_plan_rejects_context_below_128k(self):
        with self.assertRaises(ValueError):
            build_plan(65536, settings(), 1, 65536)

    def test_plan_stops_at_active_slot_limit(self):
        plan = build_plan(262144, settings(), 2, 2748350)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 2])
        self.assertEqual(plan["session"]["streams"], [1, 2])
        plan = build_plan(262144, settings(), 1, 2748350)
        self.assertEqual(plan["session"]["streams"], [1])

    def test_256k_plan_runs_one_and_the_serving_concurrency(self):
        plan = build_plan(262144, settings(), 8, 2748350)
        self.assertEqual(plan["capacity_near"]["target_prompt_tokens"], 245760)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 8])
        self.assertEqual(plan["session"]["steps"], 16)
        self.assertEqual(plan["session"]["per_stream_kv_demand"], 184320)
        self.assertEqual(plan["session"]["streams"], [1, 8])
        self.assertTrue(plan["preferred_256k_context"])

    def test_720k_plan_takes_the_largest_candidate_the_pool_holds(self):
        # 746,553 tokens hold four 184,320-token sessions: candidates
        # 1/2/4 fit, so the plan runs 1x and 4x.
        plan = build_plan(737280, settings(), 8, 746553)
        self.assertEqual(plan["capacity_near"]["streams"], [1])
        self.assertEqual(plan["session"]["streams"], [1, 4])

    def test_capacity_proof_runs_once_per_stream_count(self):
        adaptive = dataclasses.replace(
            settings(), min_samples=2, sample_agreement=0.05
        )
        def near_full(prompt: str, max_tokens: int, seed: int) -> RequestResult:
            del prompt, seed
            return RequestResult(
                prompt_tokens=122880, output_tokens=max_tokens, seconds=0.01
            )

        with contextlib.redirect_stdout(io.StringIO()):
            capacity = _measure_capacity(
                request=near_full,
                streams=(1, 2),
                settings=adaptive,
                context=131072,
                prompt_lines=4,
                target_prompt=None,
            )
        self.assertEqual(set(capacity), {"by_streams"})
        for count in ("1", "2"):
            self.assertEqual(len(capacity["by_streams"][count]["samples"]), 1)

    def test_sampling_env_is_validated_and_gates_canonicality(self):
        with mock.patch.dict(os.environ, SAMPLING, clear=False):
            self.assertTrue(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ, {**SAMPLING, "AUDIT_BENCH_MIN_SAMPLES": "3"},
            clear=False,
        ):
            parsed = Settings.from_env()
            self.assertEqual(parsed.min_samples, 3)
            self.assertFalse(parsed.canonical)
        with mock.patch.dict(
            os.environ,
            {"AUDIT_BENCH_SAMPLES": "2", "AUDIT_BENCH_MIN_SAMPLES": "3"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_SAMPLE_AGREEMENT": "0.9"}, clear=False
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()
        with mock.patch.dict(
            os.environ, {**SAMPLING, "AUDIT_BENCH_SAMPLES": "1"}, clear=False
        ):
            parsed = Settings.from_env()
            self.assertEqual(parsed.min_samples, 1)
            self.assertFalse(parsed.canonical)

    def test_declared_sampling_parses_and_is_required_for_canonical(self):
        self.assertEqual(
            parse_sampling("temperature=0.6 top_p=0.95 top_k=20"),
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        )
        self.assertEqual(parse_sampling(""), {})
        for value in ("temp=0.6", "temperature", "top_k=1.5",
                      "temperature=0.6 temperature=0.7"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_sampling(value)
        # An undeclared or temperature-less sampler is a server default,
        # not the checkpoint's: noncanonical.
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_SAMPLING": ""}, clear=False
        ):
            self.assertFalse(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_SAMPLING": "top_p=0.9"}, clear=False
        ):
            self.assertFalse(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_SAMPLING": "temperature=0"}, clear=False
        ):
            self.assertTrue(Settings.from_env().canonical)

    def test_retired_profiles_are_rejected(self):
        for profile in ("full", "decode", "replay", "replay-concurrent"):
            with self.subTest(profile=profile):
                with mock.patch.dict(
                    os.environ, {"AUDIT_BENCH_PROFILE": profile}, clear=False
                ):
                    with self.assertRaises(ValueError):
                        Settings.from_env()

    @staticmethod
    def _fake_conversation_chat(
        tool_steps: int,
        break_at: int | None = None,
        batch: int = 1,
    ):
        state = {"n": 0, "requested": 0}

        def chat(messages, tools, max_tokens, sampling=None, stream=False):
            del messages, tools, max_tokens, sampling, stream
            n = state["n"]
            state["n"] += 1
            time.sleep(0.02)
            usage: dict = {
                "prompt_tokens": 1000 * (n + 1), "completion_tokens": 50,
            }
            if n > 0:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": 1000 * n - 100
                }
            bail = break_at is not None and n >= break_at
            if state["requested"] < tool_steps and not bail:
                calls = []
                for _ in range(min(batch, tool_steps - state["requested"])):
                    state["requested"] += 1
                    calls.append({
                        "id": f"call_{state['requested']}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {"path": f"file_{state['requested']:03d}"}
                            ),
                        },
                    })
                message = {"content": "noted", "tool_calls": calls}
            else:
                message = {"content": "" if bail else "final verdict",
                           "tool_calls": []}
            return {"choices": [{"message": message}], "usage": usage}

        return chat

    def _conversation_settings(self, **overrides):
        base = dataclasses.replace(
            settings(),
            profile="conversation",
            min_samples=2,
            sample_agreement=0.5,
            conv_max_steps=3,
        )
        return dataclasses.replace(base, **overrides)

    def test_conversation_steps_fill_seven_eighths_of_context(self):
        conv = self._conversation_settings(conv_max_steps=16)
        # (131072 * 7/8 - 32768) // (8192 + 256) = 9 files; the verdict
        # cap of 16,384 still fits above the ~109k final prompt.
        self.assertEqual(conversation_steps(131072, conv), 9)
        self.assertEqual(
            conversation_steps(131072, self._conversation_settings()), 3
        )

    def test_conversation_measures_growth_reuse_and_verdict(self):
        conv = self._conversation_settings()
        sent: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            # Each sample is a fresh conversation against a fresh fake.
            active: list = [self._fake_conversation_chat(3)]

            def per_sample_chat(messages, tools, max_tokens, sampling=None,
                                stream=False):
                sent.append((sampling, stream))
                if len(messages) == 2:
                    active[0] = self._fake_conversation_chat(3)
                return active[0](messages, tools, max_tokens)

            measured = _measure_conversations(
                per_sample_chat, CALIBRATION, 131072, conv
            )
        self.assertTrue(measured["compliant"])
        self.assertEqual(len(measured["samples"]), 2)
        first = measured["samples"][0]
        self.assertEqual(first["turns"], 4)
        self.assertEqual(first["files_read"], 3)
        self.assertGreater(first["cached_fraction"], 0)
        self.assertIsNotNone(measured["cached_fraction_median"])
        self.assertGreater(
            measured["conversations_per_hour"]["median"], 0
        )
        # The model drives the loop at the declared sampling, unstreamed.
        self.assertTrue(sent)
        for item in sent:
            self.assertEqual(item, ({"temperature": 0.6}, False))

    def test_conversation_accepts_batched_tool_calls(self):
        conv = self._conversation_settings()
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_conversations(
                self._fake_conversation_chat(3, batch=3),
                CALIBRATION, 131072, conv,
            )
        first = measured["samples"][0]
        self.assertTrue(first["compliant"])
        self.assertEqual(first["turns"], 2)
        self.assertEqual(first["files_read"], 3)

    def test_conversation_records_noncompliance_without_padding(self):
        conv = self._conversation_settings()
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_conversations(
                self._fake_conversation_chat(3, break_at=1),
                CALIBRATION, 131072, conv,
            )
        self.assertFalse(measured["compliant"])
        self.assertEqual(len(measured["samples"]), 2)

    def _session_settings(self, **overrides):
        base = dataclasses.replace(
            settings(), min_samples=2, sample_agreement=0.5, conv_max_steps=3
        )
        return dataclasses.replace(base, **overrides)

    def test_session_scripts_every_turn_and_measures_the_server(self):
        requests: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                _fake_session_chat(requests), CALIBRATION, 131072,
                self._session_settings(), (1,),
            )
        one = measured["by_streams"]["1"]
        self.assertEqual(len(one["samples"]), 2)
        session = one["samples"][0]["per_stream"][0]
        self.assertEqual(session["steps"], 3)
        self.assertEqual(len(session["per_step"]), 3)
        # Four calls of 40 tokens; decode is (40 - 1) tokens over the
        # second after the first token, prefill the uncached prompt over
        # time to first token.
        self.assertEqual(session["output_tokens"], 160)
        self.assertAlmostEqual(
            session["report"]["decode_tokens_per_second"], 39.0
        )
        self.assertAlmostEqual(
            session["turn_decode_tokens_per_second"]["first"], 39.0
        )
        self.assertAlmostEqual(
            session["prefill_tokens_per_second"], (1000 + 3 * 400) / 2.0
        )
        self.assertTrue(session["complete"])
        self.assertGreater(one["sessions_per_hour"]["median"], 0)
        self.assertEqual(measured["scaling_vs_1x"]["1"], 1.0)
        # First sample: three natural-length turns, then the report, every
        # call streamed at the declared sampling with no control message.
        first = requests[:4]
        self.assertEqual(
            [item["max_tokens"] for item in first],
            [SESSION_TURN_TOKENS] * 3 + [SESSION_REPORT_TOKENS],
        )
        for item in first:
            self.assertTrue(item["stream"])
            self.assertEqual(item["sampling"], {"temperature": 0.6})
            for message in item["messages"]:
                self.assertNotIn(
                    "BENCHMARK CONTROL", message.get("content") or ""
                )
        report_messages = first[3]["messages"]
        self.assertEqual(report_messages[-1]["content"], SESSION_REPORT_REQUEST)
        # The scripted assistant turn replaced the model's own output.
        scripted = report_messages[2]
        self.assertEqual(scripted["role"], "assistant")
        self.assertEqual(
            json.loads(
                scripted["tool_calls"][0]["function"]["arguments"]
            )["path"],
            "file_001",
        )
        self.assertIn("file_001", report_messages[3]["content"])
        self.assertNotIn("reasoning", scripted)

    def test_session_resends_reasoning_only_when_preserved(self):
        requests: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            _measure_sessions(
                _fake_session_chat(requests), CALIBRATION, 131072,
                self._session_settings(preserve_reasoning=True), (1,),
            )
        report_messages = requests[3]["messages"]
        scripted = [m for m in report_messages if m["role"] == "assistant"]
        self.assertEqual(len(scripted), 3)
        for message in scripted:
            self.assertEqual(message["reasoning"], "step thought")
            self.assertEqual(message["content"], SESSION_NOTE)

    def test_preserved_reasoning_reserves_a_turn_cap_per_step(self):
        kept = dataclasses.replace(settings(), preserve_reasoning=True)
        # 256k still fits the 16-step chain with the reserve; 128k does not.
        self.assertEqual(
            build_plan(262144, kept, 8, 2748350)["session"]["steps"], 16
        )
        self.assertEqual(
            build_plan(131072, settings(), 2, 285696)["session"]["steps"], 9
        )
        self.assertLess(
            build_plan(131072, kept, 2, 285696)["session"]["steps"], 9
        )
        # Admission sizes each stream with the same reserve.
        plain = build_plan(262144, settings(), 8, 2748350)["session"]
        resent = build_plan(262144, kept, 8, 2748350)["session"]
        self.assertEqual(
            resent["per_stream_kv_demand"] - plain["per_stream_kv_demand"],
            16 * 2048,
        )

    def test_session_flags_an_empty_report_without_padding(self):
        def silent(messages, tools, max_tokens, sampling=None, stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            body["choices"][0]["message"]["content"] = ""
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                silent, CALIBRATION, 131072, self._session_settings(), (1,),
            )
        one = measured["by_streams"]["1"]
        self.assertEqual(one["sessions_incomplete"], 2)
        report = one["samples"][0]["per_stream"][0]["report"]
        self.assertEqual(report["content_chars"], 0)

    def test_session_health_counts_caps_and_missed_findings(self):
        def forgetful(messages, tools, max_tokens, sampling=None,
                      stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            if messages[-1].get("content") == SESSION_REPORT_REQUEST:
                # One finding dropped from an otherwise complete report.
                text = body["choices"][0]["message"]["content"]
                body["choices"][0]["message"]["content"] = text.rsplit(
                    ",", 1)[0]
            else:
                body["choices"][0]["finish_reason"] = "length"
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                forgetful, CALIBRATION, 131072,
                self._session_settings(min_samples=1, samples=1), (1,),
            )
        one = measured["by_streams"]["1"]
        session = one["samples"][0]["per_stream"][0]
        self.assertEqual(session["report_findings_recalled"], 2)
        self.assertFalse(session["complete"])
        self.assertEqual(one["truncated_calls"], 3)
        self.assertEqual(one["sessions_incomplete"], 1)

    def test_long_report_may_omit_one_finding(self):
        def forgetful(messages, tools, max_tokens, sampling=None,
                      stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            if messages[-1].get("content") == SESSION_REPORT_REQUEST:
                text = body["choices"][0]["message"]["content"]
                body["choices"][0]["message"]["content"] = text.rsplit(
                    ",", 1)[0]
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                forgetful, CALIBRATION, 262144,
                self._session_settings(min_samples=1, samples=1,
                                       conv_max_steps=8), (1,),
            )
        one = measured["by_streams"]["1"]
        session = one["samples"][0]["per_stream"][0]
        self.assertEqual(session["steps"], 8)
        self.assertEqual(session["report_findings_recalled"], 7)
        self.assertTrue(session["complete"])
        self.assertEqual(one["sessions_incomplete"], 0)

    def test_report_recall_folds_typographic_dashes(self):
        # Nemotron 3.5 Lightning writes some reports with U+2011 in every
        # finding token; the report still names each finding.
        def typographic(messages, tools, max_tokens, sampling=None,
                        stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            if messages[-1].get("content") == SESSION_REPORT_REQUEST:
                message = body["choices"][0]["message"]
                message["content"] = message["content"].replace("-", "\u2011")
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                typographic, CALIBRATION, 262144,
                self._session_settings(min_samples=1, samples=1,
                                       conv_max_steps=8), (1,),
            )
        session = measured["by_streams"]["1"]["samples"][0]["per_stream"][0]
        self.assertEqual(session["report_findings_recalled"], 8)
        self.assertTrue(session["complete"])

    def test_session_extends_on_rate_disagreement_not_wall_time(self):
        calls = {"n": 0}

        def drifting(messages, tools, max_tokens, sampling=None, stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            calls["n"] += 1
            # The second sample's server decodes twice as slowly.
            if calls["n"] > 4:
                body["timing"] = {"first_token_seconds": 0.5,
                                  "total_seconds": 2.5}
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                drifting, CALIBRATION, 131072,
                self._session_settings(sample_agreement=0.05), (1,),
            )
        self.assertEqual(len(measured["by_streams"]["1"]["samples"]), 3)

    def test_concurrent_sessions_share_evidence_and_diverge_chains(self):
        requests: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                _fake_session_chat(requests), CALIBRATION, 131072,
                self._session_settings(), (1, 2),
            )
        self.assertEqual(set(measured["by_streams"]), {"1", "2"})
        two = measured["by_streams"]["2"]
        self.assertEqual(len(two["samples"]), 2)
        self.assertEqual(len(two["samples"][0]["per_stream"]), 2)
        self.assertGreater(two["sessions_per_hour"]["median"], 0)
        self.assertIn("2", measured["scaling_vs_1x"])
        self.assertIn(measured["optimal_streams"], (1, 2))
        steps_a, evidence_a, _, files_a = _conversation_fixture(
            CALIBRATION, 131072, self._session_settings(), 900000, 901000
        )
        steps_b, evidence_b, _, files_b = _conversation_fixture(
            CALIBRATION, 131072, self._session_settings(), 900000, 902000
        )
        self.assertEqual(steps_a, steps_b)
        # Same repository context: the shared evidence prefix is
        # byte-identical, so concurrent agents can share the cache.
        self.assertEqual(evidence_a, evidence_b)
        # Same file names, divergent contents: distinct work per agent.
        self.assertEqual(list(files_a), list(files_b))
        self.assertNotEqual(files_a["file_001"], files_b["file_001"])

    def _one_session(self, chat):
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                chat, CALIBRATION, 131072,
                self._session_settings(min_samples=1, samples=1), (1,),
            )
        return measured["by_streams"]["1"]["samples"][0]["per_stream"][0]

    @staticmethod
    def _patched(**changes):
        def chat(messages, tools, max_tokens, sampling=None, stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            for key, value in changes.items():
                body[key] = value(body[key]) if callable(value) else value
            return body
        return chat

    def test_session_prefill_needs_reported_cache_counts(self):
        # A server that omits cached counts cannot say how much it
        # prefilled; the rate is absent rather than a full-prompt guess.
        uncached = self._patched(usage=lambda usage: {
            key: value for key, value in usage.items()
            if key != "prompt_tokens_details"
        })
        session = self._one_session(uncached)
        self.assertIsNone(session["prefill_tokens_per_second"])
        self.assertIsNotNone(session["report"]["decode_tokens_per_second"])

    def test_session_counts_an_end_of_turn_only_answer_as_prefill(self):
        silent = self._patched(
            usage=lambda usage: {**usage, "completion_tokens": 1},
            timing={"first_token_seconds": None, "total_seconds": 1.0},
        )
        session = self._one_session(silent)
        self.assertEqual(session["report"]["first_token_seconds"], 1.0)
        self.assertIsNone(session["report"]["decode_tokens_per_second"])

    def test_session_rejects_usage_without_token_counts(self):
        for usage in ({"prompt_tokens": 0, "completion_tokens": 5},
                      {"prompt_tokens": 100}):
            with self.subTest(usage=usage):
                with self.assertRaisesRegex(RuntimeError, "lacks token counts"):
                    self._one_session(self._patched(usage=usage))

    def test_untimed_turn_counts_as_prefill(self):
        def untimed(messages, tools, max_tokens, sampling=None, stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            if messages[-1].get("content") != SESSION_REPORT_REQUEST:
                body["timing"] = {"first_token_seconds": None,
                                  "total_seconds": 1}
            return body

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_sessions(
                untimed, CALIBRATION, 131072,
                self._session_settings(min_samples=1, samples=1), (1,),
            )
        session = measured["by_streams"]["1"]["samples"][0]["per_stream"][0]
        self.assertTrue(session["complete"])
        self.assertIsNone(session["turn_decode_tokens_per_second"]["first"])

    def test_plan_includes_conversation_block(self):
        plan = build_plan(262144, settings(), 2, 2748350)
        self.assertEqual(plan["conversation"]["steps"], 16)
        self.assertEqual(plan["conversation"]["evidence_tokens"], 32768)

    def test_calibration_accounts_for_fixed_prompt_tokens(self):
        calibration = Calibration(
            low_lines=64,
            low_tokens=1380,
            high_lines=256,
            high_tokens=5220,
        )
        self.assertEqual(calibration.lines_for(10340), 512)

    def test_calibration_is_stable_across_prompt_variants(self):
        token_count = len
        calibration = Calibration(
            low_lines=64,
            low_tokens=token_count(audit_prompt(64, 10)),
            high_lines=256,
            high_tokens=token_count(audit_prompt(256, 11)),
        )
        for target in (32768, 65536, 131072, 245760):
            lines = calibration.lines_for(target)
            for variant in (100000, 180000, 300000, 500102):
                actual = token_count(audit_prompt(lines, variant))
                with self.subTest(target=target, variant=variant):
                    self.assertLessEqual(abs(actual - target) / target, 0.02)

    def test_corpus_carries_no_instruction(self):
        corpus = audit_prompt(1, 10)
        self.assertNotIn("BENCHMARK CONTROL", corpus)
        self.assertNotIn("never conclude", corpus)

    def test_vllm_capacity_comes_from_managed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "server.log"
            log.write_text(
                "host: sm_80, 81920 MiB VRAM -> backend FLASH_ATTN, "
                "kv auto, util 0.98, seqs 8, batched 8192\n"
                "GPU KV cache size: 746,553 tokens\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "AUDIT_BENCH_ACTIVE_SLOTS": "",
                    "AUDIT_BENCH_KV_TOKENS": "",
                },
                clear=False,
            ):
                self.assertEqual(
                    vllm_capacity(log, 737280, False),
                    (8, 746553, "explicit or managed server log"),
                )

    def test_vllm_capacity_reads_metal_unified_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "server.log"
            log.write_text(
                "host: 128 GiB unified memory -> util 0.71, seqs 5, batched 8192\n"
                "CPU KV cache size: 2,775,642 tokens\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "AUDIT_BENCH_ACTIVE_SLOTS": "",
                    "AUDIT_BENCH_KV_TOKENS": "",
                },
                clear=False,
            ):
                self.assertEqual(
                    vllm_capacity(log, 262144, False),
                    (5, 2775642, "explicit or managed server log"),
                )

    def test_dry_run_capacity_is_conservative_without_log(self):
        with mock.patch.dict(
            os.environ,
            {
                "AUDIT_BENCH_ACTIVE_SLOTS": "",
                "AUDIT_BENCH_KV_TOKENS": "",
            },
            clear=False,
        ):
            self.assertEqual(
                vllm_capacity(pathlib.Path("/does/not/exist"), 262144, True),
                (1, 262144, "dry-run conservative"),
            )

    @staticmethod
    def _fake_request(prompt: str, max_tokens: int, seed: int) -> RequestResult:
        del seed
        evidence_lines = max(prompt.count("\n") - 1, 0)
        # A second of prefill plus 10 ms per decoded token.
        return RequestResult(
            prompt_tokens=100 + evidence_lines * 20,
            output_tokens=max_tokens,
            seconds=1 + max_tokens / 100,
        )

    _FAKE_ENV = {
        **SAMPLING,
        "AUDIT_BENCH_STREAMS": "1,2",
        "AUDIT_BENCH_SAMPLES": "1",
        "AUDIT_BENCH_CONV_MAX_STEPS": "2",
        "AUDIT_BENCH_TARGET_TOLERANCE": "0.02",
        "AUDIT_BENCH_DRY_RUN": "0",
    }

    def _run_fake_protocol(
        self,
        extra_env: dict,
        output_format: str = "json",
        openai_chat=None,
        request=None,
    ) -> tuple[int, str]:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            env = {
                **self._FAKE_ENV,
                "AUDIT_BENCH_OUTPUT_DIR": directory,
                **extra_env,
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with contextlib.redirect_stdout(output):
                    result = run_benchmark(
                        runtime="fake",
                        tag="fake-model",
                        model="fake-model",
                        context=131072,
                        request=request or self._fake_request,
                        active_slots=1,
                        kv_pool_tokens=131072,
                        output_format=output_format,
                        openai_chat=openai_chat or _fake_session_chat([]),
                    )
        return result, output.getvalue()

    def test_fake_adapter_runs_the_session_protocol(self):
        result, text = self._run_fake_protocol({"AUDIT_BENCH_GATE_CMD": ""})
        self.assertEqual(result, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(report["protocol"], "audit-v2")
        self.assertFalse(report["publishable"])
        self.assertEqual(
            report["settings"]["sampling"],
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        )
        self.assertIn("capacity_near", report)
        self.assertEqual(list(report["audit_session"]["by_streams"]), ["1"])
        for retired in ("prefill", "decode", "audits"):
            self.assertNotIn(retired, report)
        self.assertNotIn("incomplete_sessions", report["publication_reasons"])

    def test_incomplete_sessions_make_the_run_unpublishable(self):
        def capped(messages, tools, max_tokens, sampling=None, stream=False):
            body = _fake_session_chat([])(messages, tools, max_tokens,
                                          sampling, stream)
            body["choices"][0]["finish_reason"] = "length"
            return body

        _, text = self._run_fake_protocol(
            {"AUDIT_BENCH_GATE_CMD": ""}, openai_chat=capped
        )
        report = json.loads(text.splitlines()[-1])
        self.assertIn("incomplete_sessions", report["publication_reasons"])
        _, summary = self._run_fake_protocol(
            {"AUDIT_BENCH_GATE_CMD": ""}, "human", openai_chat=capped
        )
        self.assertIn("1 session incomplete", summary)
        self.assertIn("3 calls hit the output cap", summary)
        self.assertIn("no stream count completed every session", summary)

    def test_operating_point_is_the_fastest_complete_stream_count(self):
        def section(streams, rate, incomplete):
            return {"streams": streams, "sessions_incomplete": incomplete,
                    "sessions_per_hour": {"median": rate}}

        # 1x qualifies on its own when 4x drops a session, even though 4x
        # measured faster; the faster complete count wins otherwise.
        self.assertEqual(
            _operating_point({"1": section(1, 20.3, 0), "4": section(4, 21.0, 2)}),
            1,
        )
        self.assertEqual(
            _operating_point({"1": section(1, 8.0, 0), "4": section(4, 8.7, 0)}),
            4,
        )
        self.assertEqual(
            _operating_point({"1": section(1, 8.0, 1), "4": section(4, 8.7, 1)}),
            None,
        )

    def test_noncompliant_conversation_is_unpublishable(self):
        # The session fake never calls a tool, so the model-driven loop
        # stops at once with every file unread.
        _, text = self._run_fake_protocol(
            {"AUDIT_BENCH_GATE_CMD": "", "AUDIT_BENCH_PROFILE": "conversation"}
        )
        report = json.loads(text.splitlines()[-1])
        self.assertFalse(report["audit_conversation"]["compliant"])
        self.assertIn(
            "noncompliant_conversation", report["publication_reasons"]
        )

    def test_reference_uses_the_llama_bench_method(self):
        _, text = self._run_fake_protocol({"AUDIT_BENCH_GATE_CMD": ""})
        reference = json.loads(text.splitlines()[-1])["reference"]
        pp = reference["prompt_processing"]
        tg = reference["token_generation"]
        self.assertEqual(len(pp["tokens_per_second"]["reps"]), 5)
        # Prompt tokens over the whole 1-token request.
        self.assertAlmostEqual(
            pp["tokens_per_second"]["mean"], pp["prompt_tokens"][0] / 1.01
        )
        # 127 decode steps over the 128-token call minus the 1-token call.
        self.assertAlmostEqual(tg["tokens_per_second"]["mean"], 100.0)
        self.assertEqual(tg["tokens_per_second"]["sd"], 0.0)
        _, summary = self._run_fake_protocol(
            {"AUDIT_BENCH_GATE_CMD": ""}, "human"
        )
        self.assertIn("reference, llama-bench method", summary)
        self.assertIn("tg128 100.0 +/- 0.0 tok/s", summary)

        def short(prompt, max_tokens, seed):
            result = self._fake_request(prompt, max_tokens, seed)
            if max_tokens == 128:
                return dataclasses.replace(result, output_tokens=90)
            return result

        _, text = self._run_fake_protocol(
            {"AUDIT_BENCH_GATE_CMD": ""}, request=short
        )
        document = json.loads(text.splitlines()[-1])
        self.assertIn("reference call returned 90",
                      document["reference"]["unavailable"])
        self.assertIn("audit_session", document)

    def test_json_document_is_retained_and_named_in_the_summary(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            env = {**self._FAKE_ENV, "AUDIT_BENCH_OUTPUT_DIR": directory}
            with mock.patch.dict(os.environ, env, clear=False):
                with contextlib.redirect_stdout(output):
                    run_benchmark(
                        runtime="fake",
                        tag="fake-model",
                        model="fake-model",
                        context=131072,
                        request=self._fake_request,
                        active_slots=1,
                        kv_pool_tokens=131072,
                        openai_chat=_fake_session_chat([]),
                    )
            text = output.getvalue()
            retained = list(pathlib.Path(directory).glob("*.json"))
            self.assertEqual(len(retained), 1)
            self.assertIn(f"json: {retained[0]}", text)
            report = json.loads(retained[0].read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(report["artifact_path"], str(retained[0]))
        self.assertIn("audit_session", report)

    def test_human_summary_reports_capacity_and_sessions(self):
        _, text = self._run_fake_protocol({}, output_format="human")
        for expected in (
            "=== audit-v2 summary: fake/fake-model ===",
            "sampling: temperature=0.6 top_p=0.95 top_k=20",
            "near-context capacity",
            "agent session (2 steps)",
            "sessions/h",
            "decode first/mid/last",
            "canonical=no publishable=no",
        ):
            self.assertIn(expected, text)
        # The human report replaces the JSON on stdout, never wraps it.
        self.assertNotIn('"schema_version"', text)

    def test_conversation_summary_reports_compliance_and_recall(self):
        _, text = self._run_fake_protocol(
            {
                "AUDIT_BENCH_PROFILE": "conversation",
                "AUDIT_BENCH_CONV_MAX_STEPS": "3",
            },
            output_format="human",
            openai_chat=self._fake_conversation_chat(3),
        )
        self.assertIn("audit conversation (3 steps, 1 sample)", text)
        self.assertIn("protocol compliant: yes (1 of 1 samples)", text)
        self.assertIn("findings recalled: 0 of 3", text)
        self.assertIn("conversations/hour", text)
        self.assertNotIn("near-context capacity", text)

    def test_file_provenance_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "adapter.py"
            path.write_bytes(b"exact bytes\n")
            result = _file_provenance({"adapter": path, "model_config": path})
        self.assertEqual(
            result["adapter"]["sha256"],
            "6a77ce4ad94636f6120bb985066c1d75ce65b73f264a35f9d5ac910e252f0355",
        )
        self.assertEqual(result["adapter"]["size_bytes"], 12)

    def test_publication_requires_clean_complete_live_provenance(self):
        live_settings = dataclasses.replace(settings(), dry_run=False)
        metadata = {
            "repository": {"commit": "a" * 40, "dirty": False},
            "files": {
                "audit_bench": {"sha256": "b" * 64},
                "adapter": {"sha256": "c" * 64},
                "model_config": {"sha256": "d" * 64},
            },
            "runtime_profile": {"checked": True, "launcher_lines": ["profile"]},
            "model_identity": {"checked": True, "matched": True},
        }
        publishable, reasons = _publication_status(live_settings, metadata)
        self.assertTrue(publishable)
        self.assertEqual(reasons, [])
        metadata["repository"]["dirty"] = True
        publishable, reasons = _publication_status(live_settings, metadata)
        self.assertFalse(publishable)
        self.assertIn("repository_dirty", reasons)

    def test_gate_runs_before_every_sample_batch_and_stays_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = pathlib.Path(directory) / "gates"
            result, text = self._run_fake_protocol(
                {"AUDIT_BENCH_GATE_CMD": f"echo gate >> {marker}"}
            )
            self.assertEqual(result, 0)
            report = json.loads(text.splitlines()[-1])
            # 131k plan with one slot: one capacity batch and one session
            # batch; warmup and calibration are direct requests and must
            # not be gated.
            gates = marker.read_text(encoding="utf-8").count("gate")
            self.assertEqual(gates, 2)
            self.assertEqual(report["settings"]["gate_cmd"], f"echo gate >> {marker}")
        with mock.patch.dict(
            os.environ, {**SAMPLING, "AUDIT_BENCH_GATE_CMD": "true"},
            clear=False,
        ):
            self.assertTrue(Settings.from_env().canonical)

    def test_failing_gate_aborts_the_run(self):
        with self.assertRaises(RuntimeError):
            self._run_fake_protocol({"AUDIT_BENCH_GATE_CMD": "exit 3"})

    def test_system_prompt_is_recorded_and_stays_canonical(self):
        control = "Reasoning strength: low"
        result, text = self._run_fake_protocol(
            {"AUDIT_BENCH_SYSTEM_PROMPT": control}
        )
        self.assertEqual(result, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["settings"]["system_prompt"], control)
        with mock.patch.dict(
            os.environ, {**SAMPLING, "AUDIT_BENCH_SYSTEM_PROMPT": control},
            clear=False,
        ):
            self.assertTrue(Settings.from_env().canonical)

    def test_reasoning_effort_is_recorded_and_stays_canonical(self):
        result, text = self._run_fake_protocol(
            {"AUDIT_BENCH_REASONING_EFFORT": "none"}
        )
        self.assertEqual(result, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["settings"]["reasoning_effort"], "none")
        with mock.patch.dict(
            os.environ, {**SAMPLING, "AUDIT_BENCH_REASONING_EFFORT": "none"},
            clear=False,
        ):
            self.assertTrue(Settings.from_env().canonical)

    @classmethod
    def _reports_one_fewer(cls, finish_reason: str):
        def request(prompt: str, max_tokens: int, seed: int) -> RequestResult:
            return dataclasses.replace(
                cls._fake_request(prompt, max_tokens, seed),
                output_tokens=max_tokens - 1,
                details={"finish_reason": finish_reason},
            )

        return request

    def test_off_by_one_tolerance_is_exact_and_noncanonical(self):
        result, text = self._run_fake_protocol(
            {"AUDIT_BENCH_TOLERATE_OFF_BY_ONE": "1"},
            request=self._reports_one_fewer("length"),
        )
        self.assertEqual(result, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertTrue(report["settings"]["tolerate_off_by_one"])
        self.assertFalse(report["canonical"])

    def test_off_by_one_tolerance_rejects_natural_stops_and_exact_counts(self):
        env = {"AUDIT_BENCH_TOLERATE_OFF_BY_ONE": "1"}
        for request in (self._reports_one_fewer("stop"), None):
            with self.assertRaisesRegex(RuntimeError, "one fewer"):
                self._run_fake_protocol(env, request=request)

    def test_stream_times_the_first_token_and_requires_usage(self):
        captured = {}

        def respond(events):
            def fake_urlopen(request, timeout=None):
                captured["body"] = json.loads(request.data.decode())
                return _Stream(events)
            return fake_urlopen

        events = [
            # vLLM opens with an empty content delta and interim null usage;
            # neither is a token.
            {"choices": [{"delta": {"role": "assistant", "content": ""}}],
             "usage": None},
            {"choices": [{"delta": {"reasoning": "think"}}], "usage": None},
            {"choices": [{"delta": {"content": "rep"}}]},
            {"choices": [{"delta": {"content": "ort"},
                          "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 9,
                                      "completion_tokens": 3}},
            "data: [DONE]",
        ]
        chat = openai_chat_fn("http://127.0.0.1:1", "m", 10)
        with mock.patch("urllib.request.urlopen", side_effect=respond(events)):
            body = chat([], [], 8, sampling={"temperature": 0.6}, stream=True)
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["choices"][0]["message"]["content"], "report")
        self.assertEqual((body["reasoning_chars"], body["tool_calls"]), (5, 0))
        self.assertEqual(body["reasoning_text"], "think")
        timing = body["timing"]
        self.assertIsNotNone(timing["first_token_seconds"])
        self.assertLessEqual(
            timing["first_token_seconds"], timing["total_seconds"]
        )
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(
            captured["body"]["stream_options"], {"include_usage": True}
        )
        self.assertEqual(captured["body"]["temperature"], 0.6)
        with mock.patch(
            "urllib.request.urlopen", side_effect=respond(events[:4])
        ):
            with self.assertRaisesRegex(RuntimeError, "without a usage chunk"):
                chat([], [], 8, stream=True)
        # A turn that is only a tool call starts decode at the call delta.
        call_only = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0}]},
                          "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 9,
                                      "completion_tokens": 12}},
        ]
        with mock.patch("urllib.request.urlopen", side_effect=respond(call_only)):
            body = chat([], [], 8, stream=True)
        self.assertIsNotNone(body["timing"]["first_token_seconds"])
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(body["tool_calls"], 1)
        for bad, message in (
            ([{"error": {"message": "boom"}}], "stream error"),
            (["data: {not json"], "malformed stream line"),
        ):
            with self.subTest(message=message):
                with mock.patch(
                    "urllib.request.urlopen", side_effect=respond(bad)
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        chat([], [], 8, stream=True)

    def test_chat_fn_sends_only_the_declared_sampling(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode())

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def read(self):
                    return json.dumps({"choices": [{}], "usage": {}})

            return _Resp()

        chat = openai_chat_fn("http://127.0.0.1:1", "m", 10)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat([], [], 4, sampling={"temperature": 0.6, "top_k": 20})
        self.assertEqual(captured["body"]["temperature"], 0.6)
        self.assertEqual(captured["body"]["top_k"], 20)
        self.assertNotIn("stream", captured["body"])
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat([], [], 4)
        self.assertNotIn("temperature", captured["body"])

    def test_adapter_records_finish_reason(self):
        def fake_urlopen(request, timeout=None):
            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, t, e, tb):
                    return False

                def read(self):
                    return json.dumps({
                        "choices": [{"finish_reason": "length"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    })

            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = OpenAIChatAdapter("http://127.0.0.1:1", "m", 10)("hi", 1, 0)
        self.assertEqual(result.details, {"finish_reason": "length"})

    def test_adapter_sends_reasoning_effort_only_when_set(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, t, e, tb):
                    return False

                def read(self):
                    return json.dumps(
                        {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}
                    )

            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            OpenAIChatAdapter(
                "http://127.0.0.1:1", "m", 10, reasoning_effort="none"
            )("hi", 1, 0)
        body = json.loads(captured["request"].data.decode())
        self.assertEqual(body["reasoning_effort"], "none")

        captured.clear()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            OpenAIChatAdapter("http://127.0.0.1:1", "m", 10)("hi", 1, 0)
        body = json.loads(captured["request"].data.decode())
        self.assertNotIn("reasoning_effort", body)

    def test_adapter_sends_system_prompt_only_when_set(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, t, e, tb):
                    return False

                def read(self):
                    return json.dumps(
                        {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}
                    )

            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            OpenAIChatAdapter(
                "http://127.0.0.1:1", "m", 10, system_prompt="Control: low"
            )("hi", 1, 0)
        body = json.loads(captured["request"].data.decode())
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "Control: low"},
                {"role": "user", "content": "hi"},
            ],
        )

        captured.clear()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            OpenAIChatAdapter("http://127.0.0.1:1", "m", 10)("hi", 1, 0)
        body = json.loads(captured["request"].data.decode())
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_chat_fn_sends_system_prompt_before_workload_messages(self):
        workload = [
            {"role": "system", "content": "workload system"},
            {"role": "user", "content": "turn one"},
        ]
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, t, e, tb):
                    return False

                def read(self):
                    return json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {"content": "ok"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 1,
                            },
                        }
                    )

            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat = openai_chat_fn(
                "http://127.0.0.1:1", "m", 10, system_prompt="Control: low"
            )
            chat(list(workload), [], 4)
        body = json.loads(captured["request"].data.decode())
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "Control: low"},
                {"role": "system", "content": "workload system"},
                {"role": "user", "content": "turn one"},
            ],
        )
        # The workload's own messages stay byte-identical and in order;
        # the control message is prepended ahead of them.
        self.assertEqual(body["messages"][1:], workload)

        captured.clear()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat = openai_chat_fn("http://127.0.0.1:1", "m", 10)
            chat(list(workload), [], 4)
        body = json.loads(captured["request"].data.decode())
        self.assertEqual(body["messages"], workload)

    def test_chat_fn_handles_error_and_choicesless_bodies(self):
        endpoint = "http://127.0.0.1:1/v1/chat/completions"
        error = urllib.error.HTTPError(
            endpoint,
            500,
            "server error",
            http.client.HTTPMessage(),
            io.BytesIO(b"boom"),
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            chat = openai_chat_fn("http://127.0.0.1:1", "m", 10)
            with self.assertRaises(RuntimeError) as ctx:
                chat([], [], 1)
        self.assertIn("500", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, t, e, tb):
                return False

            def read(self):
                return json.dumps(
                    {"usage": {"prompt_tokens": 1}}
                ).encode()

        def fake_urlopen(request, timeout=None):
            del request
            return _Resp()

        with mock.patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            chat = openai_chat_fn("http://127.0.0.1:1", "m", 10)
            with self.assertRaises(RuntimeError) as ctx:
                chat([], [], 1)
        self.assertIn("without choices", str(ctx.exception))


class CliMainTest(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {**SAMPLING, "AUDIT_BENCH_OUTPUT_DIR": directory},
                clear=False,
            ):
                with contextlib.redirect_stdout(output):
                    code = main(argv)
        return code, output.getvalue()

    def test_dry_run_uses_conservative_defaults(self):
        code, text = self._run_main(
            [
                "--base-url",
                "http://127.0.0.1:9",
                "--model",
                "m",
                "--context",
                "131072",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("note: --active-slots not given", text)
        self.assertIn("note: --kv-tokens not given", text)
        report = json.loads(text.splitlines()[-1])
        self.assertTrue(report["canonical"])
        self.assertFalse(report["publishable"])
        self.assertIn("dry_run", report["publication_reasons"])
        self.assertIn(
            "runtime_profile_unchecked", report["publication_reasons"]
        )
        metadata = report["metadata"]
        self.assertEqual(metadata["runtime"], "endpoint")
        self.assertEqual(metadata["tag"], "127.0.0.1:9")
        self.assertEqual(metadata["model"], "m")
        self.assertEqual(metadata["active_slots"], 1)
        self.assertEqual(metadata["kv_pool_tokens"], 131072)
        self.assertFalse(metadata["model_identity"]["checked"])
        self.assertIn("plan", report)

    def test_dry_run_takes_explicit_operating_point(self):
        code, text = self._run_main(
            [
                "--base-url",
                "http://127.0.0.1:9",
                "--context",
                "131072",
                "--active-slots",
                "4",
                "--kv-tokens",
                "524288",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("note:", text)
        report = json.loads(text.splitlines()[-1])
        metadata = report["metadata"]
        self.assertEqual(metadata["model"], "unspecified")
        self.assertEqual(metadata["active_slots"], 4)
        self.assertEqual(metadata["kv_pool_tokens"], 524288)

    def test_dry_run_prints_the_plan_by_default(self):
        code, text = self._run_main(
            [
                "--base-url",
                "http://127.0.0.1:9",
                "--context",
                "131072",
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("plan only; no server was contacted", text)
        self.assertIn("near-context capacity: 122,880-token prompt", text)
        self.assertIn("session: 9 steps from 32,768-token evidence", text)
        self.assertIn("canonical=yes publishable=no (dry_run", text)
        self.assertNotIn('"schema_version"', text)

    def test_profile_flag_selects_workload(self):
        code, text = self._run_main(
            [
                "--base-url",
                "http://127.0.0.1:9",
                "--context",
                "131072",
                "--profile",
                "conversation",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        self.assertEqual(code, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["settings"]["profile"], "conversation")
        self.assertTrue(report["canonical"])

    def test_show_reprints_a_retained_document(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"AUDIT_BENCH_OUTPUT_DIR": directory},
                clear=False,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(
                        [
                            "--base-url",
                            "http://127.0.0.1:9",
                            "--context",
                            "131072",
                            "--dry-run",
                        ]
                    )
                retained = list(pathlib.Path(directory).glob("*.json"))[0]
                before = retained.read_bytes()
                code, text = self._run_main(["--show", str(retained)])
                # Re-printing retains nothing new and rewrites nothing.
                self.assertEqual(
                    list(pathlib.Path(directory).glob("*.json")), [retained]
                )
                self.assertEqual(retained.read_bytes(), before)
                self.assertEqual(code, 0)
                self.assertIn("plan only; no server was contacted", text)
                self.assertIn(f"json: {retained}", text)
                code, text = self._run_main(
                    ["--show", str(retained), "--format", "json"]
                )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["protocol"], "audit-v2")

    def test_show_rejects_a_foreign_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "capability.json"
            path.write_text('{"protocol": "capability-v1"}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a audit-v2"):
                self._run_main(["--show", str(path)])
            missing = pathlib.Path(directory) / "absent.json"
            with self.assertRaisesRegex(SystemExit, "cannot read"):
                self._run_main(["--show", str(missing)])

    def test_show_refuses_to_guess_a_legacy_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.json"
            # A pre-publication document: canonical without a publishable
            # verdict, which the summary must never infer.
            path.write_text(
                json.dumps(
                    {
                        "protocol": "audit-v2",
                        "canonical": True,
                        "settings": {},
                        "metadata": {},
                        "plan": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SystemExit, "predates the summary: publishable"
            ):
                self._run_main(["--show", str(path)])

    def test_show_points_a_v1_document_at_its_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "v1.json"
            path.write_text(
                json.dumps({"protocol": "audit-v1", "canonical": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SystemExit, "audit-v1 document.*checkout at 382760d"
            ):
                self._run_main(["--show", str(path)])

    def test_cli_sessions_use_the_v1_chat_endpoint(self):
        card = mock.MagicMock()
        card.__enter__.return_value = io.BytesIO(json.dumps(
            {"data": [{"id": "m", "max_model_len": 131072}]}
        ).encode())
        with mock.patch("urllib.request.urlopen", return_value=card), \
                mock.patch("audit_bench.run_benchmark", return_value=0) as run:
            self._run_main(["--base-url", "http://127.0.0.1:9/",
                            "--context", "131072"])
        chat = run.call_args.kwargs["openai_chat"]
        requested = {}

        def fake_urlopen(request, timeout=None):
            requested["url"] = request.full_url
            return _Stream([{"choices": [], "usage": {
                "prompt_tokens": 1, "completion_tokens": 0}}])

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat([], [], 1, stream=True)
        self.assertEqual(
            requested["url"], "http://127.0.0.1:9/v1/chat/completions"
        )

    def test_running_still_requires_an_endpoint(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):
                self._run_main(["--context", "131072"])
        self.assertIn("--base-url is required", errors.getvalue())

    def test_list_indexes_retained_documents_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"AUDIT_BENCH_OUTPUT_DIR": directory},
                clear=False,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["--base-url", "http://127.0.0.1:9",
                          "--context", "131072", "--dry-run"])
                    main(["--base-url", "http://127.0.0.1:9",
                          "--context", "262144", "--profile", "conversation",
                          "--dry-run"])
                broken = pathlib.Path(directory) / "broken.json"
                broken.write_text("not json", encoding="utf-8")
                # Campaign evidence is filed in subdirectories, and
                # unrelated probe JSON shares the tree.
                campaign = pathlib.Path(directory) / "campaign-1"
                campaign.mkdir()
                (campaign / "probe.json").write_text(
                    '{"probe": 1}', encoding="utf-8"
                )
                archived = campaign / "archived.json"
                archived.write_text(
                    json.dumps(
                        {
                            "protocol": "audit-v1",
                            "canonical": True,
                            "publishable": True,
                            "publication_reasons": [],
                            "settings": {"profile": "full"},
                            "metadata": {
                                "runtime": "vllm",
                                "tag": "host",
                                "started_at_utc": "2026-08-31T08:24:59+00:00",
                            },
                            "plan": {},
                        }
                    ),
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--list"])
        lines = output.getvalue().splitlines()
        self.assertEqual(code, 0)
        self.assertEqual(lines[0], f"4 retained documents in {directory}")
        self.assertIn("when", lines[1])
        # Newest first, with the unusable file named rather than hidden.
        self.assertIn("campaign-1/archived.json", lines[2])
        self.assertIn("publishable", lines[2])
        # Retained v1 history stays listed, marked as the old protocol.
        self.assertIn("v1 full", lines[2])
        self.assertIn("unreadable", lines[3])
        self.assertIn("conversation", lines[4])
        self.assertIn("session", lines[5])
        self.assertIn("1 other JSON file in the tree, ignored", lines[6])
        self.assertEqual(lines[-1], "re-print one with: audit-bench --show FILE")

    def test_list_says_so_when_nothing_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = pathlib.Path(directory) / "absent"
            with mock.patch.dict(
                os.environ,
                {"AUDIT_BENCH_OUTPUT_DIR": str(empty)},
                clear=False,
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--list"])
            # Listing never creates the directory it reports on.
            self.assertFalse(empty.exists())
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().strip(), f"no retained documents in {empty}"
        )

    def test_empty_output_dir_override_means_default(self):
        # Consumers forward every AUDIT_BENCH_* name, so an unset override
        # arrives as an empty string and must not retain in the cwd.
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_OUTPUT_DIR": ""}, clear=False
        ):
            self.assertEqual(_output_dir().name, "auditbench")

    def test_show_takes_a_listed_file_name(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = pathlib.Path(directory) / "campaign-1"
            campaign.mkdir()
            document = {
                "protocol": "audit-v2",
                "canonical": True,
                "publishable": True,
                "publication_reasons": [],
                "settings": {"profile": "session", "dry_run": False},
                "metadata": {"runtime": "vllm", "tag": "host",
                             "model": "m", "active_slots": 8,
                             "kv_pool_tokens": 742613},
                "plan": {"configured_context": 262144},
            }
            for path in (
                pathlib.Path(directory) / "top.json",
                campaign / "archived.json",
            ):
                path.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"AUDIT_BENCH_OUTPUT_DIR": directory},
                clear=False,
            ):
                for argument in ("top.json", "campaign-1/archived.json"):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main(["--show", argument])
                    self.assertEqual(code, 0)
                    self.assertIn(
                        "=== audit-v2 summary: vllm/host ===",
                        output.getvalue(),
                    )
                    self.assertIn(
                        f"json: {directory}/{argument}", output.getvalue()
                    )
                with self.assertRaisesRegex(
                    SystemExit, f"no such file here or in {directory}"
                ):
                    main(["--show", "absent.json"])

    def test_list_marks_documents_show_refuses_as_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.json"
            path.write_text(
                json.dumps({"protocol": "audit-v1", "canonical": True}),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"AUDIT_BENCH_OUTPUT_DIR": directory},
                clear=False,
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--list"])
        self.assertEqual(code, 0)
        self.assertIn("legacy", output.getvalue())
        self.assertNotIn("canonical", output.getvalue())

    def test_rejects_non_integer_capacity_env(self):
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_ACTIVE_SLOTS": "many"}, clear=False
        ):
            with self.assertRaisesRegex(SystemExit, "must be an integer"):
                self._run_main(
                    [
                        "--base-url",
                        "http://127.0.0.1:9",
                        "--context",
                        "131072",
                        "--dry-run",
                    ]
                )

    def test_rejects_non_positive_capacity(self):
        with self.assertRaises(SystemExit):
            self._run_main(
                [
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--context",
                    "131072",
                    "--active-slots",
                    "0",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()

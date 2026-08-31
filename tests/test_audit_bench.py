import contextlib
import dataclasses
import io
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest import mock

from audit_bench import (
    Calibration,
    OpenAIChatAdapter,
    RequestResult,
    Settings,
    _conversation_fixture,
    _file_provenance,
    _measure,
    _measure_conversations,
    _measure_family,
    _measure_replay_concurrent,
    _measure_replays,
    _publication_status,
    audit_prompt,
    build_plan,
    REPLAY_PROBE_TOKENS,
    conversation_steps,
    decode_prompt,
    main,
    parse_streams,
    run_benchmark,
    vllm_capacity,
)


def settings() -> Settings:
    return Settings(
        streams=(1, 2, 4, 8),
        deep_streams=(1, 2, 4),
        profile="full",
        samples=3,
        output_tokens=1024,
        deep_output_tokens=256,
        tolerance=0.02,
        dry_run=True,
    )


class AuditBenchTest(unittest.TestCase):
    def test_streams_are_strict_and_start_at_one(self):
        self.assertEqual(parse_streams("1,2,4,8"), (1, 2, 4, 8))
        for value in ("", "2,4", "1,4,2", "1,2,2", "1,zero"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_streams(value)

    def test_128k_plan_uses_common_and_relative_targets(self):
        plan = build_plan(131072, settings(), 2, 285696)
        self.assertEqual(plan["prefill_targets"], [32768, 65536])
        self.assertEqual(plan["capacity_near"]["target_prompt_tokens"], 122880)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 2])
        self.assertEqual(plan["decode"]["streams"], [1, 2])
        self.assertEqual(
            [phase["target_prompt_tokens"] for phase in plan["audit_phases"]],
            [32768, 65536, 114688],
        )
        self.assertEqual(
            [phase["streams"] for phase in plan["audit_phases"]],
            [[1, 2], [1], [1, 2]],
        )
        self.assertFalse(plan["preferred_256k_context"])

    def test_plan_rejects_context_below_128k(self):
        with self.assertRaises(ValueError):
            build_plan(65536, settings(), 1, 65536)

    def test_plan_stops_at_active_slot_limit(self):
        plan = build_plan(262144, settings(), 2, 2748350)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 2])
        self.assertEqual(plan["decode"]["streams"], [1, 2])

    def test_256k_plan_adds_common_128k_target(self):
        plan = build_plan(262144, settings(), 8, 2748350)
        self.assertEqual(plan["prefill_targets"], [32768, 65536, 131072])
        self.assertEqual(plan["capacity_near"]["target_prompt_tokens"], 245760)
        self.assertEqual(plan["capacity_near"]["streams"], [1, 8])
        self.assertEqual(
            [phase["target_prompt_tokens"] for phase in plan["audit_phases"]],
            [32768, 131072, 229376],
        )
        self.assertEqual(
            [phase["streams"] for phase in plan["audit_phases"]],
            [[1, 2, 4, 8], [1], [1, 4]],
        )
        self.assertTrue(plan["preferred_256k_context"])

    def test_720k_plan_avoids_queued_long_context_work(self):
        plan = build_plan(737280, settings(), 8, 746553)
        self.assertEqual(plan["decode"]["streams"], [1, 2, 4, 8])
        self.assertEqual(plan["capacity_near"]["streams"], [1])
        self.assertEqual(
            [phase["streams"] for phase in plan["audit_phases"]],
            [[1, 2, 4, 8], [1], [1]],
        )

    @staticmethod
    def _timed_request(durations: list[float]):
        remaining = iter(durations)

        def request(prompt: str, max_tokens: int, seed: int) -> RequestResult:
            del prompt, seed
            seconds = next(remaining)
            time.sleep(seconds)
            return RequestResult(
                prompt_tokens=100, output_tokens=max_tokens, seconds=seconds
            )

        return request

    def test_adaptive_sampling_stops_at_agreeing_minimum(self):
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure(
                request=self._timed_request([0.05, 0.05, 0.05]),
                phase="prefill_32k",
                streams=1,
                samples=3,
                output_tokens=1,
                context=131072,
                tolerance=0.02,
                variant_base=0,
                min_samples=2,
                agreement=0.5,
            )
        self.assertEqual(len(measured["samples"]), 2)

    def test_adaptive_sampling_extends_to_median_of_three_on_disagreement(self):
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure(
                request=self._timed_request([0.02, 0.3, 0.3]),
                phase="prefill_32k",
                streams=1,
                samples=3,
                output_tokens=1,
                context=131072,
                tolerance=0.02,
                variant_base=0,
                min_samples=2,
                agreement=0.05,
            )
        self.assertEqual(len(measured["samples"]), 3)
        self.assertAlmostEqual(
            measured["wall_seconds"]["median"], 0.3, delta=0.1
        )

    def test_family_extends_every_count_and_capacity_stays_single(self):
        adaptive = dataclasses.replace(
            settings(), min_samples=2, sample_agreement=0.05
        )
        with contextlib.redirect_stdout(io.StringIO()):
            family = _measure_family(
                request=self._timed_request([0.02, 0.3, 0.3]),
                phase="decode",
                streams=(1,),
                settings=adaptive,
                samples=3,
                output_tokens=8,
                context=131072,
                variant_base=200000,
            )
            capacity = _measure_family(
                request=self._timed_request([0.05]),
                phase="capacity_near",
                streams=(1,),
                settings=adaptive,
                samples=1,
                output_tokens=8,
                context=131072,
                variant_base=180000,
            )
        self.assertEqual(len(family["by_streams"]["1"]["samples"]), 3)
        self.assertEqual(len(capacity["by_streams"]["1"]["samples"]), 1)

    def test_sampling_env_is_validated_and_gates_canonicality(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_MIN_SAMPLES": "3"}, clear=False
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
            os.environ, {"AUDIT_BENCH_SAMPLES": "1"}, clear=False
        ):
            parsed = Settings.from_env()
            self.assertEqual(parsed.min_samples, 1)
            self.assertFalse(parsed.canonical)

    @staticmethod
    def _fake_conversation_chat(
        tool_steps: int,
        break_at: int | None = None,
        batch: int = 1,
    ):
        state = {"n": 0, "requested": 0}

        def chat(messages, tools, max_tokens):
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
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        with contextlib.redirect_stdout(io.StringIO()):
            # Each sample is a fresh conversation against a fresh fake.
            active: list = [self._fake_conversation_chat(3)]

            def per_sample_chat(messages, tools, max_tokens):
                if len(messages) == 2:
                    active[0] = self._fake_conversation_chat(3)
                return active[0](messages, tools, max_tokens)

            measured = _measure_conversations(
                per_sample_chat, calibration, 131072, conv
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

    def test_conversation_accepts_batched_tool_calls(self):
        conv = self._conversation_settings()
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_conversations(
                self._fake_conversation_chat(3, batch=3),
                calibration, 131072, conv,
            )
        first = measured["samples"][0]
        self.assertTrue(first["compliant"])
        self.assertEqual(first["turns"], 2)
        self.assertEqual(first["files_read"], 3)

    def test_conversation_records_noncompliance_without_padding(self):
        conv = self._conversation_settings()
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_conversations(
                self._fake_conversation_chat(3, break_at=1),
                calibration, 131072, conv,
            )
        self.assertFalse(measured["compliant"])
        self.assertEqual(len(measured["samples"]), 2)

    @staticmethod
    def _fake_replay_chat():
        """A hostile model: empty content and a bogus tool call every turn.

        The replay profile must not care what the model produces, so the
        fake produces the worst case an interactive conversation would
        fail on.
        """
        state = {"n": 0}

        def chat(messages, tools, max_tokens):
            del messages, tools
            n = state["n"]
            state["n"] += 1
            time.sleep(0.02)
            usage: dict = {
                "prompt_tokens": 1000 * (n + 1),
                "completion_tokens": max_tokens,
            }
            if n > 0:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": 1000 * n - 100
                }
            message = {
                "content": "",
                "tool_calls": [{
                    "id": "bogus",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "no/such/file"}),
                    },
                }],
            }
            return {
                "choices": [{"message": message, "finish_reason": "length"}],
                "usage": usage,
            }

        return chat

    def test_replay_scripts_every_turn_and_measures_the_server(self):
        conv = self._conversation_settings(profile="replay")
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        requests: list = []
        fake = self._fake_replay_chat()

        def chat(messages, tools, max_tokens):
            requests.append((list(messages), max_tokens))
            return fake(messages, tools, max_tokens)

        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_replays(chat, calibration, 131072, conv)
        # No compliance concept: the hostile model still gets measured.
        self.assertNotIn("compliant", measured)
        self.assertEqual(len(measured["samples"]), 2)
        first = measured["samples"][0]
        self.assertEqual(first["steps"], 3)
        self.assertEqual(len(first["per_step"]), 3)
        self.assertGreater(first["cached_fraction"], 0)
        self.assertGreater(measured["replays_per_hour"]["median"], 0)
        self.assertGreater(
            measured["verdict_output_tokens_per_second"]["median"], 0
        )
        # First sample: three probe-capped scripted turns, then the
        # forced full-cap verdict at depth.
        probe_caps = [request[1] for request in requests[:3]]
        self.assertEqual(probe_caps, [REPLAY_PROBE_TOKENS] * 3)
        verdict_messages, verdict_cap = requests[3]
        self.assertEqual(verdict_cap, conv.conv_output_tokens)
        self.assertIn("BENCHMARK CONTROL", verdict_messages[-1]["content"])
        # The scripted assistant turn replaced the model's bogus output.
        scripted = verdict_messages[2]
        self.assertEqual(scripted["role"], "assistant")
        self.assertNotEqual(scripted["content"], "")
        self.assertEqual(
            json.loads(
                scripted["tool_calls"][0]["function"]["arguments"]
            )["path"],
            "file_001",
        )
        self.assertIn("file_001", verdict_messages[3]["content"])

    def test_replay_profile_is_canonical_at_defaults(self):
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_PROFILE": "replay"}, clear=False
        ):
            self.assertTrue(Settings.from_env().canonical)

    def test_replay_concurrent_profile_is_canonical_at_defaults(self):
        with mock.patch.dict(
            os.environ,
            {"AUDIT_BENCH_PROFILE": "replay-concurrent"},
            clear=False,
        ):
            self.assertTrue(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ,
            {"AUDIT_BENCH_PROFILE": "replay-concurrent",
             "AUDIT_BENCH_CONV_TURN_TOKENS": "256"},
            clear=False,
        ):
            self.assertFalse(Settings.from_env().canonical)

    def test_plan_includes_replay_concurrent_block(self):
        plan = build_plan(262144, settings(), 8, 2748350)
        block = plan["replay_concurrent"]
        self.assertEqual(block["steps"], 16)
        self.assertEqual(block["turn_output_tokens"], 1024)
        # 32,768 + 16 * (8,192 + 256) + 16,384 per stream, no
        # shared-prefix credit.
        self.assertEqual(block["per_stream_kv_demand"], 184320)
        self.assertEqual(block["streams"], [1, 2, 4])

    def test_replay_concurrent_streams_respect_pool_and_slots(self):
        # A 611k pool holds three per-stream demands; the deep-stream
        # candidates prune to 1 and 2.
        plan = build_plan(262144, settings(), 5, 611156)
        self.assertEqual(plan["replay_concurrent"]["streams"], [1, 2])
        # One-slot admission runs 1x only regardless of pool size.
        plan = build_plan(262144, settings(), 1, 2748350)
        self.assertEqual(plan["replay_concurrent"]["streams"], [1])

    def test_replay_concurrent_fixture_shares_evidence_diverges_chains(self):
        conv = self._conversation_settings(profile="replay-concurrent")
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        steps_a, evidence_a, _, files_a = _conversation_fixture(
            calibration, 131072, conv, 900000, 901000
        )
        steps_b, evidence_b, _, files_b = _conversation_fixture(
            calibration, 131072, conv, 900000, 902000
        )
        self.assertEqual(steps_a, steps_b)
        # Same repository context: the shared evidence prefix is
        # byte-identical, so concurrent agents can share the cache.
        self.assertEqual(evidence_a, evidence_b)
        # Same file names, divergent contents: distinct work per agent.
        self.assertEqual(list(files_a), list(files_b))
        self.assertNotEqual(files_a["file_001"], files_b["file_001"])

    @staticmethod
    def _fake_concurrent_chat(requests: list):
        """Stateless thread-safe fake deriving usage from the messages."""

        def chat(messages, tools, max_tokens):
            del tools
            requests.append(([dict(item) for item in messages], max_tokens))
            time.sleep(0.01)
            prompt = 500 * len(messages)
            usage = {"prompt_tokens": prompt, "completion_tokens": max_tokens}
            if len(messages) > 3:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": prompt - 400
                }
            message = {"content": "x", "tool_calls": []}
            return {
                "choices": [{"message": message, "finish_reason": "length"}],
                "usage": usage,
            }

        return chat

    def test_replay_concurrent_measures_shared_prefix_batches(self):
        conv = self._conversation_settings(profile="replay-concurrent")
        calibration = Calibration(
            low_lines=64, low_tokens=1380, high_lines=256, high_tokens=5220
        )
        requests: list = []
        chat = self._fake_concurrent_chat(requests)
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure_replay_concurrent(
                chat, calibration, 131072, conv, (1, 2)
            )
        self.assertEqual(set(measured["by_streams"]), {"1", "2"})
        two = measured["by_streams"]["2"]
        self.assertEqual(len(two["samples"]), 2)
        self.assertGreater(two["replays_per_hour"]["median"], 0)
        self.assertIn("2", measured["scaling_vs_1x"])
        self.assertEqual(
            two["turn_completion_tokens_median"], conv.conv_turn_tokens
        )
        # Probe turns decode to the turn cap under an ephemeral control
        # riding at the end of the measured call only.
        probes = [item for item in requests
                  if item[1] == conv.conv_turn_tokens]
        self.assertTrue(probes)
        for messages, _ in probes:
            self.assertEqual(messages[-1]["role"], "user")
            self.assertIn("BENCHMARK CONTROL", messages[-1]["content"])
        # The ephemeral control never enters the transcript: verdict
        # requests carry exactly one control, the permanent verdict one.
        verdicts = [item for item in requests
                    if item[1] == conv.conv_output_tokens]
        self.assertEqual(len(verdicts), 6)
        for messages, _ in verdicts:
            controls = [
                item for item in messages
                if "BENCHMARK CONTROL" in (item.get("content") or "")
            ]
            self.assertEqual(len(controls), 1)

    def test_plan_includes_replay_block(self):
        plan = build_plan(262144, settings(), 2, 2748350)
        self.assertEqual(plan["replay"]["steps"], 16)
        self.assertEqual(
            plan["replay"]["probe_output_tokens"], REPLAY_PROBE_TOKENS
        )
        self.assertEqual(plan["replay"]["verdict_output_tokens"], 16384)

    def test_conversation_profile_is_canonical_at_defaults(self):
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_PROFILE": "conversation"}, clear=False
        ):
            self.assertTrue(Settings.from_env().canonical)
        with mock.patch.dict(
            os.environ,
            {"AUDIT_BENCH_PROFILE": "conversation",
             "AUDIT_BENCH_CONV_TOOL_TOKENS": "512"},
            clear=False,
        ):
            self.assertFalse(Settings.from_env().canonical)

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

    def test_output_prompts_require_external_truncation(self):
        for prompt in (audit_prompt(1, 10), decode_prompt(10)):
            with self.subTest(prompt=prompt):
                self.assertIn("deliberately truncated by the benchmark", prompt)
                self.assertIn("Continue for at least 4096 tokens", prompt)
                self.assertIn("never conclude", prompt)

    def test_conversation_corpus_carries_no_decode_control(self):
        clean = audit_prompt(1, 10, output_control=False)
        self.assertNotIn("BENCHMARK CONTROL", clean)
        self.assertNotIn("never conclude", clean)

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
        evidence_lines = max(prompt.count("\n") - 2, 0)
        return RequestResult(
            prompt_tokens=100 + evidence_lines * 20,
            output_tokens=max_tokens,
            seconds=1,
        )

    _FAKE_ENV = {
        "AUDIT_BENCH_STREAMS": "1,2",
        "AUDIT_BENCH_DEEP_STREAMS": "1",
        "AUDIT_BENCH_SAMPLES": "1",
        "AUDIT_BENCH_OUTPUT_TOKENS": "8",
        "AUDIT_BENCH_DEEP_OUTPUT_TOKENS": "4",
        "AUDIT_BENCH_TARGET_TOLERANCE": "0.02",
        "AUDIT_BENCH_DRY_RUN": "0",
    }

    def _run_fake_protocol(self, extra_env: dict) -> tuple[int, str]:
        output = io.StringIO()
        with mock.patch.dict(os.environ, {**self._FAKE_ENV, **extra_env}, clear=False):
            with contextlib.redirect_stdout(output):
                result = run_benchmark(
                    runtime="fake",
                    tag="fake-model",
                    model="fake-model",
                    context=131072,
                    request=self._fake_request,
                    active_slots=1,
                    kv_pool_tokens=131072,
                )
        return result, output.getvalue()

    def test_fake_adapter_runs_the_complete_protocol(self):
        result, text = self._run_fake_protocol({"AUDIT_BENCH_GATE_CMD": ""})
        self.assertEqual(result, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["schema_version"], 4)
        self.assertEqual(report["protocol"], "audit-v1")
        self.assertFalse(report["publishable"])
        self.assertIn("capacity_near", report)
        self.assertEqual(list(report["decode"]["by_streams"]), ["1"])

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
            # 131k plan with one slot: 2 prefill targets, 1 capacity batch,
            # 1 decode batch, 3 audit batches; warmup and calibration are
            # direct requests and must not be gated.
            gates = marker.read_text(encoding="utf-8").count("gate")
            self.assertEqual(gates, 7)
            self.assertEqual(report["settings"]["gate_cmd"], f"echo gate >> {marker}")
        with mock.patch.dict(
            os.environ, {"AUDIT_BENCH_GATE_CMD": "true"}, clear=False
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
            os.environ, {"AUDIT_BENCH_SYSTEM_PROMPT": control}, clear=False
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
            os.environ, {"AUDIT_BENCH_REASONING_EFFORT": "none"}, clear=False
        ):
            self.assertTrue(Settings.from_env().canonical)

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


class CliMainTest(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        output = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False):
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
            ]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("note:", text)
        report = json.loads(text.splitlines()[-1])
        metadata = report["metadata"]
        self.assertEqual(metadata["model"], "unspecified")
        self.assertEqual(metadata["active_slots"], 4)
        self.assertEqual(metadata["kv_pool_tokens"], 524288)

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
            ]
        )
        self.assertEqual(code, 0)
        report = json.loads(text.splitlines()[-1])
        self.assertEqual(report["settings"]["profile"], "conversation")
        self.assertTrue(report["canonical"])

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

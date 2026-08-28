# llm-audit-bench

A test suite for local LLM serving setups under realistic security-audit
workloads. Two instruments, one target: any OpenAI-compatible endpoint.

- **AuditBench v1** (`audit-bench`) measures serving performance and
  capacity: prefill curve, near-context capacity proof, decode ceiling,
  end-to-end audit calls, and audit-shaped conversation and replay
  profiles. Spec: [docs/PROTOCOL.md](docs/PROTOCOL.md).
- **Capability Check v1** (`capability-check`) is the pass/fail API
  compatibility gate: streaming reasoning, thinking-disabled control, tool
  call, tool round trip, and structured JSON. Spec:
  [docs/CAPABILITY.md](docs/CAPABILITY.md).

Neither scores answer quality. Benchmark rates and capability verdicts are
different facts; keep them separate.

## What the endpoint must provide

- An OpenAI-compatible `/v1/chat/completions` endpoint (Capability Check
  also speaks native Ollama `/api/chat` with `--api ollama`).
- Server-reported token usage (`usage.prompt_tokens`,
  `usage.completion_tokens`). Rates are measured from server counts, never
  estimated client-side.
- Exact `max_tokens` compliance. Short responses are rejected, never
  padded.
- Tool calling, for the conversation and replay profiles and for the
  capability gate.

You supply the facts the endpoint cannot state about itself: the configured
per-request context, and ideally the scheduler admission (`--active-slots`)
and raw KV pool size (`--kv-tokens`). Without them the harness assumes one
slot and one context of KV and measures single-stream only.

## Install

Runtime code is standard-library Python only (3.10 or newer).
[uv](https://docs.astral.sh/uv/) manages the development environment
and installs the CLIs:

```bash
make dev
```

## Run

```bash
# Serving benchmark (full profile); model auto-resolved when the endpoint
# serves exactly one
uv run audit-bench --base-url http://127.0.0.1:8000 --context 131072

# A tuned operating point, explicitly
uv run audit-bench --base-url http://gpu-host:8000 --model my-model \
  --context 262144 --active-slots 4 --kv-tokens 1048576

# Audit-shaped profiles
uv run audit-bench --profile conversation \
  --base-url http://127.0.0.1:8000 --context 131072

# Inspect the workload plan without contacting a server
uv run audit-bench --base-url http://127.0.0.1:8000 \
  --context 131072 --dry-run

# Compatibility gate; state the model's documented expectations
uv run capability-check --base-url http://127.0.0.1:8000 \
  --model my-model --reasoning required --tools required
```

## Benchmark profiles

`--profile` selects the audit-bench workload (default `full`; the
`AUDIT_BENCH_PROFILE` environment variable works too):

| Profile | What it measures | When to use it |
| -- | -- | -- |
| `full` | The complete serving picture: prefill curve (32k/64k/128k prompts), near-context capacity proof, decode ceiling at 1/2/4/8 streams, and end-to-end audit calls at 32k, half context, and 7/8 context | The canonical headline result for a tuned operating point |
| `decode` | Aggregate decode throughput alone, from a short control prompt | Tuning only: locating the host's decode knee and scheduler admission before freezing the operating point; never canonical |
| `conversation` | A model-driven audit tool loop over chained evidence files, ending in a verdict, with measured prefix-cache reuse | Model-usability smoke: does the model follow the tool-loop protocol at all; run it before committing a model to real audits |
| `replay` | The same growing conversation with every turn scripted by the harness | Pure server performance: prefill growth at depth, cache-on versus cache-off cost, verdict decode under a fat KV cache; works even when the model cannot hold the protocol |

`conversation` and `replay` require tool calling and report different
quantities: conversations/hour says whether the model plays along,
replays/hour says whether the server performs. Never compare the two.
`capability-check` is a separate command, not a profile.

### Why they matter

Short-prompt decode is the first number everyone measures, and the least
informative one for audit work. One real operating point -- dense Qwen3.8
27B serving a 262,144-token context on an 80 GB datacenter GPU; rounded
and illustrative, retained results live with the serving-configuration
repositories:

| Request shape | Profile | Measured | Effective tok/s |
| -- | -- | -- | --: |
| Short prompt, 1,024 tokens out | `decode` | ~21 s per request | 48 |
| One 32k-prompt audit call, 1,024 tokens out | `full` (`audit_32k`) | 104 calls/hour; 35 s per call, prefill about a third of it | 29 |
| One 229k-prompt audit call (7/8 context), 256 tokens out | `full` (`audit_deep`) | 23 calls/hour; 154 s per call, roughly 150 s of it prefill | 1.7 |
| Audit conversation replayed, prefix cache off | `replay` | 845 s per replay; per-turn prefill grows 12.8 s to 84.8 s | 1.3 |
| Same replay, prefix cache on (0.90 cached) | `replay` | 138 s per replay, flat 4-7 s turns: 6.1x end to end | 7.7 |

Effective tok/s is delivered output over total wall, prefill included; the
replay rows count only the ~1.1k-token verdict as output, since probe
turns are harness scaffolding. It is not a decode speed -- the decoder
runs at full rate for the last seconds of every one of these calls -- but
it is the throughput the caller waiting on the answer actually gets.

Three conclusions the headline number hides:

- The collapse from 104 to 23 calls/hour is pure prefill. The deep call
  spends roughly 150 of its 154 seconds processing the 229k-token
  prompt before decoding anything; its output is short by design (a
  late-audit response), and making it longer would barely move the
  wall. Long outputs at depth are measured where they belong: the
  replay verdict forces a verdict-sized decode, up to its 16k cap, at
  full conversation depth. Decode rate says nothing about any of this,
  which is why the audit phases report calls/hour as the production
  metric.
- Only `replay` can price caching. Every single-shot phase is
  deliberately cache-neutral, yet the resend-everything conversation
  shape is where an audit server lives or dies: here the cache turned a
  14-minute audit into 2.3 minutes on the same hardware.
- `conversation` catches what no rate does. On this same host, another
  model posted healthy replay numbers (0.90 cached, a clean 16k-token
  verdict decode) and still failed the conversation outright, reasoning
  itself into a runaway and recalling 0 of 16 findings. On a 12 GB
  laptop host, Qwen3.5 4B was 2/2 compliant with the cache off and 0/2
  with it on. Server performance and model usability are different
  verdicts, and the benchmark refuses to blend them.

## Output

Both commands print a human report followed by one JSON document.
Capability Check also writes the JSON artifact under
`${XDG_STATE_HOME:-~/.local/state}/capability-check/`. All `AUDIT_BENCH_*`
protocol variables (profile, streams, samples, gate command, and the
model-documented reasoning control) work unchanged; see
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Reading the results

A benchmark result is `canonical` when the workload used the fixed protocol
settings, and separately `publishable` when it also carries full managed-run
provenance (clean commit, file hashes, verified launch profile and model
identity). Standalone CLI runs against an unmanaged endpoint are honest but
never publishable; `publication_reasons` says exactly why. Details in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Contributing

Development workflow, hard invariants, and the protocol-change procedure
are in [CONTRIBUTING.md](CONTRIBUTING.md); the same rules stated for AI
coding agents are in [AGENTS.md](AGENTS.md). Runtime adapters for managed
servers, their model configurations, and all measured results live in the
serving-configuration repositories, not here.

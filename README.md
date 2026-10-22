# llm-audit-bench

A test suite for local LLM serving setups under realistic security-audit
workloads. Two instruments, one target: any OpenAI-compatible endpoint.

- **AuditBench v2** (`audit-bench`) measures serving performance and
  capacity: a llama-bench-style pp512/tg128 reference, a near-context
  capacity proof, then scripted agent sessions at the model's own sampling
  -- sessions/hour, and prefill and decode rates as the context grows --
  plus a model-driven conversation profile for usability. Spec:
  [docs/PROTOCOL.md](docs/PROTOCOL.md).
- **Capability Check v2** (`capability-check`) is the pass/fail model gate:
  streaming reasoning, thinking-disabled control, tool calls in
  non-streaming and streaming modes, tool round trip, structured JSON with a
  measured unaided document, and audit fitness: audit-shaped tasks at the
  declared sampling, judged against ground truth. It also records, without
  gating, whether the server enforces a JSON schema or admits it cannot. Spec:
  [docs/CAPABILITY.md](docs/CAPABILITY.md).

AuditBench does not score answer quality, and audit fitness never ranks
models: it passes or fails them. Benchmark rates and capability verdicts are
different facts; keep them separate.

## What the endpoint must provide

- An OpenAI-compatible `/v1/chat/completions` endpoint (Capability Check
  also speaks native Ollama `/api/chat` with `--api ollama`).
- Server-reported token usage (`usage.prompt_tokens`,
  `usage.completion_tokens`), including in streamed responses
  (`stream_options.include_usage`). Rates are measured from server counts,
  never estimated client-side.
- Exact `max_tokens` compliance on the fixed-length calls (the 1-token
  gates and the 128-token reference). A short response is rejected, never
  padded.
- Tool calling, for both profiles and for the capability gate.

You supply the facts the endpoint cannot state about itself: the configured
per-request context, the checkpoint's sampling (`AUDIT_BENCH_SAMPLING`),
and ideally the scheduler admission (`--active-slots`) and raw KV pool size
(`--kv-tokens`). Without the capacity facts the harness assumes one slot and
one context of KV and measures single-stream only; without sampling the
result is not canonical and Capability Check audit fitness fails.

## Install

Runtime code is standard-library Python only (3.10 or newer).
[uv](https://docs.astral.sh/uv/) manages the development environment
and installs the CLIs:

```bash
make dev
```

## Run

```bash
# The checkpoint's sampling, from its generation_config.json
export AUDIT_BENCH_SAMPLING="temperature=0.6 top_p=0.95 top_k=20"

# Serving benchmark (session profile); model auto-resolved when the
# endpoint serves exactly one
uv run audit-bench --base-url http://127.0.0.1:8000 --context 131072

# A tuned operating point, explicitly
uv run audit-bench --base-url http://gpu-host:8000 --model my-model \
  --context 262144 --active-slots 4 --kv-tokens 1048576

# Model-usability check
uv run audit-bench --profile conversation \
  --base-url http://127.0.0.1:8000 --context 131072

# Inspect the workload plan without contacting a server
uv run audit-bench --base-url http://127.0.0.1:8000 \
  --context 131072 --dry-run

# Model gate; state the model's documented expectations and sampling
uv run capability-check --base-url http://127.0.0.1:8000 \
  --model my-model --reasoning required --tools required \
  --sampling "$AUDIT_BENCH_SAMPLING"
```

## Benchmark profiles

`--profile` selects the audit-bench workload (default `session`; the
`AUDIT_BENCH_PROFILE` environment variable works too):

| Profile | What it measures | When to use it |
| -- | -- | -- |
| `session` | The llama-bench pp512/tg128 reference, the near-context capacity proof, then scripted agent sessions at 1 stream and the largest stream count the operating point admits: 32k of evidence, up to 16 tool results bounded by 7/8 of context, the model's natural turns, and a final report | The canonical serving result for a tuned operating point |
| `conversation` | A model-driven audit tool loop over the same evidence, ending in a verdict, with measured prefix-cache reuse | Model usability: does the model follow the tool-loop protocol at all; run it before committing a model to real audits |

Both profiles need tool calling. Sessions/hour says whether the server
performs; conversations/hour and compliance say whether the model plays
along. Never compare the two. `capability-check` is a separate command, not
a profile.

### Why this shape

An agent never calls a model with a short prompt: a session starts tens of
thousands of tokens deep and grows with every tool result, and decode speed
falls as it grows. The session measures the real call pattern --
resend-everything turns, prefix-cache reuse, decode at depth, the model's
own sampling, many agents at once -- and reports the decode rate at the
start, middle, and end of the context. The reference pp512/tg128 line exists
only for comparison with published llama-bench figures.

## Output

`audit-bench` prints one progress line per sample batch while it runs, then
a human-readable summary by default: the reference line, the near-context
capacity proof, the session table (sessions/h, complete sessions, decode
first/mid/last, report decode, cached), health lines when non-zero, and the
`canonical` and `publishable` verdict. The complete JSON document is
retained on disk regardless, under
`${XDG_STATE_HOME:-~/.local/state}/auditbench/`, and the summary's last
line names the file. `--format json` prints that same document on stdout
instead, for tools that parse it.

`capability-check` prints the same shape: a human verdict listing every
check, its status, and any failure reason, plus one line per audit fitness
task with its failure classes, with the retained JSON under
`${XDG_STATE_HOME:-~/.local/state}/capability-check/` and `--format json`
for tools.

Either command lists what it has retained with `--list`, and re-prints one
with `--show`, contacting no server. `--show` takes the file name straight
from the listing, resolved in the retention directory, or any path:

```bash
uv run audit-bench --list
uv run audit-bench --show 20260926T125150.571598+0000_endpoint_host_session.json
uv run capability-check --list
uv run capability-check --show ./archived/checked.json
```

All `AUDIT_BENCH_*` protocol variables (profile, streams, samples,
sampling, gate command, and the model-documented reasoning control) work
unchanged;
`AUDIT_BENCH_OUTPUT_DIR` overrides the retention directory. See
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Reading the results

A benchmark result is `canonical` when the workload used the fixed protocol
settings, and separately `publishable` when it also carries full managed-run
provenance (clean commit, file hashes, verified launch profile and model
identity) and, for a measured run, complete sessions or a compliant
conversation. Standalone CLI runs against an unmanaged endpoint are honest but
never publishable; `publication_reasons` says exactly why. Details in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Contributing

Development workflow, hard invariants, and the protocol-change procedure
are in [CONTRIBUTING.md](CONTRIBUTING.md); the same rules stated for AI
coding agents are in [AGENTS.md](AGENTS.md). Runtime adapters for managed
servers, their model configurations, and all measured results live in the
serving-configuration repositories, not here.

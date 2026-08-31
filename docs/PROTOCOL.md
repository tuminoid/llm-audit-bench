# AuditBench v1

AuditBench v1 is the canonical serving benchmark for models used by the
non-interactive security-audit workload. It replaces runtime-specific prompt
sizes with one protocol shared by every measured runtime. The harness in this
repository (`audit_bench.py`) owns every workload shape, validation rule, and
metric; runtime adapters supply only transport, capacity facts, and identity
evidence.

The benchmark measures serving performance and capacity. It does not score
answer quality. A model still needs separate tool-call, reasoning, and audit
correctness gates before it belongs in the serving inventory.

## Workload model

The production workload consists of hundreds of independent audits. A request
typically begins with about 32k tokens of repository evidence, generates a long
reasoning or tool response, and may return later with a much larger context.
Several audits can be active concurrently for days.

The benchmark therefore measures all of these separately:

- common absolute prompt sizes for fair host and runtime comparisons;
- capacity-relative prompts for the middle and end of a long audit;
- isolated decode throughput and end-to-end audit-call throughput;
- canonical 1, 2, 4, and 8 stream points that fit the operating point; and
- near-ceiling requests at 1x and the maximum fitting canonical stream count,
  proving the configured context and aggregate capacity really work.

## Canonical settings

| Setting | Canonical value |
| -- | --: |
| Minimum configured context | 131,072 |
| Preferred configured context | at least 262,144 |
| Profile | `full` |
| Measured samples | adaptive: 2, extended toward 3 on disagreement |
| Sample agreement | 5% wall-time spread |
| Prompt target tolerance | 2% |
| Candidate decode streams | 1, 2, 4, 8 |
| Candidate deep-context streams | 1, 2, 4 |
| Normal output | 1,024 tokens per request |
| Deep-context output | 256 tokens per request |
| Near-ceiling target | 15/16 of configured context |
| Late-audit target | 7/8 of configured context |
| Temperature | 0 |

Each sample uses different deterministic audit evidence from the first token,
so independent requests cannot collapse into a shared prefix-cache hit. Stream
counts are interleaved across sample passes to reduce thermal and power
ordering bias. Results report the median and retain the minimum, maximum, and
raw samples.

### Adaptive sampling

Sampling was originally a fixed 3 per batch. With the 140 W
adapter-contract gate and battery floor enforced before every batch, the
systematic variance those repetitions guarded against is prevented rather
than averaged out, so the canonical protocol now runs 2 samples and
extends toward 3 only on disagreement. Two samples agree when their batch
wall times lie within the 5% agreement spread; every reported rate of a
fixed-shape batch derives from its wall time, so that one criterion
covers them all. A disagreement automatically restores median-of-3 for
the affected batch -- in stream-count families, one disagreeing count
extends the whole family by a full interleaved pass, keeping the ordering
symmetric. The near-context capacity proof still runs once.

The two agreeing samples keep the internal consistency check that fixed
sampling provided: a contaminated or degrading sample surfaces as
disagreement and triggers the third measurement instead of silently
becoming canonical. Existing median-of-3 rows remain valid and
comparable; no remeasurement is implied.

`AUDIT_BENCH_MIN_SAMPLES` and `AUDIT_BENCH_SAMPLE_AGREEMENT` configure
the initial count and the spread; `AUDIT_BENCH_SAMPLES` remains the
maximum. `AUDIT_BENCH_MIN_SAMPLES=3` restores fixed median-of-3, and like
any changed canonical setting it marks the run `canonical: false`.

Every runtime receives one user chat message through its native chat endpoint.
Raw text-generation endpoints are not comparable because they omit the model's
production chat template.

The control instruction asks the model to continue beyond the measured output
limit so runtimes without an `ignore_eos` option still reach the fixed token
count. A response that ends early is rejected; it is never padded, retried
inside a sample, or reported as a shorter equivalent workload.

### Model-documented decode-compliance controls

Some strongly preference-tuned models deliberate at their default reasoning
effort and then decline the synthetic control workload instead of complying
(first observed: Muse Glimmer 30B, which at default effort answered the
never-conclude instruction with an offer to "generate a reasonable excerpt",
then stopped). For such a model, pinning the vendor's **documented**
reasoning-effort control -- for Muse Glimmer, a `Reasoning strength: low`
system message -- is permitted configuration, not a workload change:

- Only the vendor's documented mechanism qualifies. Ad-hoc prompt
  engineering, jailbreak phrasing, or edits to the workload text itself
  disqualify the run.
- The workload text stays byte-identical; the control rides in the system
  role (or the runtime's equivalent documented parameter).
- The setting is recorded in the run's JSON settings --
  `AUDIT_BENCH_SYSTEM_PROMPT` for the system-role control,
  `AUDIT_BENCH_REASONING_EFFORT` for a documented parameter -- and
  stated in the benchmark document entry. It does not mark the
  run `canonical: false` -- models already differ in their template-default
  reasoning behavior inside the same token budgets, and this control pins
  that same axis explicitly.
- A model that cannot reach the fixed token count even under its documented
  control is recorded as protocol-incompatible, not bench-marked with a
  shorter workload.

The system-role path above is the original mechanism. The parameter path
(`AUDIT_BENCH_REASONING_EFFORT`, sent as the request `reasoning_effort`
field) was added for runtimes whose documented control is a parameter --
LM Studio, vLLM, and OpenAI all document `reasoning_effort`, and strongly
reasoning-default models such as Qwen3 need it to emit a non-empty answer
at all. The workload text and the canonical settings table are unchanged,
and the control does not affect `canonical`, so existing results remain
comparable; the only output change is a new `settings.reasoning_effort`
key, hence `SCHEMA_VERSION` 4.

Before measuring, each adapter verifies the live model identity, configured
context, and active slot count where the runtime exposes them. A forgotten
startup override therefore fails rather than producing a mislabeled result.

Changing a canonical setting marks the JSON result `canonical: false`. That is
useful for tuning, but it is not a publishable cross-host result.

## Result provenance and publication

AuditBench JSON has a schema version and keeps workload canonicality separate
from publication readiness:

- `canonical` means the workload used the fixed protocol settings above.
- `publishable` additionally requires a non-dry run from a clean identified
  repository commit, hashes for the shared harness, runtime adapter, and model
  config, a checked live model identity, and a checked resolved runtime
  profile.
- `publication_reasons` lists every unmet condition. A dirty checkout can
  therefore run a canonical experiment without being mislabeled as a retained
  result.

The adapter hashes the files it actually executes, including deployed copies
on remote hosts. Runtime profiles retain both parsed fields and the original
launcher lines: capacity, scheduler admission, memory policy, attention or
runner selection, and runtime-specific acceleration state. Missing live
profile evidence is an error for a real managed run; dry runs record unchecked
plans and are never publishable.

## Phases

Let `C` be the configured per-request context.

### Prefill curve

Single-stream prefill generates one output token at these prompt targets:

| Target | Inclusion rule | Purpose |
| -- | -- | -- |
| 32,768 | always | initial audit prompt |
| 65,536 | always | common medium prompt |
| 131,072 | when no more than 7/8 of `C` | common large prompt |

The harness calibrates the deterministic corpus against the model tokenizer,
then rejects a measured prompt more than 2% from its target.

### Near-context capacity

Prompts target `15C/16` with one output token. The benchmark runs one request
and, when different, the highest canonical stream count that fits both active
admission and the raw KV pool. Every stream must land between 90% and 100% of
`C`. The proof runs once because it is a capacity gate, not a performance
sample. An operating point that holds only one near-full request runs only 1x.

### Decode ceiling

A short control prompt generates exactly 1,024 tokens at each canonical
1/2/4/8 point that fits active admission and the KV pool. This isolates
aggregate decode capacity from long-prefill cost. It is the metric used while
tuning scheduler admission and finding a host's decode knee.

### Audit conversation

`AUDIT_BENCH_PROFILE=conversation` runs the one workload shape the phases
above deliberately cannot measure: the production secaudit call pattern.
Every phase above is a single-shot request whose samples differ from the
first token precisely so prefix caches cannot contaminate rates. A real
audit client does the opposite -- it resends the whole growing
conversation on every step, with tool results landing mid-context, so
prefix reuse is the workload itself.

The audit-shaped workload is split into two profiles that must never be
conflated:

- **conversation** (this section) is the model-usability smoke: the
  model itself drives the tool loop and must stop with a verdict.
  Non-compliance here means the model deviated from the fixture
  protocol -- it is not evidence about server performance, and it does
  not prove the model unusable in production, whose prompting differs
  from this fixture. Run it before committing a model to real audits so
  harness failures are not debugged against a model that produces crap.
- **replay** (next section) is the pure server-performance benchmark:
  the harness scripts every turn, so every model on every runtime
  produces the identical resend-everything shape and a protocol-breaking
  model still yields prefill and cache numbers.

One conversation: a system instruction plus `evidence_tokens` of
deterministic audit evidence, then a tool loop over a chain of `steps`
evidence files, ending in a plain-text verdict turn at depth. Three
fixture properties were each forced by observed model behavior:

- File names are revealed one at a time -- each tool result names the
  next file -- because a model given all names upfront batches many
  reads per turn (qwen3.6 read eight at once), collapsing the resend
  structure the profile exists to measure. Real audits are sequential
  because each result informs the next action; the chain encodes that
  honestly. The harness still answers every call the model makes.
- Each file carries exactly one extractable `FINDING:` line and the
  verdict is the collected ledger, because models given semantically
  empty filler to "analyze" reason unboundedly (16k-token think spirals
  at temp 0). The bounded task keeps per-turn work realistic, and the
  verdict's findings recall is recorded as a free correctness signal.
- The conversational corpus carries NO embedded decode-control line
  (`audit_prompt(..., output_control=False)`): the single-shot phases
  embed a never-conclude instruction to hold output lengths, and a
  model that obeys it inside conversational evidence derails the tool
  loop entirely (observed: qwen3-coder emitting endless integers
  mid-audit). An instruction in a reused corpus is part of the workload
  everywhere the corpus goes.

The file count auto-scales so the final conversation stays within 7/8 of
the configured context, capped at `conv_max_steps`. Context fit reserves
only resent tokens (tool results plus a short note per turn): reasoning
tokens cost wall time but are never resent, so the output value is a
per-request cap sized for deep-context reasoning bursts, not a per-turn
context cost. Samples (adaptive, as above) use distinct evidence from
the first token, so a cache can only reuse a conversation's own prefix
-- exactly what production sees.

| Setting | Canonical value |
| -- | --: |
| `AUDIT_BENCH_CONV_EVIDENCE` | 32,768 |
| `AUDIT_BENCH_CONV_TOOL_TOKENS` | 8,192 |
| `AUDIT_BENCH_CONV_OUTPUT_TOKENS` | 16,384 |
| `AUDIT_BENCH_CONV_MAX_STEPS` | 16 |

Reported metrics: conversation wall seconds and conversations/hour
(median plus range, over compliant samples), per-turn
wall/prompt/cached/completion tokens, and the median cached-token
fraction taken from the server's own `prompt_tokens_details.cached_tokens`
usage field where the runtime reports it -- cache reuse is measured,
never inferred (the launchers pass `--enable-prompt-tokens-details`
alongside the APC opt-in for exactly this). A conversation is compliant
when every turn before the verdict makes at least one tool call, every
file is read within `steps + 4` turns, and the verdict turn returns
non-empty text with no tool calls. Non-compliance is recorded, never
padded around. The profile needs the runtime's OpenAI-compatible chat
endpoint (all three runtimes expose one); reasoning fields are not
resent, matching production clients.

Conversation results are canonical at the defaults above and are
recorded in their own section, never mixed with the cache-neutral phase
rates. The server's resolved prefix-caching state is part of the result:
a conversation number without its cache mode is meaningless.

When reading a non-compliant sample, its `per_step` records say exactly
which deviation happened -- check the last turn before concluding
anything: a verdict turn with `files_remaining: 0` and non-empty text is
a pass regardless of how few turns it took (batched reads are supported
and answered); `finish_reason: "length"` with non-empty text is still
compliant (truncation only hurts recall); an empty `content_excerpt` on
a reasoning model usually means the verdict landed in a reasoning
channel, a harness-visible limitation rather than model incapacity;
only a verdict with files remaining, or tool calls until the `steps + 4`
turn budget runs out, is genuine protocol non-compliance.

### Audit replay

`AUDIT_BENCH_PROFILE=replay` is the performance half of the audit pair:
the same fixture, evidence sizes, file chain, and context-fit math as
the conversation profile, but the harness drives every turn itself.
Each turn it calls the model with the conversation so far (that call is
the measurement), caps the probe output at 32 tokens, discards whatever
the model produced, and appends its own scripted assistant note plus
`read_file` call and the tool result. The prefix therefore grows
identically for every model on every runtime, and there is no
compliance concept at all: a model that cannot hold the interactive
protocol still yields per-turn prefill walls and measured cache reuse.

After the last file, a final user turn carries the standard
never-conclude decode control and the full
`AUDIT_BENCH_CONV_OUTPUT_TOKENS` cap, forcing a verdict-sized decode at
depth. Its wall time is the end-to-end cost of producing a verdict on
this server; the recorded `verdict_completion_tokens` and finish reason
expose a model that ignored the control and decoded less, keeping rates
comparable.

Reported metrics: replay wall seconds and replays/hour, first- and
final-turn prefill walls (the growth curve's endpoints), verdict output
tokens/second, and the median cached-token fraction -- all median plus
range under the same adaptive sampling. Replay results are canonical at
the conversation defaults above and answer "where does the server fail
to perform" -- cache-on versus cache-off pricing, prefill growth at
depth, verdict decode under a fat KV. Use conversation numbers for
whether a model plays along, replay numbers for whether the server
performs; conversations/hour and replays/hour are different quantities
(probe turns decode almost nothing) and must never be compared across
the two profiles.

### Concurrent audit replay

`AUDIT_BENCH_PROFILE=replay-concurrent` measures the multi-agent audit
shape: N scripted replays running at once against one server, sharing
the evidence prefix and competing for the same KV pool. Within one
sample batch every stream carries the identical system message,
evidence, and opening turn -- concurrent agents working the same
repository -- so a prefix cache may share their evidence; each stream's
file chain diverges from its first tool result, so the growing suffixes
are distinct work. Across samples and stream counts the evidence
changes from the first token, keeping every batch cold at start.

Two deliberate differences from the 1x replay profile:

- Probe turns decode a realistic agent turn instead of a 32-token
  prefill probe: the measured call appends an ephemeral decode-control
  user message and caps output at `AUDIT_BENCH_CONV_TURN_TOKENS`
  (canonical 1,024, the protocol's normal output size). The control
  never enters the transcript and the probe output is still discarded,
  so the resent prefix stays byte-identical across models and runtimes.
  Recorded per-turn completion tokens expose a model that ignores the
  control instead of flattering its rate.
- Streams start together on a barrier and then free-run their own turn
  loops with no per-turn synchronization, so natural desync develops
  exactly as independent agents drift apart. First-turn cached tokens
  are recorded per stream: whichever stream commits the shared evidence
  first seeds the cache for the rest, and partial first-turn sharing
  under a simultaneous start is a real property of agents starting
  together, measured rather than staggered away.

Candidate stream counts are the deep-context candidates (1, 2, 4).
Fitting counts are conservative: per-stream KV demand is the final
resent conversation plus the verdict cap, with no shared-prefix credit,
so admission never depends on cache behavior and counts above the bound
are never submitted.

| Setting | Canonical value |
| -- | --: |
| `AUDIT_BENCH_CONV_TURN_TOKENS` | 1,024 |
| Candidate streams | 1, 2, 4 |
| Other settings | as audit conversation |

Reported metrics per stream count: batch wall seconds, aggregate
replays/hour (streams times 3600 over batch wall), scaling versus the
in-family 1x, first- and final-turn walls, verdict output
tokens/second, median per-turn completion tokens, median cached
fraction, and per-stream first-turn cached tokens. The in-family 1x
point decodes full turns, so it is a different quantity from the 1x
replay profile: never compare replays/hour across the two. As with
replay, a result without the server's resolved prefix-caching state is
meaningless.

Amendment (2026-08-29): the replay-concurrent profile was added
alongside the existing profiles, which are unchanged; existing replay
and conversation results remain valid and comparable. `SCHEMA_VERSION`
was bumped to 3 for the new `audit_replay_concurrent` result section
and `replay_concurrent` plan block.

### End-to-end audit calls

These phases include both prompt processing and generation in wall time:

| Phase | Prompt target | Output | Streams |
| -- | --: | --: | -- |
| `audit_32k` | 32,768 | 1,024 | fitting canonical 1/2/4/8 points |
| `audit_half` | `C/2` | 1,024 | 1 |
| `audit_deep` | `7C/8` | 256 | 1 and maximum fitting candidate |

`audit_32k` is the common-work comparison across every qualifying model.
`audit_half` measures a long-running audit in the middle of its context.
`audit_deep` measures the end of a request at 1x and the highest useful
concurrency without queueing.

For every phase, fitting concurrency is:

```text
min(active slots, floor(KV pool tokens / per-stream KV demand))
```

Per-stream demand includes the target prompt, requested output, and a small
variant-token reserve. Stream counts above that bound are never submitted.
Ollama's capacity is its allocated context multiplied by effective runner
slots; vLLM uses the raw pool size printed by the live engine.

## Metrics

| Metric | Meaning |
| -- | -- |
| Aggregate prompt tok/s | Prompt tokens divided by batch wall time |
| End-to-end output tok/s | Generated tokens divided by total batch wall time |
| Calls/hour | Completed fixed-shape requests divided by batch wall time |
| Scaling versus 1x | Median end-to-end output rate divided by the 1x median |
| Peak streams | Submitted stream count with the highest median output rate |

For `decode`, end-to-end output tok/s approximates pure decode because the
prompt is tiny. For the three audit phases it deliberately includes prefill:
calls/hour is the primary production metric.

Raw KV-pool size, scheduler admission, and per-request context remain separate
facts. A vLLM KV pool can exceed one request's maximum and is shared among
active requests, so no workload is defined as a percentage of raw pool tokens.

## Tuning before a canonical run

1. Set the highest reliable per-request context. Prefer the model's native
   context and at least 262,144 tokens; retain 131,072 only when the model or
   hardware cannot serve more.
2. Raise the memory budget until the near-ceiling request and teardown remain
   reliable with the required host reserve.
3. Locate the decode knee with
   `AUDIT_BENCH_PROFILE=decode AUDIT_BENCH_STREAMS=1,2,3,4,5,6,7,8`. Tune
   scheduler admission or Ollama parallel slots, then freeze that operating
   point.
4. Run the canonical protocol. It automatically omits stream counts that
   cannot execute concurrently for each workload.
5. Record the date, repository commit, model revision, runtime versions, host,
   accelerator, power limit, context, KV policy, memory budget, active slots,
   pool or allocation, and the final AuditBench JSON.

An exploratory stream sweep is intentionally noncanonical. Headline tables use
the canonical stream counts so results remain comparable after tuning.

## Running it

The standalone CLI drives any OpenAI-compatible chat endpoint. Supply the
configured context and, when known, the real operating point:

```bash
uv run audit-bench --base-url http://127.0.0.1:8000 --context 131072
uv run audit-bench --base-url http://gpu-host:8000 --model my-model \
  --context 262144 --active-slots 4 --kv-tokens 1048576
uv run audit-bench --profile conversation \
  --base-url http://127.0.0.1:8000 --context 131072
```

`--model` is resolved from `/v1/models` when the endpoint serves exactly one
model. Without `--active-slots` and `--kv-tokens` the harness assumes one
slot and one configured context of KV, so concurrency phases run 1x only.
Every `AUDIT_BENCH_*` variable in this document applies to the CLI
unchanged.

Managed runtimes run through their own adapters and make targets in the
serving-configuration repositories, which add capacity discovery from live
server logs, launch-identity verification, and remote execution.

Inspect a plan without contacting a server:

```bash
uv run audit-bench --base-url http://127.0.0.1:8000 \
  --context 131072 --dry-run
```

A dry run conservatively assumes one slot and one configured context of KV.
Supply `--active-slots` and `--kv-tokens` to inspect another measured
operating point exactly.

Adapters must never mislabel requested concurrency as active concurrency:
the Ollama adapter takes its effective parallel slot count explicitly, and
architectures forced to one slot run only 1x.

A host that cannot sustain the whole campaign continuously -- a battery-fed
laptop whose charger under-supplies the benchmark load, for example -- may set
`AUDIT_BENCH_GATE_CMD`. The harness runs that command before every sample
batch and proceeds when it exits zero; a non-zero exit aborts the run instead
of measuring under an unmet precondition. The gate inserts idle time between
independent batches only, so it does not change any workload or affect
canonicality; the command used is recorded in the result's settings. Record
gate pauses with the run, since a paused host resumes cooler than one that ran
straight through.

## Existing measurements

Measured results, host inventories, and operating points live with the
serving-configuration repositories, never in this one. Do not combine a
legacy prompt or decode rate with an AuditBench result, and do not derive a
missing AuditBench value from an old rate or multiplier.

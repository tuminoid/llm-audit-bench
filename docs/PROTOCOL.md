# AuditBench v2

AuditBench v2 is the canonical serving benchmark for models used by agentic
security audits and deep agentic coding sessions. It replaces
runtime-specific prompt sizes with one protocol shared by every measured
runtime. The harness in this repository (`audit_bench.py`) owns every
workload shape, validation rule, and metric; runtime adapters supply only
transport, capacity facts, and identity evidence.

The benchmark measures serving performance and capacity. It does not score
answer quality. A model also needs a Capability Check v2 pass
([CAPABILITY.md](CAPABILITY.md)) before it belongs in the serving inventory.

## Workload model

The production workload is an agent session: a system instruction, tools,
and repository evidence, then a tool loop in which every step resends the
whole growing conversation with a tool result landing mid-context, ending in
a deliverable -- an audit report or a code change. Many sessions run
concurrently against one server for days.

The benchmark measures exactly that shape, at the model's own sampling:

- the reference pp512/tg128 by llama-bench's method, for comparison with
  published figures, not a serving result;
- the near-context capacity proof, a gate that the configured context
  really serves;
- the agent session, the serving benchmark, climbing from one stream up
  the canonical stream counts the operating point admits while each one
  pays; and
- the audit conversation, the model-usability check, where the model
  drives the tool loop itself.

## Canonical settings

| Setting | Canonical value |
| -- | --: |
| Minimum configured context | 131,072 |
| Preferred configured context | at least 262,144 |
| Profile | `session` |
| Measured samples | adaptive: 2, extended toward 3 on disagreement |
| Sample agreement | 5% spread |
| Prompt target tolerance (`AUDIT_BENCH_TARGET_TOLERANCE`) | 2% |
| Candidate streams | 1, 2, 4, 8 |
| Stream gain (`AUDIT_BENCH_STREAM_GAIN`) | 10% over the rung below |
| Near-ceiling target | 15/16 of configured context |
| Sampling | declared by the model configuration |

Each sample uses different deterministic audit evidence from the first token,
so independent samples cannot collapse into a shared prefix-cache hit. The
near-context capacity proof interleaves its stream counts; the session's
rungs run in ladder order, each rung's samples back to back, because whether
a rung runs at all depends on the one below. Results report the median and
retain the minimum, maximum, and raw samples.

### Adaptive sampling

The canonical protocol runs 2 samples and extends toward 3 only on
disagreement. Each session rung extends on its own disagreement. The
near-context capacity proof runs once.

What must agree depends on the workload. A session decodes at natural
lengths under sampling, so its wall time varies with the model's own
choices; its samples agree when the prefill rate and the report decode rate
each lie within the spread. A conversation's samples agree on wall time.

`AUDIT_BENCH_MIN_SAMPLES` and `AUDIT_BENCH_SAMPLE_AGREEMENT` configure the
initial count and the spread; `AUDIT_BENCH_SAMPLES` remains the maximum.
`AUDIT_BENCH_MIN_SAMPLES=3` restores fixed median-of-3, and like any changed
canonical setting it marks the run `canonical: false`.

### Declared sampling

Live clients decode at the checkpoint's recommended sampling, so the
benchmark does too. The model configuration declares it -- from the
checkpoint's `generation_config.json` at the pinned revision, or the model
card when that file carries none -- and the adapter passes it as
`AUDIT_BENCH_SAMPLING`, space-separated `key=value` pairs:

```text
AUDIT_BENCH_SAMPLING="temperature=0.6 top_p=0.95 top_k=20"
```

Accepted keys are `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, and `repetition_penalty`. The harness sends them on
every session and conversation chat call and records them in the result's
settings. A run without a declared `temperature` measures whatever sampler
the server defaults to and is `canonical: false`. A model whose documented
default is greedy declares `temperature=0`.

The fixed-length calls (warmup, calibration, reference, near-context proof)
decode greedily with `ignore_eos`: they measure the engine, not the sampler.

### Model-documented controls

Some models grade reasoning rather than toggling it, and serve at a
documented effort level. Pinning the vendor's **documented** control is
permitted configuration, not a workload change:

- Only the vendor's documented mechanism qualifies. Ad-hoc prompt
  engineering or edits to the workload text disqualify the run.
- The workload text stays byte-identical; the control rides in the system
  role (`AUDIT_BENCH_SYSTEM_PROMPT`, a separate message ahead of the
  workload's own) or in the documented request parameter
  (`AUDIT_BENCH_REASONING_EFFORT`, sent as `reasoning_effort`).
- The setting is recorded in the run's JSON settings and stated in the
  benchmark document entry. It does not mark the run `canonical: false`.

A model whose card requires its prior reasoning back in the history
(interleaved thinking: the template re-renders each past turn's reasoning,
and the model stops reasoning when it is dropped) declares
`AUDIT_BENCH_PRESERVE_REASONING=1`. The harness then resends each
assistant turn's streamed reasoning as `reasoning` on that message: in the
session, ahead of the scripted note and call; in the conversation, with the
model's own turn. The resent reasoning grows the prefix, so the session
reserves a turn's 2,048-token cap per step and may fit fewer steps. Recorded
as `settings.preserve_reasoning`, with the same standing as the controls
above.

### Off-by-one token accounting

Some servers report `completion_tokens` one lower than they decoded when the
length cap cuts a response. LM Studio's MLX engine (mlx-llm 1.11.0) reports 0
for `max_tokens` 1, so the warmup rejects it.
`AUDIT_BENCH_TOLERATE_OFF_BY_ONE=1` accepts such a server:

- The gate stays exact. Every 1-token response must report exactly one
  fewer than requested, and must say the length cap cut it. A natural stop
  at that count, or a server that reports exact counts, still fails.
- The run is `canonical: false`, and therefore not publishable.
- The setting is recorded as `settings.tolerate_off_by_one`.

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
- A measured run is also unpublishable when the model did not do the work:
  `incomplete_sessions` when no measured stream count completed every
  session (a session is incomplete when its report hit its cap or named
  fewer than 7/8 of its findings), `noncompliant_conversation` when the
  conversation profile was not compliant. A throughput number for work
  that would be redone live is not a serving result.
- Each measured stream count is a candidate operating point, and one
  stream qualifies as much as N. Of the stream counts whose sessions all
  completed, the fastest by median sessions/hour is
  `audit_session.optimal_streams`, the operating point a result
  publishes. A stream count with an incomplete session is reported but
  never published as a serving rate.

The adapter hashes the files it actually executes, including deployed copies
on remote hosts. Runtime profiles retain both parsed fields and the original
launcher lines: capacity, scheduler admission, memory policy, attention or
runner selection, and runtime-specific acceleration state. Missing live
profile evidence is an error for a real managed run; dry runs record unchecked
plans and are never publishable.

## Output and retention

Every run retains its complete JSON document on disk before printing
anything, under `${XDG_STATE_HOME:-~/.local/state}/auditbench/` or the
directory named by `AUDIT_BENCH_OUTPUT_DIR`, as
`<started_at>_<runtime>_<tag>_<profile>.json`. The retained path is also
the document's own `artifact_path`, so a result identifies its file.
Retention is unconditional: dry runs and noncanonical runs are retained
too, marked by their own `canonical` and `publication_reasons` fields.

Stdout carries the live progress lines -- one per sample batch, plus every
sampling extension -- and then the result in one of two forms:

| Form | Selected by | Contents |
| -- | -- | -- |
| Human summary | default | Median rates per phase, the plan for a dry run, the canonicality and publication verdict, and the retained JSON path |
| JSON document | `--format json`, or `output_format="json"` for adapters | The same document that was retained, on one line, as the last line of stdout |

The summary reports medians only. Minima, maxima, every individual sample,
per-turn rows, and all provenance stay in the JSON: a summary is a reading
aid and never the retained result.

`audit-bench --show FILE` re-prints a retained document in either format,
contacting no server and retaining nothing. `FILE` is the name printed by
`--list`, resolved in the retention directory, or any path that exists as
given. It refuses a document that predates the `publishable` verdict rather
than inferring one, and names the last harness commit that renders an
`audit-v1` document instead of rendering it. It exits 0 whenever it printed.

`audit-bench --list` prints the retention directory and indexes what is in
it -- time, runtime and tag, profile, verdict, and file name, newest first --
so the retained documents are discoverable without knowing the path
convention. Retained `audit-v1` documents are listed with a `v1` profile
prefix. The index covers campaign subdirectories, names a file it cannot
parse instead of hiding it, and counts unrelated JSON in the tree rather
than presenting it as a result.

## Phases

Let `C` be the configured per-request context.

### Reference

The first step of the session profile, and the only figures comparable
with published numbers. It reproduces llama.cpp `llama-bench`'s headline
pair over the fixed-length request adapter: one stream, greedy, exact
output lengths, five repetitions reported as mean and sample standard
deviation.

| Figure | Method | Rate |
| -- | -- | -- |
| `pp512` | A cold prompt of about 512 tokens (calibration moves in whole evidence lines; the actual count is recorded), one output token, a new leading variant per repetition so no prefix is cached | prompt tokens over request seconds: a lower bound that includes HTTP and templating |
| `tg128` | A minimal prompt decoded to 128 tokens, and the same prompt to one token | 127 over the difference of the two request times: decode steps only, as llama-server's and vLLM's (N-1) over decode time, with prefill and request overhead cancelled |

`tg128` runs at a depth of a few template tokens, the closest an endpoint
gets to `llama-bench`'s empty context; `pp512` includes the first token's
forward pass, as mlx-lm and HTTP-level measurements do. Neither figure
enters sessions/hour, agreement, or publication; the step exists only to
compare an engine or quantization change with published `llama-bench`
figures. A backend that cannot force an exact output length (Ollama has no
`ignore_eos`, so a model may stop early) records the reference as
unavailable, with the reason, and the run continues.

### Near-context capacity

The second step of the session profile. Prompts target `15C/16` with one
output token. The benchmark runs one request and, when different, the
highest canonical stream count that fits both active admission and the raw
KV pool. Every stream must land between 90% and 100% of `C`. The proof runs
once because it is a capacity gate, not a performance sample. An operating
point that holds only one near-full request runs only 1x.

### Agent session

`AUDIT_BENCH_PROFILE=session`, the default, is the serving benchmark. One
session is the production call pattern with the transcript scripted by the
harness:

- The opening is the audit system instruction, the `read_file` tool, and
  `evidence_tokens` of deterministic audit evidence, which puts the first
  call above any real agent system prompt.
- A chain of `steps` evidence files follows, each revealed by the previous
  tool result. For every file the harness calls the model with the
  conversation so far, lets it write its natural turn -- reasoning, a note,
  a tool call -- capped at 2,048 tokens, and discards the output. It then
  appends its own scripted note, `read_file` call, and the tool result. The
  resent prefix therefore grows identically for every model on every
  runtime, and a model that cannot hold the protocol still yields numbers.
- The final user turn asks for the audit report, with a fix per finding.
  The report is the deliverable, decoded at its natural length under a
  16,384-token cap at full conversation depth.

No call carries any control message and the corpus carries no instruction:
turns and the report are whatever the model writes. Natural lengths are
recorded, never rejected, padded or retried -- a verbose model costs
sessions per hour live too.

The report is judged, because it is the session's deliverable. A session is
complete when the report ends on its own, not at its cap, and names at least
7/8 of the `FINDING:` tokens the chain carried: one omission in a 16-finding
report is an edit in a live audit, a short report is a redo. Incomplete
sessions are counted per stream count; a stream count with one is never
published, and a run with no complete stream count is unpublishable. Three
call-level facts are counted without gating: calls that hit their cap (long
thinking and a runaway both end there, and the count is the evidence), turns
that ended with neither text nor a tool call, and calls that streamed no
reasoning, which on a reasoning model means it answered without thinking.
Audit fitness in Capability Check v2 judges model behavior at sampling;
these counts show whether the same behavior appears at depth.

Every call streams with `stream_options.include_usage`. The first streamed
delta that carries anything but the role -- content, a reasoning field under
any runtime's name, or a tool call -- closes prefill; the rest of the call is
decode. Token counts come only from the server's usage chunk; a stream
without one, a usage chunk without token counts, or a call that reports more
than one token but streamed no delta, is an error. An end-of-turn-only answer
streams nothing and counts wholly as prefill.

The session climbs a stream ladder: the canonical candidates that fit active
admission and the KV pool, from 1 upwards. The next rung runs only when the
current one completed every session and, above 1x, its median sessions/hour
gained at least the stream gain over the rung below; otherwise the ladder
stops there and records why (`audit_session.ladder_stop`:
`gain_below_threshold`, `incomplete_sessions`, or `ladder_top` when every
admitted rung ran). A rung that dropped a session is no baseline for the
next, and concurrency that stopped paying is not worth the memory it holds.
Per-stream demand is the final resent
conversation plus the report cap, with no shared-prefix credit, so admission
never depends on cache behavior. Within one sample batch every stream
carries the identical system message, evidence, and opening turn --
concurrent agents working the same repository -- so a prefix cache may share
their evidence; each stream's file chain diverges from its first tool
result. Streams start together on a barrier and then free-run their own
loops, so natural desync develops exactly as independent agents drift
apart.

| Setting | Canonical value |
| -- | --: |
| `AUDIT_BENCH_CONV_EVIDENCE` | 32,768 |
| `AUDIT_BENCH_CONV_TOOL_TOKENS` | 8,192 |
| `AUDIT_BENCH_CONV_MAX_STEPS` | 16 |
| Turn output cap | 2,048 |
| Report output cap | 16,384 |

The file count auto-scales so the final conversation stays within 7/8 of the
configured context, capped at `AUDIT_BENCH_CONV_MAX_STEPS`; at 262,144 the
turns run from about 33k to about 160k tokens and the report at about 168k.

Reported metrics per stream count, median plus range where sampled:

| Metric | Meaning |
| -- | -- |
| Sessions/hour | Streams times 3600 over batch wall; the production metric |
| Scaling versus 1x | Sessions/hour over the 1x median |
| Output tokens per session | Every turn plus the report, server-counted |
| Prefill tok/s | Uncached prompt tokens over time to first token, summed over a session's calls; absent when the server does not report cached tokens |
| Turn decode tok/s | Per stream, at the first, middle, and last turn: the decode curve against depth |
| Report decode tok/s | Per stream, the deliverable at full depth: the robust decode figure |
| Cached fraction | Server-reported cached prompt tokens over prompt tokens |
| Complete sessions | Sessions whose report ended on its own and recalled at least 7/8 of its findings, of all sessions |
| Truncated calls, empty turns, unreasoned calls | Counted health facts, never gating |

A short turn -- a sentence and a tool call -- times its decode over few
tokens, and a tool parser that holds deltas back until the call is parsed
moves time from decode into prefill. Turn decode rates are therefore most
reliable for reasoning models, whose turns are long; the report decode rate
holds for every model.

Per-turn rows retain prompt, cached, and completion tokens, finish reason,
time to first token, decode rate, content and reasoning length, and the
number of tool calls; reasoning text is measured, never retained. Adapters
record the server's resolved prefix-caching state in the runtime profile;
cached fractions are read against it.

### Audit conversation

`AUDIT_BENCH_PROFILE=conversation` is the model-usability check: the model
itself drives the tool loop over the same fixture and must stop with a
verdict. Non-compliance means the model deviated from the fixture protocol;
it is not evidence about server performance, and it does not prove the model
unusable in production, whose prompting differs from this fixture. Run it
before committing a model to real audits.

Three fixture properties were each forced by observed model behavior:

- File names are revealed one at a time -- each tool result names the
  next file -- because a model given all names upfront batches many reads
  per turn, collapsing the resend structure. The harness still answers
  every call the model makes.
- Each file carries exactly one extractable `FINDING:` line and the verdict
  is the collected ledger, because models given semantically empty filler
  to "analyze" reason unboundedly. The verdict's findings recall is
  recorded as a free correctness signal.
- The corpus carries no embedded instruction: a model that obeys an
  instruction inside conversational evidence derails the tool loop.

| Setting | Canonical value |
| -- | --: |
| `AUDIT_BENCH_CONV_EVIDENCE` | 32,768 |
| `AUDIT_BENCH_CONV_TOOL_TOKENS` | 8,192 |
| `AUDIT_BENCH_CONV_OUTPUT_TOKENS` | 16,384 |
| `AUDIT_BENCH_CONV_MAX_STEPS` | 16 |

Calls are not streamed and decode at the declared sampling. Reported
metrics: conversation wall seconds and conversations/hour (median plus
range, over compliant samples), per-turn wall/prompt/cached/completion
tokens, and the median cached-token fraction from the server's own
`prompt_tokens_details.cached_tokens`. A conversation is compliant when
every turn before the verdict makes at least one tool call, every file is
read within `steps + 4` turns, and the verdict turn returns non-empty text
with no tool calls. Reasoning fields are not resent, matching production
clients, except under `AUDIT_BENCH_PRESERVE_REASONING` (above).

When reading a non-compliant sample, its `per_step` records say exactly
which deviation happened -- check the last turn before concluding anything:
a verdict turn with `files_remaining: 0` and non-empty text is a pass
regardless of how few turns it took; `finish_reason: "length"` with
non-empty text is still compliant (truncation only hurts recall); an empty
`content_excerpt` on a reasoning model usually means the verdict landed in a
reasoning channel; only a verdict with files remaining, or tool calls until
the `steps + 4` turn budget runs out, is genuine protocol non-compliance.

Conversations/hour and sessions/hour are different quantities -- the
session scripts its transcript -- and are never compared.

## Tuning before a canonical run

1. Set the highest reliable per-request context. Prefer the model's native
   context and at least 262,144 tokens; retain 131,072 only when the model or
   hardware cannot serve more.
2. Raise the memory budget until the near-ceiling request and teardown remain
   reliable with the required host reserve.
3. Locate the serving concurrency with noncanonical session runs at
   `AUDIT_BENCH_STREAMS=1,N` for each candidate `N`. Tune scheduler
   admission or Ollama parallel slots, then freeze that operating point.
4. Run the canonical protocol. It automatically climbs the stream ladder
   from 1x and stops where the next rung stops paying.
5. Record the date, repository commit, model revision, runtime versions, host,
   accelerator, power limit, context, KV policy, memory budget, active slots,
   pool or allocation, and the final AuditBench JSON.

## Running it

The standalone CLI drives any OpenAI-compatible chat endpoint. Supply the
configured context, the declared sampling, and, when known, the real
operating point:

```bash
export AUDIT_BENCH_SAMPLING="temperature=0.6 top_p=0.95 top_k=20"
uv run audit-bench --base-url http://127.0.0.1:8000 --context 131072
uv run audit-bench --base-url http://gpu-host:8000 --model my-model \
  --context 262144 --active-slots 4 --kv-tokens 1048576
uv run audit-bench --profile conversation \
  --base-url http://127.0.0.1:8000 --context 131072
```

`--model` is resolved from `/v1/models` when the endpoint serves exactly one
model. Without `--active-slots` and `--kv-tokens` the harness assumes one
slot and one configured context of KV, so sessions run 1x only. Every
`AUDIT_BENCH_*` variable in this document applies to the CLI unchanged. Add
`--format json` when a tool consumes stdout; the JSON is retained on disk
either way.

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
canonicality; the command used is recorded in the result's settings.

## Existing measurements

Measured results, host inventories, and operating points live with the
serving-configuration repositories, never in this one. Do not combine an
`audit-v1` result with an `audit-v2` one, and do not derive a missing value
from an old rate or multiplier.

## Amendment: AuditBench v2 (2026-09-26)

v2 replaces v1's serving profiles. v1 results remain history under their
own protocol and are never compared with v2; every host is re-measured.

What v1 got wrong:

- Its headline decode came from a prompt of about 65 tokens. No agent call
  starts that shallow, and decode slows with depth, so the number did not
  predict live sessions.
- Nothing measured decode between a short prompt and full depth: the
  single-shot audit phases folded prefill into calls/hour, and replay probe
  turns decoded 32 tokens.
- It decoded greedily; live clients sample at the checkpoint's defaults.
- It forced output lengths with a never-conclude control message, which
  derailed one model and was refused by another as a prompt injection, so
  the decode it measured was not an agent's.

What changed:

- Removed: the `full` profile (prefill curve, decode ceiling, `audit_32k`,
  `audit_half`, `audit_deep`), the `decode` tuning profile, `replay`,
  `replay-concurrent`, the forced-length output control, and the
  `AUDIT_BENCH_DEEP_STREAMS`, `AUDIT_BENCH_OUTPUT_TOKENS`,
  `AUDIT_BENCH_DEEP_OUTPUT_TOKENS`, and `AUDIT_BENCH_CONV_TURN_TOKENS`
  settings.
- Added: the `session` profile (default) with streamed timing, and declared
  sampling (`AUDIT_BENCH_SAMPLING`), required for canonical results of both
  profiles; and the `reference` step, `llama-bench`'s pp512/tg128 method,
  labelled as not a serving result.
- Kept: the near-context capacity proof, now inside `session`; the
  `conversation` profile, now at the declared sampling; provenance,
  retention, adaptive sampling, the gate, and the documented controls.
- Session health: the report must end on its own and recall every
  finding, or the run is unpublishable; capped calls, empty turns and
  unreasoned calls are counted, never gating.
- `PROTOCOL` is `audit-v2` and `SCHEMA_VERSION` is 7.

## Amendment: 2.1.0 (2026-09-28)

The first canonical v2 campaigns changed one publication rule and three
failure modes. The workload is unchanged, so 2.0.0 and 2.1.0 results
compare; `PROTOCOL` stays `audit-v2`.

- A complete session names at least 7/8 of its findings, not every one:
  the first canonical run lost publication to one omission in 18
  sessions, and at sampling some session drops one in nearly every run.
- A reference the backend cannot force to an exact length is recorded as
  unavailable instead of aborting the run (Ollama has no `ignore_eos`).
- A session call that streams nothing counts all its time as prefill and
  has no decode rate, instead of aborting the run.
- The vLLM capacity lookup reads vLLM-Metal 0.30.0's `CPU KV cache size`
  label.

## Amendment: 2.2.0 (2026-10-20)

Two scoring and transcript fixes from the October trials. The workload's
shape is unchanged: the session transcript differs only in its tool-call
ids, a few tokens per step, so 2.1.0 and 2.2.0 rates compare; `PROTOCOL`
stays `audit-v2`.

- Finding recall folds typographic dashes (U+2010 to U+2015 and the minus
  sign) to an ASCII hyphen before matching. Nemotron 3.5 Lightning writes
  some reports with U+2011 in every finding token; such a report named all
  16 findings yet scored none, and its session counted as incomplete.
- Scripted tool-call ids are nine alphanumerics (`call00001` onwards, and
  `call{turn:03d}{index:02d}` for the conversation's fallback), the shape
  vLLM's Mistral tokenizer requires of every id in the history; `call_001`
  failed every Mistral Small 4 session on its first resent step.

## Amendment: 2.3.0 (2026-10-22)

The publication rule picks the optimal operating point instead of requiring
every stream count to complete. On a host where concurrency adds little,
such as the M5 Max, a model can complete every session at one stream yet
drop one at four; 2.2.0 then rejected the whole run, discarding the faster
and complete 1x measurement. A run is now publishable when at least one
measured stream count completed every session, and it publishes the
fastest such count (`optimal_streams`, `SCHEMA_VERSION` 8). The workload,
its settings and every measured rate are unchanged, so 2.2.0 and 2.3.0
rates compare; `PROTOCOL` stays `audit-v2`. A 2.2.0 result rejected only for
`incomplete_sessions` that has a complete stream count qualifies under this
rule from its retained `by_streams`.

2.3.0 also adds `AUDIT_BENCH_PRESERVE_REASONING` (see "Model-documented
controls"), off by default: a model whose template re-renders every past
turn's reasoning reads dropped turns as empty thoughts and stops reasoning.
Laguna-XS streamed no reasoning on 237 of 255 session calls without it and
on 1 of 255 with it. Runs without the control are unchanged.

Capability Check v2 makes its `json_schema` check advisory in the same
release and gives its workflow task the same reasoning control (see
`docs/CAPABILITY.md`, `SCHEMA_VERSION` 6).

## Amendment: 2.5.0 (2026-10-31)

The session measures a stream ladder instead of its two ends. 2.4.0 ran 1x
and the largest admitted count and never measured between them, so the
operating point could only be one of the two: on the M5 Max, 4x gained 9 to
13% over 1x for most models and cost the run's longest leg, while 2x, the
likely knee, went unmeasured. The session now climbs 1, 2, 4, 8 within
admission and stops at the first rung that gains less than 10% over the rung
below or drops a session (`AUDIT_BENCH_STREAM_GAIN`; any other value is
noncanonical). Publication is unchanged: the fastest measured rung that
completed every session is `optimal_streams`.

Each stream count's workload, settings and rate are unchanged, so 2.4.0 and
2.5.0 rates at the same stream count compare; `PROTOCOL` stays `audit-v2`.
What differs is which counts run: a 2.5.0 result may publish 2x where 2.4.0
published 4x, and lacks the rungs above where it stopped. Rungs now run their
samples back to back instead of interleaved with the other count, since a
rung's existence depends on the one below; any thermal drift therefore falls
on the later, higher rung and makes the ladder stop early rather than late.
An incomplete 1x stops the ladder, where 2.4.0 still measured the top count.
`SCHEMA_VERSION` 9 adds `audit_session.ladder_stop`.

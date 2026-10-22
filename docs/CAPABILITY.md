# Capability Check v2

Capability Check v2 is the pass/fail gate for models in a serving inventory.
Its API checks prove the response semantics coding agents need; its audit
fitness check proves the model can carry audit work at the sampling agents
run it with. AuditBench separately measures serving performance and
capacity.

Capability results never rank model quality or serving performance. Raw JSON
lives under `${XDG_STATE_HOME:-~/.local/state}/capability-check/`, or the
directory named by `CAPABILITY_OUTPUT_DIR`.

## Protocol checks

The protocol uses fixed local fixtures:

| Check | Passing condition |
| -- | -- |
| Model identity | The live server matches the requested model: the managed launch artifact for adapters, the endpoint's own model listing for standalone runs |
| Streaming reasoning | The stream completes, final content is correct, reasoning uses the dedicated field, and no raw markers leak |
| Thinking disabled | When configured, the documented control produces correct content with an empty reasoning field |
| Tool call | With automatic tool choice, one non-streaming `lookup_issue` call carries `{"issue_number": 42}` |
| Streaming tool call | With automatic tool choice, one streamed `lookup_issue` call is assembled with `{"issue_number": 42}` |
| Tool round trip | The model consumes the returned `sentinel-cobalt` value and finishes without another tool call |
| Structured JSON | The final content is the exact requested JSON object, without Markdown cleanup. The same check records, without gating, what the model manages when asked for the full fixture document with the schema stated in words (`unaided`) |
| JSON schema | Advisory: recorded with its status, never decides the verdict. The endpoint is honest about schema enforcement: it either constrains decoding to the fixture schema, or refuses the parameter with a 4xx. Accepting it and returning anything else fails the check |
| Audit fitness | Fewer than `FITNESS_FAILURE_LIMIT` (2) of the audit samples fail: three audit-shaped tasks (two for a model without tools), four samples each, at the declared sampling |

The reasoning fixture allows enough output for models whose thought phase is
longer than their answer. Results retain content excerpts and field lengths,
not reasoning text.

The OpenAI-compatible adapter accepts either `reasoning` or
`reasoning_content` as the dedicated reasoning field, in complete responses
and streaming deltas. It assembles each field before normalization; matching
copies are counted once, while conflicting nonempty copies fail. Reasoning
in either spelling must still be absent when thinking is disabled, and raw
markers in visible content still fail. `reasoning_length` reports normalized
text length; the existing `legacy_reasoning_length` retains the raw
`reasoning_content` length for wire-format evidence.

The OpenAI-compatible adapters explicitly use `tool_choice: auto` for both
tool-call checks. This exercises the model's own tool syntax and the server's
configured parser; a named choice can invoke guided decoding and bypass both.
Ollama's native `/api/chat` has no tool-choice parameter, so offering tools has
the same automatic semantics. The fixed prompt requires the call, and both
paths validate the same exact name and arguments.

### Model manners versus server enforcement

Two different facts share the word "structured", and the protocol keeps them
apart:

- **Structured JSON** is a model property. The request carries no format
  parameter, so the check measures whether the model volunteers a bare
  object when asked plainly. A fence, a preamble, or trailing prose fails.
  The same check then asks for the full fixture document with the schema
  stated in words, fairly rather than adversarially, and records the outcome
  in `unaided` without gating on it: `valid`, `json_only`, `ignored`,
  `truncated`, or `error`. That fact decides only whether a server which
  cannot constrain decoding is usable with this model, so it is recorded and
  never disqualifying -- a model needing the grammar is fine on a host that
  provides one.
- **JSON schema** is a server property. Constrained decoding masks invalid
  tokens during sampling -- `response_format` carrying a `json_schema` on
  OpenAI-compatible servers, the schema in `format` on Ollama -- so it holds
  regardless of the model's manners, and it is a function of the server
  build and its flags.

Read the pair together. `enforced` with `unaided: ignored` means the host
carries the model; `ignored` with `unaided: valid` means the server
guarantees nothing but this model complies anyway; `ignored` with
`unaided: ignored` is the combination that keeps a model off that host for
structured work.

The enforcement fixture is adversarial on purpose: a strict schema
(`verdict` enum, integer `count`, at least two `refs`, no additional
properties) paired with a prompt demanding Markdown prose, a heading, and a
fenced block. An unconstrained model that obeys the prose cannot satisfy the
schema, so a pass is evidence of the grammar rather than of good manners.

Its `enforcement` field names what happened, and is retained on failure too:

| Value | Meaning | Status |
| -- | -- | -- |
| `enforced` | Output validates against the fixture schema | passed |
| `unsupported` | The endpoint refused the parameter with a 4xx | passed |
| `json_only` | Parseable JSON that violates the accepted schema: JSON mode rather than schema mode, or a schema compiler that silently dropped keywords it cannot express | failed |
| `ignored` | The parameter was accepted and the content is not JSON at all | failed |
| `whitespace_runaway` | The server permits structural whitespace, and the model emitted it until the budget ran out: grammar-valid, no document | failed |
| `truncated` | The response hit the token budget before closing the document, so nothing is proven | failed |

A server without constrained decoding is not defective and passes by refusing
the parameter; a server that accepts it and does not honor it fails the
check, because callers would otherwise trust a guarantee that does not exist.
The check is advisory: it never fails the gate. The check runs
in the same mode as Structured JSON: the documented disable control when the
model has one, the reasoning control otherwise.

Thinking and enforcement coexist, and the fixed 16,384-token budget exists
for that reason. Servers constrain the visible content channel and leave the
reasoning channel free, so a thinking model reasons unconstrained, then emits
a schema-clean document; the budget has to cover both. A model whose thinking
crowds out the document reports `truncated`, which is a fact about that model
under a fixed workload rather than a knob to widen.

A `truncated` result is worth reading before blaming the model, because a
grammar bounds structure and not length. Where a server permits structural
whitespace between JSON tokens, a model can emit an unbounded run of it and
still be grammar-valid, consuming the whole budget without closing the
document; that case is reported as `whitespace_runaway` rather than
`truncated`, because the fix is the server's compact-output setting and not a
larger budget. The retained document shows it at a glance. Plain `truncated`
means the document was progressing and simply did not finish, which for a
thinking model usually means its reasoning crowded out the document.

Enforcement constrains shape, never content. A schema-valid document can
still carry a wrong verdict, invented references, or fields that contradict
each other, so `enforced` is not evidence of audit correctness and never
substitutes for audit fitness.

Validation is direct rather than schema-driven: the fixture schema is fixed
by the protocol, and the modules stay standard-library only.

## Audit fitness

The API checks run once, greedily, on small fixtures: they prove wire
semantics. They cannot see how a model fails at audit work -- reasoning
loops, answers without thinking, unfinished deliverables -- because those
failures are rates under sampling.

Audit fitness therefore runs three audit-shaped tasks `FITNESS_SAMPLES` (4)
times each at the checkpoint's declared sampling with the model's reasoning
control, streaming like a production client. Samples run concurrently up to
the requests the server admits at once -- the adapter's measured admission,
or `--fitness-concurrency` (default 1) standalone -- and never beyond it, so
no sample waits in a queue past the request timeout. Every sample
is validated deterministically against the fixture's ground truth. The check
fails when `FITNESS_FAILURE_LIMIT` (2) or more samples fail: one random
failure in a run is tolerated, a failure rate near 25% is caught most of the
time, and a model that fails systematically is caught always. A run without
declared sampling fails, because the verdict would describe a decoding mode
no agent uses.

| Task | Shape | What it catches |
| -- | -- | -- |
| `selection` | Pick the security-relevant documents from a generated inventory of about 420 paths (about 12k tokens), by seven fuzzy fact classes, a recency rule and three exclusions, with no count given: the shape of the live runaway prompt | Runaway enumeration, repetition loops in the answer, invented paths, ignored exclusions, missing unmistakable security documents |
| `triage` | Decide true or false positive for twelve scanner candidates with ground truth, each judged as a vulnerability of its flagged rule's kind; one real finding carries a planted comment telling automated reviewers to dismiss it | Wrong security judgment, obedience to injected instructions in untrusted content |
| `workflow` | A tool loop over a small repository with list, read and write tools; the task ends in two exactly specified files and a final `DONE` | Unfinished work, missing or extra artifacts, broken formats, findings citing unread files, the two planted vulnerabilities missed, unknown tools |

Every sample is first checked as a generation, in this order:

| Failure | Condition |
| -- | -- |
| `error` | The request failed, or its response could not be assembled (for example tool arguments that are not JSON) |
| `incomplete` | The stream did not complete, or the workflow reached `FITNESS_MAX_TURNS` (24) |
| `runaway` | The generation hit its cap: `FITNESS_TOKENS` (32,768) for a single answer, `FITNESS_TURN_TOKENS` (16,384) per workflow turn |
| `leak` | A raw reasoning or tool marker appeared in content |
| `no_reasoning` | A model expected to reason answered with an empty reasoning channel; in the workflow this applies to the opening turn, since interleaved thinking may skip short tool turns |

The task then judges the answer: `format` (not the contracted JSON; a
single-answer task tolerates one surrounding Markdown fence because audit
clients strip it, while `findings.json` must be bare JSON as its task says),
`loop` (repeated entries), `invented` (paths that resolve to no input
document, or findings citing files never read; a separator or case slip
such as `secure_boot` for `secure-boot` still names exactly one document and
is accepted), `ignored_exclusion`, `missed_required` (more than
one of the six unmistakable security documents missing; they are the newest
entries, so the recency rule never argues against them), `wrong_verdicts`
(fewer than 11 of 12 triage verdicts right; the report names the planted
comment when the model followed it), `instruction` (artifacts missing, extra
or out of section order, or a final reply other than `DONE`),
`missed_findings` (a planted vulnerability not reported at the vulnerable
line, within five lines; the unguarded export also counts where it is found
one layer down, at `export_all`), and `tool_misuse`.

The caps sit well above measured healthy passes, so reaching one is a
runaway, not a slow thinker; a report records the caps it ran under. Each
sample records its seconds, calls, completion tokens, reasoning length, and
`max_line_repeats`, the occurrences of the most repeated line of 16 or more
characters across its output: the loop evidence. A failed sample also
keeps the first and last 600 characters of content and the last 600 of
reasoning, enough to diagnose without retaining transcripts; when the
generation failed but an answer arrived (`no_reasoning`, `leak`), `answer`
records how the task judged it anyway, and a failed workflow keeps the first
1,500 characters of its `findings.json`. A workflow `read_file` of an
artifact the model already wrote returns it, since reading output back to
verify it is normal agent behavior.

The fixtures are generated deterministically from the protocol and fictional:
no repository content is sent. `CAPABILITY_FITNESS_SAMPLES` or
`--fitness-samples` changes the samples per task for a trial; `0` skips the
check. A promotion needs the default.

## Model expectations

Every model config (or standalone invocation, via `--reasoning`, `--tools`,
`--reasoning-mode`, and `--disable-mode`) declares:

- whether reasoning must be present or absent;
- whether structured tools are required;
- the documented control used to enable reasoning;
- the documented control used to disable it, or `none` when unsupported;
  and
- the checkpoint's sampling (`--sampling`, space-separated `key=value`), the
  same declaration AuditBench uses; and
- whether its card requires prior reasoning back in the history
  (`--preserve-reasoning`, or `CAPABILITY_PRESERVE_REASONING=1`): the
  workflow task then resends each turn's reasoning on its assistant message
  (`reasoning` on OpenAI-compatible servers, `thinking` on Ollama's native
  API), as `AUDIT_BENCH_PRESERVE_REASONING` does in AuditBench.

These are model facts. Do not infer them from parser names or manifest
capability lists: manifests can omit working tools, and a configured reasoning
parser does not mean thinking is enabled by default.

Controls differ by model family, so the mode names spell out the wire form
rather than an intent. `enable-thinking` and `enable-thinking-false` send the
`enable_thinking` chat-template kwarg; `reasoning-effort-high` and
`reasoning-effort-none` send `reasoning_effort`, the control used by families
that grade reasoning instead of toggling it; `think-prefix` and
`no-think-prefix` prepend an in-band marker. `reasoning_effort` travels as a
chat-template kwarg because servers forward kwargs to the template
generically, while top-level handling of that field is model-specific.

A model whose reasoning is off by default is still a reasoning model: declare
the control that turns it on. `none` means the model documents no such
control, not that the default happens to suit the check.

## How results are generated

Every run retains its complete JSON report on disk before printing
anything, then prints one of two forms on stdout: the human verdict by
default -- every check with its status, seconds, and failure or skip
reason, one line per audit fitness task (failed of sampled, by failure
class), the identity evidence, the overall result, and the retained path
-- or the one-line JSON report with `--format json`, for tools that parse
it. The verdict is a projection; the report on disk is the result.
`capability-check --show FILE` re-prints any retained report without
contacting a server, taking the file name printed by `--list` or any path,
and exits 0 whenever it printed: the gate verdict is the report's own
`status`, not that exit code. `capability-check --list` indexes the
retention directory, newest first, so a campaign can find the artifact a
results row must cite.

The results table is a projection of retained `capability-v2` JSON, not a
hand-scored judgment. Copy each cell from the named check in the raw
artifact into the results document:

| Table column | JSON source |
| -- | -- |
| Identity | `checks.model_identity.status` |
| Reasoning | `checks.streaming_reasoning.status` |
| Thinking off | `checks.thinking_disabled.status`; use `unsupported` only when it is `skipped` because the model config declares no disable mode |
| Tool call | `checks.tool_call.status` |
| Streaming tool call | `checks.streaming_tool_call.status` |
| Tool loop | `checks.tool_round_trip.status` |
| JSON | `checks.structured_json.status` |
| Unaided document | `checks.structured_json.unaided`: `valid`, `json_only`, `ignored`, `truncated`, or `error`; never gates |
| Schema enforced | `checks.json_schema.enforcement`: `enforced` or `unsupported` pass; `json_only`, `ignored`, `whitespace_runaway`, and `truncated` fail the check; never gates (`advisory_failures`) |
| Audit fitness | `checks.audit_fitness.status`, with `failed_samples` of `total_samples` and `failure_classes` |
| Result | top-level `status` |
| Model identity | Ollama manifest `identity.digest`, or Hugging Face `identity.revision` |
| Artifact | basename of top-level `artifact_path` |

Do not derive a pass from terminal prose, old benchmark notes, parser
configuration, or manifest capability labels. The printed verdict is a
reading aid, not a source: transcribe from the retained JSON. Do not copy
request timings into the results document; Capability Check is not a
performance benchmark.

## Repeatable campaign procedure

Run managed campaigns from the serving-configuration repository root. Use
only the profiles its results document lists for that host; model configs
for another host are not an implied campaign.

1. Check the managed server status. Do not stop or replace a process you do not
   own, especially on a shared remote host.
2. Start the selected profile. For Ollama, pull it through the managed server
   before checking so the live local artifact matches the current tag.
3. Run the matching capability target.
4. Always stop the server afterward and confirm resource release, including
   after a failed check.
5. Read the retained JSON named by `artifact_path`.
6. Update the host row and raw-artifact table mechanically from the JSON mapping
   above. Record failures as failures; never convert manual evidence into a
   Capability Check pass.

Managed lifecycle targets (start, capability, stop, and their remote
variants) live in the serving-configuration repositories. Standalone, against
an already running endpoint:

```bash
uv run capability-check --base-url http://127.0.0.1:8000 \
  --model <served-name> --reasoning required --tools required \
  --sampling 'temperature=0.6 top_p=0.95 top_k=20'
```

Add `--format json` when a tool consumes stdout; the report is retained on
disk either way.

Remote raw JSON remains in that host's capability state directory. Copy or
archive it with the campaign evidence before editing the results document.

### Identity drift

A digest mismatch distinguishes two cases, and only one is drift. The
Ollama adapters consult the registry's manifest header (a few KB, no
download) before deciding:

- **Stale local store**: the registry still serves the pinned digest --
  the local tag simply predates an upstream tag move. The adapter pulls
  through the managed server automatically, landing exactly the
  already-reviewed artifact, and validates again. No pin changes, no
  human decision consumed.
- **True drift**: the registry serves something other than the pin. The
  refusal stands and names all three digests (registry, pin, local).
  Drift is never permission to update a pin: retain the failed
  artifact, inspect the replacement manifest or Hugging Face revision,
  verify format, quantization, architecture, native context, and
  capabilities, then run one explicit identity override. Repin only
  after that artifact passes. If a benchmarked artifact changed, keep
  its performance numbers historical until the replacement is
  benchmarked -- and remember depth proofs are per digest.

An unreachable registry keeps the plain refusal: recovery is only
automatic when it is provably safe.

### Promotion rule

A model intended for agent workloads must pass identity, streaming reasoning,
non-streaming and streaming tool calls, tool round trip, structured JSON, and
audit fitness at the default sample count. Its reasoning result must match
the model-config expectation. A throughput result cannot override a
capability failure.

Schema enforcement is a host fact, not a model fact, and it never decides
the verdict: a failed `json_schema` check is listed in the report's
`advisory_failures`, not in `failed_checks`. `unsupported` says the server
cannot constrain decoding; the failing values mean the endpoint misrepresents
what it does. Record the mode with the host, and do not treat `enforced` as a
substitute for the model's own structured-output result.

## Amendments

- **Schema enforcement advisory** (`SCHEMA_VERSION` 6). The `json_schema`
  check runs and records its status and mode as before, but no longer
  decides the verdict or the exit code: no audit client depends on
  constrained decoding yet, and its failures are engine gaps rather than
  model faults. A failed check is listed in `advisory_failures`. Every other
  check and fixture is unchanged, so a report that failed only `json_schema`
  under `SCHEMA_VERSION` 5 is a pass under this rule. The same version adds
  the `preserve_reasoning` expectation for interleaved-thinking models; it
  defaults off, which keeps every other model's workflow unchanged.
- **Triage judged per flagged rule** (report shape unchanged,
  `SCHEMA_VERSION` 5). The triage prompt defined a true positive as
  "exploitable as written", which made the unsalted password hash (C05)
  arguably a false positive, since exploiting it needs a leaked table, and
  invited an authorization reading of the guarded invoice path (C12). A
  gate cannot count defensible answers as failures, so a true positive is
  now "a real vulnerability of the flagged rule's kind, as written".
  Candidates and ground truth are unchanged. Earlier triage failures may
  include such misses; rerun before relying on them. Passes and the other
  tasks remain comparable.
- **Audit fitness added; protocol `capability-v2`** (`SCHEMA_VERSION` 5,
  2026-09-27). A new gating check samples three audit-shaped tasks at the
  declared sampling. The API checks keep their fixtures and pass conditions,
  so their per-check results stay comparable, but a v1 pass is not a v2
  pass: rerun before promoting. Streaming requests now ask for
  `stream_options.include_usage` to record completion tokens. `--show` and
  `--list` still read v1 reports.
- **Reasoning field aliases accepted** (report shape unchanged,
  `SCHEMA_VERSION` 4). The OpenAI adapter now normalizes `reasoning_content`
  as well as `reasoning`; the former spelling alone no longer fails the
  response. Fixtures, budgets, tool validation and reasoning separation are
  unchanged. Earlier failures caused by the rejected field name, and tool
  continuations skipped because of them, require a rerun; they are not
  evidence of broken reasoning or tools. Other results remain comparable.
- **Automatic tool parsing covered** (`SCHEMA_VERSION` 4). The tool-call
  fixture now uses automatic rather than named choice, and a second check runs
  it over streaming. Named choice could let guided decoding bypass the model's
  tool syntax and the configured parser, so earlier tool-call passes do not
  establish parser compatibility and must be rerun. Other per-check results
  remain comparable.
- **Unaided document recorded** (`SCHEMA_VERSION` 3). Structured JSON keeps
  its fixture and pass condition, so its statuses stay comparable; the check
  additionally asks for the fixture document with the schema stated in words
  and records `unaided`. It cannot change a verdict, and campaigns recorded
  before it simply lack the field.
- **Schema enforcement added** (`SCHEMA_VERSION` 2). Existing checks and
  their fixtures are unchanged, so earlier per-check results remain
  comparable. The new `json_schema` check adds a key and can
  fail a host that passed before, but only when that host accepts a schema
  parameter and does not honor it: campaigns recorded before this check
  simply lack the fact, and re-running is the way to obtain it.

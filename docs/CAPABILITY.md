# Capability Check v1

Capability Check v1 is the pass/fail API compatibility gate for models in a
serving inventory. It proves response semantics needed by coding agents;
AuditBench separately measures serving performance and capacity.

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
| Tool call | Exactly one `lookup_issue` call carries `{"issue_number": 42}` |
| Tool round trip | The model consumes the returned `sentinel-cobalt` value and finishes without another tool call |
| Structured JSON | The final content is the exact requested JSON object, without Markdown cleanup. The same check records, without gating, what the model manages when asked for the full fixture document with the schema stated in words (`unaided`) |
| JSON schema | The endpoint is honest about schema enforcement: it either constrains decoding to the fixture schema, or refuses the parameter with a 4xx. Accepting it and returning anything else fails |

The reasoning fixture allows enough output for models whose thought phase is
longer than their answer. Results retain content excerpts and field lengths,
not reasoning text.

The OpenAI-compatible adapters use named `tool_choice`. Ollama's native
`/api/chat` has no equivalent force parameter, so its fixed prompt requires
the call and the same exact name and arguments are validated afterward.

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
the parameter; a server that accepts it and does not honor it fails, because
callers would otherwise trust a guarantee that does not exist. The check runs
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
substitutes for the conversation profile in AuditBench.

Validation is direct rather than schema-driven: the fixture schema is fixed
by the protocol, and the modules stay standard-library only.

## Model expectations

Every model config (or standalone invocation, via `--reasoning`, `--tools`,
`--reasoning-mode`, and `--disable-mode`) declares:

- whether reasoning must be present or absent;
- whether structured tools are required;
- the documented control used to enable reasoning; and
- the documented control used to disable it, or `none` when unsupported.

These are model facts. Do not infer them from parser names or manifest
capability lists: manifests can omit working tools, and a configured reasoning
parser does not mean thinking is enabled by default.

## How results are generated

Every run retains its complete JSON report on disk before printing
anything, then prints one of two forms on stdout: the human verdict by
default -- every check with its status, seconds, and failure or skip
reason, the identity evidence, the overall result, and the retained path
-- or the one-line JSON report with `--format json`, for tools that parse
it. The verdict is a projection; the report on disk is the result.
`capability-check --show FILE` re-prints any retained report without
contacting a server, taking the file name printed by `--list` or any path,
and exits 0 whenever it printed: the gate verdict is the report's own
`status`, not that exit code. `capability-check --list` indexes the
retention directory, newest first, so a campaign can find the artifact a
results row must cite.

The results table is a projection of retained `capability-v1` JSON, not a
hand-scored judgment. Copy each cell from the named check in the raw
artifact into the results document:

| Table column | JSON source |
| -- | -- |
| Identity | `checks.model_identity.status` |
| Reasoning | `checks.streaming_reasoning.status` |
| Thinking off | `checks.thinking_disabled.status`; use `unsupported` only when it is `skipped` because the model config declares no disable mode |
| Tool call | `checks.tool_call.status` |
| Tool loop | `checks.tool_round_trip.status` |
| JSON | `checks.structured_json.status` |
| Unaided document | `checks.structured_json.unaided`: `valid`, `json_only`, `ignored`, `truncated`, or `error`; never gates |
| Schema enforced | `checks.json_schema.enforcement`: `enforced` or `unsupported` pass; `json_only`, `ignored`, `whitespace_runaway`, and `truncated` fail |
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
  --model <served-name> --reasoning required --tools required
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

A model intended for agent workloads must pass identity, streaming, tool call,
tool round trip, and structured JSON. Its reasoning result must match the
model-config expectation. A throughput result cannot override a capability
failure.

Schema enforcement is a host fact, not a model fact: `unsupported` says the
server cannot constrain decoding, which promotes or blocks nothing by itself,
while the failing values mean the endpoint misrepresents what it does. Record
the mode with the host, and do not treat `enforced` as a substitute for the
model's own structured-output result.

## Amendments

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

# Contributing

This repository holds two measurement harnesses and their protocol
documents. Contributions are welcome when they sharpen the measurements,
the protocol text, or the tooling around them -- and unwelcome when they
grow the scope. Read the invariants below before writing code.

## Development setup

[uv](https://docs.astral.sh/uv/) manages the environment; runtime code
itself has no dependencies. All tasks run through make:

| Target | Action |
| -- | -- |
| `make dev` | Create the uv-managed venv and install the CLIs |
| `make test` | Run the unit tests |
| `make lint` | Run markdownlint and a Python compile check |
| `make all` | Lint and test |
| `make clean` | Remove the venv and build artifacts |

`make all` must pass before any change is complete.

## Repository layout

| Path | Contents |
| -- | -- |
| `src/audit_bench.py` | AuditBench v2 harness and the `audit-bench` CLI |
| `src/capability_check.py` | Capability Check v1 harness and the `capability-check` CLI |
| `tests/` | Unit tests (stdlib `unittest`) |
| `docs/PROTOCOL.md` | Canonical AuditBench v2 protocol document |
| `docs/CAPABILITY.md` | Canonical Capability Check v1 protocol document |
| `AGENTS.md` | The same rules, stated for AI coding agents |

## Hard invariants

Consumer repositories import these modules from a sibling checkout by
putting its `src/` directory on `PYTHONPATH`, or copy the two files next
to their runtime adapters on remote hosts. That imposes rules a change
must not break:

- Runtime code is Python standard library only. uv and `pyproject.toml`
  are development tooling; a third-party import in either module is a
  defect, not a convenience.
- The modules stay plain single files under `src/` with their current
  names. No renames, no splitting into packages.
- `audit_bench.py` and `capability_check.py` never import each other.
- Workload logic lives only in `audit_bench.py`: prompt generation,
  target sizes, sample counts, stream counts, validation, metrics, and
  the JSON schema. Capability fixtures and pass conditions live only in
  `capability_check.py`. Runtime adapters (in consumer repositories) own
  transport, model loading, capacity discovery, and identity evidence.
- The honesty rules are not negotiable: a fixed-length response that
  comes back short is rejected, and a natural-length one is recorded at
  its reported length, never padded or retried inside a sample; token
  counts come from server-reported usage, never client-side estimates; requested
  concurrency is never reported as active concurrency; failures are
  recorded as failures.
- The `canonical` versus `publishable` distinction survives every
  refactor: canonicality describes the workload only, publishability the
  provenance.

## Changing the protocol

[docs/PROTOCOL.md](docs/PROTOCOL.md) and
[docs/CAPABILITY.md](docs/CAPABILITY.md) are canonical; code follows the
document, not the other way around. A change
that touches a canonical setting or workload shape needs, in one commit:

1. An amendment note in the affected document stating what changed and
   whether existing results remain comparable.
2. The matching code change, with tests updated to pin the new behavior.
3. A `SCHEMA_VERSION` bump in the affected module when the emitted JSON
   shape changes.

Changing canonical defaults invalidates cross-host comparability of every
existing result at those settings. That cost is acceptable only when the
old workload measured the wrong thing -- say so explicitly in the
amendment note.

## Style

- Python: match the existing idiom; every behavior change comes with a
  test. The suites run offline -- network access is always mocked.
- Markdown: plain ASCII, no emojis, standard markdown only, prose wrapped
  within the linted line length.
- Keep changes minimal. An addition that does not trace to a concrete
  need is scope creep, and unrelated refactors are rejected regardless of
  quality.

## What does not belong here

Measured results, campaign records, host inventories, hostnames or
hardware details of real machines, runtime launchers, and model
configurations. Those live in the private serving-configuration
repositories that consume this one.

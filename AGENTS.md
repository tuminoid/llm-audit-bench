# Agent Guidance for llm-audit-bench

Rules for AI agents working in this repository. They assume the machine-wide
safety rules already apply.

## Scope

This repository holds exactly two things: the shared benchmark and
capability harnesses (`audit_bench.py`, `capability_check.py` and their
tests) and their protocol documents ([docs/PROTOCOL.md](docs/PROTOCOL.md),
[docs/CAPABILITY.md](docs/CAPABILITY.md)).

It never holds: measured results, campaign records, host inventories,
hostnames or hardware details of real machines, runtime launchers, or model
configurations. Those live in the serving-configuration repositories. A
change that adds any of them here is wrong regardless of how convenient it
is.

## Compatibility contract

Consumers depend on this repository in two ways:

- **Sibling checkout**: a serving repository points `PYTHONPATH` (or an
  explicit path variable) at this checkout's `src/` directory and imports
  `audit_bench` and `capability_check` directly.
- **Adjacent copy**: remote deploys rsync the two module files next to the
  runtime adapters, where Python's script-directory path entry resolves
  them.

Both modes impose hard rules:

- The modules stay plain single files under `src/` with their current
  names. Moving or renaming them is a breaking change for every consumer.
- Runtime code is Python standard library only. uv and `pyproject.toml`
  manage the development environment and CLI entry points; they must never
  become import-time requirements. A third-party import in either module is
  a defect.
- Each module stays self-contained: `audit_bench.py` and
  `capability_check.py` do not import each other.

## Where logic lives

- Prompt generation, target sizes, sample counts, stream counts,
  validation, metrics, and the JSON schema belong only in `audit_bench.py`.
- Capability fixtures and pass conditions belong only in
  `capability_check.py`.
- Runtime adapters (in consumer repositories) own transport, model loading,
  capacity discovery, residency checks, and identity evidence. Logic that
  would let two runtimes measure different workloads under the same name
  must not move into an adapter.

## Protocol changes

- [docs/PROTOCOL.md](docs/PROTOCOL.md) and
  [docs/CAPABILITY.md](docs/CAPABILITY.md) are the canonical protocol
  documents. Code follows the document, not the other way around.
- Changing a canonical setting or workload shape requires an explicit
  amendment note in the document stating what changed and why existing
  results remain comparable (or do not); git history carries the dates.
- A change to the emitted JSON shape bumps `SCHEMA_VERSION` in the affected
  module.
- Never weaken the honesty rules: a fixed-length response that comes back
  short is rejected, and a natural-length one is recorded at its reported
  length, never padded or retried inside a sample; token counts come from server-reported
  usage, never client-side estimates; requested concurrency is never
  reported as active concurrency; failures are recorded as failures.
- The `canonical` versus `publishable` distinction must survive every
  refactor: canonicality describes the workload only, publishability the
  provenance.

## Workflow

- All tasks run through Makefile targets: `make dev`, `make test`,
  `make lint`, `make all`, `make clean`. Do not invoke tools directly.
- `make test` and `make lint` must pass before a change is complete.
- Plain ASCII everywhere. No emojis or special unicode. Standard markdown
  only; keep lines within the linted length.

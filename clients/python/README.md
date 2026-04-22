# workbench-astra-client (Python)

Python port of the AI Workbench Astra client. Wraps the Astra Data API
for `wb_*` control-plane tables and vector-store collections, matching
the behavior of the TypeScript reference client byte-for-byte (enforced
by [`../conformance/`](../conformance/)).

## Status

**Scaffold.** The public shape is defined; implementations raise
`NotImplementedError`. This exists so Cédrick can fill it in without
touching the surrounding harness.

## Structure

```
clients/python/
├── pyproject.toml
├── README.md                                    ← you are here
├── src/workbench_astra_client/
│   ├── __init__.py                              ← re-exports WorkbenchAstraClient
│   ├── client.py                                ← top-level client
│   ├── control_plane.py                         ← wb_* table ops
│   ├── data_plane.py                            ← collection ops (Phase 1b)
│   ├── types.py                                 ← records mirroring TS types
│   └── _http.py                                 ← shared HTTPX transport
└── tests/
    ├── conftest.py                              ← fixtures (incl. mock handle)
    └── test_conformance.py                      ← runs scenarios.md
```

## Getting started (Cédrick)

1. `cd clients/python`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -e '.[dev]'`
4. In a separate terminal from the repo root: `npm run conformance:mock`
5. `pytest` — you'll see `NotImplementedError` failures. That's the
   starting point.

## Where to focus

Each `NotImplementedError` in `control_plane.py` corresponds to one
operation in [`../conformance/scenarios.md`](../conformance/scenarios.md).
Implement them in order; they all call `_http.request(...)` under the
hood. Use [`astrapy`](https://github.com/datastax/astrapy) once we're
ready to talk to real Astra — for conformance against the mock, raw
`httpx` is fine.

## Type mirroring

Python type definitions in `types.py` track the TypeScript definitions
in [`../../src/control-plane/types.ts`](../../src/control-plane/types.ts).
When the TS types change, update here in the same PR.

Naming convention:
- TS `camelCase` → Python `snake_case`.
- TS `Record<string, string>` → Python `dict[str, str]`.
- TS `readonly` → Python `@dataclass(frozen=True)` or `TypedDict` with
  `total=False` for optional fields.

## Conformance

`tests/test_conformance.py` runs every scenario against the mock and
compares captured HTTP payloads to the fixtures in
[`../conformance/fixtures/`](../conformance/fixtures/). Until fixtures
exist (they ship with PR-1a.2), the test asserts only that each
scenario runs without exception.

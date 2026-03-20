# Memory Backend Abstraction

Hermes now owns a small memory backend seam at the runner and gateway layer.

## Design intent

- `Honcho` remains the only built-in backend shipped by Hermes.
- Hermes can optionally load one out-of-tree backend from a configured Python
  factory.
- The existing `honcho_*` tool surface stays unchanged for now.
- `CLI`, `gateway`, and `ACP` all use the same backend factory and contract.
- `gateway` and `ACP` inherit it through `AIAgent`; CLI loads the same backend factory directly.

This is deliberately **not** a tool plugin. Memory owns session attach, prompt
context, prefetch, migration, and shutdown, so the abstraction lives next to
the current runner-owned Honcho lifecycle.

## Built-in behavior

If no external backend factory is configured, Hermes loads the built-in Honcho
backend. Current behavior remains unchanged.

## External backend hook

Hermes supports one explicit experimental override:

```json
{
  "hosts": {
    "hermes": {
      "experimental": {
        "memory_backend_factory": "package.module:create_backend"
      }
    }
  }
}
```

If this field is set, Hermes loads that factory instead of the built-in Honcho
backend.

The factory must return either:

- `memory_backends.base.MemoryBackendBundle`
- or a 3-tuple of `(manager, config, manifest)`

Hermes validates the returned manifest, config view, and manager methods before
accepting the backend. Invalid factories fail loudly. Hermes does not silently
fall back to Honcho.

## Manifest contract

Every backend manifest must define:

- `protocol_version`
- `backend_id`
- `display_name`
- `capabilities`
- `config_source`

The initial capability vocabulary is:

- `profile`
- `search`
- `answer`
- `conclude`
- `prefetch`
- `migrate`
- `ai_identity`

Hermes uses capabilities to hide unsupported read and management features
cleanly instead of pretending every backend supports the full Honcho surface.

## Runtime contract

The backend contract is Hermes-shaped, not generic. It matches what Hermes
already needs:

- session attach/create
- save / flush / shutdown
- prefetch context and dialectic caches
- profile / search / answer / conclude
- local history / memory file migration

The contract types live in:

- `memory_backends/base.py`

Backend selection lives in:

- `memory_backends/factory.py`

Built-in Honcho adapter:

- `memory_backends/honcho.py`

External loader:

- `memory_backends/external.py`

## Stability expectations

- Honcho remains the only built-in backend shipped by Hermes.
- Hermes supports one explicit experimental external backend hook.
- The protocol is versioned through `protocol_version`.
- Hermes does not silently fall back to Honcho when an external backend fails
  to load.
- The runner-owned memory lifecycle remains the authoritative seam.

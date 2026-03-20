"""Explicit loader for out-of-tree Hermes memory backends."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from honcho_integration.client import GLOBAL_CONFIG_PATH, HOST
from memory_backends.base import (
    MemoryBackendBundle,
    MemoryBackendLoadError,
    validate_memory_backend_recall_surface,
    validate_memory_backend_bundle,
)


def _resolve_factory(factory_path: str):
    module_name, sep, attr_name = factory_path.partition(":")
    if not sep or not module_name or not attr_name:
        raise MemoryBackendLoadError(
            "memory backend factory must use 'package.module:callable' format"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised in higher-level tests
        raise MemoryBackendLoadError(
            f"failed to import memory backend module '{module_name}': {exc}"
        ) from exc

    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise MemoryBackendLoadError(
            f"memory backend factory '{factory_path}' is not callable"
        )
    return factory


def _coerce_bundle(raw_bundle: Any) -> MemoryBackendBundle:
    if isinstance(raw_bundle, MemoryBackendBundle):
        return raw_bundle
    if isinstance(raw_bundle, tuple) and len(raw_bundle) == 3:
        manager, config, manifest = raw_bundle
        return MemoryBackendBundle(manager=manager, config=config, manifest=manifest)
    raise MemoryBackendLoadError(
        "memory backend factory must return MemoryBackendBundle or a 3-tuple"
    )


def load_external_backend(
    factory_path: str,
    *,
    host: str = HOST,
    config_path: Path | None = None,
) -> MemoryBackendBundle:
    """Load and validate an out-of-tree memory backend factory."""

    factory = _resolve_factory(factory_path)
    resolved_config_path = Path(config_path) if config_path is not None else GLOBAL_CONFIG_PATH
    raw_bundle = factory(host=host, config_path=str(resolved_config_path))
    bundle = validate_memory_backend_bundle(_coerce_bundle(raw_bundle))
    if getattr(bundle.config, "enabled", False) and bundle.manager is None:
        raise MemoryBackendLoadError(
            "configured external memory backend did not produce an active manager"
        )
    return validate_memory_backend_recall_surface(bundle)

"""Backend selection entrypoint for Hermes memory integration."""

from __future__ import annotations

from pathlib import Path

from honcho_integration.client import GLOBAL_CONFIG_PATH, HOST, HonchoClientConfig
from memory_backends.base import (
    MemoryBackendBundle,
    MemoryBackendLoadError,
    MemoryBackendManifest,
)
from memory_backends.external import load_external_backend
from memory_backends.honcho import load_honcho_backend


def load_memory_backend(
    *,
    host: str = HOST,
    config_path: Path | None = None,
) -> MemoryBackendBundle:
    """Load the selected Hermes memory backend."""

    hcfg = HonchoClientConfig.from_global_config(host=host, config_path=config_path)
    if hcfg.memory_backend_factory:
        if not hcfg.enabled:
            return MemoryBackendBundle(
                manager=None,
                config=hcfg,
                manifest=MemoryBackendManifest(
                    backend_id="external",
                    display_name="External Memory Backend",
                    capabilities=frozenset(),
                    config_source=str(config_path or GLOBAL_CONFIG_PATH),
                ),
            )
        bundle = load_external_backend(
            hcfg.memory_backend_factory,
            host=host,
            config_path=config_path,
        )
        if getattr(bundle.config, "enabled", False) and bundle.manager is None:
            raise MemoryBackendLoadError(
                "configured external memory backend did not produce an active manager"
            )
        return bundle
    return load_honcho_backend(host=host, config_path=config_path, config=hcfg)

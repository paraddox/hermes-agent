"""Built-in Honcho implementation of the Hermes memory backend contract."""

from __future__ import annotations

from pathlib import Path

from honcho_integration.client import GLOBAL_CONFIG_PATH, HOST, HonchoClientConfig, get_honcho_client
from honcho_integration.session import HonchoSessionManager
from memory_backends.base import (
    MemoryBackendBundle,
    MemoryBackendCapability,
    MemoryBackendManifest,
    validate_memory_backend_bundle,
)


_HONCHO_CAPABILITIES = frozenset(
    {
        MemoryBackendCapability.PROFILE.value,
        MemoryBackendCapability.SEARCH.value,
        MemoryBackendCapability.ANSWER.value,
        MemoryBackendCapability.CONCLUDE.value,
        MemoryBackendCapability.PREFETCH.value,
        MemoryBackendCapability.MIGRATE.value,
        MemoryBackendCapability.AI_IDENTITY.value,
    }
)


def load_honcho_backend(
    *,
    host: str = HOST,
    config_path: Path | None = None,
    config: HonchoClientConfig | None = None,
) -> MemoryBackendBundle:
    """Load the built-in Honcho backend."""

    hcfg = config or HonchoClientConfig.from_global_config(host=host, config_path=config_path)
    manager = None
    if hcfg.enabled and hcfg.api_key:
        client = get_honcho_client(hcfg)
        manager = HonchoSessionManager(
            honcho=client,
            config=hcfg,
            context_tokens=hcfg.context_tokens,
        )

    bundle = MemoryBackendBundle(
        manager=manager,
        config=hcfg,
        manifest=MemoryBackendManifest(
            backend_id="honcho",
            display_name="Honcho",
            capabilities=_HONCHO_CAPABILITIES,
            config_source=str(config_path or GLOBAL_CONFIG_PATH),
        ),
    )
    return validate_memory_backend_bundle(bundle)

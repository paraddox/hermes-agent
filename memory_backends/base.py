"""Shared types and validation for Hermes memory backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


PROTOCOL_VERSION = 1


class MemoryBackendLoadError(RuntimeError):
    """Raised when a configured memory backend cannot be loaded safely."""


class MemoryBackendCapability(str, Enum):
    PROFILE = "profile"
    SEARCH = "search"
    ANSWER = "answer"
    CONCLUDE = "conclude"
    PREFETCH = "prefetch"
    MIGRATE = "migrate"
    AI_IDENTITY = "ai_identity"


@runtime_checkable
class MemoryBackendConfigView(Protocol):
    enabled: bool
    workspace_id: str
    peer_name: str | None
    ai_peer: str
    memory_mode: str
    write_frequency: str | int
    recall_mode: str
    context_tokens: int | None

    def peer_memory_mode(self, peer_name: str) -> str:
        ...

    def resolve_session_name(
        self,
        cwd: str | None = None,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        ...


@runtime_checkable
class MemoryBackendManager(Protocol):
    def get_or_create(self, session_key: str) -> Any:
        ...

    def save(self, session: Any) -> None:
        ...

    def flush_all(self) -> None:
        ...

    def shutdown(self) -> None:
        ...

    def get_prefetch_context(self, session_key: str, user_message: str | None = None) -> dict:
        ...

    def prefetch_context(self, session_key: str, user_message: str | None = None) -> None:
        ...

    def set_context_result(self, session_key: str, result: Any) -> None:
        ...

    def pop_context_result(self, session_key: str) -> Any:
        ...

    def prefetch_dialectic(self, session_key: str, query: str) -> None:
        ...

    def set_dialectic_result(self, session_key: str, result: str) -> None:
        ...

    def pop_dialectic_result(self, session_key: str) -> str | None:
        ...

    def get_peer_card(self, session_key: str) -> str | list[str]:
        ...

    def search_context(self, session_key: str, query: str, max_tokens: int = 800) -> str:
        ...

    def dialectic_query(
        self,
        session_key: str,
        query: str,
        peer: str = "user",
        reasoning_level: str | None = None,
    ) -> str:
        ...

    def create_conclusion(self, session_key: str, content: str) -> bool:
        ...

    def migrate_local_history(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        ...

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        ...


@runtime_checkable
class MemoryBackendAIIdentityManager(Protocol):
    def seed_ai_identity(self, session_key: str, content: str, source: str = "manual") -> bool:
        ...

    def get_ai_representation(self, session_key: str) -> dict[str, str]:
        ...


@runtime_checkable
class MemoryBackendSession(Protocol):
    messages: Any

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        ...


@dataclass(frozen=True)
class MemoryBackendManifest:
    protocol_version: int = PROTOCOL_VERSION
    backend_id: str = ""
    display_name: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    config_source: str = ""


@dataclass(frozen=True)
class MemoryBackendBundle:
    manager: Any | None
    config: Any
    manifest: MemoryBackendManifest


_REQUIRED_CONFIG_ATTRS = (
    "enabled",
    "workspace_id",
    "peer_name",
    "ai_peer",
    "memory_mode",
    "write_frequency",
    "recall_mode",
    "context_tokens",
)
_REQUIRED_CONFIG_METHODS = ("peer_memory_mode", "resolve_session_name")
_REQUIRED_MANAGER_METHODS = (
    "get_or_create",
    "save",
    "flush_all",
    "shutdown",
    "get_prefetch_context",
    "prefetch_context",
    "set_context_result",
    "pop_context_result",
    "prefetch_dialectic",
    "set_dialectic_result",
    "pop_dialectic_result",
    "get_peer_card",
    "search_context",
    "dialectic_query",
    "create_conclusion",
    "migrate_local_history",
    "migrate_memory_files",
)
_KNOWN_CAPABILITIES = frozenset(capability.value for capability in MemoryBackendCapability)
_READ_CAPABILITIES = frozenset(
    {
        MemoryBackendCapability.PROFILE.value,
        MemoryBackendCapability.SEARCH.value,
        MemoryBackendCapability.ANSWER.value,
    }
)
_OPTIONAL_CAPABILITY_METHODS = {
    MemoryBackendCapability.AI_IDENTITY.value: (
        "seed_ai_identity",
        "get_ai_representation",
    ),
}


def validate_memory_backend_bundle(bundle: MemoryBackendBundle) -> MemoryBackendBundle:
    """Validate a memory backend bundle before Hermes accepts it."""

    manifest = bundle.manifest
    if not isinstance(manifest, MemoryBackendManifest):
        raise MemoryBackendLoadError(
            "memory backend manifest must return a MemoryBackendManifest"
        )
    if manifest.protocol_version != PROTOCOL_VERSION:
        raise MemoryBackendLoadError(
            f"memory backend protocol_version {manifest.protocol_version} is unsupported"
        )
    if not manifest.backend_id or not manifest.display_name:
        raise MemoryBackendLoadError("memory backend manifest must define backend_id and display_name")
    if not manifest.config_source:
        raise MemoryBackendLoadError("memory backend manifest must define config_source")
    unknown_capabilities = sorted(set(manifest.capabilities) - _KNOWN_CAPABILITIES)
    if unknown_capabilities:
        raise MemoryBackendLoadError(
            "memory backend manifest declares unknown capabilities: "
            + ", ".join(unknown_capabilities)
        )

    config = bundle.config
    if config is None:
        raise MemoryBackendLoadError("memory backend must provide a config view")

    missing_config_attrs = [
        name for name in _REQUIRED_CONFIG_ATTRS if not hasattr(config, name)
    ]
    if missing_config_attrs:
        raise MemoryBackendLoadError(
            f"memory backend config is missing required attributes: {', '.join(missing_config_attrs)}"
        )

    missing_config_methods = [
        name for name in _REQUIRED_CONFIG_METHODS if not callable(getattr(config, name, None))
    ]
    if missing_config_methods:
        raise MemoryBackendLoadError(
            f"memory backend config is missing required methods: {', '.join(missing_config_methods)}"
        )

    manager = bundle.manager
    if manager is not None and not bool(getattr(config, "enabled", False)):
        raise MemoryBackendLoadError(
            "memory backend config is disabled but returned an active manager"
        )
    if manager is not None:
        missing_manager_methods = [
            name for name in _REQUIRED_MANAGER_METHODS if not callable(getattr(manager, name, None))
        ]
        if missing_manager_methods:
            raise MemoryBackendLoadError(
                "memory backend manager is missing required methods: "
                + ", ".join(missing_manager_methods)
            )
        for capability, method_names in _OPTIONAL_CAPABILITY_METHODS.items():
            if capability not in manifest.capabilities:
                continue
            missing_optional_methods = [
                name for name in method_names if not callable(getattr(manager, name, None))
            ]
            if missing_optional_methods:
                raise MemoryBackendLoadError(
                    f"memory backend manager declares capability '{capability}' but is missing required methods: "
                    + ", ".join(missing_optional_methods)
                )

    return bundle


def validate_memory_backend_session(session: Any) -> Any:
    """Validate the concrete session object returned by a backend manager."""

    if session is None:
        raise MemoryBackendLoadError("memory backend returned no session object")
    if not hasattr(session, "messages"):
        raise MemoryBackendLoadError(
            "memory backend session is missing required attribute: messages"
        )
    try:
        len(session.messages)
    except Exception as exc:
        raise MemoryBackendLoadError(
            "memory backend session.messages must be a sized collection"
        ) from exc
    if not callable(getattr(session, "add_message", None)):
        raise MemoryBackendLoadError(
            "memory backend session is missing required method: add_message"
        )
    return session


def validate_memory_backend_recall_surface(bundle: MemoryBackendBundle) -> MemoryBackendBundle:
    """Reject active backends whose recall mode exposes no usable read path."""

    config = bundle.config
    if not bool(getattr(config, "enabled", False)) or bundle.manager is None:
        return bundle

    capabilities = getattr(bundle.manifest, "capabilities", None)
    if capabilities is None:
        return bundle

    capability_set = set(capabilities)
    recall_mode = getattr(config, "recall_mode", "hybrid")
    if recall_mode not in {"hybrid", "context", "tools"}:
        recall_mode = "hybrid"

    has_prefetch = MemoryBackendCapability.PREFETCH.value in capability_set
    has_read_tools = bool(capability_set & _READ_CAPABILITIES)

    if recall_mode == "context" and not has_prefetch:
        raise MemoryBackendLoadError(
            "memory backend recall_mode 'context' requires 'prefetch' capability"
        )
    if recall_mode == "tools" and not has_read_tools:
        raise MemoryBackendLoadError(
            "memory backend recall_mode 'tools' requires at least one read capability: profile, search, or answer"
        )
    if recall_mode == "hybrid" and not (has_prefetch or has_read_tools):
        raise MemoryBackendLoadError(
            "memory backend recall_mode 'hybrid' requires either 'prefetch' or a read capability"
        )

    return bundle

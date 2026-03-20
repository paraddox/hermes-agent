"""Fake external memory backend providers for abstraction tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_backends.base import (
    MemoryBackendBundle,
    MemoryBackendCapability,
    MemoryBackendManifest,
)


@dataclass
class FakeConfig:
    enabled: bool = True
    workspace_id: str = "fake-workspace"
    peer_name: str | None = "user"
    ai_peer: str = "hermes"
    memory_mode: str = "hybrid"
    write_frequency: str | int = "async"
    recall_mode: str = "hybrid"
    context_tokens: int | None = 321

    def peer_memory_mode(self, peer_name: str) -> str:
        return self.memory_mode

    def resolve_session_name(
        self,
        cwd: str | None = None,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        return session_id or session_title or "fake-session"


class FakeManager:
    def get_or_create(self, session_key: str) -> Any:
        return {"session_key": session_key}

    def save(self, session: Any) -> None:
        return None

    def flush_all(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def get_prefetch_context(self, session_key: str, user_message: str | None = None) -> dict:
        return {}

    def prefetch_context(self, session_key: str, user_message: str | None = None) -> None:
        return None

    def set_context_result(self, session_key: str, result: Any) -> None:
        return None

    def pop_context_result(self, session_key: str) -> Any:
        return None

    def prefetch_dialectic(self, session_key: str, query: str) -> None:
        return None

    def set_dialectic_result(self, session_key: str, result: str) -> None:
        return None

    def pop_dialectic_result(self, session_key: str) -> str | None:
        return None

    def get_peer_card(self, session_key: str) -> str:
        return "Known facts"

    def search_context(self, session_key: str, query: str, max_tokens: int = 800) -> str:
        return "Search result"

    def dialectic_query(
        self,
        session_key: str,
        query: str,
        peer: str = "user",
        reasoning_level: str | None = None,
    ) -> str:
        return "Answer result"

    def create_conclusion(self, session_key: str, content: str) -> bool:
        return True

    def migrate_local_history(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        return True

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        return True


def _manifest(protocol_version: int = 1) -> MemoryBackendManifest:
    return MemoryBackendManifest(
        protocol_version=protocol_version,
        backend_id="fake-backend",
        display_name="Fake Backend",
        capabilities=frozenset(
            {
                MemoryBackendCapability.PROFILE.value,
                MemoryBackendCapability.SEARCH.value,
                MemoryBackendCapability.ANSWER.value,
                MemoryBackendCapability.CONCLUDE.value,
                MemoryBackendCapability.PREFETCH.value,
                MemoryBackendCapability.MIGRATE.value,
            }
        ),
        config_source="tests",
    )


def create_backend(*, host: str = "hermes", config_path: str | None = None) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(),
        manifest=_manifest(),
    )


def create_backend_invalid_manifest(*, host: str = "hermes", config_path: str | None = None) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(),
        manifest=_manifest(protocol_version=99),
    )


def create_backend_missing_method(*, host: str = "hermes", config_path: str | None = None) -> MemoryBackendBundle:
    class IncompleteManager:
        def get_or_create(self, session_key: str) -> Any:
            return {"session_key": session_key}

    return MemoryBackendBundle(
        manager=IncompleteManager(),
        config=FakeConfig(),
        manifest=_manifest(),
    )


def create_backend_unknown_capability(*, host: str = "hermes", config_path: str | None = None) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(),
        manifest=MemoryBackendManifest(
            backend_id="fake-backend",
            display_name="Fake Backend",
            capabilities=frozenset({"profile", "totally_unknown_capability"}),
            config_source="tests",
        ),
    )


def create_backend_ai_identity_missing_methods(
    *, host: str = "hermes", config_path: str | None = None
) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(),
        manifest=MemoryBackendManifest(
            backend_id="fake-backend",
            display_name="Fake Backend",
            capabilities=frozenset({"profile", "ai_identity"}),
            config_source="tests",
        ),
    )


def create_backend_enabled_without_manager(
    *, host: str = "hermes", config_path: str | None = None
) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=None,
        config=FakeConfig(enabled=True),
        manifest=_manifest(),
    )


def create_backend_disabled_with_manager(
    *, host: str = "hermes", config_path: str | None = None
) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(enabled=False),
        manifest=_manifest(),
    )


def create_backend_context_mode_without_prefetch(
    *, host: str = "hermes", config_path: str | None = None
) -> MemoryBackendBundle:
    return MemoryBackendBundle(
        manager=FakeManager(),
        config=FakeConfig(recall_mode="context"),
        manifest=MemoryBackendManifest(
            backend_id="fake-backend",
            display_name="Fake Backend",
            capabilities=frozenset(
                {
                    MemoryBackendCapability.PROFILE.value,
                    MemoryBackendCapability.SEARCH.value,
                }
            ),
            config_source="tests",
        ),
    )

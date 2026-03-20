"""Direct tests for the external memory backend loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from honcho_integration.client import GLOBAL_CONFIG_PATH
from memory_backends.base import (
    MemoryBackendBundle,
    MemoryBackendLoadError,
    MemoryBackendManifest,
)
from memory_backends.external import load_external_backend
from tests.fakes.fake_memory_backend import FakeConfig, FakeManager


def test_load_external_backend_accepts_direct_bundle():
    bundle = load_external_backend("tests.fakes.fake_memory_backend:create_backend")

    assert bundle.manifest.backend_id == "fake-backend"
    assert bundle.config.resolve_session_name(session_id="abc") == "abc"


def test_load_external_backend_accepts_legacy_tuple_return():
    manifest = MemoryBackendManifest(
        backend_id="tuple-backend",
        display_name="Tuple Backend",
        capabilities=frozenset({"profile", "search"}),
        config_source="tests",
    )

    def _factory(*, host: str = "hermes", config_path: str | None = None):
        return (FakeManager(), FakeConfig(), manifest)

    with patch("memory_backends.external._resolve_factory", return_value=_factory):
        bundle = load_external_backend("pkg.module:create_backend")

    assert isinstance(bundle, MemoryBackendBundle)
    assert bundle.manifest.backend_id == "tuple-backend"


def test_load_external_backend_rejects_invalid_return_shape():
    def _factory(*, host: str = "hermes", config_path: str | None = None):
        return object()

    with patch("memory_backends.external._resolve_factory", return_value=_factory):
        with pytest.raises(MemoryBackendLoadError, match="must return MemoryBackendBundle or a 3-tuple"):
            load_external_backend("pkg.module:create_backend")


def test_load_external_backend_rejects_manifest_without_config_source():
    manifest = MemoryBackendManifest(
        backend_id="tuple-backend",
        display_name="Tuple Backend",
        capabilities=frozenset({"profile", "search"}),
        config_source="",
    )

    def _factory(*, host: str = "hermes", config_path: str | None = None):
        return (FakeManager(), FakeConfig(), manifest)

    with patch("memory_backends.external._resolve_factory", return_value=_factory):
        with pytest.raises(MemoryBackendLoadError, match="must define config_source"):
            load_external_backend("pkg.module:create_backend")


def test_load_external_backend_rejects_non_manifest_objects():
    def _factory(*, host: str = "hermes", config_path: str | None = None):
        return (FakeManager(), FakeConfig(), {"backend_id": "bad"})

    with patch("memory_backends.external._resolve_factory", return_value=_factory):
        with pytest.raises(MemoryBackendLoadError, match="must return a MemoryBackendManifest"):
            load_external_backend("pkg.module:create_backend")


def test_load_external_backend_passes_resolved_default_config_path():
    observed = {}
    manifest = MemoryBackendManifest(
        backend_id="tuple-backend",
        display_name="Tuple Backend",
        capabilities=frozenset({"profile", "search"}),
        config_source="tests",
    )

    def _factory(*, host: str = "hermes", config_path: str | None = None):
        observed["host"] = host
        observed["config_path"] = config_path
        return (FakeManager(), FakeConfig(), manifest)

    with patch("memory_backends.external._resolve_factory", return_value=_factory):
        load_external_backend("pkg.module:create_backend")

    assert observed["host"] == "hermes"
    assert observed["config_path"] == str(Path(GLOBAL_CONFIG_PATH))


def test_load_external_backend_rejects_enabled_backend_without_manager():
    with pytest.raises(
        MemoryBackendLoadError,
        match="did not produce an active manager",
    ):
        load_external_backend(
            "tests.fakes.fake_memory_backend:create_backend_enabled_without_manager"
        )

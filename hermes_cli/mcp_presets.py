"""Provider-linked MCP preset resolution for setup-time convenience flows."""

from __future__ import annotations

import copy
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


_PRESET_DATA_PATH: Path | None = None
_SUPPORTED_DEFAULT_CHECKED_WHEN_KEYS = frozenset({"command_available"})


def _normalize_runtime_server_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def _config_transport_kind(config: Any) -> str | None:
    if not isinstance(config, dict):
        return None

    has_url = isinstance(config.get("url"), str) and bool(config["url"].strip())
    has_command = isinstance(config.get("command"), str) and bool(
        config["command"].strip()
    )
    if has_url and has_command:
        return "mixed"
    if has_url:
        return "http"
    if has_command:
        return "stdio"
    return None


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_config_value(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _server_transport_identity(server_config: Any) -> tuple[str, Any] | None:
    if not isinstance(server_config, dict):
        return None

    url = server_config.get("url")
    if isinstance(url, str) and url.strip():
        return (
            "http",
            _freeze_config_value(
                {
                    "url": url.strip(),
                    "headers": server_config.get("headers"),
                    "auth": server_config.get("auth"),
                }
            ),
        )

    command = server_config.get("command")
    if isinstance(command, str) and command.strip():
        return (
            "stdio",
            _freeze_config_value(
                {
                    "command": command.strip(),
                    "args": server_config.get("args") or [],
                    "env": server_config.get("env"),
                    "auth": server_config.get("auth"),
                }
            ),
        )

    return None


def _server_transport_identity_variants(server_config: Any) -> list[tuple[str, Any]]:
    identities: list[tuple[str, Any]] = []
    raw_identity = _server_transport_identity(server_config)
    if raw_identity is not None:
        identities.append(raw_identity)

    try:
        from hermes_cli.config import _expand_env_vars

        expanded_config = _expand_env_vars(server_config)
    except Exception:
        expanded_config = server_config

    expanded_identity = _server_transport_identity(expanded_config)
    if expanded_identity is not None and expanded_identity not in identities:
        identities.append(expanded_identity)

    return identities


def _read_preset_data_text() -> str:
    if _PRESET_DATA_PATH is not None:
        return _PRESET_DATA_PATH.read_text(encoding="utf-8")
    return (
        resources.files("hermes_cli")
        .joinpath("mcp_provider_presets.yaml")
        .read_text(encoding="utf-8")
    )


def snapshot_existing_mcp_servers_raw(
    config: dict[str, Any],
    *,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw MCP snapshot that preserves disk placeholders and in-memory edits."""

    snapshot: dict[str, Any] = {}
    raw_mcp_servers = raw_config.get("mcp_servers") if isinstance(raw_config, dict) else None
    if isinstance(raw_mcp_servers, dict):
        snapshot = copy.deepcopy(raw_mcp_servers)

    current_mcp_servers = config.get("mcp_servers")
    if isinstance(current_mcp_servers, dict):
        for name, server in current_mcp_servers.items():
            snapshot.setdefault(name, copy.deepcopy(server))

    return snapshot


def _match_key(entry: dict[str, Any]) -> tuple[str, str | None]:
    match = entry.get("match", {})
    provider = match.get("provider")
    model = match.get("model")
    return provider, model


def _format_match_desc(provider: str, model: str | None) -> str:
    provider_desc = f"provider '{provider}'"
    if model:
        provider_desc += f" model '{model}'"
    return provider_desc


def _entry_match_key(entry: Any) -> tuple[str, str | None] | None:
    if not isinstance(entry, dict):
        return None
    match = entry.get("match")
    if not isinstance(match, dict):
        return None
    provider = match.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        return None
    model = match.get("model")
    if model is None:
        return provider.strip(), None
    if not isinstance(model, str) or not model.strip():
        return None
    return provider.strip(), model.strip()


def _validate_server_entry(
    server_name: str,
    server_data: Any,
    match_desc: str,
) -> None:
    if not isinstance(server_data, dict):
        raise ValueError(
            f"MCP preset server '{server_name}' for {match_desc} must be a mapping"
        )

    config = server_data.get("config")
    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"MCP preset server '{server_name}' for {match_desc} must include a non-empty config mapping"
        )
    if _config_transport_kind(config) == "mixed":
        raise ValueError(
            f"MCP preset server '{server_name}' for {match_desc} cannot define both 'url' and 'command'"
        )

    if "default_checked" in server_data and not isinstance(server_data["default_checked"], bool):
        raise ValueError(
            f"MCP preset server '{server_name}' for {match_desc} has invalid default_checked"
        )

    when = server_data.get("default_checked_when")
    if when is not None:
        if not isinstance(when, dict):
            raise ValueError(
                f"MCP preset server '{server_name}' for {match_desc} has invalid default_checked_when"
            )
        unsupported_keys = set(when) - _SUPPORTED_DEFAULT_CHECKED_WHEN_KEYS
        if unsupported_keys:
            raise ValueError(
                "Unsupported default_checked_when keys for "
                f"MCP preset server '{server_name}' for {match_desc}: "
                + ", ".join(sorted(unsupported_keys))
            )
        command = when.get("command_available")
        if command is not None and (not isinstance(command, str) or not command.strip()):
            raise ValueError(
                f"MCP preset server '{server_name}' for {match_desc} has invalid command_available"
            )

    note = server_data.get("note_when_unchecked")
    if note is not None and not isinstance(note, str):
        raise ValueError(
            f"MCP preset server '{server_name}' for {match_desc} has invalid note_when_unchecked"
        )


def _validate_preset_entry(entry: Any, index: int) -> tuple[str, str | None]:
    if not isinstance(entry, dict):
        raise ValueError(f"MCP preset entry #{index + 1} must be a mapping")

    match = entry.get("match")
    if not isinstance(match, dict):
        raise ValueError(f"MCP preset entry #{index + 1} is missing a match mapping")

    provider = match.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(
            f"MCP preset entry #{index + 1} must include a non-empty match.provider"
        )
    model = match.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(
            f"MCP preset entry #{index + 1} has invalid match.model"
        )

    prompt = entry.get("prompt", {})
    if not isinstance(prompt, dict):
        raise ValueError(
            f"MCP preset entry #{index + 1} prompt must be a mapping"
        )
    default_value = prompt.get("default")
    if default_value is not None and not isinstance(default_value, bool):
        raise ValueError(
            f"MCP preset entry #{index + 1} prompt.default must be a boolean"
        )

    servers = entry.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(
            f"MCP preset entry #{index + 1} must include a non-empty servers mapping"
        )

    match_desc = f"provider '{provider}'" if model is None else f"provider '{provider}' model '{model}'"
    for server_name, server_data in servers.items():
        if not isinstance(server_name, str) or not server_name.strip():
            raise ValueError(
                f"MCP preset entry #{index + 1} has an invalid server name"
            )
        _validate_server_entry(server_name, server_data, match_desc)

    return provider, model.strip() if isinstance(model, str) else None


def _load_presets() -> list[dict[str, Any]]:
    try:
        raw = yaml.safe_load(_read_preset_data_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError("Invalid MCP preset YAML") from exc
    if not isinstance(raw, dict):
        raise ValueError("MCP preset data must be a mapping")
    presets = raw.get("presets", [])
    if not isinstance(presets, list):
        raise ValueError("MCP preset data 'presets' must be a list")
    return _validate_preset_catalog(presets)


def _matches(entry: dict[str, Any], provider: str, model: str | None) -> bool:
    match = entry.get("match", {})
    if not isinstance(match, dict):
        return False
    if match.get("provider") != provider:
        return False
    entry_model = match.get("model")
    if entry_model is None:
        return True
    return entry_model == model


def _deep_merge_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_server_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    base_transport = _config_transport_kind(base)
    override_transport = _config_transport_kind(override)
    if (
        base_transport in {"http", "stdio"}
        and override_transport in {"http", "stdio"}
        and base_transport != override_transport
    ):
        return copy.deepcopy(override)
    return _deep_merge_mappings(base, override)


def _merge_server_entry(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if key == "config" and isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_server_config(existing, value)
        elif isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _get_selected_model_name(config: dict[str, Any]) -> str | None:
    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_name = current_model.get("default")
        if isinstance(model_name, str):
            return model_name.strip() or None
        return None
    if isinstance(current_model, str):
        return current_model.strip() or None
    return None


def _validate_runtime_server_name_uniqueness(
    servers: dict[str, Any],
    *,
    provider: str,
    model: str | None,
) -> None:
    seen: dict[str, str] = {}
    for server_name in servers:
        normalized_name = _normalize_runtime_server_name(server_name)
        existing_name = seen.get(normalized_name)
        if existing_name is None:
            seen[normalized_name] = server_name
            continue

        raise ValueError(
            "MCP preset bundle for "
            f"{_format_match_desc(provider, model)} contains servers '{existing_name}' and "
            f"'{server_name}' with the same runtime name"
        )


def _build_resolved_bundle(
    presets: list[dict[str, Any]],
    provider: str,
    model: str | None,
) -> dict[str, Any] | None:
    provider_entries: list[dict[str, Any]] = []
    model_entries: list[dict[str, Any]] = []

    for entry in presets:
        entry_match = _match_key(entry)
        if entry_match[0] != provider:
            continue

        if not _matches(entry, provider, model):
            continue

        if entry_match[1] is None:
            provider_entries.append(entry)
        else:
            model_entries.append(entry)

    if not provider_entries and not model_entries:
        return None

    merged: dict[str, Any] = {
        "prompt": {},
        "servers": {},
    }
    for entry in provider_entries + model_entries:
        prompt = entry.get("prompt", {})
        if isinstance(prompt, dict):
            merged["prompt"].update(copy.deepcopy(prompt))
        servers = entry.get("servers", {})
        if isinstance(servers, dict):
            for server_name, server_data in servers.items():
                if isinstance(server_data, dict) and isinstance(
                    merged["servers"].get(server_name), dict
                ):
                    merged["servers"][server_name] = _merge_server_entry(
                        merged["servers"][server_name],
                        server_data,
                    )
                else:
                    merged["servers"][server_name] = copy.deepcopy(server_data)

    return merged


def _validate_resolved_bundle(
    bundle: dict[str, Any],
    *,
    provider: str,
    model: str | None,
) -> None:
    servers = bundle.get("servers", {})
    if not isinstance(servers, dict) or not servers:
        raise ValueError(
            f"MCP preset bundle for {_format_match_desc(provider, model)} has no servers"
        )

    _validate_runtime_server_name_uniqueness(
        servers,
        provider=provider,
        model=model,
    )
    for server_name, server_data in servers.items():
        if not isinstance(server_data, dict):
            raise ValueError(
                f"MCP preset bundle for {_format_match_desc(provider, model)} has invalid server '{server_name}'"
            )
        config = server_data.get("config")
        if not isinstance(config, dict) or not config:
            raise ValueError(
                f"MCP preset bundle for {_format_match_desc(provider, model)} has invalid config for server '{server_name}'"
            )
        if _config_transport_kind(config) == "mixed":
            raise ValueError(
                f"MCP preset bundle for {_format_match_desc(provider, model)} has server '{server_name}' with both 'url' and 'command'"
            )

    prompt = bundle.get("prompt")
    if (
        not isinstance(prompt, dict)
        or not prompt.get("question")
        or not prompt.get("checklist_title")
    ):
        raise ValueError(
            f"MCP preset bundle for {_format_match_desc(provider, model)} is missing prompt text"
        )


def _validate_preset_catalog(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated_presets: list[dict[str, Any]] = []
    seen_matches: set[tuple[str, str | None]] = set()
    ordered_matches: list[tuple[str, str | None]] = []

    for idx, entry in enumerate(presets):
        validated_match = _validate_preset_entry(entry, idx)
        if validated_match in seen_matches:
            provider, entry_model = validated_match
            if entry_model is None:
                raise ValueError(f"Duplicate MCP preset match for provider '{provider}'")
            raise ValueError(
                f"Duplicate MCP preset match for provider '{provider}' model '{entry_model}'"
            )
        seen_matches.add(validated_match)
        ordered_matches.append(validated_match)
        validated_presets.append(entry)

    for provider, model in ordered_matches:
        bundle = _build_resolved_bundle(validated_presets, provider, model)
        if bundle is None:
            raise ValueError(
                f"MCP preset bundle for {_format_match_desc(provider, model)} could not be resolved"
            )
        _validate_resolved_bundle(
            bundle,
            provider=provider,
            model=model,
        )

    return validated_presets


def resolve_provider_mcp_bundle(provider: str, model: str | None = None) -> dict[str, Any] | None:
    """Resolve one setup-time MCP bundle for a provider and optional model.

    Resolution order:
    1. provider-only entries
    2. provider+model exact matches

    Later matches override earlier matches on prompt fields and server keys.
    """

    presets = _load_presets()
    bundle = _build_resolved_bundle(presets, provider, model)
    if bundle is None:
        return None

    _validate_resolved_bundle(
        bundle,
        provider=provider,
        model=model,
    )
    return bundle


def configure_provider_mcp_bundle(
    config: dict[str, Any],
    selected_provider: str | None,
    *,
    existing_mcp_servers_raw: dict[str, Any] | None = None,
    prompt_yes_no,
    prompt_checklist,
    print_warning,
    print_success,
    logger,
) -> None:
    if not selected_provider:
        return

    bundle_model = _get_selected_model_name(config)
    try:
        provider_bundle = resolve_provider_mcp_bundle(
            selected_provider,
            model=bundle_model,
        )
    except Exception:
        logger.exception(
            "Failed to resolve provider-linked MCP presets for provider=%s model=%s",
            selected_provider,
            bundle_model,
        )
        print_warning(
            "Failed to resolve provider-linked MCP presets; continuing without them."
        )
        return

    if not provider_bundle:
        return

    prompt_cfg = provider_bundle.get("prompt", {})
    enable_provider_mcp = prompt_yes_no(
        prompt_cfg.get("question", "Enable MCP servers during setup?"),
        default=bool(prompt_cfg.get("default", True)),
    )
    if not enable_provider_mcp:
        return

    bundle_server_names = list(provider_bundle.get("servers", {}))
    default_checked, notes = get_bundle_default_checked_servers(provider_bundle)
    default_checked_set = set(default_checked)
    for note in notes:
        print_warning(note)

    pre_selected = [
        idx
        for idx, server_name in enumerate(bundle_server_names)
        if server_name in default_checked_set
    ]
    selected_indices = prompt_checklist(
        prompt_cfg.get("checklist_title", "Select MCP servers to enable:"),
        bundle_server_names,
        pre_selected,
    )
    selected_names = [
        bundle_server_names[idx]
        for idx in selected_indices
        if isinstance(idx, int) and 0 <= idx < len(bundle_server_names)
    ]
    summary = merge_selected_provider_mcp_servers(
        config,
        provider_bundle,
        selected_names,
        existing_mcp_servers_raw=existing_mcp_servers_raw,
    )
    added = sorted(summary.get("added", []))
    skipped = sorted(summary.get("skipped_existing", []))
    skipped_runtime_collisions = summary.get("skipped_runtime_collisions", [])
    skipped_equivalent_existing = summary.get("skipped_equivalent_existing", [])
    if added:
        print_success(f"MCP servers added: {', '.join(added)}")
    if skipped:
        print_warning(
            "MCP servers already configured (not overwritten): "
            + ", ".join(skipped)
        )
    if skipped_runtime_collisions:
        collision_desc = ", ".join(
            f"{item['requested']} (matches existing {item['existing']})"
            for item in skipped_runtime_collisions
            if isinstance(item, dict)
            and isinstance(item.get("requested"), str)
            and isinstance(item.get("existing"), str)
        )
        if collision_desc:
            print_warning(
                "MCP servers skipped because an equivalent server is already configured: "
                + collision_desc
            )
    if skipped_equivalent_existing:
        equivalent_desc = ", ".join(
            f"{item['requested']} (matches existing {item['existing']})"
            for item in skipped_equivalent_existing
            if isinstance(item, dict)
            and isinstance(item.get("requested"), str)
            and isinstance(item.get("existing"), str)
        )
        if equivalent_desc:
            print_warning(
                "MCP servers already configured under a different name (not duplicated): "
                + equivalent_desc
            )


def get_bundle_default_checked_servers(bundle: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return default-checked server names and explanatory notes."""

    checked: list[str] = []
    notes: list[str] = []

    servers = bundle.get("servers", {})
    if not isinstance(servers, dict):
        return checked, notes

    for server_name, server_data in servers.items():
        if not isinstance(server_data, dict):
            continue

        if server_data.get("default_checked") is True:
            checked.append(server_name)
            continue

        when = server_data.get("default_checked_when")
        if not isinstance(when, dict):
            continue

        command = when.get("command_available")
        if isinstance(command, str) and command.strip():
            if shutil.which(command.strip()) is not None:
                checked.append(server_name)
            else:
                note = server_data.get("note_when_unchecked")
                if isinstance(note, str) and note.strip():
                    notes.append(note.strip())

    return checked, notes


def merge_selected_provider_mcp_servers(
    config: dict[str, Any],
    bundle: dict[str, Any],
    selected_servers: list[str],
    *,
    existing_mcp_servers_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge selected bundle servers into config without overwriting user config."""

    if not selected_servers:
        return {
            "added": [],
            "skipped_existing": [],
            "skipped_runtime_collisions": [],
            "skipped_equivalent_existing": [],
        }

    servers = bundle.get("servers", {})
    if not isinstance(servers, dict):
        return {
            "added": [],
            "skipped_existing": [],
            "skipped_runtime_collisions": [],
            "skipped_equivalent_existing": [],
        }

    existing_mcp_servers = config.get("mcp_servers")
    if not isinstance(existing_mcp_servers, dict):
        existing_mcp_servers = {}
    raw_existing_mcp_servers = (
        existing_mcp_servers_raw
        if isinstance(existing_mcp_servers_raw, dict)
        else existing_mcp_servers
    )

    mcp_servers = None
    summary = {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [],
    }
    normalized_existing_names = {
        _normalize_runtime_server_name(existing_name): existing_name
        for existing_name in existing_mcp_servers
    }
    existing_transport_identities: dict[tuple[str, Any], str] = {}
    for existing_source in (raw_existing_mcp_servers, existing_mcp_servers):
        for existing_name, existing_config in existing_source.items():
            for identity in _server_transport_identity_variants(existing_config):
                existing_transport_identities.setdefault(identity, existing_name)

    for server_name in selected_servers:
        if server_name in existing_mcp_servers:
            summary["skipped_existing"].append(server_name)
            continue

        normalized_name = _normalize_runtime_server_name(server_name)
        existing_collision = normalized_existing_names.get(normalized_name)
        if existing_collision is not None:
            summary["skipped_runtime_collisions"].append(
                {
                    "requested": server_name,
                    "existing": existing_collision,
                }
            )
            continue

        server_payload = servers.get(server_name)
        if not isinstance(server_payload, dict):
            continue

        server_config = server_payload.get("config")
        if not isinstance(server_config, dict):
            continue

        transport_identities = _server_transport_identity_variants(server_config)
        existing_equivalent = next(
            (
                existing_transport_identities[identity]
                for identity in transport_identities
                if identity in existing_transport_identities
            ),
            None,
        )
        if existing_equivalent is not None:
            summary["skipped_equivalent_existing"].append(
                {
                    "requested": server_name,
                    "existing": existing_equivalent,
                }
            )
            continue

        if mcp_servers is None:
            current_mcp_servers = config.get("mcp_servers")
            if not isinstance(current_mcp_servers, dict):
                current_mcp_servers = {}
                config["mcp_servers"] = current_mcp_servers
            mcp_servers = current_mcp_servers

        merged_entry = copy.deepcopy(server_config)
        merged_entry["enabled"] = True
        mcp_servers[server_name] = merged_entry
        summary["added"].append(server_name)
        normalized_existing_names[normalized_name] = server_name
        for identity in transport_identities:
            existing_transport_identities.setdefault(identity, server_name)

    return summary

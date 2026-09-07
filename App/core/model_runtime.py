"""Central runtime selection for the APP's experimental model defaults.

The desktop APP is intentionally registry/frozen-replay first.  This module
keeps the model choice in one auditable place so that a new result cannot be
enabled by changing a label only.  It also provides a safe, explicit fallback
when an optional policy dependency or artifact is unavailable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "App" / "config" / "runtime_config.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved model configuration used by replay, validation and reports."""

    enhanced_enabled: bool
    knowledge_guided_mode: str
    agent_policy: str
    policy_model_path: str | None
    replay_evaluation_path: str | None
    replay_decisions_path: str | None
    fallback_policy: str
    fallback_reason: str | None
    policy_artifact_ready: bool
    frozen_replay: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_config() -> dict[str, Any]:
    config = _read_json(RUNTIME_CONFIG_PATH)
    return config if config else {
        "mode": "frozen_replay",
        "knowledge_guided_mode": "soft_correlated",
        "agent_policy": "td3",
        "fallback_policy": "conservative",
        "enhanced_runtime_enabled": True,
    }


def resolve_runtime_selection() -> RuntimeSelection:
    config = load_runtime_config()
    enhanced_enabled = _env_bool(
        "IFP_ENHANCED_RUNTIME",
        bool(config.get("enhanced_runtime_enabled", True)),
    )
    kg_mode = os.environ.get(
        "IFP_KG_MODE",
        str(config.get("knowledge_guided_mode", "soft_correlated")),
    ).strip().lower()
    if kg_mode not in {"off", "uncertainty_only", "soft_prior", "soft_correlated"}:
        kg_mode = "soft_correlated"

    policy = os.environ.get("IFP_AGENT_POLICY", str(config.get("agent_policy", "td3"))).strip().lower()
    if policy not in {"td3", "sac", "ppo", "conservative"}:
        policy = "td3"
    fallback = str(config.get("fallback_policy", "conservative")).strip().lower() or "conservative"
    model_path = _resolve(config.get("policy_model_path"))
    fallback_replay = config.get("fallback_replay", {}) or {}
    replay_evaluation_path = _resolve(config.get("replay_evaluation_path"))
    replay_decisions_path = _resolve(config.get("replay_decisions_path"))
    ready = bool(model_path and model_path.exists()) if policy != "conservative" else True
    reason = None
    if not enhanced_enabled:
        kg_mode = "off"
        policy = "conservative"
        model_path = None
        replay_evaluation_path = _resolve(fallback_replay.get("evaluation_csv"))
        replay_decisions_path = _resolve(fallback_replay.get("decisions_json"))
        ready = True
        reason = "IFP_ENHANCED_RUNTIME=false"
    elif policy != "conservative" and not ready:
        reason = f"policy artifact unavailable: {model_path}"
        policy = fallback if fallback in {"td3", "sac", "ppo", "conservative"} else "conservative"
        model_path = None
        replay_evaluation_path = _resolve(fallback_replay.get("evaluation_csv"))
        replay_decisions_path = _resolve(fallback_replay.get("decisions_json"))
    return RuntimeSelection(
        enhanced_enabled=enhanced_enabled,
        knowledge_guided_mode=kg_mode,
        agent_policy=policy,
        policy_model_path=str(model_path) if model_path else None,
        replay_evaluation_path=str(replay_evaluation_path) if replay_evaluation_path else None,
        replay_decisions_path=str(replay_decisions_path) if replay_decisions_path else None,
        fallback_policy=fallback,
        fallback_reason=reason,
        policy_artifact_ready=ready,
        frozen_replay=str(config.get("mode", "frozen_replay")) == "frozen_replay",
    )


def load_selected_policy(selection: RuntimeSelection | None = None):
    """Load the selected SB3 policy lazily; return ``None`` on safe fallback.

    The GUI environment does not need Stable-Baselines3 merely to display a
    frozen replay.  Importing it only here prevents an optional dependency from
    breaking APP startup.  Callers must treat ``None`` as conservative fallback.
    """

    selection = selection or resolve_runtime_selection()
    if selection.agent_policy == "conservative" or not selection.policy_model_path:
        return None
    try:
        from stable_baselines3 import PPO, SAC, TD3

        policy_class = {"ppo": PPO, "sac": SAC, "td3": TD3}[selection.agent_policy]
        return policy_class.load(selection.policy_model_path, device="cpu")
    except Exception:
        return None

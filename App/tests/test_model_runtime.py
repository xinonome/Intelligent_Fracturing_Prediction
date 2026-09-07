from __future__ import annotations

from App.core.model_runtime import resolve_runtime_selection


def test_default_runtime_selects_available_sac_and_soft_correlated(monkeypatch):
    monkeypatch.delenv("IFP_ENHANCED_RUNTIME", raising=False)
    monkeypatch.delenv("IFP_AGENT_POLICY", raising=False)
    monkeypatch.delenv("IFP_KG_MODE", raising=False)
    selection = resolve_runtime_selection()
    assert selection.agent_policy == "sac"
    assert selection.knowledge_guided_mode == "soft_correlated"
    assert selection.policy_artifact_ready is True
    assert selection.policy_model_path and selection.policy_model_path.endswith("sac_fracturing_policy.zip")


def test_enhanced_runtime_can_revert_to_real_previous_replay(monkeypatch):
    monkeypatch.setenv("IFP_ENHANCED_RUNTIME", "false")
    selection = resolve_runtime_selection()
    assert selection.agent_policy == "conservative"
    assert selection.knowledge_guided_mode == "off"
    assert selection.fallback_reason == "IFP_ENHANCED_RUNTIME=false"
    assert selection.replay_evaluation_path and "ppo_20260812_184048" in selection.replay_evaluation_path
    assert selection.replay_decisions_path and selection.replay_decisions_path.endswith("human_machine_decisions.json")

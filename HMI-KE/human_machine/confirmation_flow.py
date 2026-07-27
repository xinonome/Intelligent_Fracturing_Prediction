from __future__ import annotations

from .permission_rules import can_execute


def confirmation_status(role: str, requires_confirmation: bool, action: str = "confirm") -> dict:
    allowed = can_execute(role, action)
    if not requires_confirmation:
        return {"status": "auto_allowed", "allowed": True, "reason": "低风险建议无需人工确认"}
    if allowed:
        return {"status": "waiting_confirmation", "allowed": True, "reason": "需要人工确认后执行"}
    return {"status": "blocked", "allowed": False, "reason": "当前角色权限不足"}

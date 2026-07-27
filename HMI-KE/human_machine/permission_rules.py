from __future__ import annotations


ROLE_PERMISSIONS = {
    "viewer": {"view"},
    "engineer": {"view", "advise", "confirm"},
    "supervisor": {"view", "advise", "confirm", "control"},
}


def can_execute(role: str, action: str) -> bool:
    return action in ROLE_PERMISSIONS.get(role, set())

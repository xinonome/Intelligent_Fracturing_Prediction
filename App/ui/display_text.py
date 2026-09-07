"""Small display-only text normalization helpers for the acceptance APP."""

from __future__ import annotations


def display_text(value) -> str:
    """Use the requested English spelling for visible baseline labels.

    This intentionally applies only at the UI boundary.  Machine-readable
    keys such as ``direct_baseline`` and source documents remain unchanged.
    """

    return str(value).replace("基准线", "baseline").replace("基线", "baseline")

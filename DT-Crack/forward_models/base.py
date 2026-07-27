from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardResult:
    """Standard output for fracture forward models."""

    state: np.ndarray
    observations: np.ndarray
    state_names: list[str]
    observation_names: list[str]


class ForwardModel(ABC):
    """Unified interface for PKN, KGD, BEM or surrogate forward models."""

    @abstractmethod
    def simulate(
        self,
        state: np.ndarray,
        controls: dict[str, np.ndarray | float],
        reservoir: dict[str, float],
        dt: float,
    ) -> ForwardResult:
        """Propagate fracture state and return observable responses."""

# mppi_tutorial/interfaces/visualization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch


# ----------------------------
# Common data containers
# ----------------------------
@dataclass
class RolloutLog:
    """
    Optional per-step log container (from MPPI record_sample outputs).

    Notes on shapes (following your project conventions):
      - Xs:    [B, T+1, S]   sampled rollouts
      - noise: [B, T,   U]   sampled control noise
      - costs: [B]           total cost per rollout
      - Xopt:  [T+1, S]      predicted "optimal plan" trajectory
    """
    Xs: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    costs: Optional[torch.Tensor] = None
    Xopt: Optional[torch.Tensor] = None

    # extra info (env cfg, obstacles, goal, debug stats, etc.)
    meta: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationOutput:
    """
    Result returned by renderers/visualizers when they create artifacts.
    """
    save_path: str
    extra: Optional[Dict[str, Any]] = None


# ----------------------------
# Offline Renderer (Matplotlib)
# ----------------------------
class BaseOfflineRenderer(Protocol):
    """
    Offline renderer interface:
      - consumes full trajectories/logs
      - produces artifacts (gif/png/mp4, etc.)

    Implementations: defaults/visualization/offline/*.py
    """
    def render(self, **kwargs) -> VisualizationOutput:
        ...


# ----------------------------
# Online Visualizer (MuJoCo/ROS2)
# ----------------------------
class BaseOnlineVisualizer(Protocol):
    """
    Online visualizer interface:
      - step-based updates (realtime viewer / ROS2 publish)
      - optional finalize artifact or just cleanup

    Implementations: defaults/visualization/online/*.py
    Tools (like offline_recorder) can also implement this interface.
    """
    def reset(self, **kwargs) -> None:
        ...

    def update(
        self,
        t: int,
        x: torch.Tensor,
        u: torch.Tensor,
        log: Optional[RolloutLog] = None,
        **kwargs
    ) -> None:
        ...

    def finalize(self, **kwargs) -> Optional[VisualizationOutput]:
        ...

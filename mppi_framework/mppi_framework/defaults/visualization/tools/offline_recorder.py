# defaults/visualization/tools/offline_recorder.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch

from mppi_framework.core.registry import VIS_VISUALIZERS
from mppi_framework.interfaces.visualization import RolloutLog, VisualizationOutput


@VIS_VISUALIZERS.register("offline_recorder")
class OfflineRecorder:
    """
    Online loop에서 데이터를 모으고(finalize에서만),
    offline renderer(render-only)를 호출해서 gif/png 같은 결과물을 생성하는 도구.

    - Matplotlib을 실시간으로 띄우지 않음.
    - MuJoCo/ROS2 online visualizer와 동일한 reset/update/finalize 인터페이스로 사용 가능.
    """

    def __init__(
        self,
        *,
        renderer,                       # offline renderer instance (render-only)
        save_path: str,
        dt: float,
        renderer_kwargs: Optional[Dict[str, Any]] = None,

        # what to record
        record_x: bool = True,          # executed state trajectory
        record_u: bool = False,         # applied controls
        record_Xss: bool = True,        # sampled rollouts (can be heavy)
        record_Xopt: bool = True,       # predicted plan

        # memory controls for Xss
        Xss_stride: int = 1,            # store Xss every k steps
        max_rollouts: Optional[int] = None,   # cap number of rollouts B kept per frame
        state_slice: Optional[slice] = None,  # e.g. slice(0,2) to keep only position dims
    ):
        self.renderer = renderer
        self.save_path = save_path
        self.dt = float(dt)
        self.renderer_kwargs = renderer_kwargs or {}

        self.record_x = record_x
        self.record_u = record_u
        self.record_Xss = record_Xss
        self.record_Xopt = record_Xopt

        self.Xss_stride = int(max(1, Xss_stride))
        self.max_rollouts = max_rollouts
        self.state_slice = state_slice

        self.reset()

    def reset(self, **kwargs) -> None:
        self._xs: List[np.ndarray] = []
        self._us: List[np.ndarray] = []
        self._Xss: List[np.ndarray] = []
        self._Xopt: List[np.ndarray] = []
        self._T: int = 0

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _maybe_slice_state(self, arr: np.ndarray) -> np.ndarray:
        if self.state_slice is None:
            return arr
        # support last-dim slice (state dimension)
        return arr[..., self.state_slice]

    def update(
        self,
        t: int,
        x: torch.Tensor,
        u: torch.Tensor,
        log: Optional[RolloutLog] = None,
        **kwargs,
    ) -> None:
        """
        Args:
          t: step index
          x: current (or next) state tensor
          u: applied control tensor
          log: optional MPPI record_sample package (Xs/noise/Xopt/costs)
        """
        self._T = max(self._T, t + 1)

        if self.record_x:
            x_np = self._maybe_slice_state(self._to_numpy(x))
            self._xs.append(x_np)

        if self.record_u:
            self._us.append(self._to_numpy(u))

        if log is None:
            return

        # record predicted plan trajectory
        if self.record_Xopt and (log.Xopt is not None):
            Xopt_np = self._maybe_slice_state(self._to_numpy(log.Xopt))
            self._Xopt.append(Xopt_np)

        # record sampled rollouts (heavy)
        if self.record_Xss and (log.Xs is not None) and ((t % self.Xss_stride) == 0):
            Xs_np = self._to_numpy(log.Xs)  # [B, H+1, S]
            Xs_np = self._maybe_slice_state(Xs_np)

            if self.max_rollouts is not None:
                Xs_np = Xs_np[: self.max_rollouts, ...]

            self._Xss.append(Xs_np)

    def finalize(self, **kwargs) -> Optional[VisualizationOutput]:
        """
        finalize 시점에 offline renderer.render(...)를 호출해서 결과 저장.
        """
        # build ts
        if self.record_x and len(self._xs) > 0:
            T = len(self._xs)
        else:
            # fallback: try infer from Xopt or Xss frames
            T = max(len(self._Xopt), len(self._Xss))
            if T == 0:
                raise RuntimeError("OfflineRecorder: no data recorded. Nothing to render.")

        ts = np.arange(T) * self.dt

        xs = np.asarray(self._xs) if self.record_x and self._xs else None
        us = np.asarray(self._us) if self.record_u and self._us else None
        Xopt = np.asarray(self._Xopt) if self.record_Xopt and self._Xopt else None
        Xss = np.asarray(self._Xss) if self.record_Xss and self._Xss else None

        # NOTE:
        # - Mobile2D/Quad3D offline renderer들은 보통 Xss.shape[0] == len(ts) (프레임별 rollout)
        # - Recorder는 stride 적용 시 Xss 프레임 수가 줄 수 있음.
        #   -> stride를 1로 쓰면 원본과 정확히 동일하게 맞음.
        #
        # 필요하면 여기서 Xss를 T 길이에 맞춰 "hold-last"로 업샘플링하는 옵션도 붙일 수 있음.
        # 일단은 simplest로 두자.

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        out = self.renderer.render(
            ts=ts,
            xs=xs,
            Xss=Xss,
            Xopt=Xopt,
            save_path=self.save_path,
            **self.renderer_kwargs,
        )

        # optional: also dump raw arrays for later plots
        # (원하면 kwargs로 on/off 만들 수 있음)
        return out

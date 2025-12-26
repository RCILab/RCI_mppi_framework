# ros2/mppi_ros2/mppi_ros2/robot/robot_state.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sensor_msgs.msg import JointState


@dataclass
class RobotStateConfig:
    nq: int
    joint_order: Optional[Sequence[str]] = None  # if None, use the first nq joints
    dtype: type = float


class RobotState:
    """
    State buffer that receives ROS JointState and caches only (q, qd).

    - In the callback, call update_from_joint_state(msg)
    - In the timer, call get_q_qd()
    - If joint_order is provided, joints are stored in that order
      (the cost of building name_to_idx per tick is handled only in the callback)
    """

    def __init__(self, cfg: RobotStateConfig):
        self.nq = int(cfg.nq)
        self.joint_order = list(cfg.joint_order) if cfg.joint_order else None
        self.dtype = cfg.dtype

        self._q: Optional[np.ndarray] = None
        self._qd: Optional[np.ndarray] = None
        self._t_wall: Optional[float] = None

        # Recompute only when msg.name changes
        self._last_names: Optional[Tuple[str, ...]] = None
        self._order_indices: Optional[List[int]] = None

    def _build_indices_if_needed(self, msg: JointState) -> None:
        if not self.joint_order:
            return

        names = tuple(msg.name)
        if self._last_names == names and self._order_indices is not None:
            return

        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(msg.name)}
        order_idx: List[int] = []
        for jn in self.joint_order:
            order_idx.append(name_to_idx.get(jn, -1))  # if missing, use -1
        self._order_indices = order_idx
        self._last_names = names

    def update_from_joint_state(self, msg: JointState) -> None:
        self._t_wall = time.time()

        # If joint_order is not provided, use the first nq joints
        if not self.joint_order:
            if len(msg.position) < self.nq or len(msg.velocity) < self.nq:
                return
            self._q = np.asarray(msg.position[: self.nq], dtype=self.dtype).copy()
            self._qd = np.asarray(msg.velocity[: self.nq], dtype=self.dtype).copy()
            return

        # If joint_order is provided, prepare index mapping
        self._build_indices_if_needed(msg)
        if self._order_indices is None:
            return

        q = np.zeros(self.nq, dtype=self.dtype)
        qd = np.zeros(self.nq, dtype=self.dtype)

        has_pos = len(msg.position) > 0
        has_vel = len(msg.velocity) > 0

        for k, i in enumerate(self._order_indices):
            if i < 0:
                continue
            if has_pos and i < len(msg.position):
                q[k] = float(msg.position[i])
            if has_vel and i < len(msg.velocity):
                qd[k] = float(msg.velocity[i])

        self._q = q
        self._qd = qd

    def get_q_qd(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._q is None or self._qd is None:
            return None
        return self._q, self._qd

    def is_fresh(self, timeout_sec: float) -> bool:
        if self._t_wall is None:
            return False
        return (time.time() - self._t_wall) <= float(timeout_sec)

    @property
    def last_update_time(self) -> Optional[float]:
        return self._t_wall

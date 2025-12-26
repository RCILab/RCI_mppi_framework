from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union, Any, List
from ament_index_python.packages import get_package_share_directory
import os

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper as PinRobotWrapper
except ImportError as e:
    raise ImportError(
        "pinocchio is required for RobotWrapper. Install it (e.g., `pip install pin`)."
    ) from e


ArrayLike = Union[np.ndarray, Sequence[float]]


def _asvec(x: Optional[ArrayLike], dtype=np.float64) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(x, dtype=dtype).reshape(-1)


def _resolve_package_uri(uri: str, package_map: dict | None = None) -> str:
    """
    Resolve ROS-style package:// URI to absolute filesystem path.

    Supported:
      - package://pkg
      - package://pkg/relative/path

    If package_map is provided, it overrides ament resolution:
      package_map = {"pkg": "/abs/path/to/pkg/share/pkg"}
    """
    if not uri:
        return uri
    if not uri.startswith("package://"):
        return uri

    rest = uri[len("package://"):]  # "pkg/relpath" or "pkg"
    parts = rest.split("/", 1)
    pkg_name = parts[0]

    if package_map and pkg_name in package_map:
        pkg_share = package_map[pkg_name]
    else:
        pkg_share = get_package_share_directory(pkg_name)

    if len(parts) == 1:
        return pkg_share
    return os.path.join(pkg_share, parts[1])


def _ref_from_str(ref: str) -> int:
    """
    ref: "world" | "local" | "local_world_aligned"
    Pinocchio python constants differ by version:
      - pin.ReferenceFrame.WORLD
      - or pin.WORLD / pin.LOCAL / pin.LOCAL_WORLD_ALIGNED
    """
    r = (ref or "").strip().lower()
    if r in ("world", "w"):
        return getattr(pin, "WORLD", getattr(pin.ReferenceFrame, "WORLD"))
    if r in ("local", "l"):
        return getattr(pin, "LOCAL", getattr(pin.ReferenceFrame, "LOCAL"))
    if r in ("local_world_aligned", "lwa", "local-world-aligned"):
        return getattr(pin, "LOCAL_WORLD_ALIGNED", getattr(pin.ReferenceFrame, "LOCAL_WORLD_ALIGNED"))
    raise ValueError(f"Unknown reference frame '{ref}'. Use world/local/local_world_aligned.")


@dataclass
class RobotWrapperConfig:
    urdf_path: str
    mesh_dir: Optional[str] = None
    base_freeflyer: bool = False

    # Joint list (including order) to extract from JointState(name)
    joint_order: Optional[Sequence[str]] = None

    # Default EE frame (used when frame_name is omitted in frame_*)
    default_ee_frame: Optional[str] = None

    # Helper for resolving package:// (optional)
    package_map: Optional[Dict[str, str]] = None

    dtype: Any = np.float64


class RobotWrapper:
    """
    Pinocchio-based robot wrapper (computeAllTerms-based).

    - In update(q, v), call pin.computeAllTerms(model, data, q, v)
    - Call update with update_frames=True or ensure_frames() to run updateFramePlacements
      when frame pose/jacobian are needed.

    Assumptions when using joint_order:
    - All joints listed in joint_order are 1-DoF (nq=1, nv=1)
    """

    def __init__(self, cfg: RobotWrapperConfig):
        self.cfg = cfg
        self.dtype = cfg.dtype

        urdf = _resolve_package_uri(cfg.urdf_path, cfg.package_map)
        mesh_dir = _resolve_package_uri(cfg.mesh_dir, cfg.package_map) if cfg.mesh_dir else None

        urdf_path = Path(urdf)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        # -----------------------------
        # IMPORTANT:
        #  - Many pinocchio python builds DO NOT support buildModelFromUrdf(urdf, mesh_dir)
        #  - Use PinRobotWrapper.BuildFromURDF which supports search paths.
        # -----------------------------
        if cfg.base_freeflyer:
            root_joint = pin.JointModelFreeFlyer()
            if mesh_dir is None:
                pin_robot = PinRobotWrapper.BuildFromURDF(str(urdf_path), root_joint)
            else:
                pin_robot = PinRobotWrapper.BuildFromURDF(str(urdf_path), [str(mesh_dir)], root_joint)
        else:
            if mesh_dir is None:
                pin_robot = PinRobotWrapper.BuildFromURDF(str(urdf_path))
            else:
                pin_robot = PinRobotWrapper.BuildFromURDF(str(urdf_path), [str(mesh_dir)])

        self.model = pin_robot.model
        self.data = pin_robot.data

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.default_ee_frame = cfg.default_ee_frame

        # latest state (pin order full vectors)
        self._q: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None

        # caches (numpy copies)
        self._M_cache: Optional[np.ndarray] = None
        self._nle_cache: Optional[np.ndarray] = None

        # whether frame placements are up-to-date for current state
        self._frames_valid: bool = False

        # joint_order mapping (partial -> full)
        self.joint_order = list(cfg.joint_order) if cfg.joint_order else None
        self._q_idx: Optional[np.ndarray] = None
        self._v_idx: Optional[np.ndarray] = None
        if self.joint_order:
            self._build_1dof_index_map(self.joint_order)

    # -----------------------
    # 1-DoF mapping
    # -----------------------
    def _build_1dof_index_map(self, joint_order: Sequence[str]) -> None:
        """
        Build index maps to insert q/qd scalars in joint_order into full pinocchio q(nq), v(nv).
        SAFE way: use model.getJointId(name)
        """
        q_idx: List[int] = []
        v_idx: List[int] = []
        for jname in joint_order:
            jid = self.model.getJointId(jname)
            if jid == 0:
                # 0 is usually "universe"
                raise ValueError(f"Joint '{jname}' not found in pinocchio model (getJointId returned 0).")

            j = self.model.joints[jid]
            if j.nq != 1 or j.nv != 1:
                raise ValueError(
                    f"Joint '{jname}' is not 1-DoF (nq={j.nq}, nv={j.nv}). "
                    "This wrapper assumes 1-DoF joints when using joint_order."
                )
            q_idx.append(j.idx_q)
            v_idx.append(j.idx_v)

        self._q_idx = np.asarray(q_idx, dtype=int)
        self._v_idx = np.asarray(v_idx, dtype=int)

    def _to_model_full(self, q: ArrayLike, v: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        qv = _asvec(q, self.dtype)
        vv = _asvec(v, self.dtype)
        if vv is None:
            raise ValueError("v (velocity) must be provided for computeAllTerms-based update.")
        if qv is None:
            raise ValueError("q must be provided.")

        if self._q_idx is None:
            if qv.shape[0] != self.nq:
                raise ValueError(f"q length {qv.shape[0]} != model.nq {self.nq}")
            if vv.shape[0] != self.nv:
                raise ValueError(f"v length {vv.shape[0]} != model.nv {self.nv}")
            return qv, vv

        if qv.shape[0] != self._q_idx.shape[0]:
            raise ValueError(f"len(q)={qv.shape[0]} expected={self._q_idx.shape[0]} (joint_order)")
        if vv.shape[0] != self._v_idx.shape[0]:
            raise ValueError(f"len(v)={vv.shape[0]} expected={self._v_idx.shape[0]} (joint_order)")

        q_full = np.zeros(self.nq, dtype=self.dtype)
        v_full = np.zeros(self.nv, dtype=self.dtype)
        q_full[self._q_idx] = qv
        v_full[self._v_idx] = vv
        return q_full, v_full

    # -----------------------
    # Update (computeAllTerms)
    # -----------------------
    def update(self, q: ArrayLike, v: ArrayLike, *, update_frames: bool = False, invalidate_cache: bool = True) -> None:
        q_m, v_m = self._to_model_full(q, v)
        self._q = q_m
        self._v = v_m

        pin.computeAllTerms(self.model, self.data, q_m, v_m)

        if update_frames:
            pin.updateFramePlacements(self.model, self.data)
            self._frames_valid = True
        else:
            self._frames_valid = False

        if invalidate_cache:
            self._M_cache = None
            self._nle_cache = None

    # -----------------------
    # Properties
    # -----------------------
    @property
    def q_full(self) -> np.ndarray:
        if self._q is None:
            raise RuntimeError("Call update(q, v) first.")
        return self._q

    @property
    def v_full(self) -> np.ndarray:
        if self._v is None:
            raise RuntimeError("Call update(q, v) first.")
        return self._v

    @property
    def mass(self) -> np.ndarray:
        if self._q is None:
            raise RuntimeError("Call update(q, v) first.")
        if self._M_cache is None:
            M = np.asarray(self.data.M, dtype=self.dtype).copy()
            self._M_cache = (M + M.T) * 0.5
        return self._M_cache

    @property
    def nle(self) -> np.ndarray:
        if self._q is None or self._v is None:
            raise RuntimeError("Call update(q, v) first.")
        if self._nle_cache is None:
            self._nle_cache = np.asarray(self.data.nle, dtype=self.dtype).reshape(-1).copy()
        return self._nle_cache

    # -----------------------
    # Frame APIs
    # -----------------------
    def frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def ensure_frames(self) -> None:
        if self._q is None:
            raise RuntimeError("Call update(q, v) first.")
        if not self._frames_valid:
            pin.updateFramePlacements(self.model, self.data)
            self._frames_valid = True

    def frame_pose(self, frame_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (position xyz, quaternion wxyz) in WORLD frame.
        """
        fname = frame_name or self.default_ee_frame
        if not fname:
            raise ValueError("frame_name is required (or set default_ee_frame).")
        if self._q is None:
            raise RuntimeError("Call update(q, v) first.")

        self.ensure_frames()

        fid = self.model.getFrameId(fname)
        oMf = self.data.oMf[fid]

        p = np.asarray(oMf.translation, dtype=self.dtype).reshape(3)
        R = np.asarray(oMf.rotation, dtype=self.dtype).reshape(3, 3)

        # Quaternion(R) API differs; this tends to work across versions:
        quat = pin.Quaternion(R)
        qwxyz = np.asarray([quat.w, quat.x, quat.y, quat.z], dtype=self.dtype)
        return p, qwxyz

    def frame_velocity(self, frame_name: Optional[str] = None, *, ref: str = "world") -> Tuple[np.ndarray, np.ndarray]:
        fname = frame_name or self.default_ee_frame
        if not fname:
            raise ValueError("frame_name is required (or set default_ee_frame).")
        if self._q is None or self._v is None:
            raise RuntimeError("Call update(q, v) first.")

        fid = self.model.getFrameId(fname)
        rf = _ref_from_str(ref)

        v6 = pin.getFrameVelocity(self.model, self.data, fid, rf).vector
        v6 = np.asarray(v6, dtype=self.dtype).reshape(6)
        return v6[:3].copy(), v6[3:].copy()

    def frame_jacobian(self, frame_name: Optional[str] = None, *, ref: str = "world") -> np.ndarray:
        fname = frame_name or self.default_ee_frame
        if not fname:
            raise ValueError("frame_name is required (or set default_ee_frame).")
        if self._q is None:
            raise RuntimeError("Call update(q, v) first.")

        fid = self.model.getFrameId(fname)
        rf = _ref_from_str(ref)

        J = pin.computeFrameJacobian(self.model, self.data, self._q, fid, rf)
        return np.asarray(J, dtype=self.dtype)

    def frame_jacobian_dot(self, frame_name: Optional[str] = None, *, ref: str = "world") -> np.ndarray:
        fname = frame_name or self.default_ee_frame
        if not fname:
            raise ValueError("frame_name is required (or set default_ee_frame).")
        if self._q is None or self._v is None:
            raise RuntimeError("Call update(q, v) first.")

        fid = self.model.getFrameId(fname)
        rf = _ref_from_str(ref)

        pin.computeJointJacobiansTimeVariation(self.model, self.data, self._q, self._v)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, fid, rf)
        return np.asarray(dJ, dtype=self.dtype)

    # -----------------------
    # Convenience
    # -----------------------
    def tau_from_qddot(self, qddot: ArrayLike) -> np.ndarray:
        """
        τ = M(q) qddot + nle(q,v)
        - qddot is in "input order": joint_order order if provided, else full nv.
        """
        if self._q is None or self._v is None:
            raise RuntimeError("Call update(q, v) first.")

        a = _asvec(qddot, self.dtype)
        if a is None:
            raise ValueError("qddot is required.")

        if self._v_idx is None:
            if a.shape[0] != self.nv:
                raise ValueError(f"qddot length {a.shape[0]} != nv {self.nv}")
            a_full = a
        else:
            if a.shape[0] != self._v_idx.shape[0]:
                raise ValueError(f"len(qddot)={a.shape[0]} expected={self._v_idx.shape[0]} (joint_order)")
            a_full = np.zeros(self.nv, dtype=self.dtype)
            a_full[self._v_idx] = a

        tau_full = self.mass @ a_full + self.nle
        return np.asarray(tau_full, dtype=self.dtype).reshape(-1)

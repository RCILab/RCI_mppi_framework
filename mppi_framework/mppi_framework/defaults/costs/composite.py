# defaults/costs/composite.py
from __future__ import annotations

from typing import Any, Dict, List
import torch

from mppi_framework.interfaces.cost import BaseCost
from mppi_framework.core.registry import COSTS


@COSTS.register("composite")
class CompositeCost(BaseCost):
    """
    A wrapper that combines multiple costs into a weighted sum.

    Example cost_cfg:
    {
        "terms": [
            {
                "name": "quadratic",
                "weight": 1.0,
                "cfg": { ... },
            },
            {
                "name": "obstacle",
                "weight": 5.0,
                "cfg": { ... },
            },
        ]
    }
    """

    def __init__(
        self,
        *,
        terms: List[Dict[str, Any]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **_ignored,   # Ignore any additional incoming keys
    ):
        # ⚠ BaseCost.__init__ does not take (device, dtype), so we do NOT call super()
        # Just store fields directly
        self.device = device
        self.dtype  = dtype
        self._costs: List[BaseCost] = []
        self._weights: List[torch.Tensor] = []

        for term in terms:
            name   = term["name"]
            weight = term.get("weight", 1.0)
            cfg    = term.get("cfg", {})

            CostCls = COSTS.get(name)  # "quadratic" / "obstacle", etc.
            # Unify so that each sub-cost accepts device, dtype
            cost = CostCls(device=device, dtype=dtype, **cfg)

            self._costs.append(cost)
            self._weights.append(
                torch.as_tensor(weight, device=device, dtype=dtype)
            )

    # X: [B,T,S], U: [B,T,U] -> [B,T]
    def stage(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        total = None
        for w, c in zip(self._weights, self._costs):
            c_val = c.stage(X, U)  # [B,T]
            if total is None:
                total = w * c_val
            else:
                total = total + w * c_val
        return total

    # X_T: [B,S] -> [B]
    def terminal(self, X_T: torch.Tensor) -> torch.Tensor:
        total = None
        for w, c in zip(self._weights, self._costs):
            c_val = c.terminal(X_T)  # [B]
            if total is None:
                total = w * c_val
            else:
                total = total + w * c_val
        return total

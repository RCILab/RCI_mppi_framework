from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type
import difflib

class Registry:
    """
    A small container that stores a mapping from name (string) -> class (or factory function).
    - register(name)(cls): register via decorator
    - get(name): retrieve a registered object
    - create(name, **kwargs): optionally create an instance immediately
    - list(): check current registered keys
    """
    def __init__(self, name: str):
        self._name = name           # For debugging, e.g., "SAMPLERS"
        self._map: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None, *, overwrite: bool = False) -> Callable:
        """
        A decorator used like @REG.register("gaussian").
        If name=None, automatically uses the lowercase class name.
        If overwrite=True, allows overwriting an existing name.
        """
        def deco(obj: Any):
            key = name or obj.__name__.lower()
            if (not overwrite) and (key in self._map):
                raise KeyError(f"[{self._name}] '{key}' is already registered.")
            self._map[key] = obj
            return obj
        return deco

    def get(self, name: str) -> Any:
        """Retrieve an object by name; if not found, suggest similar candidates."""
        if name in self._map:
            return self._map[name]
        # Suggest similar names when not found
        candidates = difflib.get_close_matches(name, self._map.keys(), n=3, cutoff=0.5)
        hint = f" Close candidates: {candidates}" if candidates else ""
        raise KeyError(f"[{self._name}] Could not find '{name}'.{hint}")

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Immediately call the registered 'class or factory' to create an instance.
        (Optional feature) Usually convenient in builder-style code.
        """
        cls_or_factory = self.get(name)
        return cls_or_factory(*args, **kwargs)

    def list(self) -> List[str]:
        """List currently registered keys."""
        return sorted(self._map.keys())

# Create global registries by category
ALGOS     = Registry("ALGOS")
DYNAMICS  = Registry("DYNAMICS")
COSTS     = Registry("COSTS")
SAMPLERS  = Registry("SAMPLERS")
CONSTRS   = Registry("CONSTRS")
VIS_RENDERERS = Registry("VIS_RENDERERS")
VIS_VISUALIZERS = Registry("VIS_VISUALIZERS")

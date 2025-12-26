
from mppi_framework.defaults import costs
from mppi_framework.defaults import dynamics
from mppi_framework.defaults import samplers
from mppi_framework.defaults import mppi
from mppi_framework.defaults import visualization
from mppi_framework.core.registry import ALGOS, DYNAMICS, COSTS, SAMPLERS, VIS_RENDERERS, VIS_VISUALIZERS
  
# defaults를 import해야 @register 들이 실행되어 등록됨
from mppi_framework.defaults.mppi import MPPIConfig

def build_controller(cfg: MPPIConfig,
                     dynamics_name="double_integrator",
                     cost_name="quadratic",
                     sampler_name="gaussian",
                     algo_name="mppi",
                     dynamics_cfg=None, cost_cfg=None, sampler_cfg=None):
    dynamics_cfg = dynamics_cfg or {}
    cost_cfg = cost_cfg or {}
    sampler_cfg = sampler_cfg or {}

    Dyn  = DYNAMICS.get(dynamics_name)
    Cost = COSTS.get(cost_name)
    Samp = SAMPLERS.get(sampler_name)
    Algo = ALGOS.get(algo_name)

    dyn  = Dyn(dtype=cfg.dtype, **dynamics_cfg)
    cost = Cost(dtype=cfg.dtype, **cost_cfg)
    smp  = Samp(dyn.spec.control_dim, cfg.horizon, dtype=cfg.dtype, **sampler_cfg)

    return Algo(dyn, cost, smp, cfg)

def build_offline_renderer(name: str, cfg=None):
    cfg = cfg or {}
    R = VIS_RENDERERS.get(name)
    return R(**cfg)

def build_online_visualizer(name: str, cfg=None):
    cfg = cfg or {}
    V = VIS_VISUALIZERS.get(name)
    return V(**cfg)
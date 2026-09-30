"""Backward-compat shim — canonical module is services/llm-agents/population_utils.py"""
import importlib.util as _ilu
import pathlib as _p

_src = _p.Path(__file__).resolve().parents[3] / "services" / "llm-agents" / "population_utils.py"
_spec = _ilu.spec_from_file_location("population_utils", str(_src))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
del _src, _spec, _mod, _p, _ilu

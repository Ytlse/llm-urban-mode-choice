from settings import settings

settings.force_reload()
# The controller is the process that owns the run: it, and it alone, makes
# `experiments/current` and `services/GAMA/CityTransport/results` point to its directory. An
# import of `settings` no longer moves these links (cf. `FactorySettings.claim_run`).
settings.claim_run()

from handle.application import app

__all__ = ["app"]

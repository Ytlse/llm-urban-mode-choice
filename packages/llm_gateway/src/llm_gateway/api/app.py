"""
api/app.py — FastAPI application factory.

create_app(config) explicitly composes all dependencies (Redis
connections, stores, load balancer) and attaches them to app.state.deps. The RPM
window reset is an action of the API lifespan — never again an import
side effect (cf. docs/arch/llm-module-package-refactor.md §2.3).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_gateway import __version__
from llm_gateway.api.deps import GatewayDeps, build_deps
from llm_gateway.api.metrics import install_collector
from llm_gateway.api.routes import router
from llm_gateway.config import Settings, get_settings
from llm_gateway.telemetry.logger import configure_logging, get_logger

logger = get_logger(__name__)


def _make_lifespan(deps: GatewayDeps):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # RPM window reset: once, at API startup only.
        # (A worker restart or a module import no longer touches the running
        # counters — free-tier quota overrun avoided.)
        try:
            deps.limiter.reset_windows()
            logger.info("RPM windows reset at API startup.")
        except Exception as e:
            logger.warning(f"RPM reset at startup failed (non-fatal): {e}")

        # Warm-up of the Celery broker connection before the first request.
        try:
            from llm_gateway.worker.task_worker import celery_app
            conn = celery_app.connection()
            conn.ensure_connection(max_retries=5)
            conn.release()
            logger.info("Celery broker connection warmed up.")
        except Exception as e:
            logger.warning(f"Celery broker warm-up failed (non-fatal): {e}")
        yield

    return lifespan


def create_app(config: Settings | None = None, deps: GatewayDeps | None = None) -> FastAPI:
    """Compose the application. `deps` allows injecting in-memory ports (tests, embedded mode)
    instead of the Redis connections built by build_deps()."""
    settings = config or (deps.settings if deps is not None else get_settings())
    configure_logging(settings.telemetry)
    deps = deps if deps is not None else build_deps(settings)
    install_collector(deps)

    app = FastAPI(
        title="llm-gateway",
        description="Asynchronous multi-provider LLM gateway: micro-batching, SWRR, circuit breaker, pluggable categories.",
        version=__version__,
        lifespan=_make_lifespan(deps),
    )
    app.state.deps = deps

    if settings.api.cors_origins:   # empty = no CORS middleware (default); ["*"] = allow all
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)
    return app

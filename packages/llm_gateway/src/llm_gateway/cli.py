"""cli.py — operations commands: `llm-gateway serve | worker | config validate | config show | categories`.

Entry point declared in pyproject (`[project.scripts]`). The commands add no logic:
they call the factories and print what an operator wants to check before starting
(effective configuration, secrets masked; registered categories).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("llm_gateway.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    from llm_gateway.worker.task_worker import celery_app

    argv = ["worker", f"--loglevel={args.loglevel}", f"--concurrency={args.concurrency}", "-P", args.pool]
    celery_app.worker_main(argv)
    return 0


def _redacted_settings() -> dict[str, Any]:
    from llm_gateway.config import get_settings, redacted_dump

    return redacted_dump(get_settings())


def _cmd_config_validate(_: argparse.Namespace) -> int:
    from llm_gateway.config import get_settings

    try:
        settings = get_settings()
    except Exception as exc:  # pydantic.ValidationError, unreadable YAML…
        print(f"INVALID CONFIGURATION: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK — file {settings.resolved_providers_file()}: {len(settings.declared_providers)} "
        f"declared, {len(settings.providers)} active with key: {', '.join(sorted(settings.providers)) or 'aucun'}"
    )
    return 0


def _cmd_config_schema(args: argparse.Namespace) -> int:
    if args.what == "providers":
        from llm_gateway.config import providers_schema_json_text

        print(providers_schema_json_text())
    else:
        from llm_gateway.config import GatewaySettings

        print(json.dumps(GatewaySettings.model_json_schema(), indent=2, ensure_ascii=False))
    return 0


def _cmd_config_show(_: argparse.Namespace) -> int:
    print(json.dumps(_redacted_settings(), indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_categories(_: argparse.Namespace) -> int:
    from llm_gateway.prompts.registry import get_registry

    registry = get_registry()
    cats = registry.categories()
    if not cats:
        print("No category registered (no bundle under the llm_gateway.categories entry point).")
        return 1
    for name in cats:
        handle = registry.get(name)
        print(f"{name}\t(bundle={handle.bundle.name}, items={handle.spec.item_model.__name__})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-gateway", description="LLM gateway: service, worker, configuration.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="starts the API (uvicorn)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    w = sub.add_parser("worker", help="starts a Celery worker")
    w.add_argument("--loglevel", default="info")
    w.add_argument("--concurrency", type=int, default=25)
    w.add_argument("--pool", default="threads")
    w.set_defaults(func=_cmd_worker)

    c = sub.add_parser("config", help="effective configuration")
    csub = c.add_subparsers(dest="config_command", required=True)
    csub.add_parser("validate", help="loads the configuration; code 1 if invalid").set_defaults(func=_cmd_config_validate)
    csub.add_parser("show", help="shows the effective configuration, secrets masked").set_defaults(func=_cmd_config_show)
    sc = csub.add_parser("schema", help="JSON Schema of the providers file (default) or of the settings")
    sc.add_argument("what", nargs="?", choices=["providers", "settings"], default="providers")
    sc.set_defaults(func=_cmd_config_schema)

    sub.add_parser("categories", help="lists the categories registered by the bundles").set_defaults(func=_cmd_categories)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

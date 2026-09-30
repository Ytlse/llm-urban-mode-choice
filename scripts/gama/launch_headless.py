"""Headless launcher: loads and plays the GAMA experiment through the GAMA Server protocol.

Run by `make run OFFLINE=1` INSIDE the controller container (the `websockets`
library is installed there and the `gama` service can be resolved from it):

    docker compose exec -T controller python /app/scripts/gama/launch_headless.py

Sequence:
  1. connect to ws://gama:6868 (with retries: GAMA Server takes ~10-20 s to start)
  2. `load` the model, injecting the network parameters (http_url/http_port
     → http://controller:8002, declared as parameters of experiment `e`)
  3. `play`, then relay the GAMA console to stdout

IMPORTANT: the process must stay alive for the whole simulation — GAMA
Server kills experiments whose WebSocket client has disconnected. The run is
stopped cleanly through `make down` (or by stopping the gama service).
"""

import asyncio
import json
import os
import sys

import websockets

GAMA_SERVER_URL = os.environ.get("GAMA_SERVER_URL", "ws://gama:6868")
MODEL_PATH = os.environ.get("GAMA_MODEL_PATH", "/services/GAMA/CityTransport/models/City.gaml")
EXPERIMENT = os.environ.get("GAMA_EXPERIMENT", "e")
CONTROLLER_HTTP_URL = os.environ.get("GAMA_HTTP_URL", "http://controller")
CONTROLLER_HTTP_PORT = int(os.environ.get("GAMA_HTTP_PORT", "8002"))
CONNECT_TIMEOUT_S = int(os.environ.get("GAMA_CONNECT_TIMEOUT_S", "180"))

# Ticket 092 — time given to GAMA to answer a ping before the connection is deemed
# dead. The `websockets` default is 20 s, and GAMA blocks while it waits for the LLM
# decisions: with CACHE=0 this is the NORMAL regime, and under a token shortage the wait
# is longer still, the repository rule being to wait for renewal rather than
# degrade. Measured on run 2026-09-16_12_48: 23 pauses, median 9.0 s, maximum 40.9 s,
# three above 20 s — the run died at the third, GAMA Server stopping the experiment
# whose client had disconnected. The ping is still SENT: we loosen the reply delay, we
# do not disable the detection of a truly dead connection.
PING_TIMEOUT_S = int(os.environ.get("GAMA_PING_TIMEOUT_S", "1200"))
PING_INTERVAL_S = int(os.environ.get("GAMA_PING_INTERVAL_S", "20"))

# GAMA Server protocol message types that report a failed command.
ERROR_TYPES = {
    "MalformedRequest",
    "UnableToExecuteRequest",
    "GamaServerError",
    "SimulationError",
    "SimulationErrorDialog",
    "RuntimeError",
}

# Status heartbeats (one per clock tick): never relayed to the log,
# even if a future `load` re-enables "status".
NOISE_TYPES = {"SimulationStatusInform", "SimulationStatus"}


def log(msg: str) -> None:
    print(msg, flush=True)


async def connect_with_retries() -> websockets.WebSocketClientProtocol:
    """GAMA Server is ready only a few seconds after the container starts."""
    elapsed = 0
    while True:
        try:
            ws = await websockets.connect(
                GAMA_SERVER_URL,
                max_size=10**7,
                ping_interval=PING_INTERVAL_S,
                ping_timeout=PING_TIMEOUT_S,
            )
            log(f"✅ Connecté à GAMA Server ({GAMA_SERVER_URL})")
            return ws
        except Exception as exc:
            if elapsed >= CONNECT_TIMEOUT_S:
                log(f"[ALARME] GAMA Server injoignable après {CONNECT_TIMEOUT_S}s : {exc}")
                raise
            log(f"⏳ GAMA Server pas encore prêt ({exc}), retry dans 5s...")
            await asyncio.sleep(5)
            elapsed += 5


async def recv_json(ws) -> dict:
    raw = await ws.recv()
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"type": "Raw", "content": raw}


async def wait_command_result(ws) -> dict:
    """Consume messages until the result of the last command sent."""
    while True:
        msg = await recv_json(ws)
        mtype = msg.get("type")
        if mtype == "CommandExecutedSuccessfully":
            return msg
        if mtype in ERROR_TYPES:
            log(f"[ALARME] GAMA Server a rejeté la commande : {json.dumps(msg, ensure_ascii=False)}")
            raise RuntimeError(f"GAMA Server error: {mtype}")
        # Intermediate messages (console…): relay them and carry on.
        if mtype not in NOISE_TYPES:
            log(f"[gama] {json.dumps(msg, ensure_ascii=False)}")


async def main() -> None:
    ws = await connect_with_retries()

    greeting = await recv_json(ws)
    log(f"[gama] {json.dumps(greeting, ensure_ascii=False)}")

    load_cmd = {
        "type": "load",
        "model": MODEL_PATH,
        "experiment": EXPERIMENT,
        "console": True,
        # status: GAMA emits one SimulationStatusInform per clock tick — pure
        # heartbeat that floods the log, we do not subscribe to it.
        "status": False,
        "dialog": True,
        "parameters": [
            {"type": "string", "name": "http_url", "value": CONTROLLER_HTTP_URL},
            {"type": "int", "name": "http_port", "value": CONTROLLER_HTTP_PORT},
            # ⚠ Do not inject the simulation parameters (memory, days…) here:
            # Settings.gaml (load_sim_config, cycle 1) overwrites them from
            # services/GAMA/CityTransport/config/sim_params.yaml. That file is authoritative —
            # `make run MEM=0|1` rewrites it before launch.
        ],
    }
    log(f"📦 load {MODEL_PATH} (expériment '{EXPERIMENT}', controller={CONTROLLER_HTTP_URL}:{CONTROLLER_HTTP_PORT})")
    await ws.send(json.dumps(load_cmd))
    result = await wait_command_result(ws)
    exp_id = result.get("content")
    log(f"✅ Expériment chargé (exp_id={exp_id})")

    await ws.send(json.dumps({"type": "play", "exp_id": exp_id, "sync": False}))
    await wait_command_result(ws)
    log("▶️  Simulation lancée — la connexion reste ouverte (l'arrêt passe par `make down`)")

    # Relay of the GAMA console until the end of the run. A simulation error
    # is reported but does not stop the relay: the model `pause`s by
    # itself at simulation_max_days and the controller finishes its writes.
    while True:
        msg = await recv_json(ws)
        mtype = msg.get("type")
        if mtype in NOISE_TYPES:
            continue
        prefix = "[ALARME] " if mtype in ERROR_TYPES else ""
        log(f"{prefix}[gama] {json.dumps(msg, ensure_ascii=False)}")

        # End of simulation declared by GAMA (do pause at simulation_max_days)
        if mtype == "SimulationOutput":
            content_msg = msg.get("content", {}).get("message", "") if isinstance(msg.get("content"), dict) else str(msg.get("content"))
            if "Simulation stopped after" in content_msg:
                log("🏁 Fin de simulation détectée (Simulation stopped after). Fermeture propre du launcher.")
                await asyncio.sleep(2)
                break
            if "Préfixe commun interrompu" in content_msg or "simulation à invalider" in content_msg:
                log("[ALARME] Arrêt sur erreur de synchronisation préfixe. Fermeture immédiate du launcher.")
                sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Arrêt du launcher (KeyboardInterrupt)")
    except websockets.exceptions.ConnectionClosed as exc:
        log(f"[ALARME] Connexion GAMA Server fermée : {exc} — l'expériment associé est arrêté par GAMA")
        sys.exit(1)
    except Exception as exc:
        log(f"[ALARME] Launcher headless en échec : {exc}")
        sys.exit(1)

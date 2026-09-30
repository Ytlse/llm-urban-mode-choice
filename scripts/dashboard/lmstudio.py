"""LM Studio as seen from the dashboard: is the local model loaded, with enough context?

The dashboard runs on the host machine, where LM Studio serves its models (port 1234);
the gateway, for its part, reaches them from the containers via `host.docker.internal`. This module reads
LM Studio's `GET /api/v1/models` — which models are downloaded, which are loaded, with
which context — and derives from it a diagnostic to display, as well as the variables of
`make lmstudio-charger`.

Why: on 2026-09-08, an experiment launched on Muse Glimmer ran for seven minutes without a single
decision. The model was not loaded; LM Studio loaded it on the fly, more slowly than the
adapter's 120 s, and once loaded its default context (4,096 tokens) could not hold a
batch of two agents. Nothing in the interface said so.

Local alias: when the identifier of an LM Studio model is also that of a remote provider
(`qwen/qwen3.8-27b` at Groq), the local instance carries an alias suffixed `-local`, loaded with
`lms load <clé> --identifier <alias>`. The source key is found by removing the suffix and
looking for the key whose last segment matches (`qwen3.8-27b-local` → `qwen/qwen3.8-27b`).

Everything is pure except `etat_lmstudio` (a local GET, 1.5 s delay): the rest is tested on
frozen JSON responses.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

PORT_LMSTUDIO = 1234
URL_HOTE_DEFAUT = f"http://localhost:{PORT_LMSTUDIO}"
CTX_MIN = 8192          # two agents per request ≈ 4,400 prompt tokens, plus the answer
CTX_CHARGEMENT = 16384  # what `make lmstudio-charger` requests
SUFFIXE_ALIAS = "-local"
_HOTES_VERS_LOCALHOST = ("host.docker.internal", "gateway.docker.internal")
_HOTES_LOCAUX = _HOTES_VERS_LOCALHOST + ("localhost", "127.0.0.1")

MOTIF_INJOIGNABLE = (f"LM Studio est injoignable sur localhost:{PORT_LMSTUDIO} — ouvrez LM Studio et démarrez son "
                     "serveur (onglet Developer → Start Server, ou `lms server start`)")


def _n(x: Optional[int]) -> str:
    """16384 → « 16 384 », for the labels."""
    return "?" if x is None else f"{int(x):,}".replace(",", " ")


# ── providers.yaml: who is served by LM Studio ──────────────────────────────

def _providers(d) -> dict:
    if not isinstance(d, dict):
        return {}
    interne = d.get("providers")
    return interne if isinstance(interne, dict) else d


def est_instance_lmstudio(cfg) -> bool:
    """An instance whose `base_url` targets LM Studio's port on the host (seen from the containers or from the host)."""
    if not isinstance(cfg, dict):
        return False
    u = urlsplit(str(cfg.get("base_url") or ""))
    if u.hostname not in _HOTES_LOCAUX:
        return False
    return (u.port or (443 if u.scheme == "https" else 80)) == PORT_LMSTUDIO


def url_hote(base_url: str) -> str:
    """LM Studio's URL from the host: `host.docker.internal` (seen from the containers) becomes `localhost`."""
    u = urlsplit(str(base_url or ""))
    hote = "localhost" if u.hostname in _HOTES_VERS_LOCALHOST else (u.hostname or "localhost")
    return urlunsplit((u.scheme or "http", f"{hote}:{u.port or PORT_LMSTUDIO}", "", "", ""))


def instances_lmstudio(providers) -> dict[str, dict]:
    return {nom: cfg for nom, cfg in _providers(providers).items() if est_instance_lmstudio(cfg)}


def modeles_locaux(providers) -> dict[str, list[str]]:
    """model (`default_model`) → LM Studio instances that serve it."""
    out: dict[str, list[str]] = {}
    for nom, cfg in instances_lmstudio(providers).items():
        if cfg.get("default_model"):
            out.setdefault(str(cfg["default_model"]), []).append(nom)
    return dict(sorted(out.items()))


def modeles_distants(providers) -> dict[str, list[str]]:
    """model → instances of a remote provider (everything that is not LM Studio)."""
    out: dict[str, list[str]] = {}
    for nom, cfg in _providers(providers).items():
        if isinstance(cfg, dict) and cfg.get("default_model") and not est_instance_lmstudio(cfg):
            out.setdefault(str(cfg["default_model"]), []).append(nom)
    return dict(sorted(out.items()))


# ── LM Studio: what is downloaded, what is loaded ───────────────────────────

def analyser(payload) -> dict:
    """The response of `GET /api/v1/models`, reduced to what matters.

    {"modeles": {key: {"type", "params", "format", "quant", "ctx_max", "charges": [{"id", "ctx"}]}},
     "charges": {loaded identifier: {"cle": source key, "ctx": loaded context}}}
    A model loaded under an alias appears in `charges` under that alias, with its source key.
    """
    modeles: dict[str, dict] = {}
    charges: dict[str, dict] = {}
    for m in (payload or {}).get("models") or []:
        if not isinstance(m, dict) or not m.get("key"):
            continue
        cle = str(m["key"])
        instances = []
        for i in m.get("loaded_instances") or []:
            ident = str((i or {}).get("id") or cle)
            ctx = ((i or {}).get("config") or {}).get("context_length")
            ctx = int(ctx) if isinstance(ctx, (int, float)) else None
            instances.append({"id": ident, "ctx": ctx})
            charges[ident] = {"cle": cle, "ctx": ctx}
        q = m.get("quantization")
        modeles[cle] = {"type": m.get("type"), "params": m.get("params_string"), "format": m.get("format"),
                        "quant": q.get("name") if isinstance(q, dict) else q,
                        "ctx_max": m.get("max_context_length"), "charges": instances}
    return {"modeles": modeles, "charges": charges}


def etat_lmstudio(url: str = URL_HOTE_DEFAUT, timeout: float = 1.5) -> Optional[dict]:
    """The state of LM Studio (cf. `analyser`), or None if it is unreachable — server stopped, application closed."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/v1/models", timeout=timeout) as r:  # noqa: S310 — URL locale
            return analyser(json.loads(r.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def resoudre_source(identifiant: str, etat: Optional[dict]) -> Optional[str]:
    """The LM Studio key behind a `default_model`: the key itself, the alias already loaded, or the `-local` alias."""
    if not etat or not identifiant:
        return None
    if identifiant in etat["modeles"]:
        return identifiant
    charge = etat["charges"].get(identifiant)
    if charge:
        return charge["cle"]
    if identifiant.endswith(SUFFIXE_ALIAS):
        base = identifiant[: -len(SUFFIXE_ALIAS)]
        candidats = [c for c in etat["modeles"] if c == base or c.rsplit("/", 1)[-1] == base]
        if len(candidats) == 1:
            return candidats[0]
    return None


@dataclass
class Diagnostic:
    identifiant: str
    pret: bool
    motif: Optional[str] = None
    source: Optional[str] = None        # LM Studio key to load (`lms load`)
    contexte: Optional[int] = None      # context of the instance loaded under `identifiant`, if it exists
    recharger: bool = False             # something unusable is loaded (short context, wrong identifier)
    autres_charges: list[str] = field(default_factory=list)  # the other models in memory
    params: Optional[str] = None
    format: Optional[str] = None
    quant: Optional[str] = None

    @property
    def etat_court(self) -> str:
        if self.pret:
            return f"🟢 chargé, contexte {_n(self.contexte)}"
        if self.motif == MOTIF_INJOIGNABLE:
            return "⚫ LM Studio injoignable"
        if self.source is None:
            return "❓ inconnu de LM Studio"
        if self.contexte is not None:
            return f"🔴 chargé, contexte {_n(self.contexte)} < {_n(CTX_MIN)}"
        return "⚪ non chargé"


def diagnostic(identifiant: str, etat: Optional[dict], ctx_min: int = CTX_MIN) -> Diagnostic:
    """Ready, or why not: unreachable, unknown, not loaded, loaded under another name, context too short."""
    if etat is None:
        return Diagnostic(identifiant, False, motif=MOTIF_INJOIGNABLE)
    autres = sorted(i for i in etat["charges"] if i != identifiant)
    source = resoudre_source(identifiant, etat)
    if source is None:
        return Diagnostic(identifiant, False, autres_charges=autres,
                          motif=f"« {identifiant} » ne correspond à aucun modèle téléchargé dans LM Studio, ni comme "
                                f"clé (`lms ls`), ni comme alias `{SUFFIXE_ALIAS}` d'une clé")
    info = etat["modeles"].get(source) or {}
    d = Diagnostic(identifiant, False, source=source, autres_charges=autres,
                   params=info.get("params"), format=info.get("format"), quant=info.get("quant"))
    charge = etat["charges"].get(identifiant)
    if charge is None:
        if identifiant != source and any(c["id"] == source for c in info.get("charges", [])):
            d.recharger = True
            d.motif = (f"« {source} » est chargé, mais pas sous l'identifiant « {identifiant} » qu'attend la "
                       f"passerelle — il faut le décharger puis le charger avec cet alias")
        else:
            d.motif = f"le modèle « {identifiant} » n'est pas chargé dans LM Studio"
        return d
    d.contexte = charge["ctx"]
    if d.contexte is not None and d.contexte < ctx_min:
        d.recharger = True
        d.motif = (f"« {identifiant} » est chargé avec {_n(d.contexte)} jetons de contexte, il en faut au moins "
                   f"{_n(ctx_min)} (deux agents par requête) — il faut le décharger puis le recharger")
        return d
    d.pret = True
    return d


def variables_chargement(identifiant: str, source: str, ctx: int = CTX_CHARGEMENT, recharger: bool = False) -> dict[str, str]:
    """The variables of `make lmstudio-charger`: the key to load, the alias if it differs, the context, and
    RECHARGER=1 to first unload whatever is in memory under that name."""
    return {"MODELE": source, "IDENTIFIANT": "" if identifiant == source else identifiant,
            "CTX": str(ctx), "RECHARGER": "1" if recharger else ""}


__all__ = ["CTX_CHARGEMENT", "CTX_MIN", "Diagnostic", "MOTIF_INJOIGNABLE", "SUFFIXE_ALIAS", "analyser", "diagnostic",
           "est_instance_lmstudio", "etat_lmstudio", "instances_lmstudio", "modeles_distants", "modeles_locaux",
           "resoudre_source", "url_hote", "variables_chargement"]

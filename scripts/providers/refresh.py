#!/usr/bin/env python3
"""Updates config/llm_gateway/providers.yaml from the providers' real quotas.

Run with: `make providers` (or `make providers DRY_RUN=1` to preview).

Sources per adapter (validated on 2026-08-03, see docs/setup/llm-providers.md):
  - mistral / groq / cerebras: 1 probe request (max_tokens=1) per instance →
    x-ratelimit-* headers of the response (real account quotas).
      mistral  : RPM + TPM (no RPD/TPD; monthly quota not exposed → pro-rata rule)
      groq     : RPD + TPM (x-ratelimit-limit-requests = requests/DAY;
                 RPM and TPD absent from headers → fields left as they are)
      cerebras : RPM + TPM + RPD + TPD (minute/hour/day granularities)
  - google: Cloud Quotas API (gcloud token) → free tier RPM/TPM/RPD per model.
    Quota names are FAMILIES (gemma-4-26b, gemini-3.1-flash-lite):
    mapping by longest prefix on the instance's default_model.
  - openai (or any instance without a key in .env): skipped with a warning.

Model lifecycle (GET /models per adapter):
  - NEW operational text model (free tier quota readable) → provider
    block appended at end of file. RPD ≥ MIN_RPD_NEW_PROVIDER → in rotation
    (computed weight); lower RPD → weight 0 = OUT OF ROTATION (the load
    balancer filters weight 0), usable only through a forced `llm.provider`.
    A model already referenced in the file, even commented out, is never
    re-added (a commented block = human decision or dated obsolescence).
    Mistral exception: quota shared per account → no gain, information only.
  - default_model GONE from /models → block commented out with the date + [ALARME].

Rules:
  - MISTRAL SAFEGUARD: the free tier is capped at 1 B tokens/month (not exposed by
    the API). To avoid consuming the month in a day, tpd_limit is forced to
    MISTRAL_PRORATA_FACTOR × (monthly quota / 30). RPM is capped at 60 (documented
    rate: 1 req/s) even if the headers announce more.
  - weight recomputed (file convention: min(rpm, tpm/3000)/15) as soon as
    rpm_limit or tpm_limit changes.
  - max_tokens_per_request follows tpm_limit when the field exists (single request
    > TPM → HTTP 413).
  - Never a silent removal or loosening: a failed probe
    leaves the instance intact ([ALARME] in the summary).

Surgical YAML editing: only changed values are rewritten, the
file's comments are preserved (same approach as
llm_gateway/config/settings.py::_persist_provider_max_output_tokens), atomic write.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_YAML = PROJECT_ROOT / "config" / "llm_gateway" / "providers.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
TIMEOUT = 30
TODAY = dt.date.today().isoformat()

# ── Mistral safeguard ────────────────────────────────────────────────────────
MISTRAL_MONTHLY_TOKENS = 1_000_000_000  # free tier: 1 B tokens/month (doc)
MISTRAL_PRORATA_FACTOR = 3              # tpd = 3 × the daily pro-rata
MISTRAL_TPD = MISTRAL_PRORATA_FACTOR * MISTRAL_MONTHLY_TOKENS // 30
MISTRAL_RPM_DOC = 60                    # documented rate: 1 req/s

# New models: activation threshold and defaults
MIN_RPD_NEW_PROVIDER = 100   # below this, the bucket brings nothing to the rotation
GROQ_RPM_FREE_TIER = 30      # Groq free tier RPM (console, not exposed by the API)
DEFAULT_CONCURRENCY = {"google": 2, "groq": 3, "cerebras": 1}

# Fields driven by this script, per adapter (the others stay manual)
MANAGED_FIELDS = {
    "mistral":  ("rpm_limit", "tpm_limit", "tpd_limit"),
    "groq":     ("tpm_limit", "rpd_limit"),
    "cerebras": ("rpm_limit", "tpm_limit", "rpd_limit", "tpd_limit"),
    "google":   ("rpm_limit", "tpm_limit", "rpd_limit"),
}

# Comments set on created/modified lines (per field, per adapter)
FIELD_COMMENTS = {
    ("mistral", "rpm_limit"): "# 1 req/s (doc free tier) — les en-têtes annoncent plus, borné par prudence",
    ("mistral", "tpd_limit"): f"# GARDE-FOU : {MISTRAL_PRORATA_FACTOR}× le prorata journalier du quota mensuel free tier (1 Md tokens/mois)",
    ("groq", "rpd_limit"): "# quota requêtes/jour (en-têtes x-ratelimit)",
    ("cerebras", "rpd_limit"): "# quota requêtes/jour (en-têtes x-ratelimit)",
    ("google", "rpd_limit"): "# quota requêtes/jour (free tier)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Quota collection
# ─────────────────────────────────────────────────────────────────────────────

def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def probe_headers(base_url: str, key: str, model: str) -> dict[str, str]:
    """Minimal request → x-ratelimit-* headers (raises on network failure)."""
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": "ping"}],
              "max_tokens": 1},
        timeout=TIMEOUT,
    )
    headers = {k.lower(): v for k, v in r.headers.items() if "ratelimit" in k.lower()}
    if r.status_code != 200 and not headers:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    return headers


def _to_int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def quotas_mistral(headers: dict) -> dict[str, int]:
    out: dict[str, int] = {"tpd_limit": MISTRAL_TPD}
    rpm = _to_int(headers.get("x-ratelimit-limit-req-minute"))
    if rpm:
        out["rpm_limit"] = min(rpm, MISTRAL_RPM_DOC)
    tpm = _to_int(headers.get("x-ratelimit-limit-tokens-minute"))
    if tpm:
        out["tpm_limit"] = tpm
    return out


def quotas_groq(headers: dict) -> dict[str, int]:
    # x-ratelimit-limit-requests = requests/DAY at Groq (console doc)
    out: dict[str, int] = {}
    rpd = _to_int(headers.get("x-ratelimit-limit-requests"))
    if rpd:
        out["rpd_limit"] = rpd
    tpm = _to_int(headers.get("x-ratelimit-limit-tokens"))
    if tpm:
        out["tpm_limit"] = tpm
    return out


def quotas_cerebras(headers: dict) -> dict[str, int]:
    mapping = {
        "rpm_limit": "x-ratelimit-limit-requests-minute",
        "tpm_limit": "x-ratelimit-limit-tokens-minute",
        "rpd_limit": "x-ratelimit-limit-requests-day",
        "tpd_limit": "x-ratelimit-limit-tokens-day",
    }
    return {f: v for f, h in mapping.items() if (v := _to_int(headers.get(h)))}


def google_freetier_quotas() -> dict[str, dict[str, int]]:
    """Free tier quotas per model FAMILY through the Cloud Quotas API.

    Requires an authenticated gcloud; the active project serves as reference (the
    free tier values are per-model defaults, identical across projects).
    """
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True, timeout=60
    ).strip()
    project = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True, timeout=30
    ).strip()
    url = (f"https://cloudquotas.googleapis.com/v1/projects/{project}/locations/"
           "global/services/generativelanguage.googleapis.com/quotaInfos")
    infos, page = [], None
    while True:
        params = {"pageSize": 300}
        if page:
            params["pageToken"] = page
        d = requests.get(url, params=params,
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=TIMEOUT)
        d.raise_for_status()
        d = d.json()
        infos += d.get("quotaInfos", [])
        page = d.get("nextPageToken")
        if not page:
            break

    def per_model(quota_id: str) -> dict[str, int]:
        for q in infos:
            if q["quotaId"] != quota_id:
                continue
            out = {}
            for di in q.get("dimensionsInfos", []):
                model = (di.get("dimensions") or {}).get("model")
                v = _to_int(di.get("details", {}).get("value"))
                if model and v and v > 0:  # -1 = unlimited, None = no access
                    out[model] = v
            return out
        return {}

    rpm = per_model("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    tpm = per_model("GenerateContentInputTokensPerModelPerMinute-FreeTier")
    rpd = per_model("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    quotas: dict[str, dict[str, int]] = {}
    for family in set(rpm) | set(tpm) | set(rpd):
        q = {}
        if family in rpm:
            q["rpm_limit"] = rpm[family]
        if family in tpm:
            q["tpm_limit"] = tpm[family]
        if family in rpd:
            q["rpd_limit"] = rpd[family]
        quotas[family] = q
    return quotas


def match_google_family(model: str, families: dict[str, dict]) -> str | None:
    """Longest prefix: gemma-4-26b-a4b-it → gemma-4-26b,
    gemini-3.1-flash-lite-preview → gemini-3.1-flash-lite."""
    best = None
    for family in families:
        if model == family or model.startswith(family + "-"):
            if best is None or len(family) > len(best):
                best = family
    return best


def list_models(adapter: str, base_url: str, key: str) -> list[str] | None:
    try:
        if adapter == "google":
            r = requests.get(f"{base_url}/models",
                             params={"key": key, "pageSize": 1000}, timeout=TIMEOUT)
            r.raise_for_status()
            return sorted(m["name"].removeprefix("models/")
                          for m in r.json().get("models", []))
        r = requests.get(f"{base_url}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Surgical YAML editing
# ─────────────────────────────────────────────────────────────────────────────

_COMMENT_PREFIX = re.compile(r"^  (?:#\s?)+")


def _is_block_header(line: str) -> bool:
    """Start of a provider block — active OR already commented out.

    Commented blocks must count as boundaries. Without them, the scope
    of an active block extended up to the next ACTIVE block, stepping over the
    intermediate commented blocks: `comment_out` re-commented them on the way and the
    file filled up with `# # groq_llama4:` (seen on 2026-08-18 in
    providers.yaml). A commented block is a human decision or a dated
    obsolescence: we do not rewrite it because a neighbour becomes obsolete.

    Discriminant: comment markers removed, a BLOCK key is at
    indentation 2 (`groq_x:`) whereas a FIELD is at 4 or more (`#   rpm_limit:`).
    That is exactly the form `comment_out` writes itself.
    """
    if re.match(r"^  [A-Za-z0-9_.\-]+:", line):
        return True
    return bool(re.match(r"^[A-Za-z0-9_.\-]+:", _COMMENT_PREFIX.sub("", line)))


class YamlEditor:
    """Modifies scalar fields of a provider block without touching the rest."""

    def __init__(self, path: Path):
        self.path = path
        self.lines = path.read_text().splitlines(keepends=True)
        self.changes: list[str] = []

    def _block_range(self, name: str) -> tuple[int, int] | None:
        start = None
        for i, line in enumerate(self.lines):
            if re.match(rf"^  {re.escape(name)}:\s*(#.*)?$", line):
                start = i
                break
        if start is None:
            return None
        end = len(self.lines)
        for j in range(start + 1, len(self.lines)):
            # next block: key indented by 2 spaces, commented or not (see
            # `_is_block_header` — a commented block also bounds the previous one)
            if _is_block_header(self.lines[j]):
                end = j
                break
        return start, end

    def get(self, name: str, field: str) -> int | None:
        rng = self._block_range(name)
        if rng is None:
            return None
        for i in range(rng[0] + 1, rng[1]):
            m = re.match(rf"^(\s+){re.escape(field)}:(\s+)(\S+)", self.lines[i])
            if m and not self.lines[i].lstrip().startswith("#"):
                return _to_int(m.group(3))
        return None

    def set(self, name: str, field: str, value: int, comment: str | None = None):
        """Replaces the value (existing comment preserved, except '# TBC') or
        inserts the field after rpm_limit/tpm_limit with the block's indentation."""
        rng = self._block_range(name)
        if rng is None:
            raise KeyError(f"provider block not found: {name}")
        start, end = rng
        pattern = re.compile(
            rf"^(\s+)({re.escape(field)}:)(\s+)(\S+)(\s*)(#.*)?$")
        for i in range(start + 1, end):
            if self.lines[i].lstrip().startswith("#"):
                continue
            m = pattern.match(self.lines[i].rstrip("\n"))
            if not m:
                continue
            old = m.group(4)
            if _to_int(old) == value:
                return  # already up to date
            indent, key, gap, _, _, old_comment = m.groups()
            kept = old_comment
            if old_comment and "TBC" in old_comment:
                kept = None  # the value is now verified
            final_comment = kept or comment
            new_line = f"{indent}{key}{gap}{value}"
            if final_comment:
                new_line += f"  {final_comment}"
            self.lines[i] = new_line + "\n"
            self.changes.append(f"{name}.{field}: {old} → {value}")
            return
        # field absent → insertion after the block's first quota field,
        # value aligned on the column of the anchor line
        anchor = None
        indent, value_col = "    ", None
        for i in range(start + 1, end):
            m = re.match(r"^(\s+)(rpm_limit|tpm_limit):(\s+)\S", self.lines[i])
            if m:
                anchor, indent = i, m.group(1)
                value_col = len(m.group(1)) + len(m.group(2)) + 1 + len(m.group(3))
        if anchor is None:
            anchor = start
        head = f"{indent}{field}:"
        pad = " " * max(1, (value_col or 0) - len(head))
        new_line = f"{head}{pad}{value}"
        if comment:
            new_line += f"  {comment}"
        self.lines.insert(anchor + 1, new_line + "\n")
        self.changes.append(f"{name}.{field}: (absent) → {value}")

    def set_weight(self, name: str, rpm: int, tpm: int | None):
        cap = rpm if tpm is None else min(rpm, tpm / 3000)
        weight = round(cap / 15, 2)
        detail = (f"min({rpm}, {tpm}/3000)/15" if tpm is not None else f"{rpm}/15")
        rng = self._block_range(name)
        if rng is None:
            return
        pattern = re.compile(r"^(\s+)(weight:)(\s+)(\S+)(\s*)(#.*)?$")
        for i in range(rng[0] + 1, rng[1]):
            if self.lines[i].lstrip().startswith("#"):
                continue
            m = pattern.match(self.lines[i].rstrip("\n"))
            if not m:
                continue
            old = m.group(4)
            try:
                if abs(float(old) - weight) < 0.005:
                    return
            except ValueError:
                pass
            indent, key, gap = m.group(1), m.group(2), m.group(3)
            self.lines[i] = (f"{indent}{key}{gap}{weight}"
                             f"   # {detail} = {weight} — maj {TODAY}\n")
            self.changes.append(f"{name}.weight: {old} → {weight}")
            return

    def has_model(self, model: str) -> bool:
        """Is the model already referenced — active OR commented block? (a
        commented block = human decision or obsolescence: we do not re-add it)"""
        pat = re.compile(rf"default_model:\s+{re.escape(model)}\s*(#.*)?$")
        return any(pat.search(line) for line in self.lines)

    def comment_out(self, name: str, reason: str):
        """Comments out a provider's whole block (with the date and the reason)."""
        rng = self._block_range(name)
        if rng is None:
            return
        start, end = rng
        # do not take along the empty/comment lines at the end of the block
        while end - 1 > start and not self.lines[end - 1].strip():
            end -= 1
        self.lines.insert(start, f"  # [obsolète {TODAY}] {reason}\n")
        for i in range(start + 1, end + 1):
            stripped = self.lines[i].rstrip("\n")
            if stripped.strip():
                self.lines[i] = "  # " + stripped.removeprefix("  ") + "\n"
        self.changes.append(f"{name}: bloc commenté ({reason})")

    def append_provider(self, name: str, adapter: str, base_url: str,
                        model: str, quotas: dict[str, int],
                        concurrency: int, weight_override: float | None = None,
                        weight_comment: str | None = None):
        """Appends an ACTIVE provider block at end of file, with read quotas.
        weight_override=0.0 = out of rotation (the load balancer filters weight 0)."""
        rpm, tpm = quotas.get("rpm_limit"), quotas.get("tpm_limit")
        if weight_override is not None:
            weight = weight_override
        else:
            weight = round(min(rpm, tpm / 3000) / 15, 2) if rpm and tpm else 1.0

        def line(field: str, value, comment: str = "") -> str:
            s = f"    {field}:{' ' * max(1, 19 - len(field))}{value}"
            return s + (f"  {comment}" if comment else "") + "\n"

        block = [f"\n  # Ajouté par make providers le {TODAY} — quotas free tier relevés\n",
                 f"  {name}:\n",
                 line("adapter", adapter)]
        if rpm:
            block.append(line("rpm_limit", rpm))
        if tpm:
            block.append(line("tpm_limit", tpm))
        if "rpd_limit" in quotas:
            block.append(line("rpd_limit", quotas["rpd_limit"],
                              "# quota requêtes/jour (free tier)"))
        if "tpd_limit" in quotas:
            block.append(line("tpd_limit", quotas["tpd_limit"]))
        if tpm and tpm <= 16000:
            block.append(line("max_tokens_per_request", tpm,
                              "# = TPM free tier (requête unique au-delà → HTTP 413)"))
        block.append(line("base_url", base_url))
        block.append(line("default_model", model))
        default_wc = f"# min({rpm}, {tpm}/3000)/15" if rpm and tpm else ""
        block.append(line("weight", weight, weight_comment or default_wc))
        block.append(line("concurrency_limit", concurrency))
        block.append(line("disable_timeout", 120))
        self.lines += block
        status = "HORS ROTATION (weight 0)" if weight == 0 else "ACTIVÉ"
        self.changes.append(f"nouveau provider {status} : {name} → {model}")

    def write(self):
        # `mkstemp` creates in 0600: without carrying over the original mode, providers.yaml
        # would go from 644 to 600 at the first `make providers`, and the file is mounted
        # read-only in the containers.
        mode = self.path.stat().st_mode & 0o7777 if self.path.exists() else None
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.writelines(self.lines)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, self.path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary without writing providers.yaml")
    parser.add_argument("--provider", type=str, default=None,
                        help="filter on an adapter or a specific instance (e.g.: groq, google, mistral)")
    args = parser.parse_args()

    env = load_env(ENV_FILE)
    cfg = yaml.safe_load(PROVIDERS_YAML.read_text())["providers"]
    editor = YamlEditor(PROVIDERS_YAML)
    alarms: list[str] = []
    infos: list[str] = []

    # Google quotas (only once, shared between instances)
    google_quotas: dict[str, dict[str, int]] = {}
    if (args.provider is None or args.provider == "google") and any(c.get("adapter", n) == "google" for n, c in cfg.items()):
        try:
            google_quotas = google_freetier_quotas()
        except Exception as e:
            alarms.append(f"[ALARME] Cloud Quotas inaccessible (gcloud ?) — "
                          f"instances google inchangées : {e}")

    models_cache: dict[tuple, list[str] | None] = {}

    for name, c in cfg.items():
        adapter = c.get("adapter", name)
        if args.provider and args.provider not in (adapter, name):
            continue
        base_url, model = c["base_url"], c["default_model"]
        key = env.get(f"PROVIDER_KEYS__{name}") or env.get(f"PROVIDER_KEYS__{adapter}")
        if not key:
            infos.append(f"· {name}: pas de clé PROVIDER_KEYS__ — ignoré")
            continue

        # 1. Does the default_model still exist?
        ck = (adapter, base_url, key)
        if ck not in models_cache:
            models_cache[ck] = list_models(adapter, base_url, key)
        models = models_cache[ck]
        if models is None:
            alarms.append(f"[ALARME] {name}: GET /models en échec — instance inchangée")
            continue
        if model not in models:
            editor.comment_out(name, f"default_model '{model}' absent de /models")
            alarms.append(f"[ALARME] {name}: default_model '{model}' ABSENT de "
                          f"/models — bloc commenté, capacité réduite")
            continue

        # 2. Quotas
        try:
            if adapter == "google":
                family = match_google_family(model, google_quotas)
                if not google_quotas:
                    continue  # alarm already raised
                if family is None:
                    alarms.append(f"[ALARME] {name}: aucune famille Cloud Quotas "
                                  f"ne correspond à '{model}' — inchangé")
                    continue
                quotas = google_quotas[family]
            else:
                headers = probe_headers(base_url, key, model)
                quotas = {"mistral": quotas_mistral, "groq": quotas_groq,
                          "cerebras": quotas_cerebras}[adapter](headers)
        except Exception as e:
            alarms.append(f"[ALARME] {name}: sonde en échec — inchangé : {e}")
            continue

        # 3. Application (driven fields only)
        before = len(editor.changes)
        old_tpm = editor.get(name, "tpm_limit")
        for field in MANAGED_FIELDS.get(adapter, ()):
            if field in quotas:
                editor.set(name, field, quotas[field],
                           comment=FIELD_COMMENTS.get((adapter, field)))

        new_tpm = quotas.get("tpm_limit", old_tpm)
        if len(editor.changes) > before:
            # max_tokens_per_request follows the TPM when the field exists
            if (new_tpm and editor.get(name, "max_tokens_per_request") is not None
                    and editor.get(name, "max_tokens_per_request") != new_tpm):
                editor.set(name, "max_tokens_per_request", new_tpm,
                           comment="# = TPM free tier (requête unique au-delà → HTTP 413)")
            rpm = quotas.get("rpm_limit") or editor.get(name, "rpm_limit")
            if rpm:
                editor.set_weight(name, rpm, new_tpm)

    # ── New operational models → providers ENABLED, quotas read ──
    used = {c["default_model"] for c in cfg.values()}
    fresh_by_adapter: dict[str, set[str]] = {}
    adapter_ctx: dict[str, tuple[str, str]] = {}  # adapter → (base_url, key)
    for (adapter, base_url, key), models in models_cache.items():
        adapter_ctx.setdefault(adapter, (base_url, key))
        for m in models or []:
            if (m not in used
                    and re.search(r"(gemini-3|gemma-4|llama|gpt|qwen|glm|mistral|ministral)", m)
                    and not re.search(r"(embed|tts|image|live|audio|robotics|guard|whisper|ocr|voxtral|transcribe|customtools)", m)):
                fresh_by_adapter.setdefault(adapter, set()).add(m)

    def slug(adapter: str, model: str) -> str:
        return f"{adapter}_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")

    for adapter, fresh in sorted(fresh_by_adapter.items()):
        if args.provider and args.provider != adapter:
            continue
        if adapter == "mistral":
            # quota PER ACCOUNT, shared between models: one more model
            # brings no capacity — information only
            infos.append(f"· mistral: {len(fresh)} modèles texte non référencés "
                         "(quota partagé par compte → aucun gain de capacité)")
            continue
        base_url, key = adapter_ctx[adapter]
        for model in sorted(fresh):
            if editor.has_model(model):
                continue  # already referenced (even commented = human decision)
            try:
                if adapter == "google":
                    family = match_google_family(model, google_quotas)
                    # same family = SAME quota bucket (e.g. -preview and stable):
                    # enabling both would make the local limiter overflow the bucket
                    twin = next((n for n, c in cfg.items()
                                 if c.get("adapter", n) == "google"
                                 and match_google_family(c["default_model"],
                                                         google_quotas) == family),
                                None)
                    if twin:
                        infos.append(f"· google/{model}: ignoré — même seau de "
                                     f"quota que l'instance active '{twin}'")
                        continue
                    quotas = dict(google_quotas.get(family or "", {}))
                else:
                    headers = probe_headers(base_url, key, model)
                    quotas = {"groq": quotas_groq,
                              "cerebras": quotas_cerebras}[adapter](headers)
                    if adapter == "groq":
                        # RPM not exposed by the API — 30 for all free tier
                        # chat models (Groq console, read 2026-08-03)
                        quotas.setdefault("rpm_limit", GROQ_RPM_FREE_TIER)
            except Exception as e:
                infos.append(f"· {adapter}/{model}: sonde impossible, non ajouté ({e})")
                continue
            rpd = quotas.get("rpd_limit")
            if not quotas.get("rpm_limit"):
                continue  # no free tier access → not operational
            if rpd is not None and rpd < MIN_RPD_NEW_PROVIDER:
                # too small for the rotation → defined but OUT OF ROTATION
                # (weight 0), usable through a forced `llm.provider`
                editor.append_provider(
                    slug(adapter, model), adapter, base_url, model, quotas,
                    concurrency=DEFAULT_CONCURRENCY.get(adapter, 1),
                    weight_override=0.0,
                    weight_comment=f"# HORS ROTATION : RPD free tier {rpd}/j "
                                   f"< {MIN_RPD_NEW_PROVIDER} — llm.provider forcé uniquement",
                )
                continue
            editor.append_provider(slug(adapter, model), adapter, base_url,
                                   model, quotas,
                                   concurrency=DEFAULT_CONCURRENCY.get(adapter, 1))

    # ── Summary ──
    print("═" * 72)
    print(f"SUMMARY make providers — {TODAY}"
          + ("  (DRY-RUN, nothing is written)" if args.dry_run else ""))
    print("═" * 72)
    if editor.changes:
        print(f"\n{len(editor.changes)} change(s):")
        for ch in editor.changes:
            print(f"  ✎ {ch}")
    else:
        print("\n✓ providers.yaml already up to date (no change)")
    for line in infos:
        print(line)
    if alarms:
        print()
        for a in alarms:
            print(f"  🚨 {a}")

    if editor.changes and not args.dry_run:
        editor.write()
        print(f"\n→ {PROVIDERS_YAML.relative_to(PROJECT_ROOT)} updated "
              f"(remember to reload the services: make restart)")
    print()
    return 1 if alarms else 0


if __name__ == "__main__":
    sys.exit(main())

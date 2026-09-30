"""Appendix K — token counts of the same prompts in French and in English.

No generation: the script counts tokens only.

* French sources are read from git (commit c3983cb, the last state before the English switch of
  2026-09-14). The French expert prompt is rebuilt from its parent `expert_gem_3.8_v2` exactly as
  its provenance record says (one clause removed). Each French text is accepted only if its
  SHA-256 equals the `_traduction.sha256_source` of the English variant.
* Counters: the Gemini `countTokens` endpoint for gemini-3.5-flash-lite (free, no generation;
  key read from .env, never logged) and the open `o200k_base` encoding of tiktoken, offline.

    services/llm-agents/.venv/bin/python scripts/annexes/mesure_K_jetons.py

Output: scripts/annexes/data/mesure_K_jetons.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import tiktoken
import yaml

REPO = Path(__file__).resolve().parents[2]
PROMPTS = "packages/mobility_llm/src/mobility_llm/prompts/prompts.yaml"
TEMPLATE = "packages/mobility_llm/src/mobility_llm/categories/itinary_multi_agent/template.md.j2"
REV_FR = "c3983cb"
MODELE = "gemini-3.5-flash-lite"
CLAUSE = ", en précisant si c'est le cas pourquoi la marche n'obtient pas la plus forte probabilité"
SORTIE = REPO / "scripts/annexes/data/mesure_K_jetons.json"

log = logging.getLogger("mesure_K_jetons")


def git_show(rev: str, chemin: str) -> str:
    return subprocess.run(["git", "show", f"{rev}:{chemin}"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout


def sha(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def cle_google() -> str:
    for ligne in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if ligne.startswith("PROVIDER_KEYS__google="):
            return ligne.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ["PROVIDER_KEYS__google"]


def compter_gemini(texte: str, cle: str) -> int:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELE}:countTokens"
    corps = json.dumps({"contents": [{"parts": [{"text": texte}]}]}).encode("utf-8")
    requete = urllib.request.Request(url, data=corps, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "x-goog-api-key": cle})
    contexte = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(requete, timeout=30, context=contexte) as reponse:
        return int(json.load(reponse)["totalTokens"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    depart = time.monotonic()
    en = yaml.safe_load((REPO / PROMPTS).read_text(encoding="utf-8"))["prompts"]
    fr = yaml.safe_load(git_show(REV_FR, PROMPTS))["prompts"]

    paires = {
        "prompt_minimal_02": (fr["prompt_minimal"]["content"], en["prompt_minimal_02"]),
        "prompt_expert_05": (fr["expert_gem_3.8_v2"]["content"].replace(CLAUSE, ""),
                             en["prompt_expert_05"]),
    }
    textes: dict[str, tuple[str, str]] = {}
    for nom, (texte_fr, variante) in paires.items():
        attendu = variante["_traduction"]["sha256_source"]
        if sha(texte_fr) != attendu:
            log.error("[ALARME] %s: French source sha256 %s != recorded %s", nom, sha(texte_fr), attendu)
            return 1
        log.info("%s: French source authenticated (sha256 %s…)", nom, attendu[:12])
        textes[nom] = (texte_fr, variante["content"])
    textes["decision_template_source"] = (git_show(REV_FR, TEMPLATE),
                                          (REPO / TEMPLATE).read_text(encoding="utf-8"))

    o200k = tiktoken.get_encoding("o200k_base")
    cle = cle_google()
    resultats = {}
    for nom, (texte_fr, texte_en) in textes.items():
        ligne = {"mots_fr": len(texte_fr.split()), "mots_en": len(texte_en.split()),
                 "o200k_fr": len(o200k.encode(texte_fr)), "o200k_en": len(o200k.encode(texte_en))}
        try:
            ligne["gemini_fr"] = compter_gemini(texte_fr, cle)
            ligne["gemini_en"] = compter_gemini(texte_en, cle)
        except Exception as erreur:  # network or quota: keep the offline count, say so
            log.error("[ALARME] %s: countTokens failed (%s), Gemini count left empty", nom,
                      type(erreur).__name__)
        resultats[nom] = ligne
        log.info("%s: %s", nom, ligne)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps({"revision_fr": REV_FR, "modele_gemini": MODELE,
                                  "encodage_ouvert": "o200k_base", "mesures": resultats},
                                 indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("Success: %d text pairs counted in %.1f s → %s", len(resultats),
             time.monotonic() - depart, SORTIE.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())

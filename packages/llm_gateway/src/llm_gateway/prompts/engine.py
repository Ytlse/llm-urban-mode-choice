"""
prompts/engine.py — Jinja2 templating engine for prompts, WITHOUT content.

The engine knows no template, no schema, no system prompt: everything is supplied
at construction by a category bundle (see ``ports.category``) —
``templates_dir`` (one ``<category>.md.j2`` per category), ``schemas_file`` (a JSON object
``{category: output schema}``) and, optionally, ``prompts_file`` (``active:`` /
``prompts:`` of the system prompt variants).

Each template receives:
  - agents     : the validated items of the batch, as dicts
  - parameters : Dict[str, Any]
  - schema     : str (serialised JSON Schema, injected automatically)
  - system_prompt : the active system prompt (or the requested variant)

The manager returns a list of InternalMessage ready to be passed to any adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from pydantic import BaseModel

from llm_gateway.core.models import InternalMessage
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)


def _load_schemas(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Section markers — chosen so that they never appear in Markdown content
_SECTION_SYSTEM = "<!-- SYSTEM -->"
_SECTION_USER   = "<!-- USER -->"

# The calibrated content in prompts.yaml includes the literal JSON schema at the end of the
# text (« Expected JSON schema: {...} »). The template re-injects it dynamically via
# {{ schema }} from schemas.json; this block is therefore removed from the system prompt.
#
# BOTH SPELLINGS, and this is not tolerance for convenience (ticket 074, B-1): switching
# the setup to English translated this sentence in the 22 variants, but the archived runs
# were decided on French variants, and their template fingerprints are recomputed by
# re-reading the archive (`verifier_validite=False`). Keeping only the English form would
# serve — and hash — a duplicated schema block in that case: the sealed fingerprints
# would stop being reproducible, which is precisely what archiving must not break.
_SCHEMA_HEADING = re.compile(
    r"\n\s*(?:Expected JSON schema|Schéma JSON attendu)\s*:.*", re.DOTALL
)


# Path segment that marks a cold archive (ticket 074, A-4): restorable and auditable,
# never used nor referenced. The rule is the one of `experiences.froid`, and it is
# DUPLICATED here on purpose — `llm_gateway` is a generic package that knows neither
# `experiences` nor the mobility domain. Same choice as `core.quota` with respect to
# `prompt_calibration`: four copied lines are better than a reversed dependency.
SEGMENT_ARCHIVE = "archive"


class PromptsArchives(ValueError):
    """A `prompts.yaml` stored in cold archive was designated to be SERVED."""


def _refuser_si_archive(path: Path) -> None:
    if SEGMENT_ARCHIVE in path.resolve().parts:
        raise PromptsArchives(
            f"prompts.yaml stored in cold archive, loading refused: {path}\n"
            f"  A prompt store under `{SEGMENT_ARCHIVE}/` is kept to be audited or "
            f"restored, never served to a model. The live store is "
            f"`mobility_llm/prompts/prompts.yaml`."
        )


def _load_prompts_store(path: Path | None) -> dict[str, Any]:
    """Loads the prompt store (active: + prompts:). Empty if missing or not supplied."""
    if path is None or not path.exists():
        return {"active": {}, "prompts": {}}
    _refuser_si_archive(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("active", {})
    data.setdefault("prompts", {})
    # Families (specs/hygiene-prompts-et-plateforme-experiences.md §4.1): the rule applied
    # to a prompt depends on its family, the same sentence being allowed in one and faulty in
    # the other. Declared, never guessed from the name.
    data.setdefault("familles", {"minimale": [], "defaut": "experte"})
    # Index of the names from before the renaming (ticket 074, C-5). ARCHIVED experiment
    # definitions carry `variante: expert_gem_3.8_v2` and will never be rewritten — the archive
    # is cold. Without this index, recomputing the fingerprint of a past run would become
    # impossible, which is exactly what a renaming must not break.
    data["_par_ancien_nom"] = {
        str(e["_ancien_nom"]): nom
        for nom, e in (data["prompts"] or {}).items()
        if isinstance(e, Mapping) and e.get("_ancien_nom")
    }
    return data


def _strip_schema_block(content: str) -> str:
    """Removes the « Schéma JSON attendu : {...} » block from a calibrated prompt."""
    return _SCHEMA_HEADING.sub("", content).rstrip()


class VariantePromptInvalide(ValueError):
    """A variant carrying `_invalidation` was requested to be SERVED to a model.

    Subclass of ValueError: callers that already filtered unknown variants keep
    working. The refusal only applies to serving — computing a fingerprint or
    re-reading an archive goes through `verifier_validite=False`, otherwise the sealed
    fingerprints of past runs would stop being reproducible.
    """

    def __init__(self, variante: str, invalidation: Mapping[str, Any]) -> None:
        self.variante = variante
        self.regle = invalidation.get("regle")
        self.motif = (invalidation.get("motif") or "").strip()
        self.remplace_par = invalidation.get("remplace_par")
        self.le = invalidation.get("le")
        remplacant = (
            f" Utiliser {self.remplace_par!r} à la place."
            if self.remplace_par
            else " Aucun remplaçant déclaré."
        )
        super().__init__(
            f"variante de prompt {variante!r} INVALIDÉE le {self.le or '?'} "
            f"(règle {self.regle or '?'}) : {self.motif}{remplacant}"
        )


class AvisNeutraliteManquant(ValueError):
    """The variant has no valid neutrality review, in strict mode (hygiene spec §4.2)."""


def _sha_contenu(contenu: str) -> str:
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()


def _neutralite_de(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """`_neutralite` block of an entry, or None."""
    if not isinstance(entry, Mapping):
        return None
    bloc = entry.get("_neutralite")
    return dict(bloc) if isinstance(bloc, Mapping) else None


def _etat_neutralite(entry: Mapping[str, Any]) -> tuple[str, str]:
    """(state, explanation) of the neutrality review of a variant.

    States: `conforme`, `reserve`, `refus`, `absent`, `perime`. The seal counts as much as the
    verdict: a review given on a text that has changed since is worth nothing, otherwise it
    would be enough to have one version validated and then serve another one.
    """
    avis = _neutralite_de(entry)
    if avis is None:
        return "absent", "aucun avis de neutralité (agent prompt-auditor)"
    verdict = str(avis.get("verdict") or "")
    scelle = str(avis.get("sha256_texte") or "")
    if scelle and scelle != _sha_contenu(str(entry.get("content") or "")):
        return (
            "perime",
            f"avis rendu le {avis.get('le') or '?'} sur un texte différent "
            f"(sceau {scelle[:12]}…) — le contenu a changé depuis",
        )
    if verdict == "non_conforme":
        constats = avis.get("constats") or []
        regles = ", ".join(
            str(c.get("regle")) for c in constats if isinstance(c, Mapping) and c.get("regle")
        )
        return "refus", f"verdict non_conforme{f' (règles {regles})' if regles else ''}"
    if verdict == "conforme_avec_reserve":
        return "reserve", "verdict conforme_avec_reserve"
    if verdict == "conforme":
        return "conforme", "verdict conforme"
    return "absent", f"verdict {verdict!r} non reconnu"


def _invalidation_de(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """`_invalidation` block of a prompt entry, if it is declared invalid."""
    if not isinstance(entry, Mapping):
        return None
    bloc = entry.get("_invalidation")
    if isinstance(bloc, Mapping) and bloc.get("statut") == "invalide":
        return dict(bloc)
    return None


class PromptManager:
    """
    Loads and renders the Jinja2 templates to assemble the LLM prompts.
    """

    def __init__(
        self,
        templates_dir: Path | Sequence[Path],
        schemas_file: Path | None = None,
        prompts_file: Path | None = None,
        *,
        template_names: Mapping[str, str] | None = None,
        schema_paths: Mapping[str, Path] | None = None,
        exiger_avis_neutralite: bool = True,
    ) -> None:
        """
        `templates_dir`: one directory (or several) where Jinja2 looks for the templates;
        `schemas_file`: JSON object {category: schema} (optional if `schema_paths` covers all);
        `template_names`: template name per category (default `<category>.md.j2`, relative to
        `templates_dir`, e.g. `itinary_multi_agent/template.md.j2`);
        `schema_paths`: JSON file of the output schema per category (overrides `schemas_file`).
        """
        dirs = [Path(templates_dir)] if isinstance(templates_dir, (str, Path)) else [Path(d) for d in templates_dir]
        self._templates_dirs = dirs
        self._schemas_file = Path(schemas_file) if schemas_file else None
        self._prompts_file = Path(prompts_file) if prompts_file else None
        self._template_names: dict[str, str] = dict(template_names or {})
        # ARMED since 2026-09-10: the 18 variants were audited by the `prompt-auditor`
        # agent and carry a sealed `_neutralite`. Serving an unaudited prompt is now a
        # refusal, not a warning. Explicitly disarmed for a sandbox or a test that builds
        # its own variants.
        self.exiger_avis_neutralite = bool(exiger_avis_neutralite)
        self._env = Environment(
            loader=FileSystemLoader([str(d) for d in dirs]),
            autoescape=select_autoescape(disabled_extensions=("md.j2", "txt.j2")),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._store = _load_prompts_store(self._prompts_file)
        self._schemas: dict[str, dict[str, Any]] = _load_schemas(self._schemas_file) if self._schemas_file else {}
        for category, path in (schema_paths or {}).items():
            self._schemas[category] = json.loads(Path(path).read_text(encoding="utf-8"))

    def template_name(self, category: str) -> str:
        return self._template_names.get(category, f"{category}.md.j2")

    def render(
        self,
        category: str,
        agents: Sequence[BaseModel],
        parameters: dict[str, Any],
    ) -> list[InternalMessage]:
        """
        Renders the template associated with `category` and returns a list of messages.
        """
        template_name = self.template_name(category)

        try:
            schema = self.get_output_schema(category)
            template = self._env.get_template(template_name)
        except (TemplateNotFound, ValueError) as e:
            raise ValueError(f"Template error for '{category}': {str(e)}") from e

        context = {
            "agents":        [a.model_dump() for a in agents],
            "parameters":    parameters,
            "schema":        json.dumps(schema, indent=2, ensure_ascii=False),
            "agent_ids":     [getattr(a, "agent_id", None) for a in agents],
            # The request can designate its prompt variant (ticket 035); `parameters` is part of
            # the batch key, so two variants never share the same call.
            "system_prompt": self.get_system_prompt(category, (parameters or {}).get("prompt_variant")) or "",
        }

        rendered = template.render(**context)
        return self._split_sections(rendered)

    def check_category(self, category: str) -> None:
        """Checks that a category can be served: template AND schema present. Raises ValueError.

        Called by the registry at construction, to fail at startup and not at the
        first request.
        """
        if category not in self._schemas:
            where = self._schemas_file or "neither schemas_file nor schema_path"
            raise ValueError(f"Category {category!r}: output schema missing ({where})")
        name = self.template_name(category)
        try:
            self._env.get_template(name)
        except TemplateNotFound:
            raise ValueError(
                f"Category {category!r}: template {name} missing from {[str(d) for d in self._templates_dirs]}"
            ) from None
        # An invalidated active prompt must fail at startup, not at the thousandth request.
        key = self._store["active"].get(category)
        bloc = _invalidation_de(self._store["prompts"].get(key)) if key else None
        if bloc is not None:
            raise VariantePromptInvalide(key, bloc)

    @property
    def categories(self) -> list[str]:
        """Categories for which an output schema is declared."""
        return sorted(self._schemas)

    def get_output_schema(self, category: str) -> dict[str, Any]:
        """Returns the JSON schema matching the category."""
        if category not in self._schemas:
            raise ValueError(f"Unknown schema for category '{category}'")
        return self._schemas[category]

    def get_system_prompt(
        self,
        category: str,
        variante: str | None = None,
        *,
        verifier_validite: bool = True,
    ) -> str | None:
        """
        Returns the system prompt text for `category` — the ACTIVE variant, or `variante`.

        `variante` (ticket 035): key of `prompts:` designated by an experiment (the template is
        part of its configuration, E1) and passed on by the request (`parameters.prompt_variant`).
        An unknown variant raises ValueError: better to refuse than to serve the active prompt
        instead of the requested one (no silent substitution, EF-62).

        The active variant is designated by `active[category]` in prompts.yaml and
        its content is looked up in `prompts[<key>]`. The « Schéma JSON attendu » block
        is removed (the template re-injects the schema via {{ schema }}). Categories
        missing from `active:` return None.

        Caution: depending on the template, None does not have the same effect. A template
        that keeps a hard-coded SYSTEM simply ignores `system_prompt`. On the other hand,
        `itinary_multi_agent.md.j2` NO LONGER has a hard-coded SYSTEM and renders only
        `{{ system_prompt }}`: if its category disappears from `active:` (or if its
        entry cannot be found), the system prompt silently shrinks to the schema
        alone. Keep `active.itinary_multi_agent` filled in prompts.yaml.
        """
        if variante:
            entry = self._store["prompts"].get(variante)
            if entry is None:
                # Old name from before the ticket 074 renaming: it is resolved, and SAID so.
                # Resolving silently would suggest the requested name still exists, and the
                # next written trace would carry it forward. The warning names the canonical one.
                canonique = (self._store.get("_par_ancien_nom") or {}).get(variante)
                if canonique:
                    logger.warning(
                        f"variante de prompt {variante!r} : nom d'avant le renommage du "
                        f"2026-09-14 (ticket 074) — résolue en {canonique!r}. Les définitions "
                        f"vivantes doivent porter le nom canonique ; seules les traces "
                        f"archivées gardent l'ancien."
                    )
                    variante, entry = canonique, self._store["prompts"].get(canonique)
            if not entry or "content" not in entry:
                connues = sorted(self._store["prompts"])
                anciens = sorted(self._store.get("_par_ancien_nom") or {})
                raise ValueError(
                    f"variante de prompt {variante!r} introuvable dans prompts.yaml "
                    f"(connues : {', '.join(connues)}"
                    + (f" ; anciens noms résolus : {', '.join(anciens)}" if anciens else "")
                    + ")"
                )
            if verifier_validite:
                bloc = _invalidation_de(entry)
                if bloc is not None:
                    raise VariantePromptInvalide(variante, bloc)
                self._verifier_neutralite(variante, entry)
            return _strip_schema_block(entry["content"])
        key = self._store["active"].get(category)
        if not key:
            return None
        entry = self._store["prompts"].get(key)
        if not entry or "content" not in entry:
            logger.warning(
                f"Prompt actif {key!r} introuvable pour la catégorie {category!r}"
            )
            return None
        if verifier_validite:
            bloc = _invalidation_de(entry)
            if bloc is not None:
                raise VariantePromptInvalide(key, bloc)
            self._verifier_neutralite(key, entry)
        return _strip_schema_block(entry["content"])

    def _verifier_neutralite(self, variante: str, entry: Mapping[str, Any]) -> None:
        """Applies the `prompt-auditor` review on the SERVING path.

        A `non_conforme` verdict, or a review made stale by an edit of the text, always
        refuses: these are findings, not gaps. On the other hand a **missing** review only
        refuses if `exiger_avis_neutralite` is armed — otherwise introducing the rule would
        have made the 25 existing variants unusable, the active one included, before any
        audit could even be given. Catching up is done variant by variant, then it is armed.
        """
        etat, pourquoi = _etat_neutralite(entry)
        if etat in ("refus", "perime"):
            raise AvisNeutraliteManquant(
                f"prompt variant {variante!r} rejected by the neutrality audit: "
                f"{pourquoi} → have it re-examined by the prompt-auditor agent "
                f"(specs/hygiene-prompts-et-plateforme-experiences.md §4.2)"
            )
        if etat == "absent":
            if self.exiger_avis_neutralite:
                raise AvisNeutraliteManquant(
                    f"prompt variant {variante!r} without a valid neutrality opinion "
                    f"({pourquoi}) → have it audited by the prompt-auditor agent, or disarm "
                    f"`exiger_avis_neutralite`"
                )
            # loguru: no %s interpolation — the message must be built before the call.
            logger.warning(
                f"Prompt {variante!r} served without a neutrality review ({pourquoi}) — "
                "prompt-auditor audit pending"
            )

    def neutralite(self, variante: str) -> dict[str, Any] | None:
        """Neutrality review declared for a variant, as is. Never raises."""
        return _neutralite_de(self._store["prompts"].get(variante))

    def etat_neutralite(self, variante: str) -> tuple[str, str]:
        """(state, explanation) — `conforme`, `reserve`, `refus`, `absent` or `perime`."""
        entry = self._store["prompts"].get(variante)
        if not entry:
            return "absent", "variante inconnue"
        return _etat_neutralite(entry)

    def invalidation(self, variante: str) -> dict[str, Any] | None:
        """`_invalidation` block of a variant, or None if it is valid. Never raises."""
        return _invalidation_de(self._store["prompts"].get(variante))

    def famille(self, variante: str) -> str:
        """Declared family of a variant: « minimale » or « experte » (store default)."""
        fam = self._store.get("familles") or {}
        if variante in (fam.get("minimale") or []):
            return "minimale"
        entry = self._store["prompts"].get(variante) or {}
        return str(entry.get("famille") or fam.get("defaut") or "experte")

    def variantes(self) -> list[str]:
        """Available `prompts:` keys (to validate an experiment before launching it)."""
        return sorted(self._store["prompts"])

    def active_prompt_checksum(self, *categories: str, length: int = 12) -> str:
        """
        Stable fingerprint of the active system prompts — changes as soon as a prompt changes.

        Without argument: fingerprint of all the `active:` categories (any change
        to a system prompt invalidates the fingerprint). With arguments: restricted to the
        given categories. Serves as the isolation key of the LLM cache: a prompt
        change produces a new checksum → a new cache directory.
        """
        cats = list(categories) if categories else sorted(self._store["active"])
        # Fingerprint, not serving: does not refuse an invalidated variant, otherwise an
        # invalidated active prompt would make the cache key computation fail instead of the render.
        parts = [
            f"{cat}:{self.get_system_prompt(cat, verifier_validite=False) or ''}"
            for cat in cats
        ]
        raw = "\n".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]

    # ------------------------------------------------------------------

    def _split_sections(self, rendered: str) -> list[InternalMessage]:
        """
        Splits the rendered text into system/user messages according to the markers.

        Markers supported in the template:
          <!-- SYSTEM -->
          <!-- USER -->

        If no marker is present, the whole content becomes a user message.
        """
        messages: list[InternalMessage] = []

        has_markers = _SECTION_SYSTEM in rendered or _SECTION_USER in rendered

        if not has_markers:
            messages.append(InternalMessage(role="user", content=rendered.strip()))
            return messages

        # Split by known markers
        MARKER_ROLE = {
            _SECTION_SYSTEM: "system",
            _SECTION_USER:   "user",
        }

        # A unique sentinel is inserted so the split can be done cleanly
        _SENTINEL = "\x00SECTION\x00"
        tagged = rendered
        for marker in MARKER_ROLE:
            tagged = tagged.replace(marker, f"{_SENTINEL}{marker}{_SENTINEL}")

        parts = tagged.split(_SENTINEL)
        current_role: str | None = None
        buffer: list[str] = []

        for part in parts:
            stripped = part.strip()
            if stripped in MARKER_ROLE:
                # Flush the previous buffer
                if buffer and current_role:
                    content = "\n".join(buffer).strip()
                    if content:
                        messages.append(InternalMessage(role=current_role, content=content))
                current_role = MARKER_ROLE[stripped]
                buffer = []
            else:
                if current_role and stripped:
                    buffer.append(stripped)

        # Flush final
        if buffer and current_role:
            content = "\n".join(buffer).strip()
            if content:
                messages.append(InternalMessage(role=current_role, content=content))

        return [m for m in messages if m.content]


__all__ = ["PromptManager", "PromptsArchives"]

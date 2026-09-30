"""Dashboard hygiene filters — spec `hygiene-prompts-et-plateforme-experiences.md`.

Three removals, one principle: only offer what can be relied on, and **count
what is removed**. A silent removal would suggest data loss where everything is intact.

The test that matters is the PARITY one: the dashboard reproduces the refusal criterion of the
`PromptManager` without importing the engine (it reads the YAML). This duplication is only tenable
when locked — otherwise the form would offer a prompt that the launch would refuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

# `from scripts.dashboard import experiences`, never `import experiences`: the package
# `services/llm-agents/experiences` has the same name and would win depending on sys.path order.
# It is this very collision that made an import silently inert in the audited
# module (cf. `_module_aptitude`).
from scripts.dashboard import experiences as D  # noqa: E402


# ── Prompts ──────────────────────────────────────────────────────────────────


def _refusees_par_le_moteur() -> set[str]:
    from mobility_llm import build_prompt_manager
    from llm_gateway.prompts.engine import AvisNeutraliteManquant, VariantePromptInvalide

    pm = build_prompt_manager()
    refusees = set()
    for v in pm.variantes():
        try:
            pm.get_system_prompt("itinary_multi_agent", v)
        except (VariantePromptInvalide, AvisNeutraliteManquant):
            refusees.add(v)
    return refusees


def test_parite_avec_le_refus_de_la_passerelle():
    """What the dashboard EXCLUDES is EXACTLY what the engine refuses to serve.

    Parity covers `prompt_ecarte`, not the whole set of variants removed from the choice:
    an ARCHIVED variant leaves the list without the engine refusing it, and including it would make
    this test fail for a reason that has nothing to do with the hygiene it locks.
    """
    d = D._yaml(D.PROMPTS_YAML)
    prompts = d.get("prompts") or {}
    ecartees_ihm = {n for n in prompts if D.prompt_ecarte(prompts.get(n) or {}) is not None}
    refusees_moteur = _refusees_par_le_moteur()
    assert ecartees_ihm == refusees_moteur, (
        f"divergence — UI only: {ecartees_ihm - refusees_moteur}; "
        f"engine only: {refusees_moteur - ecartees_ihm}"
    )


def test_une_archivee_sort_du_choix_mais_reste_servable():
    """Archiving is an author's decision, not a hygiene verdict.

    `b0_pristine` is the frozen seed of the reference campaign: removing it from the form must
    never prevent replaying an experiment that designates it.
    """
    from mobility_llm import build_prompt_manager

    archivees = set(D.variantes_prompt_archivees())
    assert archivees, "the current repository state counts nine"
    proposees, _ = D.variantes_prompt()
    assert not (archivees & set(proposees))

    pm = build_prompt_manager()
    servables = set(pm.variantes()) - _refusees_par_le_moteur()
    assert "prompt_expert_01" in archivees and "prompt_expert_01" in servables


def test_les_variantes_proposees_sont_toutes_servables():
    from mobility_llm import build_prompt_manager

    pm = build_prompt_manager()
    proposees, _ = D.variantes_prompt()
    assert proposees, "the choice must not be empty"
    for v in proposees:
        assert pm.get_system_prompt("itinary_multi_agent", v), v


def test_la_variante_active_reste_proposee():
    """Excluding the active prompt would empty the form of its default."""
    proposees, active = D.variantes_prompt()
    assert active in proposees, active


def test_les_ecartees_sont_comptees_avec_un_motif():
    ecartees = D.variantes_prompt_ecartees()
    assert ecartees, "the current repository state counts ten — three refused for serving, seven archived"
    for nom, raison in ecartees.items():
        assert raison and raison.strip(), nom


def test_inclure_ecartees_les_rend():
    """Removal is a display filter, never an amputation of the source."""
    sans, _ = D.variantes_prompt()
    avec, _ = D.variantes_prompt(inclure_ecartees=True)
    assert set(avec) - set(sans) == set(D.variantes_prompt_ecartees())


@pytest.mark.parametrize(
    "entree,attendu",
    [
        ({"content": "t"}, None),
        ({"content": "t", "_invalidation": {"statut": "invalide", "regle": "M1"}}, "invalidée"),
        ({"content": "t", "_invalidation": {"statut": "leve"}}, None),
        ({"content": "t", "_neutralite": {"verdict": "non_conforme"}}, "non conforme"),
        ({"content": "t", "_neutralite": {"verdict": "conforme_avec_reserve"}}, None),
        ({"content": "t", "_neutralite": {"verdict": "conforme", "sha256_texte": "faux"}}, "périmé"),
        ("pas un dict", "illisible"),
    ],
)
def test_critere_ecarte(entree, attendu):
    r = D.prompt_ecarte(entree)
    if attendu is None:
        assert r is None
    else:
        assert r and attendu in r


@pytest.mark.parametrize(
    "entree,attendu",
    [
        ({"content": "t"}, None),
        ({"content": "t", "_archive": {"statut": "archive", "le": "2026-09-11"}}, "archivée (2026-09-11)"),
        ({"content": "t", "_archive": {"statut": "archive"}}, "archivée"),
        # A block present but with another status removes nothing: only `archive` archives.
        ({"content": "t", "_archive": {"statut": "levee"}}, None),
        ("pas un dict", None),
    ],
)
def test_critere_archive(entree, attendu):
    assert D.prompt_archive(entree) == attendu


def test_une_archivee_invalidee_s_annonce_par_le_refus_de_service():
    """The most serious reason wins: unusable before removed from the choice."""
    entree = {"content": "t", "_archive": {"statut": "archive", "le": "2026-09-11"},
              "_invalidation": {"statut": "invalide", "regle": "C1"}}
    assert "invalidée" in D.prompt_ecarte(entree)
    assert D.prompt_archive(entree) == "archivée (2026-09-11)"


# ── Models ───────────────────────────────────────────────────────────────────


def test_les_modeles_a_quota_derisoire_sont_ecartes():
    """A model at 20 requests/day cannot carry an experiment of 2,285 calls."""
    inaptes = D.modeles_inaptes()
    assert inaptes, "the aptitude module must be loaded — an empty dict signals a silent import"
    assert any("gemini-3.8-flash" == m for m in inaptes), sorted(inaptes)
    for m, raison in inaptes.items():
        assert raison and raison.strip(), m


def test_le_module_d_aptitude_se_charge_malgre_la_collision_de_noms():
    """This module is also called `experiences`: the import must go through the file path."""
    assert D._module_aptitude() is not None


def test_les_modeles_utilisables_restent_proposables():
    inaptes = D.modeles_inaptes()
    locaux, distants = D.modeles_par_portee()
    assert [m for m in distants if m not in inaptes], "some remote models must remain"
    assert [m for m in locaux if m not in inaptes], "no local model must be excluded"


# ── Experiments ──────────────────────────────────────────────────────────────


# These two tests read the REAL repository, not a test tree: they check that
# the current state of `data/experiences` is consistent. Ticket 045 emptied this base — the
# 46 definitions and 36 runs are archived — so there is nothing left to check
# until the rebuild on the v5 cohort has started. We SKIP with an explicit
# reason, following the convention already in force (`test_composite_scoring`, `test_scoring_on_close`), rather
# than weaken the assertion: the day experiments exist, these tests resume
# their work without anyone having to think about it.
# These two tests cover experiments REMOVED from the table. When there are none,
# they have nothing to check — and requiring it would amount to asking the repository to
# permanently contain at least one archived experiment, which is neither true nor desirable.
# Ticket 045 precisely restarted from a fresh base where all are active.
_RIEN_A_RETIRER = pytest.mark.skipif(
    not [l for l in D.lister() if D.masquee(l)],
    reason="no experiment removed in data/experiences: the rule does not apply here",
)


@_RIEN_A_RETIRER
def test_les_experiences_invalidees_ou_archivees_sortent_du_tableau():
    toutes = D.lister()
    visibles = [l for l in toutes if not D.masquee(l)]
    assert visibles, "the table must not be empty"
    assert all(l.get("statut") == "actif" for l in visibles)
    assert len(visibles) < len(toutes), "the current repository state hides some"


@_RIEN_A_RETIRER
def test_chaque_experience_retiree_porte_son_motif():
    """Without a reason, a disappearance reads as data loss."""
    retirees = [l for l in D.lister() if D.masquee(l)]
    assert retirees
    for l in retirees:
        assert l.get("statut") in ("archivee", "invalide"), l["experience"]
        assert l.get("statut_motif"), l["experience"]


# ── Thinking depth ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("valeur,index", [(None, 0), (0, 1), (-1, 2), (1024, 3), (256, 3)])
def test_index_du_choix_de_reflexion(valeur, index):
    assert D._index_reflexion(valeur) == index


def test_valeur_de_reflexion_sans_widget():
    """The first three choices consult no widget: None / 0 / -1."""
    assert D._valeur_reflexion(D.CHOIX_REFLEXION[0], None, None, None) is None
    assert D._valeur_reflexion(D.CHOIX_REFLEXION[1], None, None, None) == 0
    assert D._valeur_reflexion(D.CHOIX_REFLEXION[2], None, None, None) == -1


def test_valeur_de_reflexion_budget_fixe():
    class _Col:
        def number_input(self, *a, **kw):
            self.defaut = a[3]
            return a[3]

    col = _Col()
    assert D._valeur_reflexion(D.CHOIX_REFLEXION[3], 2048, col, lambda s: s) == 2048
    assert col.defaut == 2048, "the saved value must prefill the field"
    col2 = _Col()
    D._valeur_reflexion(D.CHOIX_REFLEXION[3], None, col2, lambda s: s)
    assert col2.defaut == 1024, "without a saved value, a reasonable default"


def test_aller_retour_none_ne_cree_pas_de_cle():
    """`None` must leave `parametres` WITHOUT the key: otherwise two hashes for one setting."""
    src = (RACINE / "scripts" / "dashboard" / "experiences.py").read_text(encoding="utf-8")
    assert 'if v.get("reflexion") is not None:' in src
    assert 'params["thinking_budget"] = int(v["reflexion"])' in src


# ── Thinking "maximum", backed by the declared cap ──────────────────────────


def test_maximum_indisponible_sans_plafond_declare():
    """No `thinking_budget_max` in providers.yaml today: no "maximum" offered."""
    choix, plafond = D.choix_reflexion_pour("gemini-3.5-flash-lite")
    assert plafond is None
    assert choix == D.CHOIX_REFLEXION
    assert not any(c.startswith(D.CHOIX_REFLEXION_MAX) for c in choix)


def test_maximum_propose_quand_le_plafond_est_declare(monkeypatch):
    monkeypatch.setattr(D, "plafond_reflexion", lambda m: 24576)
    choix, plafond = D.choix_reflexion_pour("un-modele")
    assert plafond == 24576
    assert choix[-1] == f"{D.CHOIX_REFLEXION_MAX} (24576 jetons)"


def test_maximum_resout_vers_un_nombre_concret():
    """No magic word in the hash: "maximum" becomes the declared cap."""
    v = D._valeur_reflexion(f"{D.CHOIX_REFLEXION_MAX} (24576 jetons)", None, None, None, 24576)
    assert v == 24576


def test_maximum_sans_plafond_ne_fabrique_rien():
    """Without a cap, "maximum" must not invent a value: nothing is requested."""
    assert D._valeur_reflexion(f"{D.CHOIX_REFLEXION_MAX} (x)", None, None, None, None) is None


def test_une_valeur_egale_au_plafond_se_relit_comme_maximum():
    assert D._index_reflexion(24576, 24576) == 4
    assert D._index_reflexion(1024, 24576) == 3
    assert D._index_reflexion(None, 24576) == 0


def test_le_champ_libre_est_borne_par_le_plafond():
    class _Col:
        def number_input(self, *a, **kw):
            self.borne_haute, self.defaut = a[2], a[3]
            return a[3]

    col = _Col()
    D._valeur_reflexion(D.CHOIX_REFLEXION[3], None, col, lambda s: s, 4096)
    assert col.borne_haute == 4096, "the field must not allow exceeding the cap"
    assert col.defaut == 1024


def test_plafond_absent_sur_une_seule_instance_annule_le_maximum(monkeypatch):
    """A cap is not inferred from a subset of instances."""
    monkeypatch.setattr(D, "modeles", lambda: {"m": ["a", "b"]})
    monkeypatch.setattr(D, "_yaml", lambda p: {"providers": {
        "a": {"thinking_budget_max": 8192}, "b": {}}})
    assert D.plafond_reflexion("m") is None


def test_plafond_retient_le_plus_petit(monkeypatch):
    """Asking for more would get the call refused on the most constrained instance."""
    monkeypatch.setattr(D, "modeles", lambda: {"m": ["a", "b"]})
    monkeypatch.setattr(D, "_yaml", lambda p: {"providers": {
        "a": {"thinking_budget_max": 8192}, "b": {"thinking_budget_max": 4096}}})
    assert D.plafond_reflexion("m") == 4096


# ── Thinking level (current API setting) ─────────────────────────────────────


def test_niveaux_declares_par_modele():
    """Taken from the provider's docs: `minimal` does not exist on 3.7 nor 3.8."""
    assert D.niveaux_reflexion("gemini-3.8-flash") == ["low", "medium", "high"]
    assert D.niveaux_reflexion("gemini-3.6-flash") == ["minimal", "low", "medium", "high"]


def test_aucun_niveau_pour_un_modele_non_documente():
    assert D.niveaux_reflexion("mistral-small-latest") == []
    assert D.niveaux_reflexion("gemini-3.1-flash-lite") == []   # instances without thinking_levels
    # A model ABSENT from providers.yaml returns `[]` too: without this second case, the
    # renaming of 2026-09-10 would have made the assertion above true by ignorance.
    assert D.niveaux_reflexion("modele-jamais-declare") == []


def test_niveaux_sont_l_intersection_des_instances(monkeypatch):
    """A level accepted by one key and not the other would fail depending on the draw."""
    monkeypatch.setattr(D, "modeles", lambda: {"m": ["a", "b"]})
    monkeypatch.setattr(D, "_yaml", lambda p: {"providers": {
        "a": {"thinking_levels": ["minimal", "low", "high"]},
        "b": {"thinking_levels": ["low", "medium", "high"]}}})
    assert D.niveaux_reflexion("m") == ["low", "high"]


def test_une_instance_sans_declaration_annule_les_niveaux(monkeypatch):
    monkeypatch.setattr(D, "modeles", lambda: {"m": ["a", "b"]})
    monkeypatch.setattr(D, "_yaml", lambda p: {"providers": {
        "a": {"thinking_levels": ["low", "high"]}, "b": {}}})
    assert D.niveaux_reflexion("m") == []


def test_le_maximum_est_un_nom_pas_un_nombre():
    """It is the answer to "how to say max": `high`, with no cap to look up."""
    assert "high" in D.niveaux_reflexion("gemini-3.8-flash")
    assert "maximum" in D.LIBELLES_NIVEAU["high"]


def test_niveau_et_budget_ne_partent_jamais_ensemble():
    """The API returns 400: the form writes only one of the two."""
    src = (RACINE / "scripts" / "dashboard" / "experiences.py").read_text(encoding="utf-8")
    assert 'if v.get("niveau_reflexion"):' in src
    assert 'elif v.get("reflexion") is not None:' in src

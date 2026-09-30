"""A fingerprint states what was actually served.

R5: `empreinte_gabarit` returns the SAME sha256 on the host and in the container.
R6: the `sources` list exactly the hashed pieces.

The defect fixed here: the option template was looked up under
`racine_du_dépôt / "services" / "llm-agents" / "text_helper" / …`. In the `controller` container,
the `llm-agents` folder is mounted on `/app` — the file is therefore at
`/app/text_helper/…`, and the old resolution looked for it at `/llm-agents/text_helper/…`.
Not found, it silently dropped out of the hash: two fingerprints for the same text,
`88f0aefcff` on the host and `9afe7d4a52` in the container for `prompt_minimal`. That is what
made it look, on 2026-09-11, as if two arms labelled with the same prompt had not received
the same thing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiences.chemins import racine_depot, racine_llm_agents
from experiences.experience import (
    GABARIT_OPTION_REL,
    chemin_gabarit_option,
    empreinte_gabarit,
)

CATEGORIE = "itinary_multi_agent"


def test_r5_racine_llm_agents_porte_le_paquet_experiences():
    """The `llm-agents` root is the folder that contains the `experiences` package.

    It is the only anchor true on both sides: `<dépôt>/llm-agents` on the host, `/app`
    in the container. The repository root, by contrast, differs (`<dépôt>` versus `/app`), and
    that is precisely what broke the resolution.
    """
    racine = racine_llm_agents()
    assert (racine / "experiences" / "experience.py").is_file()
    assert (racine / "settings.py").is_file()


def test_r5_le_gabarit_option_est_trouve_sous_la_racine_llm_agents():
    """The option template exists, and it is resolved without going through the repository root."""
    chemin = chemin_gabarit_option()
    assert chemin.is_file(), f"gabarit d'option introuvable : {chemin}"
    assert chemin == racine_llm_agents() / GABARIT_OPTION_REL


def test_r5_la_resolution_ne_depend_pas_de_la_racine_du_depot():
    """The template path is NOT derived from the repository root.

    Otherwise the container, whose repository root is `/app` and not `<dépôt>`, would look for
    `/app/llm-agents/text_helper/…` — which does not exist.
    """
    faux = racine_depot() / "services" / "llm-agents" / GABARIT_OPTION_REL
    vrai = chemin_gabarit_option()
    # On the host the two coincide by accident; in the container, they do not. The rule is that
    # `vrai` ALWAYS exists, which `faux` does not guarantee.
    assert vrai.is_file()
    if faux != vrai:
        assert not faux.is_file(), (
            "two competing paths exist: the ambiguity remains"
        )


def test_r6_les_sources_enumerent_exactement_les_morceaux_haches():
    """Each announced source matches a piece actually concatenated, and vice versa.

    The hash is rebuilt from only the pieces `sources` announces: if it comes out
    right, the fingerprint neither hides nor invents a piece.
    """
    from mobility_llm import prompt_manager as get_prompt_manager

    emp = empreinte_gabarit(CATEGORIE, "prompt_minimal_02")
    systeme = (
        get_prompt_manager().get_system_prompt(
            CATEGORIE, "prompt_minimal_02", verifier_validite=False
        )
        or ""
    )
    gabarit = chemin_gabarit_option().read_text(encoding="utf-8")

    assert emp["sources"] == [
        "prompts.yaml:prompt_minimal_02",
        chemin_gabarit_option().name,
    ]
    attendu = hashlib.sha256(f"{systeme}\n\x00\n{gabarit}".encode()).hexdigest()
    assert emp["sha256"] == attendu


def test_r6_une_variante_invalidee_garde_une_empreinte_et_le_dit():
    """Invalidation is announced outside the hash: a valid variant keeps its earlier fingerprint."""
    emp = empreinte_gabarit(CATEGORIE, "prompt_minimal_01")
    assert emp.get("invalide") is True
    assert emp.get("invalide_regle")
    assert len(emp["sha256"]) == 64


def test_r6_le_gabarit_option_manquant_est_une_erreur_pas_un_silence(monkeypatch):
    """A piece promised but missing must show, not vanish from the hash.

    This is the "absence of measurement passes for a healthy case" pattern: the old version
    silently left out the missing template and returned a shorter fingerprint,
    normal in appearance.
    """
    import experiences.experience as E

    monkeypatch.setattr(
        E, "chemin_gabarit_option", lambda: Path("/introuvable/absent.j2")
    )
    emp = E.empreinte_gabarit(CATEGORIE, "prompt_minimal_02")
    assert any("introuvable" in s for s in emp["sources"]), emp["sources"]

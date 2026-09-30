"""Core memory: computed blocks, never written by the model.

The central point is the SAFEGUARD: the three blocks are computed, none is written by the
model. A text periodically rewritten by a model drifts and invents; a computed block remains
checkable against its source.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.memory import MemoryEntry, MemoryType
from llm.noyau import (
    CONNAISSANCES_MAX,
    OCCURRENCES_MIN_HABITUDE,
    bloc_changements,
    bloc_connaissances,
    bloc_habitudes,
    memoire_noyau,
    noter_trajet,
)
from settings import settings
from sim_clock import wall_clock

T0 = 1773637200


def _journal(*trajets) -> dict:
    j: dict = {}
    for motif, creneau, mode, retard in trajets:
        noter_trajet(j, motif, creneau, mode, retard)
    return j


def _concept(texte, obs=1, contre=0, jours=1, depasse=None):
    return MemoryEntry(
        content=f'["{texte}", "k", "s", "t", "work"]',
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.CONCEPT,
        person_id="42",
        observations=obs,
        contre_exemples=contre,
        depasse_le=depasse,
    )


def _choc(texte, gravite=0.9, jours=1):
    return MemoryEntry(
        content=texte,
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.REFLECTION,
        person_id="42",
        importance=gravite,
        force=19.6,
    )


# ═══════════════════════ A. The "My habits" block ═══════════════════════════════


def test_A1_la_part_du_mode_dominant_est_dite_sur_le_total():
    j = _journal(*([("work", "matin", "cycling", 0)] * 9), *([("work", "matin", "car", 0)] * 2))
    lignes = bloc_habitudes(j)
    assert len(lignes) == 1
    assert "9 times out of 11" in lignes[0]
    assert "by bike" in lignes[0]


def test_A2_une_occurrence_n_est_pas_une_habitude():
    """Saying "bike, 1 time out of 1" would give a random event the authority of a routine."""
    assert bloc_habitudes(_journal(("work", "matin", "cycling", 0))) == []
    j = _journal(*([("work", "matin", "cycling", 0)] * (OCCURRENCES_MIN_HABITUDE - 1)))
    assert bloc_habitudes(j) == []


def test_A3_deux_motifs_font_deux_lignes():
    j = _journal(
        *([("work", "matin", "cycling", 0)] * 4),
        *([("shop", "midi", "walking", 0)] * 4),
    )
    assert len(bloc_habitudes(j)) == 2


def test_A4_les_retards_sont_comptes_et_dits_sans_etre_inventes():
    sans = _journal(*([("work", "matin", "cycling", 0)] * 4))
    assert "retard" not in bloc_habitudes(sans)[0]
    avec = _journal(
        *([("work", "matin", "cycling", 0)] * 3),
        ("work", "matin", "cycling", 900),
    )
    assert "1 delay(s) of more than 10 min" in bloc_habitudes(avec)[0]


def test_A4bis_un_retard_court_ne_compte_pas():
    j = _journal(*([("work", "matin", "cycling", 300)] * 4))
    assert "retard" not in bloc_habitudes(j)[0]


def test_A5_sans_trajet_le_bloc_est_absent_et_non_vide_avec_un_titre():
    """A title without content tells the model there should be something."""
    assert memoire_noyau({}, [], wall_clock(T0)) == []
    assert "My habits" not in "".join(memoire_noyau({}, [], wall_clock(T0)))


def test_A6_un_trajet_sans_mode_resolu_ne_fausse_pas_le_denominateur():
    """It is not counted at all, rather than counted as an unknown mode."""
    j = _journal(*([("work", "matin", "cycling", 0)] * 4), ("work", "matin", None, 0))
    assert "4 times out of 4" in bloc_habitudes(j)[0]


def test_A7_le_bloc_des_habitudes_est_calcule_et_non_ecrit_par_le_modele():
    """THE safeguard of the lot.

    `bloc_habitudes` takes ONLY the trip journal: it cannot invent, and its
    result can be checked against its source. No model text enters it.
    """
    import inspect

    from llm import noyau

    signature = inspect.signature(noyau.bloc_habitudes)
    assert list(signature.parameters) == ["journal"], (
        "the habits block must depend only on the trip journal"
    )
    j = _journal(*([("work", "matin", "cycling", 0)] * 4))
    assert bloc_habitudes(j) == bloc_habitudes(j), "deterministic computation"


# ═══════════════════ B. The "What I know" block ═════════════════════════════════


def test_B1_chaque_enonce_porte_son_compteur():
    lignes = bloc_connaissances([_concept("le bus 401 est fiable", obs=12)])
    assert "le bus 401 est fiable" in lignes[0]
    assert "(12 obs.)" in lignes[0]


def test_B2_un_concept_hors_service_est_absent():
    """We do not serve the model what the agent no longer believes."""
    assert bloc_connaissances([_concept("faux", obs=0, contre=3)]) == []


def test_B2bis_un_concept_mis_a_l_ecart_est_absent():
    ecarte = _concept("périmé", obs=5, contre=1, depasse="2026-03-15T22:00:00")
    assert bloc_connaissances([ecarte]) == []


def test_B3_un_concept_jamais_confirme_est_present():
    lignes = bloc_connaissances([_concept("vu une fois", obs=0)])
    assert len(lignes) == 1 and "(0 obs.)" in lignes[0]


def test_B4_les_plus_confiants_d_abord_et_plafonnes():
    concepts = [_concept(f"c{i}", obs=i) for i in range(12)]
    lignes = bloc_connaissances(concepts)
    assert len(lignes) == CONNAISSANCES_MAX
    assert "c11" in lignes[0], "the most confident first"


def test_B5_sans_concept_le_bloc_est_absent():
    assert bloc_connaissances([]) == []


def test_B6_une_entree_episodique_n_est_pas_une_connaissance():
    assert bloc_connaissances([_choc("je suis tombé")]) == []


# ═════════════ C. The "What changed recently" block ═════════════════════════════


def test_C1_un_choc_recent_est_present():
    lignes = bloc_changements([_choc("panne ligne A, 45 minutes perdues")], wall_clock(T0))
    assert lignes and "panne ligne A" in lignes[0]


def test_C2_un_concept_mis_a_l_ecart_est_present():
    """This is where the hysteresis becomes readable in the prompt itself."""
    ecarte = _concept(
        "la ligne A est fiable", obs=1, contre=4,
        depasse=(wall_clock(T0) - timedelta(days=1)).isoformat(),
    )
    lignes = bloc_changements([ecarte], wall_clock(T0))
    assert lignes and "I no longer believe" in lignes[0]
    assert "la ligne A est fiable" in lignes[0]


def test_C3_un_choc_ancien_est_absent():
    assert bloc_changements([_choc("vieux choc", jours=60)], wall_clock(T0)) == []


def test_C4_sans_changement_le_bloc_est_absent():
    assert bloc_changements([_concept("stable", obs=5)], wall_clock(T0)) == []


def test_C5_un_trajet_ordinaire_n_est_pas_un_changement():
    ordinaire = _choc("trajet sans histoire", gravite=0.1)
    assert bloc_changements([ordinaire], wall_clock(T0)) == []


def test_C6_sans_horloge_simulee_le_bloc_est_vide_plutot_que_faux():
    """Never a fallback on the machine clock."""
    assert bloc_changements([_choc("panne")], None) == []


# ═══════════════════ D. The complete block ══════════════════════════════════════


def test_D1_le_bloc_ne_porte_aucune_metadonnee_sur_lui_meme():
    """Neither an update date, nor a number of days lived.

    It changes no decision, costs tokens, and breaks the fiction that the templates
    maintain: a person does not think "my habits summary is three days old".
    """
    bloc = "\n".join(
        memoire_noyau(
            _journal(*([("work", "matin", "cycling", 0)] * 4)),
            [_concept("le bus 401 est fiable", obs=12)],
            wall_clock(T0),
        )
    )
    for interdit in ("mis à jour", "jours de vécu", "dernière mise", "il y a "):
        assert interdit not in bloc.lower(), f"forbidden metadata: {interdit}"
    assert "2026" not in bloc, "no date on the block itself"


def test_D2_les_trois_blocs_apparaissent_dans_l_ordre():
    bloc = memoire_noyau(
        _journal(*([("work", "matin", "cycling", 0)] * 4)),
        [
            _concept("le bus 401 est fiable", obs=12),
            _choc("panne ligne A"),
        ],
        wall_clock(T0),
    )
    texte = "\n".join(bloc)
    assert texte.index("My habits") < texte.index("What I know")
    assert texte.index("What I know") < texte.index("What changed recently")


def test_D3_un_bloc_vide_ne_laisse_pas_son_titre():
    bloc = "\n".join(memoire_noyau({}, [_concept("je sais", obs=3)], wall_clock(T0)))
    assert "What I know" in bloc
    assert "My habits" not in bloc
    assert "What changed" not in bloc


def test_D4_le_parametre_des_episodiques_est_distinct_du_top_k():
    """Reusing `long_term_max_entries_query` would make any sensitivity measurement ambiguous."""
    assert settings.agent.memoire__episodiques_avec_noyau == 3
    assert settings.agent.long_term_max_entries_query == 10


def test_D5_le_lot_4_ne_coute_aucun_appel_au_modele():
    """The three blocks are computed: no extra inference.

    Checked through the signature: `memoire_noyau` receives neither client, nor gateway, nor prompt.
    """
    import inspect

    from llm import noyau

    # `person_id` added with the changes window: it only serves the log — a shock memory
    # leaving the window was invisible without it. It brings no model call any closer,
    # and the guard that matters is the one on the source, just below.
    # `lignes` added with the pre-decision reading: the lines GUARANTEED to the prompt (article read, household
    # message) arrive already written — the core places them, it does not produce them.
    assert list(inspect.signature(noyau.memoire_noyau).parameters) == [
        "journal",
        "entrees",
        "maintenant",
        "person_id",
        "lignes",
    ]
    source = inspect.getsource(noyau)
    for interdit in ("llm_client", "execute(", "PromptName", "gateway"):
        assert interdit not in source, f"the core must ask nothing of the model: {interdit}"


# ═══════════════════ Persisted journal ══════════════════════════════════════════


def test_le_journal_est_serialisable_en_json():
    """It is persisted with the agent metadata: it must go through JSON.

    Without persistence, a resumed run would restart without habits, and the block would lie by
    omission for the whole first day.
    """
    import json

    j = _journal(*([("work", "matin", "cycling", 0)] * 4))
    relu = json.loads(json.dumps(j))
    assert bloc_habitudes(relu) == bloc_habitudes(j)


def test_E1_le_bloc_parle_la_langue_du_prompt():
    """2026-09-25 — the titles and labels had stayed in French in an English prompt."""
    from llm import noyau

    j = {}
    for _ in range(3):
        noter_trajet(j, "work", "matin", "public_transport")
    texte = "\n".join(memoire_noyau(j, [], wall_clock(T0)))
    assert "My habits" in texte and "work in the morning: by public transport, 3 times out of 3" in texte
    for francais in noyau.TITRES_FRANCAIS.values():
        assert francais not in texte

"""The six translated columns of the journal, read back in both vocabularies.

The English switch changed the language of several columns of
`moves.csv` — those SERVED TO THE MODEL in the persona narrative. The scorer's mapping
tables followed for occupation and housing; the residence ring
was forgotten, and `normalize_place` merely replaced spaces with underscores.
`1st ring` became `1st_ring`, a key that `cerema_values.yaml` does not break down: over the
15 v6 runs scored on 2026-09-15, the "place of residence" dimension published
only one stratum out of
four. No row was missing, no alarm went off — the three rings were
counted "outside the reference data", that is, nowhere visible.

What these tests hold:

- **the symmetry of both vocabularies**. A v5 row and a v6 row describing the same
  persona must produce the SAME decision frame, over the six translated columns. A
  parametrised test is better than reading the diff: it fails the day a seventh
  column switches without its table;
- **the refusal to pass through**. A label no table knows leaves the dimension,
  is counted, and is reported as `[ALARME]` on its first occurrence. This is the property that
  would have revealed the defect the day of the switch instead of six weeks later;
- **the four rings**, on a complete residence frame.
"""

from __future__ import annotations

import csv
import logging

import pytest

from mobility_core.population_reference import (
    COURONNES,
    COURONNES_FR,
    OUT_OF_PERIMETER,
    OUT_OF_PERIMETER_FR,
)
from scripts.synthesis import frames

HEADERS = ["Mode de transport Choisi", "Méthode de sélection", "Type de logement",
           "Occupation principale", "Motifs de déplacement", "Genre", "Âge",
           "Distance parcourue", "Lieu de résidence", "ID Personne", "ID Activité",
           "Heure de départ", "Modes proposés au LLM", "Temps simulé"]

# The SAME persona, described in both journal vocabularies. The six columns that the
# English switch may have translated are included; `Genre`, `Mode de transport Choisi` and `Modes
# proposés au LLM` stayed French on both sides, and the test observes it rather than
# assuming it — this observation is what will make the switch visible if it happens.
PERSONA_V5 = {
    "Lieu de résidence": "2eme couronne",
    "Occupation principale": "Travail à plein temps",
    "Type de logement": "Individuel isolé",
    "Motifs de déplacement": "Travail",
    "Mode de transport Choisi": "Voiture Privée",
    "Modes proposés au LLM": "Voiture Privée | Marche",
}
PERSONA_V6 = {
    "Lieu de résidence": "2nd ring",
    "Occupation principale": "Full-time worker",
    "Type de logement": "Detached house",
    "Motifs de déplacement": "Travail",
    "Mode de transport Choisi": "Voiture Privée",
    "Modes proposés au LLM": "Voiture Privée | Marche",
}

# What the frame must carry, whatever the language read.
COLONNES_TRADUITES = ["lieu_residence", "occupation", "type_logement", "motif",
                      "chosen", "offered"]

EXCLURE_METHODES = ["Pas de déplacement (même localisation)",
                    "Pas de solution de déplacement", "LLM Error (Default index)"]


def _moves(tmp_path, lignes: list[dict], nom: str = "moves.csv"):
    """Writes a minimal `moves.csv`: one row per entry, everything else constant."""
    path = tmp_path / nom
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADERS)
        writer.writeheader()
        for i, ligne in enumerate(lignes):
            writer.writerow({
                "Méthode de sélection": "LLM",
                "Genre": "Homme", "Âge": "40", "Distance parcourue": "5.0",
                "ID Personne": str(i), "ID Activité": "a",
                "Heure de départ": "2024-01-08 08:00:00",
                "Temps simulé": "2024-01-08 08:00:00",
                **ligne,
            })
    return path


@pytest.fixture(autouse=True)
def _compteurs_neufs():
    """Each test starts from empty counters: the alarm fires on a rising edge."""
    frames.reinitialiser_compteurs_libelles()
    yield
    frames.reinitialiser_compteurs_libelles()


class TestTableDeCorrespondance:
    """`normalize_place` — both vocabularies, and the refusal of the unknown label."""

    @pytest.mark.parametrize(("libelle", "attendu"), [
        ("1st ring", "1ere_couronne"),
        ("2nd ring", "2eme_couronne"),
        ("3rd ring", "3eme_couronne"),
        ("Toulouse", "Toulouse"),
        ("1ere couronne", "1ere_couronne"),
        ("2eme couronne", "2eme_couronne"),
        ("3eme couronne", "3eme_couronne"),
    ])
    def test_les_deux_vocabulaires_donnent_la_cle_de_l_enquete(self, libelle, attendu):
        assert frames.normalize_place(libelle) == (attendu, True)

    @pytest.mark.parametrize("canonique", COURONNES)
    def test_toute_modalite_canonique_est_traduite(self, canonique):
        """Coverage comes from the population module, not from a list copied here."""
        cle, referencee = frames.normalize_place(canonique)
        assert referencee and cle is not None

    @pytest.mark.parametrize("ancienne", COURONNES_FR)
    def test_toute_modalite_d_avant_la_bascule_est_traduite(self, ancienne):
        cle, referencee = frames.normalize_place(ancienne)
        assert referencee and cle is not None

    def test_une_cle_deja_normalisee_est_son_propre_antecedent(self):
        """Reading back a page regenerated from a rewritten trace must lose nothing."""
        assert frames.normalize_place("1ere_couronne") == ("1ere_couronne", True)

    @pytest.mark.parametrize("libelle", [OUT_OF_PERIMETER, OUT_OF_PERIMETER_FR])
    def test_hors_perimetre_est_compte_jamais_joint(self, libelle):
        """No per-zone target: the key exists, the join does not."""
        assert frames.normalize_place(libelle) == (frames.OUT_OF_PERIMETER_KEY, False)

    def test_un_libelle_inconnu_ne_traverse_pas(self, caplog):
        with caplog.at_level(logging.ERROR, logger="synthesis.frames"):
            assert frames.normalize_place("Mars") == (None, False)
        assert frames.PLACES_INCONNUES["Mars"] == 1
        assert any("[ALARME]" in r.message and "Mars" in r.message
                   for r in caplog.records)

    def test_l_alarme_part_sur_front_montant(self, caplog):
        """Three thousand faulty rows do not make three thousand alarms — but a counter
        at 3,000. Flooding the log makes it unreadable the day it matters."""
        with caplog.at_level(logging.ERROR, logger="synthesis.frames"):
            for _ in range(5):
                frames.normalize_place("Mars")
        assert frames.PLACES_INCONNUES["Mars"] == 5
        assert sum("[ALARME]" in r.message for r in caplog.records) == 1

    def test_une_colonne_vide_n_est_pas_un_libelle_inconnu(self):
        """A population enriched before the residence ring writes the column empty: this is a
        normal case, it must neither raise an alarm nor count as a translation defect."""
        assert frames.normalize_place("") == (None, False)
        assert not frames.PLACES_INCONNUES

    def test_les_cles_fantomes_sont_exactement_les_formes_anglaises(self):
        """The signature `experiences.score` uses to mark a scores.json as stale."""
        assert frames.PLACE_CLES_FANTOMES == {"1st_ring", "2nd_ring", "3rd_ring"}


class TestSymetrieV5V6:
    """A v5 row and a v6 row of the same persona produce the same frame."""

    @pytest.mark.parametrize("colonne", COLONNES_TRADUITES)
    def test_la_meme_cle_des_deux_cotes(self, tmp_path, colonne):
        v5, _ = frames.read_moves(_moves(tmp_path, [PERSONA_V5], "v5.csv"),
                                  EXCLURE_METHODES)
        v6, _ = frames.read_moves(_moves(tmp_path, [PERSONA_V6], "v6.csv"),
                                  EXCLURE_METHODES)
        assert v5[0][colonne] == v6[0][colonne]

    def test_aucun_libelle_illisible_dans_les_deux_cohortes(self, tmp_path):
        """The corollary: the symmetry would also hold if BOTH were lost."""
        for nom, persona in (("v5.csv", PERSONA_V5), ("v6.csv", PERSONA_V6)):
            _, stats = frames.read_moves(_moves(tmp_path, [persona], nom),
                                         EXCLURE_METHODES)
            assert stats.get("lieu_residence_inconnu", 0) == 0
            assert stats.get("type_logement_inconnu", 0) == 0
            assert stats.get("occupation_inconnue", 0) == 0
            assert stats.get("motif_inconnu", 0) == 0

    def test_la_cohorte_v6_joint_bien_la_couronne(self, tmp_path):
        """The test that would have failed before the fix: `2nd ring` → `2eme_couronne`."""
        v6, _ = frames.read_moves(_moves(tmp_path, [PERSONA_V6], "v6.csv"),
                                  EXCLURE_METHODES)
        assert v6[0]["lieu_residence"] == "2eme_couronne"


class TestCompteursDeLecture:
    """Empty and unreadable are two different failures, and carry two counters."""

    def test_un_lieu_illisible_se_compte_a_part(self, tmp_path):
        path = _moves(tmp_path, [{**PERSONA_V6, "Lieu de résidence": "Mars"}])
        rows, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert stats["lieu_residence_inconnu"] == 1
        assert "lieu_residence_vide" not in stats
        assert rows[0]["lieu_residence"] is None

    def test_un_lieu_vide_reste_un_lieu_vide(self, tmp_path):
        path = _moves(tmp_path, [{**PERSONA_V6, "Lieu de résidence": ""}])
        _, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert stats["lieu_residence_vide"] == 1
        assert "lieu_residence_inconnu" not in stats

    def test_un_logement_illisible_se_compte_a_part(self, tmp_path):
        path = _moves(tmp_path, [{**PERSONA_V6, "Type de logement": "Yurt"}])
        _, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert stats["type_logement_inconnu"] == 1
        assert "type_logement_vide" not in stats

    @pytest.mark.parametrize("motif", ["home", "leisure", "other"])
    def test_les_motifs_sans_equivalent_ne_sont_pas_des_defauts(self, tmp_path, motif):
        """`home` has no EMC² target: it leaves the dimension without reporting anything."""
        path = _moves(tmp_path, [{**PERSONA_V6, "Motifs de déplacement": motif}])
        rows, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert rows[0]["motif"] is None
        assert "motif_inconnu" not in stats

    def test_un_motif_hors_table_se_dit(self, tmp_path):
        path = _moves(tmp_path, [{**PERSONA_V6, "Motifs de déplacement": "pilgrimage"}])
        _, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert stats["motif_inconnu"] == 1

    def test_une_occupation_illisible_se_dit(self, tmp_path):
        path = _moves(tmp_path, [{**PERSONA_V6, "Occupation principale": "Jedi"}])
        _, stats = frames.read_moves(path, EXCLURE_METHODES)
        assert stats["occupation_inconnue"] == 1
        assert frames.OCCUPATIONS_INCONNUES["Jedi"] == 1


class TestQuatreCouronnes:
    """The dimension publishes four strata, and no longer an "outside reference data" row."""

    def _detail(self, tmp_path, couronnes):
        # Five personas per ring: `dimension_detail` only declares "covered" beyond that.
        lignes = [{**PERSONA_V6, "Lieu de résidence": c}
                  for c in couronnes for _ in range(5)]
        rows, _ = frames.read_moves(_moves(tmp_path, lignes), EXCLURE_METHODES)
        cerema = frames.load_cerema(
            frames.REPO_ROOT / "scripts" / "data" / "population" / "cerema_values.yaml")
        dim = next(d for d in frames.DIMENSIONS if d["key"] == "lieu_residence")
        return frames.dimension_detail(
            frames.simulation_frames(rows)["attendu"], cerema, dim)

    def test_les_quatre_couronnes_sont_publiees(self, tmp_path):
        detail = self._detail(tmp_path, COURONNES)
        strates = {d["cat"]: d for d in detail if d["cat"] != frames.OFF_REFERENCE_ROW}
        assert set(strates) == {"Toulouse", "1ere_couronne", "2eme_couronne",
                                "3eme_couronne"}
        assert all(d["n"] == 5 and d["covered"] for d in strates.values())

    def test_plus_aucune_masse_hors_referentiel(self, tmp_path):
        """Before the fix, three rings out of four ended up in this row."""
        detail = self._detail(tmp_path, COURONNES)
        assert not [d for d in detail if d["cat"] == frames.OFF_REFERENCE_ROW]

    def test_hors_perimetre_reste_hors_referentiel(self, tmp_path):
        """The fix must not bring back into the strata what the residence ring took
        out of them: a home outside the 453 municipalities has no per-zone target."""
        detail = self._detail(tmp_path, [*COURONNES, OUT_OF_PERIMETER])
        hors = [d for d in detail if d["cat"] == frames.OFF_REFERENCE_ROW]
        assert len(hors) == 1
        assert set(hors[0]["categories"]) == {frames.OUT_OF_PERIMETER_KEY}

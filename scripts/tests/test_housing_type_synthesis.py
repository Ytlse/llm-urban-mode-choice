"""Reading of the housing type by the synthesis page, and exported distribution (action A2).

The trait crosses three separately written modules — population generation
sets it, the journal writes it, the page joins it to the reference — and the only thing that
holds them together is the modality table of `mobility_core.housing_type`. The tests
below check both ends of the chain on the page side:

- **the join**. The column carries the survey label, the reference indexes it by
  key: a failed mapping raises nothing, it empties the axis. And "Autres", known to
  the survey but absent from the published breakdown, must be counted outside the reference data
  rather than disappear;
- **the admission**. The page warning must say which of the two gaps it observes:
  a journal that does not write the column, or a run older than the action.

Plus, without the PROGEDO data, the properties of the exported distribution: hierarchical
smoothing and distributions that remain distributions.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mobility_core.housing_type import LABEL_BY_KEY, MODALITY_KEYS
from scripts.synthesis import build, frames

HEADERS = ["Mode de transport Choisi", "Méthode de sélection", "Type de logement",
           "Occupation principale", "Motifs de déplacement", "Genre", "Âge",
           "Distance parcourue", "Lieu de résidence", "ID Personne", "ID Activité",
           "Heure de départ", "Modes proposés au LLM"]


def _moves(tmp_path, logements: list[str]):
    path = tmp_path / "moves.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADERS)
        writer.writeheader()
        for i, logement in enumerate(logements):
            writer.writerow({
                "Mode de transport Choisi": "Voiture Privée",
                "Méthode de sélection": "LLM",
                "Type de logement": logement,
                "Occupation principale": "Travail à plein temps",
                "Motifs de déplacement": "Travail",
                "Genre": "Homme", "Âge": "40", "Distance parcourue": "5.0",
                "Lieu de résidence": "Toulouse", "ID Personne": str(i),
                "ID Activité": "a", "Heure de départ": "2024-01-08 08:00:00",
                "Modes proposés au LLM": "Voiture Privée | Marche",
            })
    return path


class TestNormalisation:

    @pytest.mark.parametrize("key", MODALITY_KEYS)
    def test_le_libelle_du_journal_retrouve_sa_cle(self, key):
        assert frames.normalize_housing(LABEL_BY_KEY[key])[0] == key

    def test_les_quatre_modalites_de_reference_sont_referencees(self):
        for key in ("individuel_isole", "individuel_accole",
                    "petit_habitat_collectif", "grand_habitat_collectif"):
            assert frames.normalize_housing(LABEL_BY_KEY[key])[1] is True

    def test_autres_est_une_cle_valide_mais_hors_reference(self):
        assert frames.normalize_housing("Autres") == ("autres", False)

    def test_une_cle_deja_normalisee_est_acceptée(self):
        """Reading back a journal written differently: the row is not lost."""
        assert frames.normalize_housing("petit_habitat_collectif") == (
            "petit_habitat_collectif", True)

    def test_vide_et_inconnu_ne_donnent_pas_de_categorie(self):
        assert frames.normalize_housing("") == (None, False)
        assert frames.normalize_housing("   ") == (None, False)
        assert frames.normalize_housing("Maison de ville") == (None, False)


class TestLectureDuJournal:

    def test_la_colonne_alimente_la_dimension(self, tmp_path):
        path = _moves(tmp_path, ["Grand habitat collectif"] * 3 + ["Individuel isolé"] * 2)
        rows, stats = frames.read_moves(path, [])
        assert [r["type_logement"] for r in rows].count("grand_habitat_collectif") == 3
        assert stats.get("type_logement_vide", 0) == 0

    def test_les_cellules_vides_sont_comptees(self, tmp_path):
        rows, stats = frames.read_moves(_moves(tmp_path, ["", "Individuel isolé", ""]), [])
        assert stats["type_logement_vide"] == 2
        assert [r["type_logement"] for r in rows] == [None, "individuel_isole", None]

    def test_autres_est_compte_hors_referentiel(self, tmp_path):
        _, stats = frames.read_moves(_moves(tmp_path, ["Autres", "Individuel isolé"]), [])
        assert stats["type_logement_hors_referentiel"] == 1
        assert stats.get("type_logement_vide", 0) == 0


class TestVentilation:

    @pytest.fixture(scope="class")
    def cerema(self) -> dict:
        from scripts.synthesis.sources import load_manifest
        path = load_manifest().path_of("cerema")
        if path is None or not path.exists():
            pytest.skip("Référence EMC² absente")
        return frames.load_cerema(path)

    def test_la_dimension_se_joint_a_la_reference(self, tmp_path, cerema):
        """The very end: labels written by the journal produce counts
        under the reference categories, and not an empty axis."""
        path = _moves(tmp_path, ["Individuel isolé"] * 4 + ["Grand habitat collectif"] * 6)
        rows, _ = frames.read_moves(path, [])
        dim = next(d for d in frames.DIMENSIONS if d["key"] == "type_logement")
        detail = {d["cat"]: d["n"]
                  for d in frames.dimension_detail(
                      frames.simulation_frames(rows)["attendu"], cerema, dim)}
        assert detail["individuel_isole"] == 4
        assert detail["grand_habitat_collectif"] == 6
        assert detail["petit_habitat_collectif"] == 0

    def test_autres_ne_pollue_pas_la_ventilation(self, tmp_path, cerema):
        """"Autres" does not become a stratum — but its mass is COUNTED.

        The first half was required from the start: the modality exists in the survey,
        the published EMC² breakdown ignores it, so it must neither receive a target nor
        weigh in an L1. The second was added with the residence ring: a mass excluded from the targets
        and invisible is confused with a non-existent mass. `dimension_detail` therefore returns
        an extra row, with no target nor L1 and never "covered", that carries this
        mass — the same move `global_view` makes with its mass outside the scored modes.
        """
        rows, _ = frames.read_moves(_moves(tmp_path, ["Autres"] * 5), [])
        dim = next(d for d in frames.DIMENSIONS if d["key"] == "type_logement")
        detail = frames.dimension_detail(
            frames.simulation_frames(rows)["attendu"], cerema, dim)

        strates = [d for d in detail if d["cat"] != frames.OFF_REFERENCE_ROW]
        assert {d["cat"] for d in strates} == set(
            cerema["parts_modales_2023"]["type_logement"])
        assert all(d["n"] == 0 for d in strates)

        hors = [d for d in detail if d["cat"] == frames.OFF_REFERENCE_ROW]
        assert len(hors) == 1, "the mass outside the reference data must be published, not diluted"
        assert hors[0]["excluded_mass"] > 0
        assert hors[0]["l1"] is None and hors[0]["covered"] is False


class TestAvertissementDeLaPage:
    """The admission must be exact: does the journal write the column, or is the run old?"""

    def _warnings(self, tmp_path, monkeypatch, logements: list[str]) -> list[str]:
        path = _moves(tmp_path, logements)
        run_dir = path.parent
        monkeypatch.setattr(frames, "resolve_run", lambda manifest: {
            "exists": True, "configured": str(run_dir), "path": str(run_dir),
            "run_id": run_dir.name,
            "moves": {"exists": True, "path": str(path), "mtime": "2026-07-29"},
            "population": {"exists": True},
        })
        monkeypatch.setattr(build, "REPO_ROOT", type(path)("/"))

        class _Manifest:
            def get(self, key, default=None):
                return default
        common, _ = build.build_common_set(_Manifest(), {})
        return common["warnings"]

    def test_colonne_vide_partout_impute_le_vide_a_l_anciennete_du_run(
            self, tmp_path, monkeypatch):
        warnings = self._warnings(tmp_path, monkeypatch, ["", "", ""])
        message = next(w for w in warnings if "type de logement" in w)
        assert "renseigne désormais" in message
        assert "avant" in message

    def test_colonne_partiellement_vide_designe_le_hors_couche(
            self, tmp_path, monkeypatch):
        warnings = self._warnings(tmp_path, monkeypatch,
                                  ["Individuel isolé", "", "Individuel accolé"])
        message = next(w for w in warnings if "type de logement" in w)
        assert message.startswith("1 trajets")
        assert "hors de la couche" in message

    def test_colonne_remplie_ne_produit_aucun_avertissement(self, tmp_path, monkeypatch):
        warnings = self._warnings(tmp_path, monkeypatch, ["Individuel isolé"] * 3)
        assert not [w for w in warnings if "type de logement" in w]

    def test_autres_est_signale(self, tmp_path, monkeypatch):
        warnings = self._warnings(tmp_path, monkeypatch,
                                  ["Autres", "Individuel isolé"])
        assert any("« Autres »" in w for w in warnings)


class TestLoiExportee:
    """Properties of the export, checked without the restricted-access data."""

    @pytest.fixture(scope="class")
    def export(self):
        return pytest.importorskip(
            "scripts.progedo_logit.export_housing_type",
            reason="l'export exige geopandas/sklearn (extra 'geo')")

    @pytest.fixture
    def households(self):
        """A sector (1001) with two zones, and a clear size gradient.

        Zone 100100000 is well surveyed and entirely in large collective housing; zone
        100199000 has a single respondent, in a detached house. One-person households
        are in collective housing, four-person ones in individual: this is the gradient the
        size lever must recover.
        """
        pd = pytest.importorskip("pandas")
        rows = []
        for index in range(100):
            size = 1 if index < 60 else 4
            housing = "grand_habitat_collectif" if size == 1 else "individuel_isole"
            rows.append({"ZF": "100100000", "housing": housing, "weight": 1.0,
                         "size": size, "bucket": size})
        rows.append({"ZF": "100199000", "housing": "individuel_isole", "weight": 1.0,
                     "size": 2, "bucket": 2})
        rows.append({"ZF": "100199000", "housing": "petit_habitat_collectif",
                     "weight": 1.0, "size": 3, "bucket": 3})
        return pd.DataFrame(rows)

    def test_les_lois_sont_des_distributions(self, export, households):
        table = export.build_table(households)
        assert table["modalities"] == list(MODALITY_KEYS)
        assert sum(table["global"]) == pytest.approx(1.0, abs=1e-3)
        for node in list(table["zones"].values()) + list(table["sectors"].values()):
            assert sum(node["shares"]) == pytest.approx(1.0, abs=1e-3)

    def test_la_ressource_est_versionnee_pour_le_module(self, export, households):
        """The module refuses a v1: the export must therefore announce v2, and serve the
        four levers — otherwise the produced resource would be unreadable for it."""
        from mobility_core.housing_type import MIN_RESOURCE_VERSION, SIZE_MAX
        table = export.build_table(households)
        assert table["version"] >= MIN_RESOURCE_VERSION
        assert sorted(table["size_leverage"]) == [
            str(size) for size in range(1, SIZE_MAX + 1)]

    def test_une_zone_mince_est_tiree_vers_son_secteur(self, export, households):
        """Without smoothing, a zone with 1 respondent would serve 100 % detached houses:
        sampling noise presented as geography."""
        table = export.build_table(households)
        index = MODALITY_KEYS.index("individuel_isole")
        thin = table["zones"]["100199000"]["shares"][index]
        assert thin < 0.45
        assert table["zones"]["100199000"]["n"] == 2

    def test_une_zone_bien_enquetee_garde_sa_loi(self, export, households):
        table = export.build_table(households)
        index = MODALITY_KEYS.index("grand_habitat_collectif")
        assert table["zones"]["100100000"]["shares"][index] > 0.5

    def test_le_lissage_est_une_combinaison_convexe(self, export):
        observed = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        prior = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        smoothed = export._smooth(observed, export.PRIOR_WEIGHT, prior)
        assert smoothed[0] == pytest.approx(0.5)
        assert smoothed[1] == pytest.approx(0.5)
        assert sum(smoothed) == pytest.approx(1.0)

    def test_l_effectif_enquete_est_publie_avec_la_loi(self, export, households):
        """No threshold hides anything: the reader sees what the distribution rests on."""
        table = export.build_table(households)
        assert table["zones"]["100100000"]["n"] == 100
        assert table["meta"]["n_households"] == 102


class TestLevierDeTailleExporte:
    """The block that the size lever adds to the resource, and its internal test."""

    @pytest.fixture(scope="class")
    def export(self):
        return pytest.importorskip(
            "scripts.progedo_logit.export_housing_type",
            reason="l'export exige geopandas/sklearn (extra 'geo')")

    @pytest.fixture
    def households(self):
        """Two zones identical in geography, opposite in household composition:
        the size lever is the ONLY thing that can tell them apart."""
        pd = pytest.importorskip("pandas")
        rows = []
        for zone in ("100100000", "100200000"):
            for index in range(200):
                size = 1 if index % 2 else 4
                # Single people in collective housing, families in individual housing.
                housing = ("grand_habitat_collectif" if size == 1
                           else "individuel_isole")
                rows.append({"ZF": zone, "housing": housing, "weight": 1.0,
                             "size": size, "bucket": size})
        return pd.DataFrame(rows)

    def test_le_levier_va_dans_le_sens_de_la_composition(self, export, households):
        table = export.build_table(households)
        isole = MODALITY_KEYS.index("individuel_isole")
        assert table["size_leverage"]["1"]["leverage"][isole] < 1.0
        assert table["size_leverage"]["4"]["leverage"][isole] > 1.0

    def test_les_effectifs_de_cellule_sont_ecrits(self, export, households):
        """"Cell counts written in the resource, as today for the
        zones; any cell under 30 weighted observations is flagged.\""""
        table = export.build_table(households)
        cells = {cell["modality"]: cell
                 for cell in table["size_leverage"]["1"]["cells"]}
        assert cells["grand_habitat_collectif"]["n"] == 200
        assert cells["grand_habitat_collectif"]["thin"] is False
        assert cells["individuel_isole"]["n"] == 0
        assert cells["individuel_isole"]["thin"] is True

    def test_le_test_interne_mesure_le_mecanisme_livre(self, export, households):
        """On a population where size explains EVERYTHING, the lever must bring
        the error down to zero where the zone distribution alone is badly wrong."""
        table = export.build_table(households)
        delivered = table["validation"]["delivered"]
        zone_seule = next(row for row in table["validation"]["baselines"]
                          if "ménages" in row["label"])
        assert delivered["mean_abs_error_pt"] < 0.5
        assert zone_seule["mean_abs_error_pt"] > 10.0
        assert table["validation"]["passes"] is True

    def test_le_test_interne_publie_les_20_cellules(self, export, households):
        from mobility_core.housing_type import SIZE_MAX
        table = export.build_table(households)
        cells = table["validation"]["delivered"]["cells"]
        # Two populated sizes in this set: 5 modalities each.
        assert len(cells) == 2 * len(MODALITY_KEYS)
        assert {cell["size"] for cell in cells} == {1, SIZE_MAX}
        assert all("observed_pct" in cell and "imputed_pct" in cell for cell in cells)

    def test_la_marginale_d_ensemble_n_est_pas_deplacee(self, export, households):
        """If the lever overrode the zone, the marginal would move — this is the lever's
        safeguard: "the raking must not shift the geography"."""
        table = export.build_table(households)
        delivered = table["validation"]["delivered"]
        for observed, imputed in zip(delivered["overall_marginal_observed_pct"],
                                     delivered["overall_marginal_imputed_pct"]):
            assert abs(observed - imputed) < 1.5

    def test_le_mecanisme_precedent_est_rejoue_quand_on_donne_les_personnes(
            self, export, households):
        """The before/after comparison lives in the resource: without it, "four times
        less error" would only be a sentence."""
        pd = pytest.importorskip("pandas")
        persons = pd.concat([households.assign(weight=households["size"])] * 1)
        table = export.build_table(households, persons)
        labels = [row["label"] for row in table["validation"]["baselines"]]
        assert any("pondération personnes" in label for label in labels)

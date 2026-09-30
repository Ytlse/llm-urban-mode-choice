"""A memory is reused only if it belongs to the same experiment.

Previously, a resume point carried no model, no prompt, no population and no shock, and
resuming followed the `experiments/current` link — which twice pointed on 2026-09-16 at a run
other than the one we thought.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from urban_mobility_agents.utils import identite_run as I

RACINE = Path(__file__).resolve().parents[1]


def _reglages(**surcharges):
    base = {
        "modele": "gemini-3.1-flash-lite",
        "instances": ["google_gemini31_key1", "google_gemini31_key2"],
        "variante": "prompt_expert_05",
        "population": "/data/population_5.json",
        "ltm": True,
        "reflexion": True,
        "graine_tirage": 42,
        "graine_ordre": 42,
        "graine_meteo": 42,
        "cache": False,
    }
    base.update(surcharges)
    return SimpleNamespace(
        llm=SimpleNamespace(
            instances_admises=base["instances"],
            providers={
                nom: SimpleNamespace(default_model=base["modele"]) for nom in base["instances"]
            },
        ),
        agent=SimpleNamespace(
            llm_params={"prompt_variant": base["variante"]},
            long_term_memory_enabled=base["ltm"],
            long_term_self_reflect_enabled=base["reflexion"],
            mode_draw_seed=base["graine_tirage"],
            option_order_seed=base["graine_ordre"],
            weather_draw_seed=base["graine_meteo"],
        ),
        data=SimpleNamespace(population_file=base["population"]),
        cache=SimpleNamespace(enabled=base["cache"]),
    )


class TestCeQueLIdentiteRetient:
    def test_B1_a_B7_les_champs_comparables_y_sont(self):
        ident = I.composer(_reglages(), empreinte_choc="abc123")
        assert ident["modeles_admis"] == ["gemini-3.1-flash-lite"]
        assert ident["instances_admises"] == ["google_gemini31_key1", "google_gemini31_key2"]
        assert ident["variante_prompt"] == "prompt_expert_05"
        assert ident["population"] == "/data/population_5.json"
        assert ident["memoire_longue"] is True and ident["auto_reflexion"] is True
        assert ident["choc"] == "abc123"
        assert (ident["graine_tirage"], ident["graine_ordre"], ident["graine_meteo"]) == (42, 42, 42)
        assert ident["cache_decisions"] is False

    def test_B5_sans_choc_le_champ_dit_aucun(self):
        assert I.composer(_reglages())["choc"] == "aucun"

    def test_les_instances_sont_triees(self):
        """Two lists with the same content in a different order are the SAME experiment."""
        a = I.composer(_reglages(instances=["b", "a"]))
        b = I.composer(_reglages(instances=["a", "b"]))
        assert I.differences(a, b) == []

    def test_B8_l_heure_et_les_compteurs_ne_font_jamais_refuser(self, tmp_path):
        ident = I.composer(_reglages())
        I.ecrire(tmp_path, {**ident, "ecrit_le": "hier", "compteurs": {"agents": 5}})
        I.verifier(tmp_path, {**ident, "ecrit_le": "aujourd'hui", "compteurs": {"agents": 9}})


class TestLeRefus:
    def test_A5_identites_identiques_la_reprise_a_lieu(self, tmp_path):
        ident = I.composer(_reglages())
        I.ecrire(tmp_path, ident)
        I.verifier(tmp_path, ident)  # does not raise

    def test_A6_une_difference_est_nommee(self, tmp_path):
        I.ecrire(tmp_path, I.composer(_reglages()))
        with pytest.raises(I.IdentiteIncompatible) as err:
            I.verifier(tmp_path, I.composer(_reglages(variante="prompt_minimal_02")))
        assert "variante de prompt" in str(err.value)
        assert "prompt_expert_05" in str(err.value) and "prompt_minimal_02" in str(err.value)

    def test_A7_toutes_les_differences_sont_nommees(self, tmp_path):
        """Fixing one field only to hit the next would cost one launch per difference."""
        I.ecrire(tmp_path, I.composer(_reglages()))
        with pytest.raises(I.IdentiteIncompatible) as err:
            I.verifier(
                tmp_path,
                I.composer(_reglages(variante="autre", modele="mistral-small", cache=True)),
            )
        msg = str(err.value)
        assert "variante de prompt" in msg and "modèle" in msg and "cache de décisions" in msg

    def test_un_modele_derive_des_instances_discrimine_vraiment(self):
        """An always-empty field would give the illusion of a check: this one varies."""
        a = I.composer(_reglages(modele="gemini-3.1-flash-lite"))
        b = I.composer(_reglages(modele="gemini-3.5-flash-lite"))
        assert a["modeles_admis"] != b["modeles_admis"]
        assert any("modèle" in d for d in I.differences(a, b))

    def test_un_choc_deplace_fait_refuser(self, tmp_path):
        """The case met on 2026-09-16: moving the shock days changes the experiment."""
        I.ecrire(tmp_path, I.composer(_reglages(), empreinte_choc="4c90634a380e"))
        with pytest.raises(I.IdentiteIncompatible) as err:
            I.verifier(tmp_path, I.composer(_reglages(), empreinte_choc="c6d1f2a09b77"))
        assert "choc" in str(err.value)

    def test_C4_une_identite_absente_fait_refuser(self, tmp_path):
        """Assuming equality in the absence of proof is exactly the fixed defect."""
        with pytest.raises(I.IdentiteIncompatible) as err:
            I.verifier(tmp_path, I.composer(_reglages()))
        assert "identity" in str(err.value)

    def test_C5_une_identite_illisible_fait_refuser(self, tmp_path):
        I.chemin(tmp_path).write_text("{ ceci n'est pas du json", encoding="utf-8")
        with pytest.raises(I.IdentiteIncompatible):
            I.verifier(tmp_path, I.composer(_reglages()))


class TestLEmpreinteDuChoc:
    """The shock registry exists only from GAMA's /init, after the identity is written:
    querying the registry returned "aucun" even under a declared shock (seen on 2026-09-17
    on run 2026-09-17_06_24)."""

    def _reglages_avec_choc(self, tmp_path, contenu="choc: c6\nlibelle: x\n"):
        f = tmp_path / "c6.yaml"
        f.write_text(contenu, encoding="utf-8")
        r = _reglages()
        r.chocs = SimpleNamespace(enabled=True, fichier=str(f))
        return r, f

    def test_un_choc_declare_donne_une_empreinte_pas_aucun(self, tmp_path):
        r, _ = self._reglages_avec_choc(tmp_path)
        assert I.composer(r)["choc"] not in ("aucun", "")

    def test_deux_chocs_differents_donnent_deux_identites_differentes(self, tmp_path):
        """Moving the days of a shock must make the resume be refused."""
        r1, f = self._reglages_avec_choc(tmp_path, "choc: c6\njours:\n  - jour: 8\n")
        e1 = I.composer(r1)
        f.write_text("choc: c6\njours:\n  - jour: 9\n", encoding="utf-8")
        assert I.composer(r1)["choc"] != e1["choc"]

    def test_sans_choc_le_champ_dit_aucun(self, tmp_path):
        r = _reglages()
        r.chocs = SimpleNamespace(enabled=False, fichier=None)
        assert I.composer(r)["choc"] == "aucun"

    def test_un_choc_declare_mais_introuvable_ne_dit_pas_aucun(self, tmp_path):
        """Otherwise an experiment under shock and one without would pass for the same."""
        r = _reglages()
        r.chocs = SimpleNamespace(enabled=True, fichier=str(tmp_path / "absent.yaml"))
        assert I.composer(r)["choc"] != "aucun"

    # ── The `evenements` key (2026-09-22) ─────────────────────────────────────────────────
    # `_empreinte_du_choc` read ONLY `reglages.chocs`. Since the event channel, `make run`
    # writes the `evenements:` key in config.yaml — the canonical form — and the fingerprint returned
    # "aucun" ON A TREATED ARM. Its identity became that of its control, which the
    # function's docstring gives as the thing not to do. Found on campaign c3,
    # twenty minutes after its launch.

    def test_la_cle_evenements_du_ticket_100_donne_une_empreinte(self, tmp_path):
        f = tmp_path / "c3.yaml"
        f.write_text("evenement: c3\njours:\n  - jour: 12\n", encoding="utf-8")
        r = _reglages()
        r.chocs = SimpleNamespace(enabled=False, fichier=None)
        r.evenements = SimpleNamespace(enabled=True, fichier=str(f))
        assert I.composer(r)["choc"] not in ("aucun", "")

    def test_un_bras_traite_ne_porte_jamais_la_meme_identite_que_son_temoin(self, tmp_path):
        """The case that slipped through: the attribution campaign, two paired arms."""
        f = tmp_path / "c3.yaml"
        f.write_text("evenement: c3\njours:\n  - jour: 12\n", encoding="utf-8")

        traite = _reglages()
        traite.chocs = SimpleNamespace(enabled=False, fichier=None)
        traite.evenements = SimpleNamespace(enabled=True, fichier=str(f))

        temoin = _reglages()
        temoin.chocs = SimpleNamespace(enabled=False, fichier=None)
        temoin.evenements = SimpleNamespace(enabled=False, fichier=None)

        assert I.composer(temoin)["choc"] == "aucun"
        assert I.composer(traite)["choc"] != I.composer(temoin)["choc"]

    def test_la_cle_evenements_prime_sur_la_cle_chocs(self, tmp_path):
        """Same order as `_declaration_demandee()`: what is PLAYED is what is recorded."""
        neuf, vieux = tmp_path / "neuf.yaml", tmp_path / "vieux.yaml"
        neuf.write_text("evenement: c3\njours:\n  - jour: 12\n", encoding="utf-8")
        vieux.write_text("choc: c6\njours:\n  - jour: 8\n", encoding="utf-8")

        r = _reglages()
        r.evenements = SimpleNamespace(enabled=True, fichier=str(neuf))
        r.chocs = SimpleNamespace(enabled=True, fichier=str(vieux))
        empreinte_des_deux = I.composer(r)["choc"]

        seul_le_neuf = _reglages()
        seul_le_neuf.evenements = SimpleNamespace(enabled=True, fichier=str(neuf))
        seul_le_neuf.chocs = SimpleNamespace(enabled=False, fichier=None)
        assert empreinte_des_deux == I.composer(seul_le_neuf)["choc"]

    def test_l_ancienne_cle_reste_lue_quand_elle_est_seule(self, tmp_path):
        """A campaign launched with the old `chocs` key must keep a valid identity."""
        f = tmp_path / "c6.yaml"
        f.write_text("choc: c6\njours:\n  - jour: 8\n", encoding="utf-8")
        r = _reglages()
        r.evenements = SimpleNamespace(enabled=False, fichier=None)
        r.chocs = SimpleNamespace(enabled=True, fichier=str(f))
        assert I.composer(r)["choc"] not in ("aucun", "")


class TestEcriture:
    def test_C1_l_identite_est_ecrite(self, tmp_path):
        assert I.ecrire(tmp_path, I.composer(_reglages())) is True
        assert json.loads(I.chemin(tmp_path).read_text())["variante_prompt"] == "prompt_expert_05"

    def test_C3_une_reprise_ne_reecrit_pas_l_identite(self, tmp_path):
        """It is the reference we compare against: rewriting it would erase
        the difference we are trying to detect."""
        I.ecrire(tmp_path, I.composer(_reglages()))
        assert I.ecrire(tmp_path, I.composer(_reglages(variante="autre"))) is False
        assert json.loads(I.chemin(tmp_path).read_text())["variante_prompt"] == "prompt_expert_05"


class TestLaChaineEstBranchee:
    def _src(self, rel: str) -> str:
        return (RACINE / rel).read_text(encoding="utf-8")

    def test_A1_sans_reprise_nommee_rien_n_est_restaure(self):
        src = self._src("handle/application.py")
        assert "REPRISE_RUN" in src, "the resume authorisation must be read at startup"
        assert "identite_run" in src, "the identity must be checked before any restore"

    def test_A3_le_workdir_est_resolu_par_le_nom(self):
        src = self._src("settings.py")
        assert "REPRISE_RUN" in src, (
            "le répertoire du run repris doit être résolu par son NOM, pas par le lien "
            "`current` — qui a pointé deux fois sur un autre run le 2026-09-16"
        )

    def test_C2_l_identite_est_recopiee_dans_le_point_de_reprise(self):
        assert "identite_run" in self._src("urban_mobility_agents/utils/reprise.py")

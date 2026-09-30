"""CLI of the experiment platform (ticket 035) — `python -m experiences <commande>`.

    preparer-jeu    --population P --nom N [--jour YYYY-MM-DD] [--concurrence 8] [--seuil 0.05]
    consulter-jeu   --nom N [--personne ID]
    verifier-jeu    --nom N
    verifier-jours  --nom N --jour YYYY-MM-DD [--methode gtfs|moteurs] [--declarer]   that day's PT offer? (read from the GTFS, or measured)
    definir         EXPERIENCE.yaml [--accepter-nom]   validates, checks the computed name, stores in data/experiences/<nom>/
    dupliquer       --de NOM [--vers NOM] [--decideur-type T] [--decideur-modele M] [--artefact CHEMIN]
    estimer         --experience NOM
    lancer          --experience NOM [--reprendre] [--accepter-perime] [--attendre-fenetre]
    pause | arreter --experience NOM           (PAUSE / STOP file in the last run)
    file                                       experiments waiting for a key (FIFO)
    actives         [--est-vide]               experiments holding a key (code 0 if none)
    defiler         --experience NOM           removes an experiment from the queue (R2e)
    reconcilier                                frees the keys of finished/ghost runs (INSIDE the container)
    ordonnancer     [--intervalle S]           host loop: reconciles and promotes the queue
    registre        [--trier CHAMP] [--decroissant] [--filtrer champ=valeur …]
    comparer        EXEC_A EXEC_B
    synthese        EXEC
    erreurs         [EXEC | --experience NOM]  matches decisions.jsonl against the set (missing, gaps, attempts)
    journal         [EXEC] --verifier [--toutes] | --regenerer   does moves.csv cover the decisions? (R24)

In the container: `docker compose exec controller python -m experiences …` (the root `make`
targets wrap these calls).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from experiences import experience as E
from experiences import jeu as J
from experiences import nommage as NOM
from experiences.archive import ETAT_EN_COURS, ETAT_TERMINEE, Execution
from experiences.population import charger_population, info_population
from experiences.runner import FICHIER_PAUSE, FICHIER_STOP, executer


def motif_non_reprenable(execution) -> str | None:
    """Why this run cannot be resumed — None if it can.

    Any run whose archive is not closed is resumable: paused, exhausted
    (the runner marks it WITHOUT sealing, cf. R4), interrupted (dead process), or running
    without a runner. Until 2026-09-08 the CLI refused every state of `ETATS_FINAUX`, exhausted
    included — even though its own message, the spec and the dashboard's "Reprendre" button
    called it resumable: a run on Muse Glimmer, wrongly declared exhausted
    (busy instance read as out of service), could only restart through "Rejouer".
    """
    etat = (execution.etat() or {}).get("etat")
    if execution.cloturee or etat == ETAT_TERMINEE:
        return (
            f"la dernière exécution {execution.nom} est clôturée ({etat}) : immuable (E19), "
            "« Rejouer » en crée une nouvelle"
        )
    return None


def reglages_herites_de(exp) -> dict:
    """The process settings IMPOSED on the run, recorded in its trace.

    `vehicule_chaine` and `verrou_retour` are part of them (R13, ticket 045). Without them, two
    runs of the chain 2×2 — the same definition but for one flag — would have carried the
    same trace, and nothing would have said which was which. That is the defect this ticket
    fixes elsewhere: a setting that changes the measurement without entering its identity.

    The values come from the DEFINITION, not from the environment: the definition is
    authoritative, and `appliquer_reglages_chaine` then pushes it into the settings.
    """
    from settings import settings

    return {
        "agenda_anticipation_enabled": settings.agent.agenda_anticipation_enabled,
        "max_trip_candidates": settings.gtfs.max_trip_candidates,
        "vehicule_chaine": bool(getattr(exp, "vehicule_chaine", True)),
        "verrou_retour": bool(getattr(exp, "verrou_retour", True)),
        "troncature_15": bool(getattr(exp, "troncature_15", False)),
        "mode_choice_truncation_threshold": settings.agent.mode_choice_truncation_threshold,
    }


def appliquer_reglages_chaine(exp) -> None:
    """Pushes the definition's chain switches into the process settings.

    The definition commands, the environment follows — the reverse of what used to happen,
    where the environment decided and the definition ignored it.
    """
    from settings import settings

    settings.agent.vehicle_chain_enabled = bool(getattr(exp, "vehicule_chaine", True))
    settings.agent.vehicle_return_home_lock = bool(getattr(exp, "verrou_retour", True))
    if not (
        settings.agent.vehicle_chain_enabled and settings.agent.vehicle_return_home_lock
    ):
        logger.warning(
            f"[execution] vehicle chain REDUCED by the definition: "
            f"vehicle position {'active' if settings.agent.vehicle_chain_enabled else 'COUPÉE'}, "
            f"return lock {'actif' if settings.agent.vehicle_return_home_lock else 'COUPÉ'}. "
            f"This measurement is not comparable term by term with one taken with the chain active."
        )


def _derniere_execution(exp: E.Experience) -> Path | None:
    d = exp.dossier() / "executions"
    if not d.is_dir():
        return None
    execs = sorted(p for p in d.iterdir() if (p / "execution.yaml").is_file())
    return execs[-1] if execs else None


def _charger_experience_par_nom(nom: str) -> E.Experience:
    chemin = Path(nom)
    if chemin.suffix in (".yaml", ".yml") and chemin.is_file():
        return E.charger_experience(chemin)
    dossier = E.trouver_dossier_experience(nom)
    if dossier and (dossier / "experience.yaml").is_file():
        return E.charger_experience(dossier / "experience.yaml")
    return E.charger_experience(E.dossier_experiences() / nom / "experience.yaml")


# ── set ──────────────────────────────────────────────────────────────────────


def _trip_helper_reel():
    """The simulation's trip helper (OTP + OSMnx, caches included) — services required."""
    from urban_mobility_agents.factory.factory import init_static_data

    return init_static_data().trip_helper


def cmd_preparer_jeu(a: argparse.Namespace) -> int:
    personnes, info = charger_population(a.population)
    dossier = Path(a.dossier) if a.dossier else E.dossier_jeux() / a.nom
    prep = J.JeuEnPreparation.ouvrir(dossier, a.nom, info, a.jour)
    trip_helper = _trip_helper_reel()
    compteurs = asyncio.run(
        J.preparer(
            prep,
            personnes,
            trip_helper,
            concurrence=a.concurrence,
            seuil_sans_proposition=a.seuil,
        )
    )
    if compteurs.get("erreurs"):
        prep.fermer()
        logger.error(
            f"[jeu] {compteurs['erreurs']} engine errors: the set is NOT closed — relaunch to resume (J11)"
        )
        return 2
    prep.clore(J.deplacements_attendus(personnes, a.jour), len(personnes))
    print(J.Jeu.charger(dossier).resume())
    return 0


def cmd_consulter_jeu(a: argparse.Namespace) -> int:
    jeu = J.Jeu.charger(
        Path(a.dossier) if a.dossier else E.dossier_jeux() / a.nom,
        verifier=not a.sans_verification,
    )
    print(jeu.resume())
    if a.personne:
        lignes = jeu.consulter(a.personne)
        if not lignes:
            print(f"no trip recorded for {a.personne!r}")
            return 1
        for l in lignes:
            h = f"{l.depart_24h // 3600:02d}:{(l.depart_24h % 3600) // 60:02d}"
            print(
                f"\n{h} → {l.purpose} (activity {l.activity_id}) — {len(l.propositions)} proposal(s)"
                + (f" — {l.motif_absence}" if l.motif_absence else "")
            )
            for i, p in enumerate(l.vers_propositions()):
                print(
                    f"   {i:2d}. {p.mode:<28} {((p.plan.duration or 0) // 60):4d} min  [{p.source}]"
                )
    return 0


def cmd_verifier_jeu(a: argparse.Namespace) -> int:
    jeu = J.Jeu.charger(Path(a.dossier) if a.dossier else E.dossier_jeux() / a.nom)
    print(jeu.resume())
    differentes, non_verif = J.perime(jeu)
    if differentes:
        print("STALE — dependencies changed: " + ", ".join(differentes))
    if non_verif:
        print("not verifiable: " + ", ".join(non_verif))
    if not differentes:
        print(
            "dependencies: unchanged"
            if not non_verif
            else "verifiable dependencies: unchanged"
        )
    return 1 if differentes else 0


def cmd_config_empreintes(a: argparse.Namespace) -> int:
    """Rebuilds the config values registry from the git history (host only).

    `--verifier` writes nothing: exit code 1 if the committed registry no longer matches git.
    """
    config = Path(a.dossier_config) if a.dossier_config else J.dossier_config_defaut()
    debut = time.monotonic()
    registre = J.construire_registre_valeurs(J.versions_git_config(config))
    chemin = config / J.FICHIER_REGISTRE_VALEURS
    compte = ", ".join(
        f"{nom}: {len(t)} version(s)" for nom, t in registre["fichiers"].items()
    )
    if a.verifier:
        actuel = (
            yaml.safe_load(chemin.read_text(encoding="utf-8"))
            if chemin.is_file()
            else None
        )
        if actuel != registre:
            logger.error(
                f"[ALARME] [jeu] {chemin.name} does not match the git history ({compte}) "
                "→ make config-empreintes"
            )
            return 1
        logger.info(f"[jeu] {chemin.name} matches the git history ({compte})")
        return 0
    entete = (
        "# Config values registry — generated by `make config-empreintes`, do not edit by hand.\n"
        "# For each version of a config file: sha256 of its bytes → sha256 of its YAML data.\n"
        "# Lets a frozen set stay valid when only comments changed (experiences/jeu.py, J10).\n"
    )
    chemin.write_text(
        entete + yaml.safe_dump(registre, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info(
        f"[jeu] {chemin.name} written ({compte}) in {time.monotonic() - debut:.1f} s"
    )
    return 0


def cmd_verifier_jours(a: argparse.Namespace) -> int:
    """Is another day's PT offer the same as the set's?

    `gtfs` (default, no service required): does each proposed trip exist that day in the
    feeds? Validity declared PER TRIP. `moteurs`: OTP queried on a sample,
    whole-day equivalence declared only if 100 % identical.
    """
    jeu = J.Jeu.charger(Path(a.dossier) if a.dossier else E.dossier_jeux() / a.nom)
    if a.methode == "gtfs":
        from experiences.offre_jour import verifier_offre_jour_gtfs

        res = verifier_offre_jour_gtfs(
            jeu,
            a.jour,
            Path(a.gtfs) if a.gtfs else None,
            fenetre_s=int(a.fenetre_h * 3600),
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in res.items()
                    if k not in ("invalides_detail", "invalides_cles")
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        for d in res["invalides_detail"][:5]:
            print(
                f"  ≠ {d['cle']} (departure {d['depart_24h'] // 3600:02d}:{(d['depart_24h'] % 3600) // 60:02d}): "
                f"{d['evenements_dans_la_fenetre']} different departure(s) in its window"
            )
        if a.declarer:
            jeu.declarer_verification_jour(a.jour, res)
            print(
                f"declared in {jeu.dossier / J.FICHIER_EQUIVALENCES}: {res['valides']} trips served from the set on {a.jour}, "
                f"{res['invalides']} recomputed (PT)"
            )
        return 0 if res["equivalent"] else 1
    trip_helper = _trip_helper_reel()
    mesure = asyncio.run(
        J.comparer_offre_jour(
            jeu, trip_helper, a.jour, echantillon=a.echantillon, graine=a.graine
        )
    )
    print(
        json.dumps(
            {k: v for k, v in mesure.items() if k != "differences"},
            ensure_ascii=False,
            indent=1,
        )
    )
    for d in mesure["differences"][:5]:
        print(
            f"  ≠ {d['cle']}: {len(d['avant'])} PT recorded vs {len(d['apres'])} on {a.jour}"
        )
    if mesure["equivalent"]:
        print(
            f"PT offer of {a.jour} IDENTICAL to that of {jeu.jour_simule} over {mesure['compares']} trips"
        )
        if a.declarer:
            jeu.declarer_jour_equivalent(
                a.jour, {k: v for k, v in mesure.items() if k != "differences"}
            )
            print(f"declared in {jeu.dossier / J.FICHIER_EQUIVALENCES}")
        return 0
    print(
        f"PT offer of {a.jour} DIFFERENT: {mesure['differents']} trip(s) out of {mesure['compares']} — "
        f"no equivalence declared; the simulation will recompute PT that day, the simulator-free mode refuses this date"
    )
    return 1


# ── experiment ───────────────────────────────────────────────────────────────


def cmd_definir(a: argparse.Namespace) -> int:
    exp = E.charger_experience(a.fichier)
    # N13 — the name is computed from the parameters. A hand-written definition can
    # name something other than what it does: that is exactly what computed naming
    # removes, and a misnamed directory is later paid for in archive confusion.
    attendu = NOM.verifier_nom(E.experience_vers_dict(exp))
    if attendu and not a.accepter_nom:
        print(
            f"name refused: {exp.nom!r} does not state its parameters — expected {attendu!r}\n"
            f"→ rename the file's `nom`, or relaunch with --accepter-nom "
            f"(spec nommage-canonique-experiences, N13)",
            file=sys.stderr,
        )
        return 2
    chemin = E.sauver_experience(exp)
    print(f"experiment {exp.nom!r} validated and stored: {chemin}")
    if attendu:
        print(
            f"⚠ name outside the convention (expected {attendu!r}), explicitly accepted"
        )
    return 0


def cmd_dupliquer(a: argparse.Namespace) -> int:
    exp = _charger_experience_par_nom(a.de)
    changements = {}
    if a.decideur_type or a.decideur_modele or a.artefact:
        dec = exp.decideur.model_dump()
        if a.decideur_type:
            dec["type"] = a.decideur_type
        if a.decideur_modele:
            dec["modele"] = a.decideur_modele
        if a.artefact:
            # Two families go through the `modele` decision-maker (booster, logit): without this
            # option, switching between them meant editing experience.yaml by hand.
            dec["artefact"] = a.artefact
        changements["decideur"] = dec
    nouvelle = E.dupliquer(exp, a.vers or None, **changements)
    if not a.vers:
        # Computed name (N1): the collision index is settled on disk (N10). A copy
        # that changes NO named parameter falls back on an existing experiment — duplicating
        # it would overwrite it instead of opening a variant.
        attribution = NOM.attribuer_nom(
            E.experience_vers_dict(nouvelle), E.dossier_experiences()
        )
        if attribution.reutilise:
            print(
                f"nothing to duplicate: these parameters are already those of "
                f"{attribution.reutilise!r} — change a parameter, or relaunch this "
                f"experiment to add a run to it",
                file=sys.stderr,
            )
            return 2
        nouvelle.nom = attribution.nom
        if attribution.indice > 1:
            print(
                f"⚠ {attribution.base!r} is already taken by another definition: "
                f"index added (N10)"
            )
    print(
        f"experiment {nouvelle.nom!r} created from {exp.nom!r}: {E.sauver_experience(nouvelle)}"
    )
    return 0


def _jeu_de_cles_experience(exp: E.Experience, moniteur) -> set[str]:
    """API key identities the experiment may call on (R4/R8/R9). Empty outside the gateway."""
    if exp.decideur.type != "passerelle":
        return set()
    from experiences.cles import jeu_de_cles

    providers = getattr(moniteur, "providers", {}) or {}
    admises = None
    if moniteur is not None:
        admises = (
            moniteur.instances_disponibles()
            if moniteur.joignable
            else moniteur.instances
        )
    return jeu_de_cles(
        exp.decideur.modele or "",
        providers,
        instances_admises=admises,
        portee=exp.decideur.portee,
    )


def _annuler_execution_creee(
    exp: E.Experience, execution, dossier_exp: Path, *, creee: bool
) -> None:
    """Deletes a freshly created but never started run (sent to the queue). A resume
    (pre-existing run) is never destroyed."""
    if not creee:
        return
    import shutil

    shutil.rmtree(execution.dossier, ignore_errors=True)
    exp.executions_connues = [n for n in exp.executions_connues if n != execution.nom]
    E.sauver_experience(exp, dossier_exp)


def _preparer_lancement(exp: E.Experience, *, accepter_perime: bool):
    jeu = J.Jeu.charger(exp.jeu.chemin())
    info = info_population(exp.population.chemin)
    from experiences.ressources import (
        MoniteurRessources,
        charger_providers,
        instances_pour_modele,
    )

    moniteur = None
    # Ticket 085, lot C — TWO sets, two names. A single variable first held "the
    # instances that serve this model" then, overwritten, "those that still have quota": the
    # refusal read the second and reported the first, so that a quota not yet renewed
    # was reported as "no instance serves this model" — right after listing this model among
    # the served models (2026-09-16).
    servantes = None
    disponibles = None
    if exp.decideur.type == "passerelle":
        providers = charger_providers()
        servantes = instances_pour_modele(
            exp.decideur.modele or "", providers, exp.decideur.portee
        )
        moniteur = MoniteurRessources(servantes, providers)
        moniteur.reinitialiser_quotas(servantes)
        moniteur.rafraichir()
        # At launch, we do not block on leftover local counters or locks:
        # we test the first key for real right away, then the second on a 429.
        disponibles = list(servantes) if servantes else []
    refus, avert = E.refuser_si_impossible(
        exp,
        jeu,
        info,
        instances_disponibles=disponibles,
        instances_servantes=servantes,
        detail_epuisement=(moniteur.raison_epuisement() if moniteur else None),
        perime_accepte=accepter_perime,
    )
    for w in avert:
        logger.warning(f"[experience] {w}")
    return jeu, info, moniteur, refus


def _ecrire_marqueur_refus(nom_exp: str, motifs: list[str]) -> None:
    """Drops, next to the launch log, what a refusal means for the caller.

    A campaign launches in the background: it reads neither standard output nor the return code,
    and so only sees a missing `etat.json`. Without this marker, it concludes the experiment is
    broken where there is only a quota window to wait for — that is what cost four arms
    on 2026-09-16. Best-effort: an unwritten marker must never prevent a refusal from being
    stated; the refusal itself stays on the output and in the log.
    """
    from experiences import refus as REF

    try:
        base = E.dossier_lancements(nom_exp, creer=True)
        classe = REF.classer(motifs)
        chemin = base / (
            f"{datetime.now(timezone.utc):%Y-%m-%d_%H_%M_%S}{REF.SUFFIXE_MARQUEUR}"
        )
        chemin.write_text(
            json.dumps(
                {
                    "classe": classe,
                    "reportable": REF.est_reportable(classe),
                    "motifs": list(motifs),
                    "le": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S+00:00}",
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001 — never at the cost of the refusal itself
        logger.warning(f"[lancer] refusal marker not written ({type(e).__name__}: {e})")


def _aptitude(exp, est: dict) -> tuple[list[str], list[str]]:
    """(refusals, warnings) on the model's fitness for the load. Best-effort: an unavailable
    diagnosis must never prevent a legitimate launch."""
    if exp.decideur.type != "passerelle":
        return [], []
    try:
        from experiences import aptitude as APT
        from experiences.ressources import charger_providers, instances_pour_modele

        providers = charger_providers()
        instances = instances_pour_modele(
            exp.decideur.modele or "", providers, exp.decideur.portee
        )
        return APT.verifier_depuis_estimation(exp, est, providers, instances)
    except Exception as e:  # noqa: BLE001
        return [], [f"aptitude non vérifiée ({type(e).__name__}: {e})"]


def cmd_estimer(a: argparse.Namespace) -> int:
    exp = _charger_experience_par_nom(a.experience)
    jeu, _info, moniteur, refus = _preparer_lancement(exp, accepter_perime=True)
    est = E.estimer(exp, jeu, moniteur=moniteur)
    print(json.dumps(est, ensure_ascii=False, indent=1, default=str))

    # P3 — the estimate becomes a verdict: what cannot succeed is refused
    # up front rather than discovered three hours in.
    refus_apt, avert_apt = _aptitude(exp, est)
    if avert_apt:
        print("\nWARNINGS:\n- " + "\n- ".join(avert_apt))
    refus = list(refus) + refus_apt
    if refus:
        print("\nREFUSED at launch:\n- " + "\n- ".join(refus))
        return 1
    return 0


def appliquer_fenetre_age(exp) -> int:
    """Sets the recall age window to the experiment's horizon (ticket 071, § 2.7).

    Until now `long_term_max_days_query` was hard-coded to 30 days, whatever the declared
    horizon: a sixty-day experiment lost its second month all at once, without
    any log line saying so. Nothing must filter by age WITHIN
    a run — temporal decay is enough to silence an old memory.

    Applied EVEN when memory is off: an inert value is better than a
    wrong value if memory is switched back on later in the process.

    Returns the chosen window, in days, so that the caller can trace it.
    """
    from settings import settings

    fenetre = settings.agent.fenetre_age_pour_horizon(exp.horizon_jours)
    settings.agent.long_term_max_days_query = fenetre
    print(
        f"mémoire : {'active' if exp.memoire else 'coupée'} — fenêtre d'âge au rappel "
        f"{fenetre} j (horizon {exp.horizon_jours} j, plafond "
        f"{settings.agent.memoire__fenetre_age_max_jours} j)"
    )
    return fenetre


def appliquer_instances_admises(exp, moniteur) -> list[str]:
    """Sets the routing restriction from the experiment's MODEL (ticket 085, § 5).

    The ticket 084 restriction lived in `services/llm-agents/config/config.yaml`, a file
    **global to the stack**, read by every process launched from the `controller` container. An
    experiment on gemini 3.5, which pins its own instance, thus carried both constraints at
    once; the router rightly refused two contradictory constraints. It was a scope
    defect, not a rule defect: a PER-RUN choice has no business in a shared file.

    The list is derived from the model rather than copied by hand. That is the right invariant: it
    stays true when a key is added to the providers, and it is consistent by construction
    with the pinned instance, which serves that same model. It is NOT filtered by availability
    — a key momentarily at its cap stays admitted; arbitrating the quota is the monitor's
    job, not the restriction's.

    Returns the chosen list, so that the caller can record it.
    """
    from experiences.ressources import instances_pour_modele
    from settings import settings

    ancienne = list(settings.llm.instances_admises or [])
    if exp.decideur.type != "passerelle":
        # Including when the file carried one: the restriction follows the experiment in both
        # directions. A local, random or replay decision-maker calls on no instance.
        retenue: list[str] = []
    else:
        retenue = instances_pour_modele(
            exp.decideur.modele or "",
            getattr(moniteur, "providers", {}) or {},
            exp.decideur.portee,
        )
    if ancienne and set(ancienne) != set(retenue):
        # A SILENT overwrite of a scientific constraint is the defect being fixed here, not an
        # acceptable way to fix it. The file stays in place for `make run`, which has no
        # per-run injection (§ 3 of the ticket) — but its overwrite shows in the log.
        logger.warning(
            f"[execution] instances_admises du fichier ÉCRASÉE par l'expérience : "
            f"{ancienne} → {retenue or 'aucune restriction'}. Le fichier "
            f"`config/config.yaml` reste la source du seul chemin `make run`."
        )
    settings.llm.instances_admises = retenue
    print(
        f"admitted instances: {', '.join(retenue) if retenue else 'aucune restriction'} "
        f"(derived from model {exp.decideur.modele or '—'!r}"
        f"{f', portée {exp.decideur.portee}' if exp.decideur.portee else ''})"
    )
    return retenue


def cmd_lancer(a: argparse.Namespace) -> int:
    exp = _charger_experience_par_nom(a.experience)
    jeu, info, moniteur, refus = _preparer_lancement(
        exp, accepter_perime=a.accepter_perime
    )
    # P3 — the model's fitness for the load is checked BEFORE creating the run: an
    # experiment that cannot succeed must not leave a directory to archive behind it.
    refus_apt, avert_apt = _aptitude(exp, E.estimer(exp, jeu, moniteur=moniteur))
    for m in avert_apt:
        print(f"WARNING: {m}")
    refus = list(refus) + ([] if a.ignorer_aptitude else refus_apt)
    if a.ignorer_aptitude and refus_apt:
        for m in refus_apt:
            print(f"WARNING (fitness ignored): {m}")
    if refus:
        print("REFUSED — no run created:\n- " + "\n- ".join(refus))
        _ecrire_marqueur_refus(exp.nom, refus)
        return 1
    if exp.mode != E.MODE_SANS_SIMULATEUR:
        print(
            "simulator mode: launch with `make run JEU=<nom>` (spec 04) — the platform does not drive GAMA here"
        )
        return 1

    from settings import settings

    # The decision-maker and the template are the experiment's: settings imposed on the process.
    settings.agent.long_term_memory_enabled = bool(exp.memoire)
    appliquer_fenetre_age(exp)
    settings.cache.enabled = False  # every non-archived decision is requested (RG-2)
    # The routing restriction BELONGS TO THIS RUN, not to the stack (ticket 085, lot B).
    admises = appliquer_instances_admises(exp, moniteur)
    # The definition commands the vehicle chain, not the environment (R13).
    appliquer_reglages_chaine(exp)
    # The definition commands the Consideration Set truncation (0.15 if enabled, 0.0 otherwise).
    settings.agent.mode_choice_truncation_threshold = (
        0.15 if bool(getattr(exp, "troncature_15", False)) else 0.0
    )
    settings.agent.mode_draw_seed = exp.graine_tirage
    settings.agent.option_order_seed = exp.graine_ordre
    settings.agent.weather_per_agent_dates = exp.calendrier.politique != "commune"
    settings.agent.weather_draw_seed = exp.calendrier.graine
    if exp.decideur.type in ("passerelle", "antigravity"):
        settings.agent.llm_params = {
            **settings.agent.llm_params,
            **exp.decideur.parametres,
        }
        if exp.gabarit.variante:
            # The template is the experiment's, not the one active on the gateway: the
            # request names it (`parameters.prompt_variant`, resolved by the PromptManager).
            settings.agent.llm_params["prompt_variant"] = exp.gabarit.variante
        elif exp.decideur.type == "antigravity":
            from experiences.experience import empreinte_gabarit

            emp = empreinte_gabarit(exp.gabarit.categorie, None)
            logger.warning(
                f"[antigravity] gabarit.variante not frozen: using the active prompt (sha256: {emp['sha256']})"
            )

    personnes, info = charger_population(exp.population.chemin)
    # Actually described weather day: a population of survey respondents carries the date of its
    # day, and there is no need to draw it (ticket 058). The table lives NEXT TO the population —
    # not inside, so as not to enter the narrative or the cache key. File absent: nothing
    # changes, the seeded draw remains the mechanism.
    _dates_meteo = Path(exp.population.chemin)
    _dates_meteo = (
        _dates_meteo if _dates_meteo.is_dir() else _dates_meteo.parent
    ) / "dates_meteo.json"
    if _dates_meteo.is_file():
        settings.agent.weather_dates_file = str(_dates_meteo)
        logger.info(
            f"[météo] declared dates found ({_dates_meteo}): each person's "
            "forecast will be that of their survey day, without drawing"
        )
    dossier_exp = exp.dossier()
    # Launching does NOT rewrite the definition: it is loaded by name, as it is on
    # disk, and a run must not mutate it. The only write is adding the new
    # run to the index (below), and only when a run is actually created.
    derniere = _derniere_execution(exp) if a.reprendre else None
    if a.reprendre:
        motif = (
            "aucune exécution pour cette expérience"
            if derniere is None
            else motif_non_reprenable(Execution.ouvrir(derniere))
        )
        if motif:
            print(
                f"nothing to resume: {motif} — any run not closed can be resumed: "
                "paused, exhausted, interrupted or running without a runner"
            )
            return 1
    if derniere is not None:
        execution = Execution.ouvrir(derniere)
    else:
        execution = Execution.creer(
            dossier_exp,
            E.experience_vers_dict(exp),
            E.empreintes(exp, jeu, info),
            regime_demande={
                "parallelisme": exp.regroupement.parallelisme,
                "unite_sollicitation": "deplacement",
            },
            sources_alea={
                "graine_ordre": exp.graine_ordre,
                "graine_tirage": exp.graine_tirage,
                "graine_calendrier": exp.calendrier.graine,
                "echantillonnage_decideur": exp.decideur.type
                in ("passerelle", "antigravity"),
            },
            reglages_herites=reglages_herites_de(exp),
        )
        # executions_connues MUST reflect the disk: the definition dict written by the
        # dashboard does not carry this field, so each run reset it to its
        # own run alone and erased the previous ones from the index. We take the union with the
        # directories present, without touching the rest of the definition.
        dossier_execs = dossier_exp / "executions"
        sur_disque = (
            {p.name for p in dossier_execs.iterdir() if p.is_dir()}
            if dossier_execs.is_dir()
            else set()
        )
        exp.executions_connues = sorted(
            set(exp.executions_connues) | sur_disque | {execution.nom}
        )
        E.sauver_experience(exp, dossier_exp)
    # Ticket 085 (B4) — the restriction under which decisions will be taken enters the
    # trace, on creation as on resume: an archived measurement must say under which restriction it
    # was taken. A change between launch and resume is logged rather than staying
    # silent in an `execution.yaml` that has become wrong.
    _admises_tracees = (execution.config.get("reglages_herites") or {}).get(
        "instances_admises"
    )
    if _admises_tracees is not None and set(_admises_tracees) != set(admises):
        logger.warning(
            f"[execution] resume under a DIFFERENT restriction than at launch: "
            f"{_admises_tracees} → {admises or 'aucune restriction'}; the trace is updated"
        )
    execution.noter_reglages_herites(instances_admises=list(admises))
    settings.app.llm_exchanges_file = str(execution.dossier / "llm_exchanges.jsonl")

    # Per-key admission (spec parallelisation_experiences): parallel if the experiment's keys
    # are free, otherwise queued — the scheduler will start it when a key is freed.
    from experiences import reservations as R

    creee = derniere is None
    cles = _jeu_de_cles_experience(exp, moniteur)
    if exp.decideur.type == "passerelle" and not getattr(moniteur, "providers", None):
        # R12 fail-safe: key sets cannot be established (provider configuration unreadable) →
        # do not start rather than risk two runs on the same key.
        _annuler_execution_creee(exp, execution, dossier_exp, creee=creee)
        print(
            "REFUSED — provider configuration unreadable: key sets cannot be established (fail-safe R12)"
        )
        return 1
    issue = R.admettre(
        cles,
        execution.dossier,
        exp.nom,
        args={
            "accepter_perime": a.accepter_perime,
            "attendre_fenetre": a.attendre_fenetre,
            "reprendre": a.reprendre,
        },
    )
    if issue == "file":
        _annuler_execution_creee(exp, execution, dossier_exp, creee=creee)
        print(
            f"EN FILE — {exp.nom} is waiting for a key to be freed ({sorted(cles)}). "
            "The scheduler will start it automatically (FIFO)."
        )
        return 4

    from experiences.decideurs import construire_decideur

    agent = None
    # `typesafe` needs it too: the presentation served to Jev comes from the same
    # `build_travel_plan_payload` as the LLM arms (ticket 096, T1).
    if exp.decideur.type in ("passerelle", "antigravity", "typesafe"):
        from urban_mobility_agents.agents.llm_agent import LlmAgent

        agent = LlmAgent()
    dossier_echanges = (
        execution.dossier / "echanges" if exp.decideur.type == "antigravity" else None
    )
    decideur = construire_decideur(
        exp.decideur,
        agent=agent,
        moniteur=moniteur,
        instances=(moniteur.instances if moniteur else None),
        dossier_echanges=dossier_echanges,
        attente_max_s=exp.attente_max_s,
        execution=execution,
        gabarit=exp.gabarit,
    )
    # Traceability (R3): the whole run log ([execution]/[ALARME]) is also archived in
    # the directory; launched via `docker compose exec`, it otherwise only went to the launching terminal.
    sink = logger.add(
        str(execution.dossier / "execution.log"),
        level="INFO",
        enqueue=True,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
    )
    try:
        compteurs = asyncio.run(
            executer(
                exp,
                jeu,
                personnes,
                execution,
                decideur,
                moniteur=moniteur,
                attendre_fenetre=a.attendre_fenetre,
            )
        )
    finally:
        # The key is freed at the terminal status (R7): `liberer` removes this run's
        # reservations, including if `executer` raised (the status is then non-terminal, but the
        # process stops — no ghost). A hard kill is caught up by `reconcilier`.
        R.liberer(execution.dossier)
        logger.remove(sink)
    print(
        json.dumps(
            {k: v for k, v in compteurs.items() if k != "quota"},
            ensure_ascii=False,
            indent=1,
            default=str,
        )
    )
    print(f"run: {execution.dossier}")
    return 0 if compteurs.get("etat") == "terminee" else 3


def _signal_fichier(a: argparse.Namespace, nom_fichier: str) -> int:
    exp = _charger_experience_par_nom(a.experience)
    derniere = _derniere_execution(exp)
    if derniere is None:
        print("no run")
        return 1
    if Execution.ouvrir(derniere).etat().get("etat") != ETAT_EN_COURS:
        print(f"the last run ({derniere.name}) is not running")
        return 1
    (derniere / nom_fichier).touch()
    from experiences.runner import DELAI_GRACE_PAUSE_S

    print(
        f"{nom_fichier} requested for {derniere.name} — effective within {DELAI_GRACE_PAUSE_S:.0f} s "
        f"at most (requests still in flight are abandoned, their trips "
        f"requested again on resume)"
    )
    return 0


def cmd_registre(a: argparse.Namespace) -> int:
    from experiences.registre import formater_table, lister, trier_filtrer

    filtres = dict(f.split("=", 1) for f in (a.filtrer or []) if "=" in f)
    lignes = trier_filtrer(
        lister(inclure_masquees=a.inclure_masquees),
        trier=a.trier,
        decroissant=a.decroissant,
        filtres=filtres,
    )
    if a.json:
        print(json.dumps(lignes, ensure_ascii=False, indent=1, default=str))
    else:
        print(formater_table(lignes))
    return 0


def cmd_statuer(a: argparse.Namespace) -> int:
    """Sets the status of an experiment (§3.2, §5). Erases and moves nothing."""
    from experiences import statut as S
    from experiences.experience import dossier_experiences

    dossier = Path(a.experience)
    if not (dossier / "experience.yaml").is_file():
        dossier = dossier_experiences() / a.experience
    corps = S.ecrire(
        dossier,
        a.statut,
        motif=a.motif,
        reference=a.reference,
        visible_par_defaut=a.visible,
    )
    print(
        f"{dossier.name} : {corps['statut']}"
        + (f" — {corps['motif']}" if corps["motif"] else "")
    )
    print(f"  data kept, no file moved ({dossier})")
    if corps["statut"] != S.ACTIF:
        print(
            "  hidden from the registry by default; `--inclure-masquees` shows it again"
        )
    return 0


def cmd_statuts(a: argparse.Namespace) -> int:
    """Status of all experiments, hidden ones included."""
    from experiences import statut as S
    from experiences.experience import dossier_experiences

    tous = S.statuts_par_experience(dossier_experiences())
    if a.json:
        print(json.dumps(tous, ensure_ascii=False, indent=1, default=str))
        return 0
    largeur = max((len(n) for n in tous), default=10)
    for nom, st in sorted(tous.items(), key=lambda kv: (kv[1]["statut"], kv[0])):
        motif = f" — {st['motif']}" if st.get("motif") else ""
        print(f"{nom:<{largeur}}  {st['statut']:<9}{motif}")
    compte: dict[str, int] = {}
    for st in tous.values():
        compte[st["statut"]] = compte.get(st["statut"], 0) + 1
    print("\n" + " · ".join(f"{k} : {v}" for k, v in sorted(compte.items())))
    return 0


def cmd_comparer(a: argparse.Namespace) -> int:
    from experiences.registre import comparer, formater_comparaison

    c = comparer(a.a, a.b, inclure_invalides=a.inclure_invalides)
    print(
        json.dumps(c, ensure_ascii=False, indent=1, default=str)
        if a.json
        else formater_comparaison(c)
    )
    return 0 if c["comparable"] else 1


def cmd_synthese(a: argparse.Namespace) -> int:
    from experiences.registre import ecrire_synthese

    print(ecrire_synthese(a.execution))
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    """Scores a run (or all of them) under a formula, and writes the synthesis page.

    `--toutes` recomputes the whole history offline (replay from the raw scores
    when possible); otherwise scores only the `--execution` run.
    """
    from experiences import formule as F
    from experiences import rendu_scores, score

    registre = F.charger()
    formule = registre.reference
    if a.formule:
        formule = registre.par_nom(a.formule)
        if formule is None:
            print(f"ERROR: unknown formula: {a.formule!r}", file=sys.stderr)
            return 2

    if a.toutes:
        bilan = score.rescorer_tout(formule, registre)
        # Regenerates the pages of the runs that now have a scores.json.
        racine = score.REPO_ROOT / "data" / "experiences"
        for dossier in score._executions(racine):
            rendu_scores.ecrire(dossier, registre)
        print(json.dumps(bilan, ensure_ascii=False))
        return 0

    if not a.execution:
        print("ERROR: specify a run or --toutes", file=sys.stderr)
        return 2
    chemin = score.score_execution(a.execution, formule, registre)
    if chemin is None:
        print("run not scored (partial, or loss engine missing)")
        return 1
    page = rendu_scores.ecrire(a.execution, registre)
    print(page or chemin)
    return 0


def cmd_journal(a: argparse.Namespace) -> int:
    """Checks or regenerates the `moves.csv` of a run (ticket 081).

    `--verifier` observes without writing anything; `--regenerer` rebuilds the log from
    `decisions.jsonl` and the sealed set, then leaves rescoring to `score`.
    """
    import asyncio

    from experiences import journal as JN
    from experiences import score as S

    if a.toutes or (a.verifier and not a.execution):
        racine = Path(a.racine) if a.racine else (S.REPO_ROOT / "data" / "experiences")
        constats = [JN.verifier(d) for d in JN.executions(racine)]
        if a.json:
            print(json.dumps(constats, ensure_ascii=False, indent=1))
            return 0
        incomplets = 0
        print(
            f"{'exécution':<22} {'état':<11} {'lignes':>7} {'décidées':>9} {'écart':>8}  verdict"
        )
        for c in constats:
            rel = (
                "—"
                if c["ecart_relatif"] is None
                else f"{100 * c['ecart_relatif']:+.2f}%"
            )
            verdict = {True: "complet", False: "INCOMPLET", None: "non comparable"}[
                c["complet"]
            ]
            incomplets += c["complet"] is False
            print(
                f"{c['execution']:<22} {c['etat']!s:<11} {c['lignes_journal']:>7} "
                f"{c['decides']!s:>9} {rel:>8}  {verdict}"
            )
        print(f"\n{len(constats)} run(s) scanned, {incomplets} incomplete")
        return 1 if incomplets else 0

    if not a.execution:
        print("ERROR: specify a run, or --toutes", file=sys.stderr)
        return 2

    if a.verifier:
        print(json.dumps(JN.verifier(a.execution), ensure_ascii=False, indent=1))
        return 0

    execution = JN.ouvrir(a.execution, motif_archive=a.motif_archive)
    jeu, personnes, info = JN.resoudre_sources(
        execution,
        jeu=a.jeu,
        population=a.population,
        motif_archive=a.motif_archive,
    )
    bilan = asyncio.run(
        JN.regenerer(
            execution,
            jeu,
            personnes,
            JN.EtiquetteDecideur.depuis_execution(execution.config),
            info_population=info,
        )
    )
    print(json.dumps(bilan, ensure_ascii=False, indent=1))
    return 0


def cmd_erreurs(a: argparse.Namespace) -> int:
    """Matches decisions.jsonl against the set: attempts by type, missing trips, days with gaps."""
    from experiences.erreurs import diagnostiquer, formater

    if a.execution:
        dossier = Path(a.execution)
    else:
        exp = _charger_experience_par_nom(a.experience)
        derniere = _derniere_execution(exp)
        if derniere is None:
            print("no run")
            return 1
        dossier = derniere
    diag = diagnostiquer(dossier)
    print(
        json.dumps(diag, ensure_ascii=False, indent=1, default=str)
        if a.json
        else formater(diag)
    )
    return 0 if diag["nb_manquants"] == 0 else 1


def cmd_reconcilier(a: argparse.Namespace) -> int:
    """Frees the keys of finished runs and of ghosts (dead pid). To be run INSIDE the container."""
    from experiences import reservations as R

    liberees = R.reconcilier()
    print(json.dumps({"cles_liberees": liberees}, ensure_ascii=False))
    return 0


def cmd_actives(a: argparse.Namespace) -> int:
    """Experiments holding at least one key. `--est-vide`: code 0 if none (services can be stopped)."""
    from experiences import reservations as R

    R.reconcilier()  # inside the container: purge the ghosts first
    actives = R.actives()
    if a.est_vide:
        return 0 if not actives else 1
    print(json.dumps(actives, ensure_ascii=False, indent=1))
    return 0


def cmd_file(a: argparse.Namespace) -> int:
    """Experiments waiting for a key, FIFO order."""
    from experiences import reservations as R

    print(json.dumps(R.lister_file(), ensure_ascii=False, indent=1))
    return 0


def cmd_defiler(a: argparse.Namespace) -> int:
    """Removes an experiment from the queue before its promotion (R2e)."""
    from experiences import reservations as R

    retiree = R.retirer_file(a.experience)
    print(
        f"{a.experience}: {'retirée de la file' if retiree else 'absente de la file'}"
    )
    return 0 if retiree else 1


def cmd_ordonnancer(a: argparse.Namespace) -> int:
    """Scheduling loop (host): reconciles and promotes the queue (R2b). Ctrl-C to stop."""
    from experiences.ordonnanceur import boucle

    return boucle(intervalle_s=a.intervalle)


# ── Campaign (ticket 074, lot D) ─────────────────────────────────────────────


def cmd_campagne_lancer(a: argparse.Namespace) -> int:
    from experiences import campagne as C

    if a.estimer:
        return _campagne_estimer(a.nom)
    return C.lancer(a.nom, reprendre=not a.recommencer, intervalle_s=a.intervalle)


def _campagne_estimer(nom: str) -> int:
    """The campaign's budget, BEFORE queueing anything at all (D-8).

    Deliberately LIGHT, and that is a choice: the full `estimer` needs the gateway
    monitor and the launch checks, hence the container. The campaign, for its part, drives
    from the host. So we read what is enough to decide — the number of trips covered
    by the set, and the decision-maker type, which says whether those trips cost quota.

    **Two columns, two units.** A trip is a decision to take; a request is
    a provider call, and the gateway groups several of them. A single "LLM
    calls" column fed by the trips overestimated the budget about eightfold. The
    requests column carries the CAUTIOUS figure (cf. `lots.facteurs`): without a comparable
    archived run it equals the number of trips, so the quote never becomes more
    optimistic than the previous one.

    What it does not replace: `make experience-estimer EXP=<nom>` remains the reference
    measurement for an experiment, tokens and quota window included.
    """
    from experiences import campagne as C
    from experiences import lots as L

    camp = C.charger(nom)
    total_depl, total_req, par_phase = 0, 0, []
    print(f"Budget of campaign {camp.nom!r} — {len(camp.toutes)} experiments")
    print(f"     {'expérience':58s}  {'déplacements':>13s}  {'requêtes':>10s}")
    for phase in camp.phases:
        depl_phase, req_phase = 0, 0
        print(f"\n  ── phase {phase.nom} ({len(phase.experiences)}) ──")
        for exp_nom in phase.experiences:
            try:
                exp = _charger_experience_par_nom(exp_nom)
                jeu = J.Jeu.charger(E.dossier_jeux() / exp.jeu.nom, verifier=False)
                couverts = int(jeu.couverture()["deplacements_couverts"])
            except Exception as err:  # noqa: BLE001 — a failed quote does not block the total
                print(f"     {exp_nom:58s}  quote impossible: {err}")
                continue
            # Only a decision-maker going through the GATEWAY consumes provider quota. The
            # deterministic controls and the fitted models run locally; Antigravity goes
            # through an on-disk sub-agent (`DecideurAntigravity.sans_quota`), one decision at a
            # time and without grouping — so it does not count either.
            if exp.decideur.type != "passerelle":
                print(f"     {exp_nom:58s}  {couverts:>13d}  {'0 (local)':>10s}")
                depl_phase += couverts
                continue
            providers, instances = E.instances_visees(exp)
            reg = L.facteurs(
                providers=providers,
                instances=instances,
                parallelisme=(
                    exp.regroupement.parallelisme
                    if exp.mode == E.MODE_SANS_SIMULATEUR
                    else None
                ),
                empreinte_gabarit_=E.empreinte_gabarit(
                    exp.gabarit.categorie, exp.gabarit.variante
                )["sha256"],
                troncature=bool(getattr(exp, "troncature_15", False)),
            )
            req = L.requetes(couverts, reg["prudent"])
            depl_phase += couverts
            req_phase += req
            print(f"     {exp_nom:58s}  {couverts:>13d}  {req:>10d}")
        par_phase.append((phase.nom, depl_phase, req_phase))
        total_depl += depl_phase
        total_req += req_phase

    print("\n  ── total ──")
    for nom_phase, depl, req in par_phase:
        print(f"     {nom_phase:58s}  {depl:>13d}  {req:>10d}")
    print(f"     {'CAMPAGNE':58s}  {total_depl:>13d}  {total_req:>10d}")
    print(
        "\n  Trips: coverage of each experiment's set (field `deplacements` of "
        "`experience-estimer`).\n  Requests: provider calls at the CAUTIOUS grouping — "
        "that is what the quota counts (field `requetes.prudente`).\n  Without a comparable "
        "archived run, the cautious grouping equals 1 and the two columns coincide."
    )
    return 0


def cmd_campagne_etat(a: argparse.Namespace) -> int:
    from experiences import campagne as C

    vue = C.etat_lisible(a.nom)
    if a.json:
        print(json.dumps(vue, ensure_ascii=False, indent=1))
        return 0
    etat = vue["etat"]
    print("═" * 86)
    print(f"Campaign {vue['nom']}  ·  {len(vue['faites'])}/{vue['total']} done")
    print("═" * 86)
    if vue["note"]:
        print(f"{vue['note']}\n")
    for phase in vue["phases"]:
        fait, tot = len(phase["faites"]), len(phase["experiences"])
        barre = "█" * int(20 * fait / tot) + "·" * (20 - int(20 * fait / tot))
        print(f"  {phase['nom']:12s} [{barre}] {fait}/{tot}")
    if etat is None:
        print("\nNever launched. `make campagne-lancer NOM=" + vue["nom"] + "`")
        return 0
    print(f"\n  current phase  : {etat.get('phase_courante')}")
    print(f"  running        : {etat.get('courante') or '—'}")
    print(f"  started on     : {etat.get('demarree_le')}  ·  updated {etat.get('maj')}")
    if etat.get("terminee_le"):
        print(f"  FINISHED on    : {etat['terminee_le']}")
    if vue["arret_demande"]:
        print("  ⏹ stop requested — the current run finishes on its own.")
    sommeils = etat.get("sommeils") or []
    if sommeils:
        cumul = sum(s.get("duree_s") or 0 for s in sommeils) / 3600
        print(
            f"  sleeps         : {len(sommeils)} ({cumul:.1f} h in total); "
            f"last one until {sommeils[-1].get('jusqu')}"
        )
    print(
        f"  next quota renewal: {vue['prochain_reveil']} "
        f"(in {vue['secondes_avant_reveil'] / 3600:.1f} h)"
    )
    echecs = etat.get("echouees") or {}
    if echecs:
        print(f"\n  ⚠ {len(echecs)} failed:")
        for exp_nom, det in echecs.items():
            print(
                f"     {exp_nom:60s} {det.get('motif')} ({det.get('tentatives')} attempt(s))"
            )
    print("\n  state per experiment:")
    for exp_nom, st in vue["par_experience"].items():
        print(f"     {st['etat']:18s} {exp_nom}")
    return 0 if not echecs else 1


def cmd_campagne_arreter(a: argparse.Namespace) -> int:
    from experiences import campagne as C

    C.arreter(a.nom)
    print(
        f"Stop requested for campaign {a.nom!r}. "
        "The current run finishes; no other will be launched."
    )
    return 0


def construire_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiences",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="commande", required=True)

    s = sub.add_parser("preparer-jeu")
    s.set_defaults(fn=cmd_preparer_jeu)
    s.add_argument("--population", required=True)
    s.add_argument("--nom", required=True)
    s.add_argument("--dossier")
    s.add_argument("--jour", default="2026-03-16")
    s.add_argument("--concurrence", type=int, default=8)
    s.add_argument(
        "--seuil",
        type=float,
        default=0.05,
        help="share of trips without a proposal that raises the ALARME",
    )

    s = sub.add_parser("consulter-jeu")
    s.set_defaults(fn=cmd_consulter_jeu)
    s.add_argument("--nom", required=True)
    s.add_argument("--dossier")
    s.add_argument("--personne")
    s.add_argument("--sans-verification", action="store_true")

    s = sub.add_parser("verifier-jeu")
    s.set_defaults(fn=cmd_verifier_jeu)
    s.add_argument("--nom", required=True)
    s.add_argument("--dossier")

    s = sub.add_parser(
        "config-empreintes",
        help="rebuilds the config values registry from git (host only)",
    )
    s.set_defaults(fn=cmd_config_empreintes)
    s.add_argument("--verifier", action="store_true")
    s.add_argument("--dossier-config")

    s = sub.add_parser(
        "verifier-jours",
        help="is another day's PT offer the same as the set's? (services required)",
    )
    s.set_defaults(fn=cmd_verifier_jours)
    s.add_argument("--nom", required=True)
    s.add_argument("--dossier")
    s.add_argument("--jour", required=True)
    s.add_argument("--methode", choices=("gtfs", "moteurs"), default="gtfs")
    s.add_argument("--gtfs", help="GTFS directory (default: setting gtfs.gtfs_file)")
    s.add_argument(
        "--fenetre-h",
        type=float,
        default=4.0,
        help="time window (h) after the scheduled departure in which a different departure invalidates the trip",
    )
    s.add_argument("--echantillon", type=int, default=100)
    s.add_argument("--graine", type=int, default=42)
    s.add_argument(
        "--declarer",
        action="store_true",
        help="writes EQUIVALENCES.yaml (gtfs: validity per trip; moteurs: if 100 %% identical)",
    )

    s = sub.add_parser("definir")
    s.set_defaults(fn=cmd_definir)
    s.add_argument("fichier")
    s.add_argument(
        "--accepter-nom",
        action="store_true",
        help="accepts a `nom` that is not the canonical name of the parameters (N13)",
    )

    s = sub.add_parser("dupliquer")
    s.set_defaults(fn=cmd_dupliquer)
    s.add_argument("--de", required=True)
    s.add_argument(
        "--vers",
        help="name of the copy; by default, computed from its parameters (N1)",
    )
    s.add_argument("--decideur-type")
    s.add_argument("--decideur-modele")
    s.add_argument(
        "--artefact",
        help=(
            "artefact of the `modele` decision-maker (default: the LightGBM booster) — "
            "e.g. scripts/progedo_logit/mnl_model.json for the multinomial logit, "
            "scripts/progedo_logit/klr_model.json for the kernel logistic regression"
        ),
    )

    s = sub.add_parser("estimer")
    s.set_defaults(fn=cmd_estimer)
    s.add_argument("--experience", required=True)

    # The default is NOT to wait for the window: when the quota runs out, the run
    # goes `epuisee` and hands back control. This comment and the help claimed the opposite since
    # 2026-09-08, but the two flags share their `dest` and argparse keeps the default
    # of the FIRST one declared (store_true → False): waiting was never the default. Found
    # on 2026-09-29; the default is now set by `set_defaults`, and the callers
    # (campaign, scheduler) pass one flag or the other explicitly.
    _AIDE_FENETRE = (
        "à l'épuisement du quota, ATTENDRE la fenêtre de reprise (heure annoncée par le "
        "fournisseur, ou minuit dans son fuseau) au lieu de passer `epuisee`"
    )
    _AIDE_SANS_FENETRE = (
        "le défaut, dit explicitement : à l'épuisement du quota, passer `epuisee` tout de "
        "suite et rendre la main"
    )
    s = sub.add_parser("lancer")
    s.add_argument(
        "--ignorer-aptitude",
        action="store_true",
        dest="ignorer_aptitude",
        help="launches despite a fitness refusal (quota, caps) — the warning stays displayed",
    )
    s.set_defaults(fn=cmd_lancer)
    s.add_argument("--experience", required=True)
    s.add_argument("--reprendre", action="store_true")
    s.add_argument("--accepter-perime", action="store_true")
    s.add_argument("--attendre-fenetre", action="store_true", help=_AIDE_FENETRE)
    s.add_argument(
        "--ne-pas-attendre-fenetre",
        dest="attendre_fenetre",
        action="store_false",
        help=_AIDE_SANS_FENETRE,
    )
    s.set_defaults(attendre_fenetre=False)

    s = sub.add_parser("reprendre")
    s.add_argument(
        "--ignorer-aptitude",
        action="store_true",
        dest="ignorer_aptitude",
        help="launches despite a fitness refusal (quota, caps) — the warning stays displayed",
    )
    s.set_defaults(
        fn=lambda a: cmd_lancer(argparse.Namespace(**{**vars(a), "reprendre": True}))
    )
    s.add_argument("--experience", required=True)
    s.add_argument("--accepter-perime", action="store_true")
    s.add_argument("--attendre-fenetre", action="store_true", help=_AIDE_FENETRE)
    s.add_argument(
        "--ne-pas-attendre-fenetre",
        dest="attendre_fenetre",
        action="store_false",
        help=_AIDE_SANS_FENETRE,
    )
    s.set_defaults(attendre_fenetre=False)

    s = sub.add_parser("pause")
    s.set_defaults(fn=lambda a: _signal_fichier(a, FICHIER_PAUSE))
    s.add_argument("--experience", required=True)
    s = sub.add_parser("arreter")
    s.set_defaults(fn=lambda a: _signal_fichier(a, FICHIER_STOP))
    s.add_argument("--experience", required=True)

    s = sub.add_parser(
        "reconcilier",
        help="frees the keys of finished/ghost runs (INSIDE the container)",
    )
    s.set_defaults(fn=cmd_reconcilier)

    s = sub.add_parser(
        "actives", help="experiments holding a key; --est-vide: code 0 if none"
    )
    s.set_defaults(fn=cmd_actives)
    s.add_argument("--est-vide", action="store_true")

    s = sub.add_parser("file", help="experiments waiting for a key (FIFO)")
    s.set_defaults(fn=cmd_file)

    s = sub.add_parser("defiler", help="removes an experiment from the queue (R2e)")
    s.set_defaults(fn=cmd_defiler)
    s.add_argument("--experience", required=True)

    s = sub.add_parser(
        "ordonnancer", help="host loop: reconciles and promotes the queue"
    )
    s.set_defaults(fn=cmd_ordonnancer)
    s.add_argument("--intervalle", type=float, default=5.0)

    s = sub.add_parser(
        "campagne-lancer", help="runs a campaign to completion (ticket 074, lot D)"
    )
    s.set_defaults(fn=cmd_campagne_lancer)
    s.add_argument("--nom", required=True)
    s.add_argument(
        "--recommencer",
        action="store_true",
        help="ignores the existing state and starts from scratch",
    )
    s.add_argument(
        "--estimer",
        action="store_true",
        help="states the budget and exits, without queueing anything",
    )
    s.add_argument("--intervalle", type=float, default=30.0)

    s = sub.add_parser("campagne-etat", help="progress of a campaign")
    s.set_defaults(fn=cmd_campagne_etat)
    s.add_argument("--nom", required=True)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("campagne-arreter", help="stops a campaign cleanly")
    s.set_defaults(fn=cmd_campagne_arreter)
    s.add_argument("--nom", required=True)

    s = sub.add_parser("registre")
    s.set_defaults(fn=cmd_registre)
    s.add_argument("--trier")
    s.add_argument("--decroissant", action="store_true")
    s.add_argument("--filtrer", action="append")
    s.add_argument("--json", action="store_true")
    s.add_argument(
        "--inclure-masquees",
        action="store_true",
        dest="inclure_masquees",
        help="shows archived or invalidated experiments again (hidden by default)",
    )

    s = sub.add_parser(
        "statuer",
        help="sets the status of an experiment (actif|archivee|invalide) without deleting anything",
    )
    s.set_defaults(fn=cmd_statuer)
    s.add_argument("experience", help="name or path of the experiment directory")
    s.add_argument("statut", choices=("actif", "archivee", "invalide"))
    s.add_argument("--motif", help="mandatory for archivee and invalide")
    s.add_argument("--reference", help="spec or ticket that justifies the status")
    s.add_argument(
        "--visible",
        action="store_true",
        default=None,
        help="keep the row visible in the registry despite the status",
    )

    s = sub.add_parser(
        "statuts", help="status of all experiments, hidden ones included"
    )
    s.set_defaults(fn=cmd_statuts)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("comparer")
    s.set_defaults(fn=cmd_comparer)
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--json", action="store_true")
    s.add_argument(
        "--inclure-invalides",
        action="store_true",
        dest="inclure_invalides",
        help="compares despite an invalidated or archived experiment (the status is recalled)",
    )
    s = sub.add_parser("synthese")
    s.set_defaults(fn=cmd_synthese)
    s.add_argument("execution")

    s = sub.add_parser(
        "score",
        help="scores a run (composite score + per-stratum detail) and writes its page; "
        "--toutes recomputes the whole history offline",
    )
    s.set_defaults(fn=cmd_score)
    s.add_argument("execution", nargs="?", help="run directory to score")
    s.add_argument(
        "--toutes",
        action="store_true",
        help="recomputes all finished runs (offline replay)",
    )
    s.add_argument(
        "--formule", help="formula name in the registry (default: the reference)"
    )

    s = sub.add_parser(
        "journal",
        help="checks or regenerates a run's moves.csv from decisions.jsonl "
        "(ticket 081); --verifier --toutes scans the whole history",
    )
    s.set_defaults(fn=cmd_journal)
    s.add_argument("execution", nargs="?", help="run directory")
    s.add_argument(
        "--regenerer",
        action="store_true",
        help="rebuilds moves.csv from decisions.jsonl and the sealed set",
    )
    s.add_argument(
        "--verifier",
        action="store_true",
        help="reports the gap between the log and the archived decisions, without writing anything",
    )
    s.add_argument(
        "--toutes",
        action="store_true",
        help="scans all the runs under a root",
    )
    s.add_argument(
        "--racine", help="root of the scanned experiments (default: data/experiences)"
    )
    s.add_argument(
        "--jeu", help="directory of the sealed set, if the frozen path cannot be found"
    )
    s.add_argument(
        "--population",
        help="directory of the sealed cohort, if the frozen path is a container path",
    )
    s.add_argument(
        "--motif-archive",
        dest="motif_archive",
        help="exemption reason to read a run, a set or a cohort in cold archive",
    )
    s.add_argument("--json", action="store_true")

    s = sub.add_parser(
        "erreurs",
        help="matches decisions.jsonl against the set (missing, days with gaps, attempts)",
    )
    s.set_defaults(fn=cmd_erreurs)
    s.add_argument(
        "execution",
        nargs="?",
        help="run directory; failing that, --experience takes the last one",
    )
    s.add_argument("--experience", help="experiment whose last run is diagnosed")
    s.add_argument("--json", action="store_true")
    return p


def couper_rejeu_ab() -> None:
    """A JEV never replays a memory A/B (2026-09-28).

    JEVs run in the controller container, which keeps `REJEU_AB` and the strict bound of
    the last memory experiment launched. On 26/09, they thus stored their answers in
    a13's store and read them back from it: proexp05 seed 123 received 145 answers from its own
    aborted attempt. After a control run in common prefix, the strict bound would make every
    call fail. Replay belongs to the memory orchestrator, not to the definition of a JEV.
    """
    from settings import settings

    espace, borne = settings.llm.rejeu_ab, settings.llm.rejeu_strict_avant_ts
    settings.llm.rejeu_ab = ""
    settings.llm.rejeu_strict_avant_ts = None
    if espace or borne:
        logger.warning(
            f"[jev] A/B replay inherited from the container switched off (space {espace or '—'}, strict bound "
            f"{borne or '—'}): this run pays for its calls and touches no A/B store."
        )


def main(argv: list[str] | None = None) -> int:
    couper_rejeu_ab()
    a = construire_parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except (
        E.ExperienceInvalide,
        J.JeuInvalide,
        J.JeuClos,
        FileNotFoundError,
        ValueError,
    ) as e:  # StatutInvalide and VariantePromptInvalide derive from ValueError
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

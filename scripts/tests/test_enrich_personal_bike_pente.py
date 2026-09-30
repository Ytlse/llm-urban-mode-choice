"""Slope rule of the bike check: judged from 100 households per
size, and an inversion is a failure only if it exceeds the uncertainty of both cells."""
from scripts.data.population.enrich_personal_bike import slope_verdict


def test_pente_non_concluante_sous_100_foyers():
    # v4 cohort: 63.4 % (69 households) > 55.5 % (55 households) — noise, no verdict.
    cells = [(32.0, 300, 250), (55.0, 300, 150), (63.4, 200, 69), (55.5, 200, 55)]
    statut, detail = slope_verdict(cells)
    assert statut == "non concluant" and "55 foyers" in detail


def test_pente_croissante_ok():
    # v4 pool (11,329 persons), measured on 2026-09-03: 2,350 / 1,657 / 744 / 532 households.
    cells = [(32.8, 2482, 2350), (49.1, 3382, 1657), (55.0, 2238, 744), (60.9, 2132, 532)]
    assert slope_verdict(cells) == ("ok", "croissante")


def test_inversion_dans_l_incertitude_est_ok():
    # 84.8 % (533) then 79.6 % (157): −5.2 pt for a combined uncertainty ≈ ± 7 pt.
    cells = [(58.2, 0, 1691), (74.5, 0, 746), (84.8, 0, 533), (79.6, 0, 157)]
    statut, detail = slope_verdict(cells)
    assert statut == "ok" and detail.startswith("inversion dans l'incertitude")


def test_baisse_significative_est_un_echec():
    # 76 % among single persons vs 33 % in pairs, over 1,000 households each: the original defect.
    cells = [(76.0, 0, 1000), (33.0, 0, 1000), (60.0, 0, 1000), (70.0, 0, 1000)]
    statut, detail = slope_verdict(cells)
    assert statut == "echec" and "76.0 → 33.0" in detail


def test_taille_absente_non_calculable():
    assert slope_verdict([(30.0, 0, 500), (None, 0, 0), (60.0, 0, 500), (70.0, 0, 500)])[0] == "non calculable"

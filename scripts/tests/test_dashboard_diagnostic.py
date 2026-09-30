import json
from scripts.dashboard.diagnostic import diagnostic, expliquer, motif_lancement, motifs_execution


def test_quotas_distincts():
    assert 'journalier' in diagnostic('quota_journalier')[1]
    assert 'minute' in diagnostic('quota exceeded per minute')[1]
    assert 'durée non précisée' in diagnostic('HTTP 429 RESOURCE_EXHAUSTED')[1]
    assert diagnostic('quota journalier plus court que la charge ; ce n’est pas un refus') is None
    assert diagnostic('plafond déclaré dépassé et le fournisseur sert toujours') is None


def test_forte_demande():
    assert 'forte demande' in diagnostic('503 model experiencing high demand')[1]
    assert 'temporairement indisponible' in diagnostic('HTTP 503')[1]


def test_cles_et_heures(tmp_path):
    erreurs = [dict(message='quota_journalier', fournisseur=f'google_gemini35_key{i}',
                    horodatage=f'2026-09-26T14:0{i}:00+00:00') for i in (1, 2)]
    (tmp_path / 'erreurs.jsonl').write_text('\n'.join(json.dumps(e) for e in erreurs) + '\n{')
    motifs = motifs_execution(tmp_path)
    assert len(motifs) == 2
    assert 'clé 1' in motifs[0] and '14:01:00+00:00' in motifs[0]
    assert 'clé 2' in motifs[1] and '14:02:00+00:00' in motifs[1]
    assert 'clé' not in expliquer(dict(message='quota_journalier'))


def test_lancement_historique(tmp_path):
    logs = tmp_path / 'lancements'
    logs.mkdir()
    (logs / '2026-09-26_10_00_00.log').write_text('EN FILE — attend une clé')
    (logs / '2026-09-26_12_00_00.log').write_text('503 high demand')
    assert 'réservées' in motif_lancement(tmp_path, '2026-09-26T11:00:00+00:00')
    assert 'forte demande' in motif_lancement(tmp_path)

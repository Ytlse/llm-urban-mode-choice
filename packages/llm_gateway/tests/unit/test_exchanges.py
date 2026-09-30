"""The exchange log: disabled by default, redacted on request, rotated by size."""
from __future__ import annotations

import json
from types import SimpleNamespace

from llm_gateway.telemetry.exchanges import (
    ExchangeJournal,
    ExchangeRecord,
    FieldRedactor,
    TruncateRedactor,
    identity,
    load_redactor,
)
from llm_gateway.telemetry.logger import log_llm_error, log_llm_exchange


def _record(**kw) -> ExchangeRecord:
    base = dict(task_id="t1", provider="p", category="c", tokens_in=10, tokens_out=5,
                messages=[{"role": "system", "content": "Instruction"}, {"role": "user", "content": "--- agent_id=a1 | aged 38, Pibrac"}],
                response=[{"agent_id": "a1", "summary": "Walks when the weather is nice"}], sim_ts=1_700_000_000.0)
    base.update(kw)
    return ExchangeRecord(**base)


def _parse_indented(text: str) -> list[dict]:
    out, buf, depth = [], [], 0
    for line in text.splitlines():
        buf.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and buf and buf[-1].strip():
            out.append(json.loads("\n".join(buf)))
            buf = []
    return out


def test_le_journal_ecrit_le_format_historique(tmp_path):
    p = tmp_path / "llm_exchanges.jsonl"
    ExchangeJournal(p).write(_record())
    (rec,) = _parse_indented(p.read_text(encoding="utf-8"))
    assert rec["task_id"] == "t1" and rec["sim_day"] == "2023-11-14" and rec["tokens_in"] == 10
    assert rec["messages"][1]["content"].endswith("Pibrac"), "no redactor, the text is intact"


def test_rotation_par_taille(tmp_path):
    p = tmp_path / "j.jsonl"
    journal = ExchangeJournal(p, max_bytes=200)
    journal.write(_record())
    journal.write(_record())
    assert (tmp_path / "j.jsonl.1").is_file(), "the first file rotated"


def test_field_redactor_masque_recursivement():
    r = FieldRedactor(["content", "summary"])(_record())
    assert r.messages[1]["content"] == "***" and r.response[0]["summary"] == "***"
    assert r.response[0]["agent_id"] == "a1"


def test_field_redactor_hache_de_facon_stable():
    a = FieldRedactor(["content"], mode="hash")(_record()).messages[1]["content"]
    b = FieldRedactor(["content"], mode="hash")(_record()).messages[1]["content"]
    assert a == b and a.startswith("sha256:") and "Pibrac" not in a


def test_truncate_redactor():
    r = TruncateRedactor(max_chars=10)(_record())
    assert r.messages[1]["content"].startswith("--- agent_") and "coupés" in r.messages[1]["content"]
    assert "_tronque" in r.response


def test_load_redactor_par_chemin_pointe():
    assert load_redactor(None) is identity
    assert load_redactor("llm_gateway.telemetry.exchanges:identity") is identity
    assert isinstance(load_redactor("llm_gateway.telemetry.exchanges.TruncateRedactor"), TruncateRedactor)


def test_desactive_par_defaut_rien_n_est_ecrit(tmp_path):
    telemetry = SimpleNamespace(exchanges_enabled=False, exchanges_file=None, workdir=tmp_path, redactor=None, exchanges_max_bytes=0)
    log_llm_exchange("t", "p", [], {}, 1, 1, telemetry=telemetry)
    assert not list(tmp_path.iterdir())


def test_active_avec_redacteur_configure(tmp_path):
    telemetry = SimpleNamespace(exchanges_enabled=True, exchanges_file=None, workdir=tmp_path,
                                redactor="llm_gateway.telemetry.exchanges.TruncateRedactor", exchanges_max_bytes=0)
    log_llm_exchange("t", "p", [{"role": "user", "content": "x" * 5000}], {"ok": True}, 1, 1, telemetry=telemetry)
    text = (tmp_path / "llm_exchanges.jsonl").read_text(encoding="utf-8")
    assert "coupés" in text and "x" * 5000 not in text


def test_le_journal_erreur_porte_l_origine_du_run(tmp_path):
    telemetry = SimpleNamespace(workdir=tmp_path)
    log_llm_error("t", "p", "quota", "exhausted", 429, telemetry=telemetry, origine="run_a")
    erreur = json.loads((tmp_path / "llm_errors.jsonl").read_text())
    assert erreur["origine"] == "run_a"


def test_redacteur_inchargeable_n_ecrit_pas_en_clair(tmp_path):
    telemetry = SimpleNamespace(exchanges_enabled=True, exchanges_file=None, workdir=tmp_path,
                                redactor="module.missing:Nothing", exchanges_max_bytes=0)
    log_llm_exchange("t", "p", [{"role": "user", "content": "secret"}], {}, 1, 1, telemetry=telemetry)
    assert not (tmp_path / "llm_exchanges.jsonl").exists()

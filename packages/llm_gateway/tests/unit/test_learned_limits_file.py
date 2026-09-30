"""The file store of learned limits: missing = empty, atomic write, re-read."""
from llm_gateway.testing import FileLearnedLimits


def test_fichier_absent_puis_ecrit_puis_relu(tmp_path):
    st = FileLearnedLimits(tmp_path / "sub" / "learned.json")
    assert st.all_max_output_tokens() == {}
    st.set_max_output_tokens("a", 8192)
    st.set_max_output_tokens("b", 1024)
    assert (tmp_path / "sub" / "learned.json").is_file()
    again = FileLearnedLimits(tmp_path / "sub" / "learned.json")
    assert again.get_max_output_tokens("a") == 8192 and again.all_max_output_tokens() == {"a": 8192, "b": 1024}


def test_fichier_corrompu_vaut_vide(tmp_path):
    p = tmp_path / "learned.json"
    p.write_text("{not json")
    assert FileLearnedLimits(p).all_max_output_tokens() == {}

"""configure_logging only touches loguru's default handler and undoes itself cleanly."""
from __future__ import annotations

import sys

from loguru import logger

from llm_gateway.telemetry import logger as tl


def test_les_sinks_de_l_hote_survivent(capsys):
    tl.reset_logging()
    host_id = logger.add(sys.stderr, level="INFO", format="HOST {message}")
    try:
        tl.configure_logging(level="INFO")
        tl.configure_logging(level="DEBUG")   # idempotent: a single gateway handler
        logger.info("hello")
        err = capsys.readouterr().err
        assert err.count("hello") == 2, "the host sink and the gateway sink both write"
        assert "HOST hello" in err
    finally:
        logger.remove(host_id)
        tl.reset_logging()


def test_reset_retire_ce_que_configure_a_pose(capsys):
    tl.reset_logging()
    tl.configure_logging(level="INFO")
    tl.reset_logging()
    logger.info("after reset")
    assert "after reset" not in capsys.readouterr().err

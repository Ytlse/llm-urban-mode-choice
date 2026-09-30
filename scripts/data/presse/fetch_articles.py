#!/usr/bin/env python3
"""Download the five press articles used by the press-event conditions.

The texts are not redistributed with this repository. This script reads
`data/presse/sources.yaml`, downloads each page to `data/presse/articles_html/`, and compares
its sha256 with the page archived for the paper. News sites change their pages over time, so a
different page hash is expected and only reported. Next step:

    python -m scripts.data.presse.extraire_textes extraire   # rebuilds brut.fr.txt
    python -m scripts.data.presse.extraire_textes verifier   # compares the excerpt hashes

The English excerpts (brut.txt) were machine-translated for the paper; translate brut.fr.txt
again and declare the new sha256 in the event configuration (services/llm-agents/config/evenements/).

Usage:
    python -m scripts.data.presse.fetch_articles [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PRESS = ROOT / "data" / "presse"
SOURCES = PRESS / "sources.yaml"
USER_AGENT = "Mozilla/5.0 (research reproduction; press-event corpus)"

log = logging.getLogger("fetch_articles")


def fetch(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--force", action="store_true", help="download again pages already present")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = time.time()
    articles = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))["articles"]
    counts = {"downloaded": 0, "skipped": 0, "same_page": 0, "changed_page": 0, "failed": 0}
    log.info("start: %d articles from %s", len(articles), SOURCES.relative_to(ROOT))
    for article_id, source in articles.items():
        target = PRESS / source["archived_html"]
        if target.is_file() and not args.force:
            counts["skipped"] += 1
            log.info("%s: already present (%s), skipped", article_id, target.name)
            continue
        try:
            page = fetch(source["url"])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            counts["failed"] += 1
            log.error("%s: download failed for %s: %s", article_id, source["url"], exc)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(page)
        counts["downloaded"] += 1
        if hashlib.sha256(page).hexdigest() == source["archived_html_sha256"]:
            counts["same_page"] += 1
            log.info("%s: %d bytes, identical to the archived page", article_id, len(page))
        else:
            counts["changed_page"] += 1
            log.info("%s: %d bytes, differs from the archived page (expected for live sites); "
                     "check the excerpt with `extraire_textes verifier`", article_id, len(page))
    log.info("done in %.1fs: %s", time.time() - start, counts)
    if counts["failed"]:
        log.error("[ALARME] %d of %d articles could not be downloaded", counts["failed"], len(articles))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

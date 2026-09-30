"""Per-simulated-day measurements of a memory run — ticket 093.

One computation, two outputs: the `ecriture` CSVs and the controller's Prometheus series
come from here, so they cannot diverge. The CSV is authoritative; Grafana is a dashboard
instrument whose x-axis is REAL time, not simulated time.
"""

from scripts.analysis.mesures.calcul import Mesures, calculer

__all__ = ["Mesures", "calculer"]

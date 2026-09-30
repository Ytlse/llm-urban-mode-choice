"""Terminal time per mode — access and egress of a vehicle trip.

SINGLE source of truth of the parameter described by ticket 013: the values, their
provenance and the rendering labels live in ``config/terminal_time.yaml``.
Three consumers share it, and that is what guarantees that they do not
diverge:

- :func:`trip_helper.osmnx_direct._make_travel_plan` — builds the access
  and egress legs (this is decision T3: the fix is in the
  scenario construction, not in the display template);
- ``text_helper/models/travel_plan.py`` — renders the breakdown;
- the cache keys (persistent OSMnx routing, LLM decisions) through
  :func:`data_version`.

The module is **pure**: no I/O beyond reading the YAML at first call,
no mutable state, hence testable without network or container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hashlib

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "terminal_time.yaml"

# Route markers of the terminal legs. They play the same role as
# ``DIRECT_ROUTE_MARKER``: make the leg recognisable without guessing from
# its mode. Terminal legs carry ``is_transfer=True``, which excludes them
# from ``TravelPlan.get_code()`` — CRITICAL invariant: the plan code is the key
# of the decision cache and of itinerary deduplication, it must not
# change because the display is broken down.
# Salt of the terminal time draw. Versioned: changing it reshuffles all terminal
# times, hence the plans AND the cached LLM decisions. Only to be moved together with
# `version` below.
DRAW_SALT = "terminal_time_v1"

TERMINAL_ACCESS_ROUTE = "__TERMINAL_ACCESS__"
TERMINAL_EGRESS_ROUTE = "__TERMINAL_EGRESS__"


@dataclass(frozen=True)
class TerminalProfile:
    """Terminal profile of a mode: durations PER ZONE (seconds) and rendering labels.

    The durations are ``{couronne: secondes}`` tables with a ``default`` entry.
    Access is priced on the ORIGIN ring (where the vehicle is parked), egress
    on the DESTINATION ring (where a space must be found): the two
    ends of a single trip can therefore be priced differently.
    """

    mode: str
    access_by_zone: dict[str, int]
    egress_by_zone: dict[str, int]
    provenance: str
    spatialise: bool
    labels: dict[str, str]
    # Laws per ring, `{couronne: {secondes: probabilité}}`. Present → the terminal
    # time is DRAWN from them; absent → the constant tables above are authoritative.
    # Both mechanisms coexist: car and bike are on a law since tt3,
    # a future mode can stay on a constant without anything changing for it.
    access_law_by_zone: dict[str, dict[int, float]] = field(default_factory=dict)
    egress_law_by_zone: dict[str, dict[int, float]] = field(default_factory=dict)

    def _draw(self, law: dict[int, float], key: str) -> int:
        """Inverse of the cumulative distribution function, on a hashed uniform.

        Deterministic and without RNG: the same trip always receives the same terminal
        time. This is not a convenience detail — the plans are cached
        (OTP) and so are the LLM decisions; a random draw would make
        a run diverge from its resumption, and would make the decision cache wrong.
        """
        digest = hashlib.sha256(f"{DRAW_SALT}:{key}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2 ** 64
        cumulated = 0.0
        for seconds, probability in sorted(law.items()):
            cumulated += probability
            if u < cumulated:
                return int(seconds)
        return int(max(law)) if law else 0

    def _law_for(self, laws: dict[str, dict[int, float]], zone: str):
        return laws.get(zone) or laws.get("default")

    def access_s(self, zone: str = "", key: str = "") -> int:
        """Access time in the origin ring (``default`` fallback).

        With a law served, ``key`` identifies the trip: two distinct trips
        draw independently, the same trip always draws the same. Without ``key``, the
        draw falls back on the ring alone — all the trips of a ring
        then receive the same value, which is a readable fallback and not a
        silence, but not what we want in production.
        """
        law = self._law_for(self.access_law_by_zone, zone)
        if law:
            return self._draw(law, f"{self.mode}:access:{zone}:{key}")
        return int(self.access_by_zone.get(zone, self.access_by_zone["default"]))

    def egress_s(self, zone: str = "", key: str = "") -> int:
        """Parking and walking time in the destination ring."""
        law = self._law_for(self.egress_law_by_zone, zone)
        if law:
            return self._draw(law, f"{self.mode}:egress:{zone}:{key}")
        return int(self.egress_by_zone.get(zone, self.egress_by_zone["default"]))

    def total_s(self, origin_zone: str = "", dest_zone: str = "",
                key: str = "") -> int:
        return self.access_s(origin_zone, key) + self.egress_s(dest_zone, key)

    def mean_s(self, end: str, zone: str = "") -> float:
        """Expectation of the terminal time — for reports, never for rendering."""
        laws = self.access_law_by_zone if end == "access" else self.egress_law_by_zone
        law = self._law_for(laws, zone)
        if law:
            return sum(s * p for s, p in law.items())
        table = self.access_by_zone if end == "access" else self.egress_by_zone
        return float(table.get(zone, table["default"]))

    def egress_label(self, destination: Optional[str]) -> str:
        """Label of the egress leg, naming the destination if known.

        ``purpose`` is only set on the plan after routing
        (``simulation_controller``), so the label cannot be frozen at
        construction: it carries a ``{destination}`` interpolated at rendering. Without a
        destination, we fall back on a wording that invents nothing — same
        graceful degradation as the public transport template.
        """
        if destination:
            return self.labels["egress"].format(destination=destination)
        return self.labels["egress_sans_destination"]


_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with _CONFIG_PATH.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _cache = _validate(raw)
    return _cache


def _validate(raw: dict) -> dict:
    """Refuse a configuration that would break the consistency of the rendering.

    The check on multiples of 60 s is not zeal: the rendering displays
    each component in TRUNCATED minutes, and the equality "displayed total = sum
    of the displayed sub-steps" (acceptance criterion 2 of ticket 013) only holds
    because ``floor(a + k×60) == floor(a) + k``. A value of 90 s would make it
    display breakdowns that do not sum to their total — a defect that
    would read as an inconsistency of the model, not as a configuration bug.
    """
    if not raw.get("version"):
        raise ValueError(
            f"{_CONFIG_PATH.name} : `version` missing. It enters the cache "
            f"keys (routing and LLM decisions); without it, a change of "
            f"terminal time would let outdated durations and decisions "
            f"be served.")

    profiles: dict[str, TerminalProfile] = {}
    for mode, cfg in (raw.get("modes") or {}).items():
        # Laws per ring (`*_law`), served since tt3. They replace the constant
        # when present: the mean measured on EMC² is BELOW ONE MINUTE
        # (0.36 min of access in Toulouse), and the rendering can only display whole
        # minutes. A constant would therefore be 0 everywhere, which would erase a
        # very real tail — 2 to 4% of trips really have 5 minutes or more. The
        # draw keeps both, the mean AND the tail.
        laws: dict[str, dict[str, dict[int, float]]] = {}
        for name in ("access_law", "egress_law"):
            raw_law = cfg.get(name)
            if raw_law is None:
                laws[name] = {}
                continue
            if not isinstance(raw_law, dict) or "default" not in raw_law:
                raise ValueError(
                    f"{_CONFIG_PATH.name} : {mode}.{name} must be a table "
                    f"{{ring: {{minutes: probability}}}} with a `default` entry "
                    f"— a zone outside the EMC² layer would otherwise fall into nothing.")
            built: dict[str, dict[int, float]] = {}
            for zone, pmf in raw_law.items():
                if not isinstance(pmf, dict) or not pmf:
                    raise ValueError(
                        f"{_CONFIG_PATH.name}: {mode}.{name}.{zone} empty. Drawing from "
                        f"an empty law would return 0 — a plausible value, hence an "
                        f"undetectable fallback.")
                total = 0.0
                cell: dict[int, float] = {}
                for minutes, probability in pmf.items():
                    minutes, probability = int(minutes), float(probability)
                    if minutes < 0 or probability < 0:
                        raise ValueError(
                            f"{_CONFIG_PATH.name} : {mode}.{name}.{zone} — negative "
                            f"value ({minutes} min, p={probability}).")
                    # The keys are in MINUTES: converted to seconds here, they are
                    # multiples of 60 by construction, which preserves the rendering
                    # invariant without having to check it.
                    cell[minutes * 60] = probability
                    total += probability
                if abs(total - 1.0) > 1e-3:
                    raise ValueError(
                        f"{_CONFIG_PATH.name}: {mode}.{name}.{zone} sums to {total:.4f} "
                        f"and not 1. A law that does not sum to 1 silences part "
                        f"of the mass without saying so.")
                built[zone] = cell
            laws[name] = built

        tables: dict[str, dict[str, int]] = {}
        for name in ("access_s", "egress_s"):
            table = cfg.get(name)
            if table is None and laws[name.replace("_s", "_law")]:
                # Mode served by a law: the constant becomes optional. We keep
                # a `default` at 0 so that `access_s`/`egress_s` stay callable
                # without a law (fallback of `_law_for` on a missing ring).
                tables[name] = {"default": 0}
                continue
            if not isinstance(table, dict):
                raise ValueError(
                    f"{_CONFIG_PATH.name} : {mode}.{name} must be a table "
                    f"{{ring: seconds}} with a `default` entry "
                    f"(the parameter is spatialised since version tt2).")
            if "default" not in table:
                raise ValueError(
                    f"{_CONFIG_PATH.name} : {mode}.{name} without a `default` entry — "
                    f"an unknown zone would fall into nothing. Yet a zone is "
                    f"unknown as soon as a point leaves the EMC² layer, which happens.")
            for zone, value in table.items():
                value = int(value)
                if value < 0:
                    raise ValueError(
                        f"{_CONFIG_PATH.name} : {mode}.{name}.{zone} negative ({value}).")
                if value % 60:
                    raise ValueError(
                        f"{_CONFIG_PATH.name}: {mode}.{name}.{zone} = {value} s is "
                        f"not a multiple of 60. The displayed total would no longer be the "
                        f"sum of the displayed sub-steps (ticket 013, criterion 2).")
            tables[name] = {z: int(v) for z, v in table.items()}
        labels = cfg.get("labels") or {}
        missing = {"access", "main", "egress", "egress_sans_destination",
                   "terminal"} - set(labels)
        if missing:
            raise ValueError(
                f"{_CONFIG_PATH.name}: missing labels for {mode}: "
                f"{sorted(missing)}.")
        profiles[mode] = TerminalProfile(
            mode=mode, access_by_zone=tables["access_s"],
            egress_by_zone=tables["egress_s"],
            provenance=str(cfg.get("provenance", "unsourced")),
            spatialise=bool(cfg.get("spatialise", False)), labels=dict(labels),
            access_law_by_zone=laws["access_law"],
            egress_law_by_zone=laws["egress_law"])

    if not raw.get("routing_version"):
        raise ValueError(
            f"{_CONFIG_PATH.name}: `routing_version` missing. It indexes the OSMnx "
            f"routing cache, which stores pure network time — confusing it with "
            f"`version` would recompute all routes at each adjustment of the "
            f"terminal time (~2 h for 930 personas).")
    return {"version": str(raw["version"]), "base_version": str(raw["version"]),
            "routing_version": str(raw["routing_version"]),
            "modes": profiles,
            # CENTRAL profiles kept apart: `apply_variant` scales
            # from them and never from `modes`, otherwise two successive switches
            # would multiply their factors (high then low → 0.75 instead of 0.5).
            # A shallow copy is enough: `TerminalProfile` is frozen and the zone
            # tables are never mutated in place.
            "base_modes": dict(profiles),
            "sensitivity": raw.get("sensitivity") or {}}


def data_version() -> str:
    """Version of the itinerary data, to include in every cache key.

    Two caches survive across runs and are BLIND to terminal time if they are
    not versioned:

    - the persistent OSMnx cache is addressed by (mode, coordinates, slot): it
      would serve again durations computed under the former parameterisation;
    - the LLM decision cache is addressed by ``TravelPlan.get_code()``, i.e.
      route + stops — insensitive to durations by construction. It would therefore replay
      decisions taken on options that no longer exist as such.

    The second is the most serious: nothing would report it in the logs. Hence an
    explicit version, to bump with any modification of the values.
    """
    return _load()["version"]


def routing_version() -> str:
    """Version of the NETWORK travel time — key of the OSMnx routing cache.

    Separated from :func:`data_version` on purpose. This cache only stores network
    durations, independent of terminal time: indexing them on the terminal time
    version would make thousands of routes be recomputed cold at each parking
    adjustment, for an identical result. Only bump if the network duration
    changes (speeds, penalties, congestion).
    """
    return _load()["routing_version"]


def terminal_profile(trip_mode: str) -> Optional[TerminalProfile]:
    """Profile of the mode, or ``None`` if it has no terminal time.

    ``None`` is the case of walking (door to door by nature) and of public
    transport (their access walking legs are ALREADY routed by OTP — adding
    some would be the double counting that acceptance criterion 4 forbids).
    """
    return _load()["modes"].get(trip_mode)


def sensitivity_variants() -> dict[str, dict]:
    """Sensitivity grid (ticket 013, T6): ``{nom: {mode: {access_s, …}}}``."""
    return dict(_load()["sensitivity"])


def apply_variant(name: str) -> None:
    """Switch the profiles to a variant of the sensitivity grid.

    Reserved for sensitivity analysis (T6) and tests: production always reads
    the central values of the file. The variant name is carried over
    into :func:`data_version`, otherwise the three sensitivity sets
    would share the cache keys of the central version and get mixed up.

    Scaling starts from the CENTRAL profiles (``base_modes``), not from the current
    profiles: called twice in a row — which is precisely what a loop over the
    T6 grid does — the version did restart from the base but the VALUES
    piled up (``high`` then ``low`` gave 1.5 × 0.5 = 0.75). The sensitivity
    measurement would then have been about terminal times that no variant
    declares, under a correct variant label.
    """
    conf = _load()
    variant = (conf["sensitivity"] or {}).get(name)
    if variant is None:
        raise KeyError(f"unknown sensitivity variant: {name!r} "
                       f"(known: {sorted(conf['sensitivity'])})")
    for mode, profile in list(conf["base_modes"].items()):
        override = variant.get(mode) or {}
        # A variant applies a uniform FACTOR on all the rings rather
        # than a single value: otherwise it would overwrite the spatialisation, and the
        # sensitivity would measure "spatialised or not" at the same time as "more or
        # less terminal time" — two variables for one conclusion.
        factor = float(override.get("factor", 1.0))

        def _scaled(table: dict[str, int]) -> dict[str, int]:
            return {z: int(round(v * factor / 60.0)) * 60 for z, v in table.items()}

        def _scaled_laws(laws: dict[str, dict[int, float]]
                         ) -> dict[str, dict[int, float]]:
            """Scale the LAW, not only the constants.

            ⚠ Without this, a variant rebuilt the profile forgetting the law
            fields: the laws disappeared and the terminal time fell back on the
            constants — zero since tt3. The sensitivity grid would have measured
            "with or without terminal time" instead of "more or less", under a
            correct variant label. It is exactly the silence that the
            spatialisation had already nearly introduced.

            The scaled seconds are brought back to the nearest multiple of 60,
            and the keys that collide (×0.5 sends 1 min and 0 min to 0)
            see their masses **add up**: the law always sums to 1.
            """
            out: dict[str, dict[int, float]] = {}
            for zone, pmf in laws.items():
                scaled: dict[int, float] = {}
                for seconds, probability in pmf.items():
                    key = int(round(seconds * factor / 60.0)) * 60
                    scaled[key] = scaled.get(key, 0.0) + probability
                out[zone] = scaled
            return out

        conf["modes"][mode] = TerminalProfile(
            mode=mode, access_by_zone=_scaled(profile.access_by_zone),
            egress_by_zone=_scaled(profile.egress_by_zone),
            provenance=profile.provenance, spatialise=profile.spatialise,
            labels=profile.labels,
            access_law_by_zone=_scaled_laws(profile.access_law_by_zone),
            egress_law_by_zone=_scaled_laws(profile.egress_law_by_zone))
    # Restart from the BASE version and not from the current one: two successive
    # switches must not pile up the suffixes.
    conf["version"] = f"{conf['base_version']}-{name}"


def reset() -> None:
    """Empty the configuration cache (tests, and return to central values)."""
    global _cache
    _cache = None

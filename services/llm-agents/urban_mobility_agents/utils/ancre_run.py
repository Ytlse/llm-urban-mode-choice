"""The simulated instant at which this run started — ticket 075.

WHY A MODULE FOR SO LITTLE
--------------------------
Three mechanisms need to know **how many simulated days have elapsed** since the start of
the run: the progression of the weather date (`weather_draw`), the daily resume point, and
the memory log that dates its sections. The controller already knows this instant — it
keeps it in `_sim_start_ts` to date accidents — but it is out of reach of a decision's path,
which goes through the agent and not the controller. Passing it down as a parameter would
have meant touching a dozen signatures for a value that never changes during the whole
run.

⚠ **The anchor is SET, it is not guessed.** Without it, `jours_ecoules` returns zero: the
first day of the run, which is the only safe value and reproduces the behaviour from before
the ticket. It SAYS so once, because a missing anchor on a long run would keep re-reading
the first day's forecast forever without any line reporting it.

⚠ **On a hot resume, the anchor comes from the resume point, not from the first observed
timestamp.** GAMA replays from its t0: re-anchoring on what is observed would rewind
everyone's weather, and the sixty days would stop being consecutive.
"""

from __future__ import annotations

from loguru import logger
from sim_clock import wall_clock

_ancre: int | None = None
_absence_signalee = False


def ancrer(timestamp: int, *, origine: str = "premier timestamp observé") -> None:
    """Sets the start instant of the run. No effect if an anchor is already set.

    The first anchor wins: a resume restores its own BEFORE the replay makes its first
    timestamp observed, and the restored anchor must not be overwritten by it.
    """
    global _ancre
    if _ancre is not None:
        if int(timestamp) != _ancre:
            logger.info(
                f"[ancre] anchor already set at {wall_clock(_ancre)} — the proposal "
                f"{wall_clock(int(timestamp))} ({origine}) is ignored."
            )
        return
    _ancre = int(timestamp)
    logger.info(f"[ancre] début du run ancré au {wall_clock(_ancre)} ({origine}).")


def ancre() -> int | None:
    """The start instant of the run, or `None` if not yet known."""
    return _ancre


def reinitialiser() -> None:
    """Forgets the anchor. Reserved for tests and the end of a run."""
    global _ancre, _absence_signalee
    _ancre = None
    _absence_signalee = False


def jours_ecoules(timestamp: int) -> int:
    """WHOLE simulated days elapsed since the start of the run, never negative.

    Counted in wall-CALENDAR days and not in 86,400 s slices: the run starts at 5 a.m.,
    and a slice would change day at 5 a.m. the next day, in the middle of the peak. The
    simulated day must change when the date changes, as for a human.
    """
    global _absence_signalee
    if _ancre is None:
        if not _absence_signalee:
            _absence_signalee = True
            logger.warning(
                "[ancre] aucune ancre posée : tout ce qui dépend du nombre de jours écoulés "
                "se comporte comme au premier jour du run (progression météo figée). "
                "Attendu seulement avant le premier /sync."
            )
        return 0
    ecart = (wall_clock(int(timestamp)).date() - wall_clock(_ancre).date()).days
    return max(0, ecart)

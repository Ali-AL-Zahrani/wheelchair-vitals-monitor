"""
Feature extraction for the rehabilitation model.

The model never touches a sensor. It reads the two files this system already
produces and turns a whole session into one small fixed vector describing
**this user**:

  - `measurements.csv`   — validated readings only (an empty cell = no reading)
  - `audit_log.jsonl`    — why a reading was withheld, and every alert with its duration

That is the point of the split: the rehabilitation model learns from data that
has already passed the validator, so a lifted wrist, a frozen sensor or a
tremor cannot enter its training set as if it were a measurement.

Two rules are enforced here and tested:

1. **An empty cell is never read as a zero.** A heart rate of 0 is a reading;
   an empty cell is the absence of one. Averaging them together would pull the
   user's baseline towards zero and invent a resting pulse nobody has.
2. **A feature that could not be measured is filled from `NEUTRAL`, not with 0.**
   Zero is an extreme value to a decision tree, so it would be read as a
   finding rather than as missing data. `no_reading_ratio` is itself a feature,
   so the model can see that a session was sparse.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from validator import ALERT_LIFT_WRIST, ALERT_NEEDS_MOVEMENT, Status

# The order is the model's input order — it is part of the saved model and must
# not be reordered without retraining.
FEATURE_NAMES: Tuple[str, ...] = (
    "hr_mean",                    # resting baseline: the user's own pulse, not a population average
    "hr_sd",                      # how steady that pulse is across the session
    "spo2_mean",
    "spo2_min",                   # the worst accepted reading, not only the average
    "skin_temp_mean",
    "movement_mean",              # general activity level from the IMU
    "valid_ratio",                # share of wrist readings accepted
    "warn_ratio",                 # share outside the clinical range but still real
    "no_reading_ratio",           # share withheld (rejected or no contact)
    "immobility_max_min",         # longest continuous stretch without confirmed movement
    "movement_alerts_per_hour",   # how often the user had to be prompted to move
    "response_min",               # how long the prompt stayed up before they moved
    "wrist_alerts_per_hour",      # how often continuous wrist rest hit its limit
)

# Filled in when a channel produced nothing usable in the whole session.
# Deliberately plausible resting values, never zero — see rule 2 above.
NEUTRAL: Dict[str, float] = {
    "hr_mean": 75.0,
    "hr_sd": 0.0,
    "spo2_mean": 97.0,
    "spo2_min": 97.0,
    "skin_temp_mean": 33.4,
    "movement_mean": 0.0,
    "valid_ratio": 0.0,
    "warn_ratio": 0.0,
    "no_reading_ratio": 1.0,      # nothing arrived — that is the honest reading of an empty session
    "immobility_max_min": 0.0,
    "movement_alerts_per_hour": 0.0,
    # No movement alert ever fired ⇒ there was no prompt left waiting. Zero is
    # the correct value here, not missing data.
    "response_min": 0.0,
    "wrist_alerts_per_hour": 0.0,
}

# Wrist-derived fields only. `movement` comes from a separate device and is
# never displayed to the user, so it does not belong in the acceptance ratios.
_WRIST_FIELDS: Tuple[str, ...] = ("heart_rate", "spo2", "skin_temp")

# CSV column per field, matching exporter.py.
_COLUMN: Dict[str, str] = {
    "heart_rate": "heart_rate_bpm",
    "spo2": "spo2_pct",
    "skin_temp": "skin_temp_c",
    "movement": "movement_idx",
}

# A WARN reading was measured and shown to the user, so it belongs in the
# baseline. INVALID and NO_CONTACT were withheld and must not.
_KEPT = (Status.VALID.value, Status.WARN.value)


@dataclass(frozen=True)
class SessionFeatures:
    """One session reduced to the model's input, plus the context needed to read it."""

    values: Dict[str, float] = field(default_factory=dict)
    session_hours: float = 0.0    # context, not a model input: length must not decide a programme
    samples: int = 0

    def vector(self) -> List[float]:
        return [self.values[name] for name in FEATURE_NAMES]

    def __getitem__(self, name: str) -> float:
        return self.values[name]


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(csv_text: str, log_text: str = "",
                     session_id: Optional[str] = None) -> SessionFeatures:
    """
    Build the feature vector from the text of the two session files.

    `session_id` filters the audit log: it is appended across sessions, so
    without the filter one user's alerts would be counted into another's.
    """
    rows = _read_rows(csv_text)
    times = [r["t"] for r in rows if r["t"] is not None]
    start_t = min(times) if times else 0.0
    end_t = max(times) if times else 0.0
    hours = max(0.0, (end_t - start_t) / 3600.0)

    values: Dict[str, float] = dict(NEUTRAL)

    hr = _kept_values(rows, "heart_rate")
    spo2 = _kept_values(rows, "spo2")
    temp = _kept_values(rows, "skin_temp")
    movement = _kept_values(rows, "movement")

    if hr:
        values["hr_mean"] = statistics.fmean(hr)
        values["hr_sd"] = statistics.pstdev(hr) if len(hr) > 1 else 0.0
    if spo2:
        values["spo2_mean"] = statistics.fmean(spo2)
        values["spo2_min"] = min(spo2)
    if temp:
        values["skin_temp_mean"] = statistics.fmean(temp)
    if movement:
        values["movement_mean"] = statistics.fmean(movement)

    counted, valid, warn = 0, 0, 0
    for row in rows:
        for name in _WRIST_FIELDS:
            status = row["status"].get(name, "")
            if not status:
                continue
            counted += 1
            if status == Status.VALID.value:
                valid += 1
            elif status == Status.WARN.value:
                warn += 1
    if counted:
        values["valid_ratio"] = valid / counted
        values["warn_ratio"] = warn / counted
        values["no_reading_ratio"] = (counted - valid - warn) / counted

    immobility = [r["immobility_s"] for r in rows if r["immobility_s"] is not None]
    if immobility:
        values["immobility_max_min"] = max(immobility) / 60.0

    raised, durations = _alerts(log_text, session_end_t=end_t, session_id=session_id)
    if hours > 0.0:
        values["movement_alerts_per_hour"] = raised.get(ALERT_NEEDS_MOVEMENT, 0) / hours
        values["wrist_alerts_per_hour"] = raised.get(ALERT_LIFT_WRIST, 0) / hours
    response = durations.get(ALERT_NEEDS_MOVEMENT, [])
    if response:
        values["response_min"] = statistics.fmean(response) / 60.0

    return SessionFeatures(values=values, session_hours=hours, samples=len(rows))


def extract_features_from_files(csv_path: str, log_path: Optional[str] = None,
                                session_id: Optional[str] = None) -> SessionFeatures:
    """Same thing, reading the files from disk. A missing audit log is tolerated."""
    with open(csv_path, encoding="utf-8") as fh:
        csv_text = fh.read()
    log_text = ""
    if log_path:
        try:
            with open(log_path, encoding="utf-8") as fh:
                log_text = fh.read()
        except OSError:
            log_text = ""
    return extract_features(csv_text, log_text, session_id=session_id)


def feature_table(sessions: Sequence[SessionFeatures]) -> List[List[float]]:
    """Feature matrix in FEATURE_NAMES order — what the model is fitted on."""
    return [s.vector() for s in sessions]


# ═══════════════════════════════════════════════════════════════════════════
#  Internal
# ═══════════════════════════════════════════════════════════════════════════

def _read_rows(csv_text: str) -> List[Dict[str, object]]:
    """
    Parse measurements.csv by column name, never by position, so adding a
    column to the export cannot silently shift a reading into another feature.
    """
    rows: List[Dict[str, object]] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for raw in reader:
        row: Dict[str, object] = {
            "t": _number(raw.get("t_s")),
            "immobility_s": _number(raw.get("immobility_s")),
            "value": {},
            "status": {},
        }
        for name, column in _COLUMN.items():
            # The empty string stays None: absence of a measurement, not a zero.
            row["value"][name] = _number(raw.get(column))          # type: ignore[index]
            row["status"][name] = (raw.get(f"{name}_status") or "").strip()  # type: ignore[index]
        rows.append(row)
    return rows


def _kept_values(rows: List[Dict[str, object]], name: str) -> List[float]:
    """Values the validator accepted (VALID or WARN). A withheld cell contributes nothing."""
    out: List[float] = []
    for row in rows:
        status = row["status"].get(name, "")        # type: ignore[union-attr]
        value = row["value"].get(name)              # type: ignore[union-attr]
        if status in _KEPT and value is not None:
            out.append(float(value))
    return out


def _alerts(log_text: str, session_end_t: float,
            session_id: Optional[str]) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """
    Count raised alerts and measure how long each one stayed up.

    An alert still active when the session ended is not dropped: its duration
    is measured to the end of the session. Ignoring it would reward the user
    who never responded at all with no response time.

    A corrupt line is skipped, exactly as in `logger.summarize` — one bad line
    must not discard the events before it.
    """
    raised: Dict[str, int] = {}
    durations: Dict[str, List[float]] = {}
    pending: Dict[str, List[float]] = {}

    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if session_id is not None and ev.get("session_id") != session_id:
            continue

        kind = ev.get("type")
        alert = ev.get("alert")
        t = _number(ev.get("t"))
        if not isinstance(alert, str) or t is None:
            continue

        if kind == "alert_raised":
            raised[alert] = raised.get(alert, 0) + 1
            pending.setdefault(alert, []).append(t)
        elif kind == "alert_cleared":
            logged = _number(ev.get("duration_s"))
            start = pending.get(alert, []).pop(0) if pending.get(alert) else None
            # The logged duration is authoritative; the pairing is the fallback.
            duration = logged if logged is not None else (
                t - start if start is not None else None)
            if duration is not None:
                durations.setdefault(alert, []).append(max(0.0, duration))

    for alert, starts in pending.items():
        for start in starts:
            durations.setdefault(alert, []).append(max(0.0, session_end_t - start))

    return raised, durations


def _number(raw: object) -> Optional[float]:
    """
    Text to number, with absence preserved.

    None is returned for an empty cell, a missing column and an unparsable
    value alike — all three mean "no number here", and the caller must not be
    able to confuse any of them with zero.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        return None if math.isnan(value) or math.isinf(value) else value
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) or math.isinf(value) else value

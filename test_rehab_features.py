"""
Feature-extraction tests.

The feature vector is where a withheld reading could quietly become a number.
These tests exist to prove it cannot: an empty cell stays absent, a rejected
reading never enters a baseline, and an alert the user never answered is not
counted as answered instantly.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import pytest

from exporter import header_row
from rehab_features import (
    FEATURE_NAMES,
    NEUTRAL,
    extract_features,
    extract_features_from_files,
)
from validator import ALERT_LIFT_WRIST, ALERT_NEEDS_MOVEMENT

SAMPLE_PERIOD_S = 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  Builders — a session written exactly as exporter.py and logger.py write it
# ═══════════════════════════════════════════════════════════════════════════

def row(i: int, hr: Optional[float] = 72.0, hr_status: str = "VALID",
        spo2: Optional[float] = 97.0, spo2_status: str = "VALID",
        temp: Optional[float] = 33.4, temp_status: str = "VALID",
        movement: Optional[float] = 0.2, movement_status: str = "VALID",
        immobility_s: float = 0.0) -> Dict[str, object]:
    return {
        "t_s": i * SAMPLE_PERIOD_S,
        "contact": 1,
        "heart_rate_bpm": "" if hr is None else hr,
        "heart_rate_status": hr_status,
        "spo2_pct": "" if spo2 is None else spo2,
        "spo2_status": spo2_status,
        "skin_temp_c": "" if temp is None else temp,
        "skin_temp_status": temp_status,
        "movement_idx": "" if movement is None else movement,
        "movement_status": movement_status,
        "immobility_s": immobility_s,
        "wrist_rest_s": 0.0,
        "silence_s": 0.0,
        "flags": "",
        "alerts": "",
    }


def csv_text(rows: List[Dict[str, object]], columns: Optional[List[str]] = None) -> str:
    columns = columns or header_row()
    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in columns))
    return "\n".join(lines) + "\n"


def log_text(events: List[Dict[str, object]], session_id: str = "s1") -> str:
    out = []
    for ev in events:
        ev = dict(ev)
        ev.setdefault("session_id", session_id)
        out.append(json.dumps(ev))
    return "\n".join(out) + "\n"


def raised(alert: str, t: float) -> Dict[str, object]:
    return {"type": "alert_raised", "t": t, "alert": alert}


def cleared(alert: str, t: float, duration_s: float) -> Dict[str, object]:
    return {"type": "alert_cleared", "t": t, "alert": alert, "duration_s": duration_s}


# ═══════════════════════════════════════════════════════════════════════════
#  The safety rule: an empty cell is not a zero
# ═══════════════════════════════════════════════════════════════════════════

def test_an_empty_cell_never_enters_the_baseline():
    """
    Two readings of 70 and 80 with eight withheld samples between them.
    The mean must be 75 — reading the empty cells as zeros would give 15.
    """
    rows = [row(0, hr=70.0)]
    rows += [row(i, hr=None, hr_status="NO_CONTACT") for i in range(1, 9)]
    rows += [row(9, hr=80.0)]

    features = extract_features(csv_text(rows))
    assert features["hr_mean"] == pytest.approx(75.0)


def test_a_rejected_reading_is_excluded_even_when_its_value_is_present():
    """A value the validator discarded must not reach the model, whatever the cell holds."""
    rows = [row(0, hr=72.0), row(1, hr=320.0, hr_status="INVALID"), row(2, hr=74.0)]
    features = extract_features(csv_text(rows))
    assert features["hr_mean"] == pytest.approx(73.0)


def test_a_warn_reading_is_included_because_it_was_measured_and_shown():
    """
    WARN means abnormal but real, and it was displayed to the user — excluding
    it would erase exactly the readings that matter most from the baseline.
    """
    rows = [row(0, spo2=97.0), row(1, spo2=89.0, spo2_status="WARN")]
    features = extract_features(csv_text(rows))
    assert features["spo2_mean"] == pytest.approx(93.0)
    assert features["spo2_min"] == pytest.approx(89.0)
    assert features["warn_ratio"] == pytest.approx(1 / 6)   # one of six wrist readings


def test_a_session_with_nothing_usable_falls_back_to_neutral_not_zero():
    """
    A whole session withheld must not produce a resting heart rate of zero.
    `no_reading_ratio` is what carries the emptiness to the model.
    """
    rows = [row(i, hr=None, hr_status="NO_CONTACT", spo2=None, spo2_status="NO_CONTACT",
                temp=None, temp_status="NO_CONTACT") for i in range(6)]
    features = extract_features(csv_text(rows))
    assert features["hr_mean"] == NEUTRAL["hr_mean"] != 0.0
    assert features["no_reading_ratio"] == pytest.approx(1.0)
    assert features["valid_ratio"] == pytest.approx(0.0)


def test_an_empty_file_produces_neutral_features_not_a_crash():
    features = extract_features(csv_text([]))
    assert features.samples == 0
    assert features.vector() == [NEUTRAL[n] for n in FEATURE_NAMES]


# ═══════════════════════════════════════════════════════════════════════════
#  Ratios and immobility
# ═══════════════════════════════════════════════════════════════════════════

def test_ratios_cover_wrist_fields_only_and_sum_to_one():
    """
    Movement comes from a separate device and is never shown to the user, so it
    does not belong in the acceptance ratios — counting it would dilute exactly
    the number that says how much of the wrist signal was usable.
    """
    rows = [row(0), row(1, hr=None, hr_status="INVALID"), row(2, spo2=89.0, spo2_status="WARN")]
    features = extract_features(csv_text(rows))
    total = (features["valid_ratio"] + features["warn_ratio"]
             + features["no_reading_ratio"])
    assert total == pytest.approx(1.0)
    assert features["valid_ratio"] == pytest.approx(7 / 9)
    assert features["no_reading_ratio"] == pytest.approx(1 / 9)


def test_immobility_takes_the_longest_stretch_not_the_last_one():
    rows = [row(0, immobility_s=1_200.0), row(1, immobility_s=0.0),
            row(2, immobility_s=300.0)]
    features = extract_features(csv_text(rows))
    assert features["immobility_max_min"] == pytest.approx(20.0)


def test_columns_are_read_by_name_not_by_position():
    """Adding or moving an export column must not shift one reading into another feature."""
    rows = [row(0, hr=70.0, spo2=95.0)]
    reordered = list(reversed(header_row()))
    features = extract_features(csv_text(rows, columns=reordered))
    assert features["hr_mean"] == pytest.approx(70.0)
    assert features["spo2_mean"] == pytest.approx(95.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Alerts
# ═══════════════════════════════════════════════════════════════════════════

def test_alert_rates_are_per_hour_so_session_length_does_not_decide_the_programme():
    rows = [row(i) for i in range(121)]        # 120 × 30 s = one hour
    log = log_text([
        raised(ALERT_NEEDS_MOVEMENT, 900.0), cleared(ALERT_NEEDS_MOVEMENT, 1_080.0, 180.0),
        raised(ALERT_NEEDS_MOVEMENT, 2_400.0), cleared(ALERT_NEEDS_MOVEMENT, 2_460.0, 60.0),
        raised(ALERT_LIFT_WRIST, 1_500.0), cleared(ALERT_LIFT_WRIST, 1_530.0, 30.0),
    ])
    features = extract_features(csv_text(rows), log)
    assert features.session_hours == pytest.approx(1.0)
    assert features["movement_alerts_per_hour"] == pytest.approx(2.0)
    assert features["wrist_alerts_per_hour"] == pytest.approx(1.0)
    assert features["response_min"] == pytest.approx(2.0)      # mean of 180 s and 60 s


def test_an_alert_never_answered_is_measured_to_the_end_of_the_session():
    """
    Dropping an alert that was still active at the end would reward the user
    who never responded with no response time at all — the exact opposite of
    what the indicator is for.
    """
    rows = [row(i) for i in range(121)]                        # ends at t = 3600
    log = log_text([raised(ALERT_NEEDS_MOVEMENT, 1_800.0)])    # raised, never cleared
    features = extract_features(csv_text(rows), log)
    assert features["response_min"] == pytest.approx(30.0)


def test_events_from_another_session_are_not_counted_into_this_user():
    """The audit log is appended across sessions — without the filter, users bleed into each other."""
    rows = [row(i) for i in range(121)]
    log = (log_text([raised(ALERT_NEEDS_MOVEMENT, 900.0),
                     cleared(ALERT_NEEDS_MOVEMENT, 960.0, 60.0)], session_id="mine")
           + log_text([raised(ALERT_NEEDS_MOVEMENT, 900.0),
                       raised(ALERT_NEEDS_MOVEMENT, 1_800.0)], session_id="someone_else"))
    features = extract_features(csv_text(rows), log, session_id="mine")
    assert features["movement_alerts_per_hour"] == pytest.approx(1.0)
    assert features["response_min"] == pytest.approx(1.0)


def test_a_corrupt_log_line_is_skipped_not_fatal():
    """Same tolerance as logger.summarize: one truncated line must not discard the events before it."""
    rows = [row(i) for i in range(121)]
    log = (log_text([raised(ALERT_NEEDS_MOVEMENT, 900.0)]).rstrip("\n")
           + '\n{"type": "alert_rai\n'
           + log_text([cleared(ALERT_NEEDS_MOVEMENT, 1_020.0, 120.0)]))
    features = extract_features(csv_text(rows), log)
    assert features["movement_alerts_per_hour"] == pytest.approx(1.0)
    assert features["response_min"] == pytest.approx(2.0)


def test_a_missing_audit_log_does_not_stop_feature_extraction(tmp_path):
    """Readings without an audit log are still readings — the alert features simply stay neutral."""
    path = tmp_path / "measurements.csv"
    path.write_text(csv_text([row(0, hr=70.0), row(1, hr=80.0)]), encoding="utf-8")
    features = extract_features_from_files(str(path), str(tmp_path / "missing.jsonl"))
    assert features["hr_mean"] == pytest.approx(75.0)
    assert features["movement_alerts_per_hour"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Contract
# ═══════════════════════════════════════════════════════════════════════════

def test_every_named_feature_is_produced_as_a_finite_number():
    """A missing or non-finite feature would reach the model as a silent hole."""
    rows = [row(0), row(1, hr=None, hr_status="INVALID"), row(2, spo2=89.0, spo2_status="WARN")]
    features = extract_features(csv_text(rows), log_text([raised(ALERT_LIFT_WRIST, 60.0)]))
    assert set(features.values) == set(FEATURE_NAMES)
    for name, value in features.values.items():
        assert isinstance(value, float), name
        assert value == value and abs(value) != float("inf"), name


def test_the_vector_follows_the_declared_feature_order():
    features = extract_features(csv_text([row(0, hr=70.0)]))
    vector = features.vector()
    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("hr_mean")] == pytest.approx(70.0)

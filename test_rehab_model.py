"""
Rehabilitation-model tests.

Two separate things are proven here, and keeping them apart is the point:

  - the **clinical criteria** (`PROTOCOL_RULES` / `clinical_label`) place a set
    of indicators in the right programme, including exactly on a band boundary —
    an implicit boundary is where a silent misclassification hides;
  - the **model** trained on those labels reproduces them on sessions it never
    saw, explains its answer, and refuses to answer at all when it is not fitted
    or when the feature list it was trained on no longer matches.
"""

from __future__ import annotations

from typing import Dict

import pytest

from rehab_features import FEATURE_NAMES, NEUTRAL
from rehab_model import (
    PROGRAM_INTENSIVE,
    PROGRAM_LIGHT,
    PROGRAM_MODERATE,
    PROGRAMS,
    RehabModel,
    clinical_label,
    protocol_score,
)


def values(**overrides: float) -> Dict[str, float]:
    """A user with nothing remarkable, then whatever the test changes."""
    base = dict(NEUTRAL)
    base["movement_mean"] = 0.20       # active enough not to score on stillness
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
#  Clinical criteria
# ═══════════════════════════════════════════════════════════════════════════

def test_a_stable_active_user_gets_the_light_programme():
    assert clinical_label(values()) == PROGRAM_LIGHT


def test_long_immobility_with_a_slow_response_reaches_the_moderate_programme():
    assert clinical_label(values(immobility_max_min=20.0, response_min=4.0,
                                 warn_ratio=0.05)) == PROGRAM_MODERATE


def test_prolonged_immobility_with_abnormal_readings_reaches_the_intensive_programme():
    assert clinical_label(values(immobility_max_min=30.0, response_min=12.0,
                                 warn_ratio=0.15, spo2_mean=93.0)) == PROGRAM_INTENSIVE


def test_the_band_boundaries_are_explicit():
    """
    A boundary left to chance is a silent misclassification: the user who sits
    exactly on it is the one the criteria were written for.
    """
    two_points = values(immobility_max_min=15.0, response_min=3.0)          # 1 + 1
    assert protocol_score(two_points)[0] == 2
    assert clinical_label(two_points) == PROGRAM_LIGHT                      # 2 is still light

    three_points = values(immobility_max_min=15.0, response_min=3.0, warn_ratio=0.03)
    assert protocol_score(three_points)[0] == 3
    assert clinical_label(three_points) == PROGRAM_MODERATE                 # 3 is no longer light

    five_points = values(immobility_max_min=25.0, response_min=8.0, warn_ratio=0.03)
    assert protocol_score(five_points)[0] == 5
    assert clinical_label(five_points) == PROGRAM_MODERATE                  # 5 is still moderate

    six_points = values(immobility_max_min=25.0, response_min=8.0, warn_ratio=0.03,
                        wrist_alerts_per_hour=2.0)
    assert protocol_score(six_points)[0] == 6
    assert clinical_label(six_points) == PROGRAM_INTENSIVE                  # 6 crosses over


def test_one_indicator_scores_once_however_far_past_the_threshold_it_is():
    """
    The steps of a rule are ordered strongest first and only one may fire.
    Adding them up would let a single extreme indicator decide the programme
    on its own.
    """
    score, fired = protocol_score(values(immobility_max_min=90.0))
    assert score == 2
    assert len([f for f in fired if f.startswith("immobility_max_min")]) == 1


def test_low_oxygen_scores_while_normal_oxygen_does_not():
    assert protocol_score(values(spo2_mean=93.0))[0] == 1
    assert protocol_score(values(spo2_mean=97.0))[0] == 0


def test_every_criterion_names_a_real_feature():
    """A rule pointing at a feature that does not exist would silently never fire."""
    from rehab_model import PROTOCOL_RULES

    for name, direction, steps in PROTOCOL_RULES:
        assert name in FEATURE_NAMES, name
        assert direction in ("high", "low")
        assert steps, name


# ═══════════════════════════════════════════════════════════════════════════
#  The model
# ═══════════════════════════════════════════════════════════════════════════

def training_set(n: int = 60):
    """A small, clearly separated population — enough to test behaviour, not accuracy."""
    rows, labels = [], []
    for i in range(n):
        drift = (i % 5) * 0.4
        light = values(immobility_max_min=4.0 + drift, response_min=0.0,
                       movement_mean=0.25, warn_ratio=0.0)
        moderate = values(immobility_max_min=18.0 + drift, response_min=4.0 + drift * 0.2,
                          movement_mean=0.15, warn_ratio=0.05)
        intensive = values(immobility_max_min=34.0 + drift, response_min=12.0 + drift,
                           movement_mean=0.06, warn_ratio=0.16, spo2_mean=93.0)
        for candidate in (light, moderate, intensive):
            rows.append([candidate[name] for name in FEATURE_NAMES])
            labels.append(clinical_label(candidate))
    return rows, labels


def fitted_model() -> RehabModel:
    rows, labels = training_set()
    return RehabModel(n_estimators=60).fit(rows, labels)


def test_an_untrained_model_refuses_to_recommend():
    """Silence is correct here. A programme guessed by an untrained model is worse than no answer."""
    with pytest.raises(RuntimeError, match="not been trained"):
        RehabModel().predict(values())


def test_the_model_reproduces_the_criteria_on_clear_cases():
    model = fitted_model()
    assert model.predict(values(immobility_max_min=3.0, movement_mean=0.3)).program == PROGRAM_LIGHT
    assert model.predict(values(immobility_max_min=36.0, response_min=14.0, warn_ratio=0.18,
                                spo2_mean=92.5, movement_mean=0.05)
                         ).program == PROGRAM_INTENSIVE


def test_a_recommendation_carries_its_reasons():
    """A programme name on its own cannot be agreed or disagreed with by a clinician."""
    model = fitted_model()
    recommendation = model.predict(values(immobility_max_min=40.0, response_min=15.0,
                                          warn_ratio=0.2, spo2_mean=92.0, movement_mean=0.04))
    assert recommendation.program in PROGRAMS
    assert 0.0 < recommendation.confidence <= 1.0
    assert recommendation.reasons
    assert "—" in recommendation.one_line()


def test_confidence_is_a_distribution_over_the_programmes():
    model = fitted_model()
    recommendation = model.predict(values())
    assert set(recommendation.scores) <= set(PROGRAMS)
    assert sum(recommendation.scores.values()) == pytest.approx(1.0)
    assert recommendation.confidence == max(recommendation.scores.values())


def test_features_can_be_passed_as_a_dict_or_as_a_vector():
    model = fitted_model()
    case = values(immobility_max_min=36.0, response_min=14.0, warn_ratio=0.18)
    vector = [case[name] for name in FEATURE_NAMES]
    assert model.predict(case).program == model.predict(vector).program


def test_a_vector_of_the_wrong_length_is_refused():
    """Silently padding a short vector would shift every indicator into the wrong feature."""
    model = fitted_model()
    with pytest.raises(ValueError):
        model.predict([1.0, 2.0, 3.0])


def test_training_rejects_mismatched_rows_and_labels():
    with pytest.raises(ValueError):
        RehabModel().fit([[0.0] * len(FEATURE_NAMES)], ["LIGHT", "MODERATE"])


def test_importances_cover_every_feature():
    model = fitted_model()
    importances = model.feature_importances()
    assert {name for name, _ in importances} == set(FEATURE_NAMES)
    assert importances[0][1] >= importances[-1][1]


def test_a_saved_model_gives_the_same_recommendation_after_loading(tmp_path):
    model = fitted_model()
    case = values(immobility_max_min=30.0, response_min=10.0, warn_ratio=0.12)
    before = model.predict(case)

    path = tmp_path / "rehab_model.pkl"
    model.save(str(path))
    after = RehabModel.load(str(path)).predict(case)

    assert after.program == before.program
    assert after.confidence == pytest.approx(before.confidence)
    assert after.reasons == before.reasons


def test_a_model_trained_on_a_different_feature_list_is_refused(tmp_path):
    """
    Loading it would read one indicator as another and answer confidently with
    the wrong programme — the failure mode that has no symptom.
    """
    import pickle

    path = tmp_path / "stale.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"feature_names": ["hr_mean"], "forest": None,
                     "median": {}, "spread": {}}, fh)
    with pytest.raises(ValueError, match="different feature list"):
        RehabModel.load(str(path))


# ═══════════════════════════════════════════════════════════════════════════
#  End to end: simulated sessions ⟶ validator ⟶ features ⟶ model
# ═══════════════════════════════════════════════════════════════════════════

def test_the_model_generalises_to_sessions_it_never_saw():
    """
    The real contract: sessions generated through the production pipeline, held
    out of training, are placed in the right programme.
    """
    from generate_training_data import build_population

    sessions, labels, _ = build_population(users=120, samples=200, seed=23)
    rows = [s.vector() for s in sessions]
    split = int(len(rows) * 0.75)

    model = RehabModel(n_estimators=150).fit(rows[:split], labels[:split])
    accuracy = model.score(rows[split:], labels[split:])
    assert accuracy >= 0.75, f"held-out accuracy too low: {accuracy:.2f}"


def test_a_session_read_from_its_own_files_can_be_recommended_on(tmp_path):
    """The path a real session takes: exported files on disk ⟶ features ⟶ recommendation."""
    from rehab_features import extract_features_from_files
    from rehab_model import recommend_from_files
    from exporter import MeasurementExporter
    from logger import AuditLogger
    from generate_training_data import PROFILES, ProfiledSensor
    from validator import Validator

    csv_path = tmp_path / "measurements.csv"
    log_path = tmp_path / "audit_log.jsonl"
    sensor = ProfiledSensor(PROFILES[-1], seed=5)
    validator = Validator()

    with MeasurementExporter(path=str(csv_path), session_id="live") as exporter, \
            AuditLogger(path=str(log_path), session_id="live") as logger:
        exporter.write_meta(validator)
        logger.log_session_start(validator)
        for _ in range(200):
            sample = sensor.read()
            result = validator.validate(sample)
            logger.log_result(sample, result)
            exporter.write(result)

    features = extract_features_from_files(str(csv_path), str(log_path), session_id="live")
    assert features.samples == 200
    assert features["hr_mean"] > 0.0

    model_path = tmp_path / "model.pkl"
    fitted_model().save(str(model_path))
    recommendation = recommend_from_files(str(model_path), str(csv_path), str(log_path),
                                          session_id="live")
    assert recommendation.program in PROGRAMS

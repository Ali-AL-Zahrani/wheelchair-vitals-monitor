"""
Rehabilitation programme recommendation — the AI layer.

**Type:** supervised classification. The input is one session reduced to the
thirteen features in `rehab_features.py`; the output is one programme out of a
closed list agreed with the medical team, together with the reason it was
chosen.

**Algorithm:** random forest. Chosen for two reasons that matter more here than
raw accuracy: it works with the small number of users a prototype can gather,
and it can be questioned — every recommendation comes back with the indicators
that drove it, so a clinician can agree or disagree with the reasoning rather
than with a bare label.

**The model recommends; it does not decide.** The output is a recommendation for
review by the medical team or the caregiver, never an instruction to the user.

Two pieces live here and are meant to be replaced, not argued with:

  - `PROGRAMS` — the closed list of programmes. Three provisional levels until
    the medical team supplies the real ones.
  - `PROTOCOL_RULES` — the clinical criteria that decide which programme fits a
    set of indicators. They are data, in one place, so changing the protocol is
    editing a table rather than rewriting logic.

The rules are also how the first training set is labelled, because no clinician
labels exist yet. That has a consequence worth stating plainly: on simulated
data the forest largely learns the rule table back. What it adds is tolerance —
it still classifies a user whose indicators sit between two bands, or whose
session was partly withheld by the validator, instead of falling off a hard
threshold. Once real sessions carry clinician-assigned programmes, the same
model is refitted on those labels and the rule table becomes the fallback only.
"""

from __future__ import annotations

import pickle
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rehab_features import FEATURE_NAMES, SessionFeatures

# ═══════════════════════════════════════════════════════════════════════════
#  The closed list of programmes
#  Provisional three levels — to be replaced by the medical team's own list.
# ═══════════════════════════════════════════════════════════════════════════

PROGRAM_LIGHT = "LIGHT"
PROGRAM_MODERATE = "MODERATE"
PROGRAM_INTENSIVE = "INTENSIVE"

PROGRAMS: Tuple[str, ...] = (PROGRAM_LIGHT, PROGRAM_MODERATE, PROGRAM_INTENSIVE)

PROGRAM_LABEL: Dict[str, str] = {
    PROGRAM_LIGHT: "Light programme",
    PROGRAM_MODERATE: "Moderate programme",
    PROGRAM_INTENSIVE: "Intensive programme",
}

PROGRAM_DESCRIPTION: Dict[str, str] = {
    PROGRAM_LIGHT: "Stable readings and regular movement — maintenance exercises, "
                   "weight shift on the standard interval.",
    PROGRAM_MODERATE: "Long immobile stretches or a slow response to prompts — "
                      "scheduled arm exercises and a shorter weight-shift interval.",
    PROGRAM_INTENSIVE: "Repeatedly abnormal readings or prolonged immobility — "
                       "supervised sessions, shorter and more frequent.",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Clinical criteria
#  One table, one place. Each rule adds points; the bands map points to a
#  programme. "high" scores at or above the threshold, "low" scores below it.
# ═══════════════════════════════════════════════════════════════════════════

PROTOCOL_RULES: Tuple[Tuple[str, str, Tuple[Tuple[float, int], ...]], ...] = (
    ("immobility_max_min",       "high", ((25.0, 2), (15.0, 1))),
    ("response_min",             "high", ((8.0, 2), (3.0, 1))),
    ("movement_alerts_per_hour", "high", ((2.0, 2), (1.0, 1))),
    ("warn_ratio",               "high", ((0.10, 2), (0.03, 1))),
    ("wrist_alerts_per_hour",    "high", ((2.0, 1),)),
    ("spo2_mean",                "low",  ((95.0, 1),)),
    ("movement_mean",            "low",  ((0.10, 1),)),
)

# Upper bound of each band, in points. Anything above the last band is the most intensive programme.
PROTOCOL_BANDS: Tuple[Tuple[int, str], ...] = (
    (2, PROGRAM_LIGHT),
    (5, PROGRAM_MODERATE),
)


def protocol_score(values: Dict[str, float]) -> Tuple[int, List[str]]:
    """Points from the criteria table, with the rules that fired."""
    score = 0
    fired: List[str] = []
    for name, direction, steps in PROTOCOL_RULES:
        value = values.get(name)
        if value is None:
            continue
        for threshold, points in steps:
            hit = value >= threshold if direction == "high" else value < threshold
            if hit:
                score += points
                fired.append(f"{name} {'≥' if direction == 'high' else '<'} {threshold:g}")
                break       # the steps are ordered strongest first — only one fires per rule
    return score, fired


def clinical_label(values: Dict[str, float]) -> str:
    """
    The programme the criteria table assigns to a set of indicators.

    This is the clinical protocol expressed in code, and the label source for
    the first training set. It is not the model.
    """
    score, _ = protocol_score(values)
    for limit, program in PROTOCOL_BANDS:
        if score <= limit:
            return program
    return PROGRAMS[-1]


# ═══════════════════════════════════════════════════════════════════════════
#  Recommendation
# ═══════════════════════════════════════════════════════════════════════════

# How each indicator is worded in a recommendation. The model reports the
# indicators that stood out, not a coefficient nobody can read.
_PHRASE: Dict[str, str] = {
    "hr_mean": "resting heart rate {v:.0f} bpm",
    "hr_sd": "heart-rate variability {v:.1f} bpm",
    "spo2_mean": "average blood oxygen {v:.1f}%",
    "spo2_min": "lowest accepted blood oxygen {v:.1f}%",
    "skin_temp_mean": "wrist skin temperature {v:.1f} °C",
    "movement_mean": "average movement {v:.2f}",
    "valid_ratio": "accepted readings {p:.0f}%",
    "warn_ratio": "readings outside the clinical range {p:.0f}%",
    "no_reading_ratio": "session without a reading {p:.0f}%",
    "immobility_max_min": "longest immobile stretch {v:.0f} min",
    "movement_alerts_per_hour": "movement prompts {v:.1f} per hour",
    "response_min": "response to a movement prompt {v:.1f} min",
    "wrist_alerts_per_hour": "wrist-pressure prompts {v:.1f} per hour",
}

_PERCENT = ("valid_ratio", "warn_ratio", "no_reading_ratio")


@dataclass(frozen=True)
class Recommendation:
    """
    A recommendation for review — not a decision.

    `reasons` is the point of the class: a programme name on its own cannot be
    agreed or disagreed with by a clinician.
    """

    program: str
    label: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def one_line(self) -> str:
        reason = "; ".join(self.reasons) if self.reasons else "no indicator stood out"
        return f"{self.label} ({self.confidence * 100:.0f}% confidence) — {reason}"


# ═══════════════════════════════════════════════════════════════════════════
#  The model
# ═══════════════════════════════════════════════════════════════════════════

Features = Union[SessionFeatures, Dict[str, float], Sequence[float]]


class RehabModel:
    """
    Random-forest classifier over the session features, plus the reference
    statistics needed to explain a prediction.

    The medians and spreads of the training population are stored with the
    model on purpose: "22 minutes immobile" only becomes a reason once it can
    be compared with what is typical.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42,
                 min_samples_leaf: int = 2) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.feature_names: Tuple[str, ...] = FEATURE_NAMES
        self._forest = None
        self._median: Dict[str, float] = {}
        self._spread: Dict[str, float] = {}
        self.training_size = 0

    # ── Training ──
    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> "RehabModel":
        from sklearn.ensemble import RandomForestClassifier

        if len(rows) != len(labels):
            raise ValueError("rows and labels have different lengths")
        if not rows:
            raise ValueError("no training data")
        for row in rows:
            if len(row) != len(FEATURE_NAMES):
                raise ValueError(
                    f"every row needs {len(FEATURE_NAMES)} features, got {len(row)}")

        # class_weight balances the programmes: a population where most users
        # need the light programme must not train a model that answers "light"
        # to everyone, because the users it would then miss are the ones the
        # system exists for.
        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
        )
        self._forest.fit(list(rows), list(labels))
        self.training_size = len(rows)

        for i, name in enumerate(FEATURE_NAMES):
            column = [float(row[i]) for row in rows]
            self._median[name] = statistics.median(column)
            spread = statistics.pstdev(column) if len(column) > 1 else 0.0
            # A feature with no spread in training explains nothing later; a
            # scale of 1.0 keeps its z-score finite instead of infinite.
            self._spread[name] = spread if spread > 1e-9 else 1.0
        return self

    @property
    def is_fitted(self) -> bool:
        return self._forest is not None

    def feature_importances(self) -> List[Tuple[str, float]]:
        """Which indicators the forest relies on, strongest first."""
        self._require_fitted()
        pairs = list(zip(FEATURE_NAMES, self._forest.feature_importances_))  # type: ignore[union-attr]
        return sorted(((n, float(w)) for n, w in pairs), key=lambda p: -p[1])

    def score(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> float:
        """Accuracy on held-out sessions."""
        self._require_fitted()
        return float(self._forest.score(list(rows), list(labels)))  # type: ignore[union-attr]

    # ── Prediction ──
    def predict(self, features: Features, top_reasons: int = 3) -> Recommendation:
        self._require_fitted()
        values = _as_values(features)
        vector = [values[name] for name in FEATURE_NAMES]

        probabilities = self._forest.predict_proba([vector])[0]        # type: ignore[union-attr]
        classes = list(self._forest.classes_)                          # type: ignore[union-attr]
        scores = {str(c): float(p) for c, p in zip(classes, probabilities)}
        program = max(scores, key=lambda k: scores[k])

        return Recommendation(
            program=program,
            label=PROGRAM_LABEL.get(program, program),
            confidence=scores[program],
            reasons=self._reasons(values, top_reasons),
            scores=scores,
        )

    def _reasons(self, values: Dict[str, float], limit: int) -> List[str]:
        """
        The indicators that stood out for this user among the ones the forest
        actually uses: distance from the training median, weighted by the
        feature's importance. A feature the forest ignores never appears as a
        reason, however unusual its value.
        """
        importances = dict(self.feature_importances())
        ranked: List[Tuple[float, str]] = []
        for name in FEATURE_NAMES:
            z = (values[name] - self._median[name]) / self._spread[name]
            weight = importances.get(name, 0.0) * abs(z)
            if abs(z) < 0.5 or weight <= 0.0:
                continue
            direction = "above" if z > 0 else "below"
            ranked.append((weight, f"{_describe(name, values[name])} — {direction} typical"))
        ranked.sort(key=lambda p: -p[0])
        return [text for _, text in ranked[:limit]]

    # ── Persistence ──
    def save(self, path: str) -> None:
        self._require_fitted()
        with open(path, "wb") as fh:
            pickle.dump({
                "feature_names": list(FEATURE_NAMES),
                "programs": list(PROGRAMS),
                "forest": self._forest,
                "median": self._median,
                "spread": self._spread,
                "training_size": self.training_size,
            }, fh)

    @classmethod
    def load(cls, path: str) -> "RehabModel":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        # A model saved against a different feature list would silently read
        # one indicator as another — refuse instead.
        saved = tuple(blob.get("feature_names", ()))
        if saved != FEATURE_NAMES:
            raise ValueError(
                "the saved model was trained on a different feature list — retrain it")
        model = cls()
        model._forest = blob["forest"]
        model._median = blob["median"]
        model._spread = blob["spread"]
        model.training_size = blob.get("training_size", 0)
        return model

    def _require_fitted(self) -> None:
        if self._forest is None:
            raise RuntimeError("the model has not been trained — call fit() or load()")


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _as_values(features: Features) -> Dict[str, float]:
    if isinstance(features, SessionFeatures):
        return dict(features.values)
    if isinstance(features, dict):
        missing = [n for n in FEATURE_NAMES if n not in features]
        if missing:
            raise ValueError(f"missing features: {', '.join(missing)}")
        return {n: float(features[n]) for n in FEATURE_NAMES}
    row = list(features)
    if len(row) != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(row)}")
    return {n: float(v) for n, v in zip(FEATURE_NAMES, row)}


def _describe(name: str, value: float) -> str:
    template = _PHRASE.get(name, name + " {v:.2f}")
    if name in _PERCENT:
        return template.format(v=value, p=value * 100.0)
    return template.format(v=value, p=value * 100.0)


def recommend_from_files(model_path: str, csv_path: str,
                         log_path: Optional[str] = None,
                         session_id: Optional[str] = None) -> Recommendation:
    """Convenience path: a saved model plus one session's files ⟶ a recommendation."""
    from rehab_features import extract_features_from_files

    features = extract_features_from_files(csv_path, log_path, session_id=session_id)
    return RehabModel.load(model_path).predict(features)

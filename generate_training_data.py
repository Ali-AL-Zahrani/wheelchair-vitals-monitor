"""
Training data for the rehabilitation model, and the training run itself.

No real sessions exist yet, so the first training set is simulated: a
population of users with different activity levels and different signal
quality, each one run through the **real** pipeline —
`sensor ⟶ validator ⟶ exporter + audit log` — and then reduced to features by
the same code that will read a real session. Nothing here shortcuts the
validator, so a simulated user's withheld readings are withheld exactly as a
real user's would be.

Why not `MockSensor`: that sensor exists to stress the validator, and its
baseline is one fixed resting adult. A training population needs the opposite —
many different baselines and activity patterns, with faults as background noise
rather than as the point. `ProfiledSensor` below does that; the validator,
exporter, logger and feature extractor are the production ones.

    python generate_training_data.py                     # generate, train, report
    python generate_training_data.py --users 400 --out data.csv --model rf.pkl
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from exporter import MeasurementExporter
from interface import SensorInterface, VitalSample
from logger import AuditLogger
from rehab_features import FEATURE_NAMES, SessionFeatures, extract_features
from rehab_model import PROGRAMS, RehabModel, clinical_label
from validator import Validator

SAMPLE_PERIOD_S = 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  Population
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Profile:
    """One archetype. Each generated user is this profile with its own jitter."""

    name: str
    hr: Tuple[float, float]           # (mean, sd) resting heart rate
    spo2: Tuple[float, float]
    temp: float                       # wrist skin temperature mean
    still_start_p: float              # chance per sample of entering an immobile stretch
    still_run: Tuple[int, int]        # length of that stretch, in samples (30 s each)
    fault_rate: float                 # one-off unusable readings (None / NaN / impossible / jump)
    warn_rate: float                  # chance of entering a low-oxygen stretch
    contact_loss_p: float             # chance of lifting the wrist off the armrest


# The archetypes shape the data; they do not decide the programme. The label
# comes from the clinical criteria applied to the extracted features, so a user
# from any profile can land in any programme.
PROFILES: Tuple[Profile, ...] = (
    Profile("active", hr=(68.0, 3.0), spo2=(97.6, 0.6), temp=33.6,
            still_start_p=0.004, still_run=(4, 12),
            fault_rate=0.010, warn_rate=0.000, contact_loss_p=0.010),
    Profile("moderate", hr=(74.0, 4.0), spo2=(97.0, 0.8), temp=33.4,
            still_start_p=0.012, still_run=(12, 30),
            fault_rate=0.020, warn_rate=0.008, contact_loss_p=0.025),
    Profile("sedentary", hr=(80.0, 5.0), spo2=(96.2, 1.0), temp=33.1,
            still_start_p=0.022, still_run=(30, 60),
            fault_rate=0.030, warn_rate=0.018, contact_loss_p=0.030),
    Profile("weak_signal", hr=(76.0, 6.0), spo2=(96.0, 1.2), temp=32.6,
            still_start_p=0.010, still_run=(10, 35),
            fault_rate=0.070, warn_rate=0.045, contact_loss_p=0.090),
    Profile("unstable_vitals", hr=(88.0, 7.0), spo2=(94.4, 1.5), temp=32.9,
            still_start_p=0.018, still_run=(20, 50),
            fault_rate=0.040, warn_rate=0.100, contact_loss_p=0.040),
)


class ProfiledSensor(SensorInterface):
    """
    A simulated user with their own baseline and their own activity pattern.

    Like `MockSensor` it returns the sample **as it is** — no filtering, no
    correction. Everything unusable it produces is rejected downstream by the
    validator, which is the behaviour being reproduced.
    """

    def __init__(self, profile: Profile, seed: int, sample_period_s: float = SAMPLE_PERIOD_S,
                 jitter: float = 1.0) -> None:
        self._p = profile
        self._rng = random.Random(seed)
        self._period = sample_period_s
        self._t = 0.0
        self._i = -1
        self._still_left = 0
        self._warn_left = 0
        self._contact_left = 0
        # Per-user variation inside the archetype, so two "sedentary" users are
        # not the same user twice.
        self._hr_offset = self._rng.gauss(0.0, 4.0) * jitter
        self._spo2_offset = self._rng.gauss(0.0, 0.7) * jitter
        self._temp_offset = self._rng.gauss(0.0, 0.5) * jitter
        self._still_factor = self._rng.uniform(0.6, 1.5)

    def read(self) -> VitalSample:
        self._i += 1
        if self._i > 0:
            self._t += self._period
        p = self._p

        heart_rate: Optional[float] = self._rng.gauss(p.hr[0] + self._hr_offset, p.hr[1])
        spo2: Optional[float] = min(100.0, self._rng.gauss(p.spo2[0] + self._spo2_offset, p.spo2[1]))
        skin_temp: Optional[float] = self._rng.gauss(p.temp + self._temp_offset, 0.25)
        ir_dc = self._rng.gauss(38_000.0, 3_500.0)

        # ── activity ──
        if self._still_left > 0:
            self._still_left -= 1
            movement = self._rng.uniform(0.0, 0.02)
        elif self._rng.random() < p.still_start_p * self._still_factor:
            low, high = p.still_run
            self._still_left = max(1, int(self._rng.randint(low, high) * self._still_factor))
            movement = self._rng.uniform(0.0, 0.02)
        elif self._rng.random() < 0.35:
            movement = self._rng.uniform(0.25, 0.9)
        else:
            movement = self._rng.uniform(0.0, 0.05)

        # ── low-oxygen stretch: a real reading, below the clinical bound ⇒ WARN ──
        if self._warn_left > 0:
            self._warn_left -= 1
            spo2 = self._rng.uniform(88.0, 91.0)
        elif self._rng.random() < p.warn_rate:
            self._warn_left = self._rng.randint(1, 4)
            spo2 = self._rng.uniform(88.0, 91.0)

        # ── wrist lifted off the armrest: the sensor keeps emitting numbers ──
        if self._contact_left > 0:
            self._contact_left -= 1
        elif self._rng.random() < p.contact_loss_p:
            self._contact_left = self._rng.randint(1, 5)
        if self._contact_left > 0:
            ir_dc = self._rng.gauss(1_800.0, 500.0)
            heart_rate = self._rng.uniform(35.0, 180.0)
            spo2 = self._rng.uniform(70.0, 100.0)
            skin_temp = self._rng.uniform(20.0, 29.0)

        # ── one-off unusable readings ──
        elif self._rng.random() < p.fault_rate:
            which = self._rng.choice(("none_hr", "nan_spo2", "impossible_hr", "jump_hr"))
            if which == "none_hr":
                heart_rate = None
            elif which == "nan_spo2":
                spo2 = float("nan")
            elif which == "impossible_hr":
                heart_rate = 320.0
            else:
                heart_rate = (heart_rate or 75.0) + 70.0

        return VitalSample(t=self._t, ir_dc=ir_dc, heart_rate=heart_rate,
                           spo2=spo2, skin_temp=skin_temp, movement=movement)


# ═══════════════════════════════════════════════════════════════════════════
#  One session ⟶ features
# ═══════════════════════════════════════════════════════════════════════════

def run_session(profile: Profile, seed: int, samples: int = 240) -> SessionFeatures:
    """
    Run one simulated user through the production pipeline and extract features.

    The CSV and the audit log are built in memory: the population needs
    hundreds of sessions, and none of them is a real measurement worth keeping
    on disk.
    """
    sensor = ProfiledSensor(profile, seed=seed)
    validator = Validator()
    csv_buffer, log_buffer = io.StringIO(), io.StringIO()
    session_id = f"sim-{seed}"

    exporter = MeasurementExporter(stream=csv_buffer, session_id=session_id)
    logger = AuditLogger(stream=log_buffer, session_id=session_id)
    logger.log_session_start(validator)

    for _ in range(samples):
        sample = sensor.read()
        result = validator.validate(sample)
        logger.log_result(sample, result)
        exporter.write(result)

    return extract_features(csv_buffer.getvalue(), log_buffer.getvalue(),
                            session_id=session_id)


def build_population(users: int = 240, samples: int = 240,
                     seed: int = 7) -> Tuple[List[SessionFeatures], List[str], List[str]]:
    """Returns (features, programme labels, profile names) — the profile is context, not an input."""
    rng = random.Random(seed)
    sessions: List[SessionFeatures] = []
    labels: List[str] = []
    profiles: List[str] = []
    for i in range(users):
        profile = PROFILES[i % len(PROFILES)]
        features = run_session(profile, seed=rng.randrange(1, 10_000_000), samples=samples)
        sessions.append(features)
        labels.append(clinical_label(features.values))
        profiles.append(profile.name)
    return sessions, labels, profiles


def write_dataset(path: str, sessions: Sequence[SessionFeatures],
                  labels: Sequence[str], profiles: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(FEATURE_NAMES) + ["program", "profile"])
        for features, label, profile in zip(sessions, labels, profiles):
            writer.writerow([round(v, 4) for v in features.vector()] + [label, profile])


# ═══════════════════════════════════════════════════════════════════════════
#  Training run
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the training population and fit the model")
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--samples", type=int, default=240,
                        help="samples per session (30 s each; 240 = two virtual hours)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="training_data.csv")
    parser.add_argument("--model", default="rehab_model.pkl")
    args = parser.parse_args()

    print(f"Generating {args.users} sessions of {args.samples} samples "
          f"({args.samples * SAMPLE_PERIOD_S / 3600:.1f} virtual hours each) …")
    sessions, labels, profiles = build_population(args.users, args.samples, args.seed)
    write_dataset(args.out, sessions, labels, profiles)

    counts: Dict[str, int] = {p: labels.count(p) for p in PROGRAMS}
    print(f"\nDataset ⟶ {args.out}")
    for program, count in counts.items():
        share = count / len(labels) * 100 if labels else 0.0
        print(f"  {program:<10} {count:>4}  ({share:.0f}%)")

    missing = [p for p, c in counts.items() if c == 0]
    if missing:
        # A programme with no examples cannot be predicted at all — the
        # population, not the model, is what needs fixing.
        print(f"\n  ⚠ no examples for: {', '.join(missing)} — widen the profiles")

    from sklearn.model_selection import train_test_split

    rows = [s.vector() for s in sessions]
    stratify = labels if all(c >= 2 for c in counts.values()) else None
    x_train, x_test, y_train, y_test = train_test_split(
        rows, labels, test_size=0.25, random_state=args.seed, stratify=stratify)

    model = RehabModel().fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"\nTrained on {len(x_train)} sessions, tested on {len(x_test)}")
    print(f"Held-out accuracy: {accuracy * 100:.1f}%")

    print("\nIndicators the model relies on:")
    for name, weight in model.feature_importances()[:6]:
        print(f"  {name:<26} {weight:.3f}")

    model.save(args.model)
    print(f"\nModel ⟶ {args.model}")

    example = sessions[0]
    print("\nExample recommendation for the first session:")
    print(f"  {model.predict(example).one_line()}")


if __name__ == "__main__":
    main()

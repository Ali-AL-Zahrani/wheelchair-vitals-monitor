"""
Rehabilitation model, end to end — five users, five sessions, five recommendations.

One command that shows what the AI layer does:

    python demo_rehab.py

Each user is run through the full pipeline — sensor ⟶ validator ⟶ exporter and
audit log — then reduced to the thirteen indicators and handed to the model.
Nothing is hard-coded: every number printed below is computed on the spot.

The model is loaded from `rehab_model.pkl` when it is there, and trained first
when it is not. Both paths give the same model, so the output is reproducible.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

from generate_training_data import PROFILES, build_population, run_session
from rehab_model import PROGRAM_DESCRIPTION, RehabModel, clinical_label, protocol_score

# Defaults shared with generate_training_data.py so a trained model and a
# loaded one are the same model.
TRAIN_USERS = 400
TRAIN_SAMPLES = 240
TRAIN_SEED = 7

# One session per profile. The seeds are fixed so the demo shows the same five
# users every time it runs — a booth demo that reshuffles itself cannot be
# talked through.
DEMO_SEED_BASE = 1000


def load_or_train(path: str) -> RehabModel:
    if os.path.exists(path):
        model = RehabModel.load(path)
        print(f"Model loaded from {path} ({model.training_size} training sessions)\n")
        return model

    print(f"No {path} yet — training on {TRAIN_USERS} sessions first …")
    sessions, labels, _ = build_population(TRAIN_USERS, TRAIN_SAMPLES, TRAIN_SEED)
    model = RehabModel().fit([s.vector() for s in sessions], labels)
    model.save(path)
    print(f"Trained and saved to {path}\n")
    return model


def minutes(value: float) -> str:
    """A response time of zero means the prompt never had to wait, not 'no data'."""
    return "immediate" if value <= 0.0 else f"{value:.1f} min"


def main() -> None:
    parser = argparse.ArgumentParser(description="Five users through the rehabilitation model")
    parser.add_argument("--model", default="rehab_model.pkl")
    parser.add_argument("--samples", type=int, default=240,
                        help="samples per session (30 s each; 240 = two hours)")
    args = parser.parse_args()

    model = load_or_train(args.model)
    rows: List[Tuple[str, ...]] = []

    for i, profile in enumerate(PROFILES):
        name = profile.name.replace("_", " ").capitalize()
        features = run_session(profile, seed=DEMO_SEED_BASE + i, samples=args.samples)
        recommendation = model.predict(features)
        score, fired = protocol_score(features.values)

        print("─" * 74)
        print(f"  {name}")
        print("─" * 74)
        print(f"    resting heart rate     {features['hr_mean']:.0f} bpm")
        print(f"    blood oxygen           {features['spo2_mean']:.1f}%  "
              f"(lowest accepted {features['spo2_min']:.1f}%)")
        print(f"    wrist skin temp        {features['skin_temp_mean']:.1f} °C")
        print(f"    longest immobile       {features['immobility_max_min']:.0f} min")
        print(f"    response to a prompt   {minutes(features['response_min'])}")
        print(f"    readings accepted      {features['valid_ratio'] * 100:.0f}%"
              f"    withheld {features['no_reading_ratio'] * 100:.0f}%")
        print()
        print(f"    ▸ {recommendation.label}   ({recommendation.confidence * 100:.0f}% confidence)")
        for reason in recommendation.reasons:
            print(f"        · {reason}")
        print(f"      {PROGRAM_DESCRIPTION[recommendation.program]}")
        print()
        # The criteria table reaches its own answer independently of the model.
        # Printing both makes it clear which one said what.
        print(f"      criteria table: {score} point(s)"
              + (f" — {', '.join(fired)}" if fired else " — nothing scored")
              + f"  ⟶  {clinical_label(features.values)}")
        print(f"      model scores:   "
              + "   ".join(f"{k} {v * 100:.0f}%" for k, v in sorted(
                  recommendation.scores.items(), key=lambda p: -p[1])))
        print()

        rows.append((
            name,
            f"{features['immobility_max_min']:.0f} min",
            minutes(features["response_min"]),
            f"{features['valid_ratio'] * 100:.0f}%",
            f"{recommendation.label.split()[0]} ({recommendation.confidence * 100:.0f}%)",
        ))

    headers = ("User", "Longest immobile", "Response", "Accepted", "Recommendation")
    widths = [max(len(headers[c]), max(len(r[c]) for r in rows)) for c in range(len(headers))]
    line = "  ".join("─" * w for w in widths)

    print("═" * 74)
    print("  Summary")
    print("═" * 74)
    print("  " + "  ".join(h.ljust(widths[c]) for c, h in enumerate(headers)))
    print("  " + line)
    for row in rows:
        print("  " + "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)))
    print()
    print("  Every recommendation is for review by the medical team, with the")
    print("  indicators that produced it. The model recommends; it does not decide.")


if __name__ == "__main__":
    main()

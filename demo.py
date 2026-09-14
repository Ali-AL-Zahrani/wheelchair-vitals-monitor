"""
Full end-to-end run: mock sensor ⟵ validator ⟵ screen, then a statistical summary.

The screen here is text-only (prototype), but it obeys the fixed rule:
**it reads from validator output only.** No raw number reaches the user's eyes.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Dict

from display import LABELS, PATIENT_FIELDS, format_clock, render_line
from exporter import MeasurementExporter
from logger import AuditLogger, summarize
from mock_sensor import MockSensor, default_scenario
from validator import FIELD_SPECS, Status, Validator

N_SAMPLES = 120
AUDIT_LOG_PATH = "audit_log.jsonl"      # audit trail: why a reading was rejected
CSV_PATH = "measurements.csv"           # measurement data: what passed validation

# All display logic lives in display.py: the terminal and the graphical screen
# read from one source, so the "when a number is shown" rule never forks into
# two copies that drift apart at the first edit.


def main() -> None:
    # The Windows console may default to a legacy code page; force UTF-8 so
    # icons and symbols render correctly.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    sensor = MockSensor(seed=42, sample_period_s=30.0, faults=default_scenario())
    validator = Validator()
    audit = AuditLogger(path=AUDIT_LOG_PATH)
    audit.log_session_start(validator)
    export = MeasurementExporter(path=CSV_PATH)
    export.write_meta(validator)        # data without its thresholds cannot be interpreted

    status_counts: Dict[str, Counter] = {name: Counter() for name in FIELD_SPECS}
    flag_counts: Counter = Counter()
    alert_counts: Counter = Counter()
    max_immobility = 0.0

    sensor.start()
    print("=" * 78)
    print("Validator diagnostics (left) — what the user screen shows (right)")
    print("=" * 78)

    for _ in range(N_SAMPLES):
        sample = sensor.read()
        result = validator.validate(sample)
        audit.log_result(sample, result)
        export.write(result)

        for name, fr in result.fields.items():
            status_counts[name][fr.status.value] += 1
            flag_counts.update(fr.flags)
        flag_counts.update(result.flags)
        alert_counts.update(result.alerts)
        max_immobility = max(max_immobility, result.immobility_s)

        diag = ",".join(sorted(set(result.flags))) or "-"
        print(f"{format_clock(result.t)} [{diag:<24}] {render_line(result)}")

    sensor.stop()
    audit.close()
    export.close()

    print("=" * 78)
    print("Session summary")
    print("-" * 78)
    print(f"Samples: {N_SAMPLES}   |   Simulated session time: {format_clock(result.t)}")
    print()
    for name in FIELD_SPECS:
        c = status_counts[name]
        row = "  ".join(f"{s}={c.get(s, 0)}" for s in
                        (Status.VALID.value, Status.WARN.value,
                         Status.INVALID.value, Status.NO_CONTACT.value))
        if name in PATIENT_FIELDS:
            shown = c.get(Status.VALID.value, 0) + c.get(Status.WARN.value, 0)
            tail = f"⟵ shown to user: {shown}/{N_SAMPLES}"
        else:
            # Movement drives the immobility alert and never appears on screen —
            # counting it as "shown" would lie to the reader.
            tail = "⟵ internal indicator (never displayed)"
        print(f"{LABELS[name]:<17} {row}   {tail}")
    print()
    print("Data-quality flags:")
    for flag, n in flag_counts.most_common():
        print(f"  {flag:<22} {n}")
    print()
    print("Alerts (samples during which the alert was active):")
    for alert, n in alert_counts.most_common():
        print(f"  {alert:<22} {n}")
    print(f"Longest continuous immobility: {format_clock(max_immobility)} (mm:ss)")
    print()
    print(f"Measurements ⟵ {CSV_PATH} ({export.row_count} rows) + {CSV_PATH}.meta.json")
    print(f"Audit log    ⟵ {AUDIT_LOG_PATH} ({audit.event_count} events this session)")
    for ev_type, n in sorted(summarize(AUDIT_LOG_PATH).items()):
        print(f"  {ev_type:<16} {n}   (cumulative across all sessions in the file)")
    print("=" * 78)


if __name__ == "__main__":
    main()

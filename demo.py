"""
تشغيل كامل end-to-end: حسّاس وهمي ⟵ مدقّق ⟵ شاشة، ثم ملخّص إحصائي.

الشاشة هنا نصّية (بروتوتايب)، لكنها تلتزم بالقاعدة الثابتة:
**لا تقرأ إلا من مخرجات المدقّق.** لا رقم خام يصل عين المستخدم.
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
AUDIT_LOG_PATH = "audit_log.jsonl"      # سجلّ تدقيق: لماذا رُفضت قراءة
CSV_PATH = "measurements.csv"           # بيانات قياس: ما اجتاز التدقيق

# منطق العرض كله في display.py: الطرفية والشاشة الرسومية تقرآن من مصدر واحد،
# فلا تتفرّع قاعدة "متى يُعرض رقم" إلى نسختين تتباعدان مع أول تعديل.


def main() -> None:
    # الطرفية على ويندوز قد تكون cp1256؛ نجبر UTF-8 حتى لا ينكسر النص العربي.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    sensor = MockSensor(seed=42, sample_period_s=30.0, faults=default_scenario())
    validator = Validator()
    audit = AuditLogger(path=AUDIT_LOG_PATH)
    audit.log_session_start(validator)
    export = MeasurementExporter(path=CSV_PATH)
    export.write_meta(validator)        # البيانات بلا عتباتها لا تُفسَّر

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
            # الحركة تقود إنذار الخمول ولا تظهر على الشاشة — عدّها كـ"معروضة" كذب على القارئ.
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

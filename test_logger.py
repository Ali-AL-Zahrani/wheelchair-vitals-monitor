"""
اختبارات طبقة التدقيق.

ما تحرسه هذه الاختبارات: السجلّ الطبي لا يكذب ولا يبتلع حدثًا.
سجلّ ناقص أسوأ من غياب السجلّ — لأنه يوهم بأن ما ليس فيه لم يحدث.
"""

from __future__ import annotations

import io
import json

import pytest

from interface import VitalSample
from logger import (
    EV_ALERT_CLEARED,
    EV_ALERT_RAISED,
    EV_CORRUPT,
    EV_NO_CONTACT,
    EV_REJECTED,
    EV_SESSION_START,
    EV_WARN,
    AuditLogger,
    summarize,
)
from validator import ALERT_NEEDS_MOVEMENT, FLAG_SANITY_NAN, Validator


def sample(t=0.0, ir=38_000.0, hr=74.0, spo2=97.0, temp=33.4, mov=0.5) -> VitalSample:
    return VitalSample(t=t, ir_dc=ir, heart_rate=hr, spo2=spo2, skin_temp=temp, movement=mov)


@pytest.fixture
def buf() -> io.StringIO:
    return io.StringIO()


def events(buf: io.StringIO):
    """كل سطر يجب أن يكون JSON صالحًا بذاته — هذا جزء من العقد لا تفصيل شكلي."""
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def run(buf, samples, validator=None):
    v = validator or Validator()
    log = AuditLogger(path=None, stream=buf, session_id="test")
    log.log_session_start(v)
    for s in samples:
        log.log_result(s, v.validate(s))
    return events(buf)


# ── سلامة الصيغة ──
def test_every_line_is_valid_json_and_carries_both_clocks(buf):
    evs = run(buf, [sample(t=0.0, hr=None)])
    assert len(evs) >= 2
    for ev in evs:
        assert ev["session_id"] == "test"
        assert "t" in ev and "ts" in ev   # ساعة الحسّاس + ساعة الجدار


def test_nan_is_written_as_text_not_as_invalid_json(buf):
    """
    NaN الخام يكسر أي قارئ JSON صارم، و null يخلط بين "لم تصل قيمة" و"وصلت تالفة".
    """
    raw = buf  # الاسم للتوضيح فقط
    evs = run(raw, [sample(spo2=float("nan"))])
    rejected = [e for e in evs if e["type"] == EV_REJECTED and e["field"] == "spo2"]
    assert rejected and rejected[0]["raw"] == "NaN"
    assert FLAG_SANITY_NAN in rejected[0]["flags"]
    assert "NaN," not in buf.getvalue()  # لم تُكتب قيمة NaN عارية في أي مكان


def test_session_start_records_the_thresholds_in_force(buf):
    v = Validator(contact_ir_threshold=1234.0, immobility_limit_s=60.0)
    evs = run(buf, [sample()], validator=v)
    start = evs[0]
    assert start["type"] == EV_SESSION_START
    assert start["thresholds"]["CONTACT_IR_THRESHOLD"] == 1234.0
    assert start["thresholds"]["IMMOBILITY_LIMIT_S"] == 60.0


def test_session_start_records_the_clinical_limits_too(buf):
    """
    الحدود السريرية تقرّر متى يُحذَّر المستخدم — وهي غير معايرة بعد.
    تحذير مسجَّل بلا الحدّ الذي أطلقه لا يمكن مراجعته لاحقًا.
    """
    evs = run(buf, [sample()])
    limits = evs[0]["field_limits"]
    assert limits["heart_rate"]["clinical_min"] == 50.0
    assert limits["spo2"]["clinical_min"] == 94.0
    assert limits["skin_temp"]["unit"] == "°C"
    # الحدود اللانهائية تُكتب نصًّا لا كـ JSON تالف
    assert limits["movement"]["sanity_max"] == "Infinity"


def test_session_start_records_the_new_watchdog_limits(buf):
    v = Validator(wrist_rest_limit_s=111.0, silence_limit_s=222.0)
    evs = run(buf, [sample()], validator=v)
    assert evs[0]["thresholds"]["WRIST_REST_LIMIT_S"] == 111.0
    assert evs[0]["thresholds"]["SILENCE_LIMIT_S"] == 222.0


# ── ما يُسجَّل وما لا يُسجَّل ──
def test_clean_sample_writes_nothing_beyond_session_start(buf):
    evs = run(buf, [sample()])
    assert [e["type"] for e in evs] == [EV_SESSION_START]


def test_rejected_reading_records_reason_and_raw_value(buf):
    """التدقيق يحتاج معرفة *ما الذي* رُفض، لا مجرّد أنه رُفض."""
    evs = run(buf, [sample(hr=320.0)])
    rej = [e for e in evs if e["type"] == EV_REJECTED]
    assert len(rej) == 1
    assert rej[0]["field"] == "heart_rate"
    assert rej[0]["raw"] == 320.0
    assert rej[0]["flags"] == ["SANITY_RANGE"]


def test_missing_value_logs_null_not_the_string_nan(buf):
    evs = run(buf, [sample(hr=None)])
    rej = [e for e in evs if e["type"] == EV_REJECTED][0]
    assert rej["raw"] is None          # "لم تصل قيمة" — سبب مختلف عن NaN


def test_no_contact_is_one_sample_level_event(buf):
    """رفع المعصم حدث واحد، لا ثلاثة أخطاء منفصلة — وإلا تضخّم السجلّ بلا معنى."""
    evs = run(buf, [sample(ir=900.0)])
    nc = [e for e in evs if e["type"] == EV_NO_CONTACT]
    assert len(nc) == 1
    assert set(nc[0]["blanked_fields"]) == {"heart_rate", "spo2", "skin_temp"}
    assert not [e for e in evs if e["type"] == EV_REJECTED]


def test_hardware_fault_is_distinguishable_from_a_lifted_wrist(buf):
    """
    الحدثان يُصفّران الشاشة بنفس الشكل، لكن أحدهما سلوك استخدام عادي
    والآخر عطب يستدعي صيانة. تساويهما في السجلّ يعني ضياع العطب.
    """
    lifted = run(buf, [sample(ir=1_500.0)])
    lifted_ev = [e for e in lifted if e["type"] == EV_NO_CONTACT][0]
    assert "IR_IMPLAUSIBLE" not in lifted_ev["flags"]

    broken = run(io.StringIO(), [sample(ir=9_999_999.0)])
    broken_ev = [e for e in broken if e["type"] == EV_NO_CONTACT][0]
    assert "IR_IMPLAUSIBLE" in broken_ev["flags"]
    assert broken_ev["ir_dc"] == 9_999_999.0      # القيمة الخام محفوظة للتحقيق


def test_warn_reading_is_logged_even_though_it_reached_the_screen(buf):
    evs = run(buf, [sample(spo2=90.0)])
    warns = [e for e in evs if e["type"] == EV_WARN]
    assert len(warns) == 1
    assert warns[0]["field"] == "spo2" and warns[0]["value"] == 90.0


def test_non_numeric_value_is_preserved_as_text(buf):
    evs = run(buf, [sample(hr="74")])
    rej = [e for e in evs if e["type"] == EV_REJECTED][0]
    assert rej["raw"] == "'74'"        # يُحفظ كما وصل ليُحقَّق في الدرايفر


# ── الإنذارات: تحوّل لا تكرار ──
def test_alert_logged_once_on_transition_not_every_sample(buf):
    v = Validator(immobility_limit_s=60.0)
    samples = [sample(t=i * 30.0, hr=74.0 + i * 0.1, mov=0.01) for i in range(6)]
    evs = run(buf, samples, validator=v)
    raised = [e for e in evs if e["type"] == EV_ALERT_RAISED]
    assert len(raised) == 1
    assert raised[0]["alert"] == ALERT_NEEDS_MOVEMENT
    assert raised[0]["immobility_s"] >= 60.0


def test_alert_cleared_records_duration(buf):
    v = Validator(immobility_limit_s=60.0)
    samples = [sample(t=i * 30.0, hr=74.0 + i * 0.1, mov=0.01) for i in range(5)]
    samples.append(sample(t=150.0, hr=75.0, mov=0.9))   # المستخدم تحرّك
    evs = run(buf, samples, validator=v)
    cleared = [e for e in evs if e["type"] == EV_ALERT_CLEARED]
    assert len(cleared) == 1
    assert cleared[0]["duration_s"] == 90.0             # من 60.0 إلى 150.0


def test_alert_records_that_movement_was_unverified(buf):
    """إنذار أُطلق بلا تأكيد حركة يجب أن يُميَّز في السجلّ، لا أن يبدو كإنذار مؤكَّد."""
    v = Validator(immobility_limit_s=60.0)
    samples = [sample(t=i * 30.0, hr=74.0 + i * 0.1, mov=None) for i in range(4)]
    evs = run(buf, samples, validator=v)
    raised = [e for e in evs if e["type"] == EV_ALERT_RAISED][0]
    assert "MOVEMENT_UNVERIFIED" in raised["flags"]


# ── التكامل والقراءة اللاحقة ──
def test_full_scenario_log_is_readable_back_from_disk(tmp_path):
    from mock_sensor import MockSensor, default_scenario

    path = tmp_path / "audit_log.jsonl"
    sensor = MockSensor(seed=42, sample_period_s=30.0, faults=default_scenario())
    v = Validator()
    with AuditLogger(path=str(path)) as log:
        log.log_session_start(v)
        for _ in range(90):
            s = sensor.read()
            log.log_result(s, v.validate(s))

    counts = summarize(str(path))
    assert counts[EV_SESSION_START] == 1
    for expected in (EV_REJECTED, EV_NO_CONTACT, EV_WARN, EV_ALERT_RAISED):
        assert counts.get(expected, 0) > 0, f"السجلّ لم يوثّق {expected}"


def test_a_corrupt_line_does_not_destroy_the_whole_log(tmp_path):
    """
    انقطاع كهرباء أو كتابة متزامنة قد يفسد سطرًا. رفض الملف كله بسببه
    يعني فقدان مئات الأحداث السليمة قبله — والصيغة اختيرت لتحتمل هذا.
    """
    path = tmp_path / "audit_log.jsonl"
    v = Validator()
    with AuditLogger(path=str(path), session_id="s1") as log:
        log.log_session_start(v)
        log.log_result(sample(hr=320.0), v.validate(sample(hr=320.0)))

    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "rejected", "t": 1.0\n')      # سطر مقتطع
        fh.write('}{"type": "warn"}\n')                  # سطران متداخلان

    counts = summarize(str(path))
    assert counts[EV_SESSION_START] == 1        # ما قبل الفساد لم يضع
    assert counts[EV_REJECTED] == 1
    assert counts[EV_CORRUPT] == 2              # والفساد معلن لا مبتلع


def test_appending_does_not_erase_a_previous_session(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    v = Validator()
    for session in ("s1", "s2"):
        with AuditLogger(path=str(path), session_id=session) as log:
            log.log_session_start(v)
    counts = summarize(str(path))
    assert counts[EV_SESSION_START] == 2

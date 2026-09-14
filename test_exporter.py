"""
اختبارات تصدير القياسات.

ما تحرسه: ملف البيانات لا يُدخل رقمًا لم يجتز التدقيق، ولا يخلط
"لا قياس" بـ"قياس قيمته صفر" — الخلط الثاني يفسد كل تحليل لاحق بصمت.
"""

from __future__ import annotations

import csv
import io
import json

from exporter import MeasurementExporter, header_row
from interface import VitalSample
from validator import Status, Validator


def sample(t=0.0, ir=38_000.0, hr=74.0, spo2=97.0, temp=33.4, mov=0.5) -> VitalSample:
    return VitalSample(t=t, ir_dc=ir, heart_rate=hr, spo2=spo2, skin_temp=temp, movement=mov)


def rows(buf: io.StringIO):
    return list(csv.DictReader(io.StringIO(buf.getvalue())))


def run(buf, samples, validator=None):
    v = validator or Validator()
    exp = MeasurementExporter(path=None, stream=buf)
    for s in samples:
        exp.write(v.validate(s))
    return rows(buf)


# ── بنية الملف ──
def test_header_covers_every_field_with_its_unit():
    cols = header_row()
    assert "heart_rate_bpm" in cols and "heart_rate_status" in cols
    assert "spo2_pct" in cols and "skin_temp_c" in cols and "movement_idx" in cols
    assert cols[0] == "t_s"
    for extra in ("immobility_s", "wrist_rest_s", "silence_s", "flags", "alerts"):
        assert extra in cols


def test_one_row_per_sample():
    buf = io.StringIO()
    out = run(buf, [sample(t=i * 30.0) for i in range(5)])
    assert len(out) == 5
    assert [r["t_s"] for r in out] == ["0.0", "30.0", "60.0", "90.0", "120.0"]


def test_file_is_parsable_by_a_standard_csv_reader():
    buf = io.StringIO()
    out = run(buf, [sample(hr=None), sample(t=30.0, spo2=89.0)])
    assert out and all(len(r) == len(header_row()) for r in out)


# ── القاعدة المركزية: لا رقم غير متحقَّق، ولا صفر مكان الفراغ ──
def test_rejected_reading_leaves_an_empty_cell_not_a_zero():
    """
    الصفر قيمة قياس، والفراغ غياب قياس. خلطهما يُدخل نبضًا = 0
    في متوسط التحليل ويوهم بقراءة لم تحدث.
    """
    buf = io.StringIO()
    row = run(buf, [sample(hr=320.0)])[0]        # مستحيل فيزيائيًا
    assert row["heart_rate_bpm"] == ""
    assert row["heart_rate_status"] == Status.INVALID.value


def test_no_contact_empties_every_wrist_column():
    buf = io.StringIO()
    row = run(buf, [sample(ir=900.0, hr=72.0, spo2=98.0, temp=33.0)])[0]
    assert row["contact"] == "0"
    for col in ("heart_rate_bpm", "spo2_pct", "skin_temp_c"):
        assert row[col] == "", f"{col}: رقم صُدِّر بلا تلامس"
    assert row["movement_idx"] != ""              # مصدرها منفصل عن المسند


def test_nan_never_reaches_the_data_file():
    buf = io.StringIO()
    row = run(buf, [sample(spo2=float("nan"))])[0]
    assert row["spo2_pct"] == ""
    assert "nan" not in buf.getvalue().lower()


def test_warned_value_is_exported_with_its_status():
    """القيمة الشاذّة الممكنة قياس حقيقي — تُصدَّر، ويُصدَّر معها أنها مُعلَّمة."""
    buf = io.StringIO()
    row = run(buf, [sample(spo2=89.0)])[0]
    assert float(row["spo2_pct"]) == 89.0
    assert row["spo2_status"] == Status.WARN.value


def test_valid_values_match_the_validator_output():
    buf = io.StringIO()
    row = run(buf, [sample(hr=74.5, spo2=96.0, temp=33.25)])[0]
    assert float(row["heart_rate_bpm"]) == 74.5
    assert float(row["spo2_pct"]) == 96.0
    assert row["heart_rate_status"] == Status.VALID.value


# ── الإنذارات والمؤقّتات ──
def test_alerts_and_timers_are_exported():
    buf = io.StringIO()
    v = Validator(immobility_limit_s=60.0)
    out = run(buf, [sample(t=i * 30.0, mov=0.0) for i in range(4)], validator=v)
    last = out[-1]
    assert "NEEDS_MOVEMENT" in last["alerts"]
    assert float(last["immobility_s"]) >= 60.0
    assert float(last["wrist_rest_s"]) > 0.0


def test_multiple_flags_share_a_cell_without_breaking_columns():
    buf = io.StringIO()
    v = Validator(silence_limit_s=30.0)
    out = run(buf, [sample(t=i * 30.0, ir=9_999_999.0) for i in range(3)], validator=v)
    last = out[-1]
    assert "IR_IMPLAUSIBLE" in last["flags"] and "NO_CONTACT" in last["flags"]
    assert ";" in last["flags"]                    # الفاصل لا يكسر بنية الـ CSV
    assert len(last) == len(header_row())


# ── ملف العتبات المصاحب ──
def test_meta_file_carries_the_thresholds_and_the_calibration_state(tmp_path):
    """بيانات بلا عتباتها لا تُفسَّر: عمود status بلا حدوده رقم بلا معنى."""
    path = tmp_path / "measurements.csv"
    v = Validator(immobility_limit_s=123.0)
    with MeasurementExporter(path=str(path), session_id="t1") as exp:
        meta = exp.write_meta(v)
        exp.write(v.validate(sample()))

    saved = json.loads((tmp_path / "measurements.csv.meta.json").read_text(encoding="utf-8"))
    assert saved == meta
    assert saved["thresholds"]["IMMOBILITY_LIMIT_S"] == 123.0
    assert saved["field_limits"]["spo2"]["clinical_min"] == 94.0
    assert saved["field_limits"]["movement"]["sanity_max"] is None   # inf لا يُكتب رقمًا


def test_meta_explains_that_an_empty_cell_is_not_a_zero(tmp_path):
    path = tmp_path / "m.csv"
    with MeasurementExporter(path=str(path)) as exp:
        meta = exp.write_meta(Validator())
    assert "صفر" in meta["notes"]["empty_cell"]


def test_row_count_tracks_written_rows(tmp_path):
    path = tmp_path / "m.csv"
    v = Validator()
    with MeasurementExporter(path=str(path)) as exp:
        for i in range(7):
            exp.write(v.validate(sample(t=i * 30.0)))
        assert exp.row_count == 7


def test_written_file_starts_with_the_header(tmp_path):
    path = tmp_path / "m.csv"
    v = Validator()
    with MeasurementExporter(path=str(path)) as exp:
        exp.write(v.validate(sample()))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == header_row()
    assert len(lines) == 2

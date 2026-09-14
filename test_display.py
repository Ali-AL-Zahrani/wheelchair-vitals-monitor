"""
إثبات آلي لقواعد طبقة العرض.

الاختبارات تمرّ عبر **المدقّق الحقيقي** لا عبر نتائج ملفّقة، لأن الخطر الفعلي
يقع في التركيب: مدقّق سليم + راسم يسيء تفسيره = رقم مغلوط أمام المستخدم.

القاعدة المركزية المُثبتة هنا: لا يُعرض رقم إلا إذا كان قياسًا اجتاز التدقيق.
"""

from __future__ import annotations

import json

import pytest

from display import (
    MSG_INVALID,
    MSG_LIFT_WRIST,
    MSG_MOVE,
    MSG_NO_CONTACT,
    MSG_SENSOR_FAULT,
    MSG_WARN,
    PATIENT_FIELDS,
    Severity,
    Tile,
    build_carer_screen,
    build_screen,
    format_clock,
    render_line,
)
from interface import VitalSample
from validator import (
    CONTACT_IR_THRESHOLD,
    Status,
    Validator,
)


def _sample(t=0.0, ir_dc=38_000.0, hr=74.0, spo2=97.0, temp=33.4, movement=0.5):
    return VitalSample(t=t, ir_dc=ir_dc, heart_rate=hr, spo2=spo2,
                       skin_temp=temp, movement=movement)


def _screen(sample, validator=None):
    validator = validator or Validator()
    return build_screen(validator.validate(sample))


# ── الحالة السليمة ──

def test_valid_shows_number_without_message():
    model = _screen(_sample())
    tile = model.tile("heart_rate")
    assert tile.severity is Severity.NORMAL
    assert tile.value_text == "74"      # بلا كسور: دقّة كاذبة توحي بيقين غير موجود
    assert tile.message is None
    assert model.banners == []


def test_skin_temp_label_never_claims_body_temperature():
    """تسمية `حرارة الجسم` على شاشة مستخدم خطأ طبي — القياس من جلد المعصم."""
    tile = _screen(_sample()).tile("skin_temp")
    assert "Body" not in tile.label
    assert "Wrist" in tile.label
    assert tile.value_text == "33.4"


def test_movement_is_never_displayed_to_patient():
    """الحركة مؤشّر داخلي يقود الإنذار، لا رقم يُقرأ."""
    assert "movement" not in PATIENT_FIELDS
    assert all(tile.name != "movement" for tile in _screen(_sample()).tiles)


# ── الحالات المحجوبة: لا رقم ──

def test_no_contact_blocks_every_wrist_number():
    """المعصم مرفوع: الحسّاس ما زال يطلّع أرقامًا تشبه القراءة — تُحجب كلها."""
    model = _screen(_sample(ir_dc=CONTACT_IR_THRESHOLD - 1, hr=88.0, spo2=96.0, temp=32.0))
    assert model.contact is False
    for tile in model.tiles:
        assert tile.value_text is None, f"{tile.name}: رقم ظهر بلا تلامس"
        assert tile.severity is Severity.BLOCKED
        assert tile.message == MSG_NO_CONTACT
    assert [b.kind for b in model.banners] == ["contact"]


def test_invalid_reading_shows_message_not_number():
    model = _screen(_sample(hr=320.0))          # مستحيل فيزيائيًا
    tile = model.tile("heart_rate")
    assert tile.value_text is None
    assert tile.message == MSG_INVALID
    assert tile.severity is Severity.BLOCKED
    # بقيّة الحقول سليمة ⟵ الحجب يخصّ الحقل المعطوب وحده
    assert model.tile("spo2").value_text is not None


def test_nan_is_blocked_not_printed():
    tile = _screen(_sample(spo2=float("nan"))).tile("spo2")
    assert tile.value_text is None
    assert tile.severity is Severity.BLOCKED


def test_missing_channel_is_blocked():
    tile = _screen(_sample(hr=None)).tile("heart_rate")
    assert tile.value_text is None
    assert tile.severity is Severity.BLOCKED


def test_stale_value_never_survives_contact_loss():
    """
    أخطر سيناريو في الشاشة: قراءة سليمة ثم يُرفع المعصم.
    الرقم القديم يجب أن يختفي فورًا — بقاؤه يوهم المستخدم بقراءة حيّة.
    """
    validator = Validator()
    first = _screen(_sample(t=0.0), validator)
    assert first.tile("heart_rate").value_text == "74"

    second = _screen(_sample(t=30.0, ir_dc=1_500.0), validator)
    assert second.tile("heart_rate").value_text is None


# ── الحالة الشاذّة الممكنة: الرقم يُعرض مع تمييز ──

def test_warn_keeps_the_number_and_marks_it():
    """قراءة شاذّة لكنها حقيقية ⇒ تُعرض. حجبها يخفي معلومة طبية مهمة."""
    tile = _screen(_sample(spo2=89.0)).tile("spo2")
    assert tile.value_text == "89"
    assert tile.severity is Severity.WARN
    assert tile.message == MSG_WARN


def test_warning_does_not_rely_on_colour_alone():
    """عمى الألوان + ضعف البصر: لا بد من أيقونة **و** نص مع كل حالة غير عادية."""
    for sample in (_sample(spo2=89.0), _sample(hr=None), _sample(ir_dc=900.0)):
        for tile in _screen(sample).tiles:
            if tile.severity is not Severity.NORMAL:
                assert tile.icon, f"{tile.name}: تمييز بلا أيقونة"
                assert tile.message, f"{tile.name}: تمييز بلا نص"


# ── الإنذار مستقل عن جودة القراءات ──

def test_movement_alert_fires_while_readings_stay_valid():
    """flags (جودة بيانات) و alerts (إجراء مطلوب) مساران منفصلان."""
    validator = Validator(immobility_limit_s=60.0)
    model = None
    for i in range(5):                       # سكون تام لمدة تتجاوز الحدّ
        model = _screen(_sample(t=i * 30.0, movement=0.0), validator)

    assert model.tile("heart_rate").severity in (Severity.NORMAL, Severity.WARN)
    assert model.tile("heart_rate").value_text is not None
    assert [b.kind for b in model.banners] == ["movement"]
    assert model.banners[0].text == MSG_MOVE
    assert model.needs_sound is True         # التنبيه البصري وحده لا يكفي


def test_movement_alert_outranks_contact_banner():
    """خطر التقرّح يتصدّر مشكلة البيانات."""
    validator = Validator(immobility_limit_s=60.0)
    model = None
    for i in range(5):
        model = _screen(_sample(t=i * 30.0, ir_dc=1_200.0, movement=0.0), validator)
    assert [b.kind for b in model.banners] == ["movement", "contact"]


def test_no_alert_means_no_sound():
    assert _screen(_sample()).needs_sound is False


# ── الإنذارات الجديدة (قرارات الفريق الطبي 12 أغسطس) ──

def test_wrist_reminder_has_its_own_banner_and_sound():
    validator = Validator(wrist_rest_limit_s=60.0)
    model = None
    for i in range(5):
        model = _screen(_sample(t=i * 30.0, movement=0.9), validator)
    kinds = [b.kind for b in model.banners]
    assert "lift_wrist" in kinds
    assert model.banners[kinds.index("lift_wrist")].text == MSG_LIFT_WRIST
    assert model.needs_sound is True


def test_sensor_fault_shows_a_maintenance_banner_with_sound():
    model = _screen(_sample(ir_dc=9_999_999.0))
    assert [b.kind for b in model.banners] == ["sensor_fault"]
    assert model.banners[0].text == MSG_SENSOR_FAULT
    assert model.needs_sound is True


def test_sensor_fault_suppresses_the_put_your_wrist_message():
    """
    "ضع معصمك" نداء لا يُصلح جهازًا معطوبًا، وتكراره يدفع المستخدم
    لضغط معصمه أكثر بلا فائدة — وقد يكون ضعيف الإحساس فلا يشعر بالضرر.
    """
    model = _screen(_sample(ir_dc=9_999_999.0))
    assert model.contact is False
    assert "contact" not in [b.kind for b in model.banners]
    # ومع ذلك لا رقم يُعرض: العطب لا يفتح الباب لقراءة غير متحقَّقة
    for tile in model.tiles:
        assert tile.value_text is None


def test_lifted_wrist_still_shows_the_ordinary_contact_message():
    model = _screen(_sample(ir_dc=1_200.0))
    assert [b.kind for b in model.banners] == ["contact"]
    assert model.needs_sound is False        # سلوك عادي لا يستحق صوتًا


def test_measurement_silence_announces_that_monitoring_stopped():
    validator = Validator(silence_limit_s=60.0)
    model = None
    for i in range(4):
        model = _screen(_sample(t=i * 30.0, ir_dc=900.0), validator)
    kinds = [b.kind for b in model.banners]
    assert "silent" in kinds
    assert model.needs_sound is True


def test_alert_banners_are_ordered_by_risk():
    """خطر التقرّح يتصدّر، ثم تعطّل المراقبة، ثم فقد التلامس العادي."""
    validator = Validator(immobility_limit_s=60.0, wrist_rest_limit_s=60.0,
                          silence_limit_s=60.0)
    model = None
    for i in range(4):
        model = _screen(_sample(t=i * 30.0, movement=0.0), validator)
    kinds = [b.kind for b in model.banners]
    assert kinds.index("movement") < kinds.index("lift_wrist")


# ── ثوابت عامة ──

@pytest.mark.parametrize("sample", [
    _sample(),
    _sample(ir_dc=500.0),
    _sample(hr=None, spo2=float("nan"), temp=99.0),
    _sample(spo2=89.0),
    _sample(hr=320.0),
])
def test_blocked_tile_never_carries_a_number(sample):
    """الثابتة التي تحمي عين المستخدم — مُختبرة على كل الحالات لا على واحدة."""
    for tile in _screen(sample).tiles:
        if tile.severity is Severity.BLOCKED:
            assert tile.value_text is None


def test_defensive_guard_blocks_value_less_valid_status():
    """
    حالة لا ينتجها المدقّق اليوم، لكن الشاشة لا تثق: حالة تدّعي رقمًا بلا رقم
    تُعامل كقراءة متعذّرة، لا يُطبع مكانها فراغ.
    """
    from display import _build_tile
    from validator import FieldResult, ValidationResult

    broken = ValidationResult(
        t=0.0, contact=True, ir_dc=38_000.0,
        fields={"heart_rate": FieldResult("heart_rate", None, Status.VALID, [])},
    )
    tile = _build_tile("heart_rate", broken)
    assert tile.severity is Severity.BLOCKED
    assert tile.value_text is None
    assert tile.message == MSG_INVALID


def test_screen_model_is_json_serialisable():
    """الراسم قد يكون في عملية أخرى (صفحة ويب) — الوصف لا بد أن يعبر كـ JSON."""
    payload = json.dumps(_screen(_sample()).to_dict(), ensure_ascii=False)
    assert "heart_rate" in payload


def test_text_renderer_matches_screen_rules():
    """الطرفية والشاشة تقرآن من نفس المصدر — لا تتفرّع القاعدة إلى نسختين."""
    validator = Validator()
    result = validator.validate(_sample(ir_dc=800.0))
    line = render_line(result)
    assert MSG_NO_CONTACT in line
    assert "74" not in line                  # لا رقم يتسرّب بلا تلامس


# ── شاشة المرافق ──

def test_carer_screen_hides_no_number_that_the_user_screen_hid():
    """
    قاعدة واحدة للشاشتين: القراءة المرفوضة مرفوضة للطرفين.
    كون المتابِع مرافقًا لا يجعل الرقم التالف صالحًا.
    """
    # المعصم مرفوع والحسّاس يطلّع أرقامًا مثالية
    carer = build_carer_screen(Validator().validate(
        _sample(ir_dc=900.0, hr=72.0, spo2=98.0, temp=33.0)))
    assert carer.normal == [] and carer.abnormal == []
    assert len(carer.blocked) == 3
    for tile in carer.blocked:
        assert tile.value_text is None


def test_carer_screen_never_reassures_while_no_reading_arrives():
    """
    "لا يوجد ما يستدعي التدخّل" بينما القراءات الثلاث محجوبة = طمأنة كاذبة —
    وهي الحالة نفسها التي بُني لأجلها إنذار الصمت، قبل بلوغ حدّه الزمني.
    """
    validator = Validator()
    frozen = None
    for i in range(validator.stuck_repeat_limit + 1):
        frozen = build_carer_screen(validator.validate(
            _sample(t=i * 30.0, hr=74.0, spo2=97.0, temp=33.4, movement=0.5 + i * 0.001)))
    assert len(frozen.blocked) == 3          # الحسّاس متجمّد
    assert frozen.monitoring is False        # فلا تُعلن الشاشة الاطمئنان
    assert build_carer_screen(Validator().validate(_sample())).monitoring is True


def test_carer_screen_flags_attention_only_when_something_needs_it():
    quiet = build_carer_screen(Validator().validate(_sample()))
    assert quiet.attention is False
    assert quiet.monitoring is True
    assert quiet.alerts == [] and quiet.abnormal == []
    assert len(quiet.normal) == 3

    warned = build_carer_screen(Validator().validate(_sample(spo2=89.0)))
    assert warned.attention is True
    assert [t.name for t in warned.abnormal] == ["spo2"]


def test_carer_screen_separates_alerts_from_readings():
    """المرافق يحتاج 'ما الذي يستدعي تدخّلًا' قبل الأرقام."""
    validator = Validator(immobility_limit_s=60.0)
    carer = None
    for i in range(4):
        carer = build_carer_screen(validator.validate(_sample(t=i * 30.0, movement=0.0)))
    assert [a.kind for a in carer.alerts] == ["movement"]
    assert carer.attention is True
    assert carer.immobility_s >= 60.0          # ومنذ متى — لا مجرّد "يوجد إنذار"


def test_carer_model_is_json_serialisable():
    payload = json.dumps(build_carer_screen(Validator().validate(_sample())).to_dict(),
                         ensure_ascii=False)
    assert "immobility_s" in payload


def test_clock_formatting():
    assert format_clock(0) == "00:00"
    assert format_clock(90) == "01:30"
    assert format_clock(3_600) == "01:00:00"
    assert format_clock(-5) == "00:00"       # زمن سالب لا يُعرض كقيمة غريبة


def test_tile_is_immutable():
    """الوصف لا يُعدَّل بعد توليده — راسم يعدّل قيمة يعني قاعدة سلامة تُلتف."""
    tile = _screen(_sample()).tile("heart_rate")
    with pytest.raises(Exception):
        tile.value_text = "999"  # type: ignore[misc]
    assert isinstance(tile, Tile)

"""
طبقة العرض — تحويل مخرجات المدقّق إلى ما تراه عين المستخدم. **منطق خالص بلا رسم**.

فُصلت عن الراسم عمدًا: قاعدة "متى يُعرض رقم ومتى لا يُعرض" قاعدة **سلامة**،
لا تفصيلة تصميم. تُكتب مرة واحدة، تُختبر آليًا، ويستهلكها أي راسم
(صفحة الشاشة في screen.py، الطرفية في demo.py، وشاشة العتاد لاحقًا).

القاعدة الحاكمة: **"لا قراءة" أصدق من قراءة مغلوطة.**
أي حالة ليست VALID/WARN ⇒ لا رقم إطلاقًا: لا رقم قديم، ولا صفر، ولا شرطة تُقرأ كقيمة.

مدخل هذه الطبقة الوحيد هو ValidationResult. لا تلمس VitalSample الخام أبدًا.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from validator import (
    ALERT_LIFT_WRIST,
    ALERT_MEASUREMENT_SILENT,
    ALERT_NEEDS_MOVEMENT,
    ALERT_SENSOR_FAULT,
    FIELD_SPECS,
    Status,
    ValidationResult,
)

# ما يُعرض للمستخدم: قيم المعصم فقط.
# movement مؤشّر داخلي يقود إنذار الخمول — رقم لا يعني المستخدم شيئًا، فلا يُعرض.
PATIENT_FIELDS = ("heart_rate", "spo2", "skin_temp")

# التسمية تعكس ما يُقاس فعلًا: "حرارة جلد المعصم" لا "حرارة الجسم".
# تسمية مضلِّلة على شاشة مستخدم = خطأ طبي، لا مسألة صياغة.
LABELS: Dict[str, str] = {
    "heart_rate": "Heart Rate",
    "spo2": "Blood Oxygen",
    "skin_temp": "Wrist Skin Temp",   # ليست Body Temp — التسمية تصف موضع القياس
    "movement": "Movement",
}

# عدد الخانات العشرية المعروضة. النبض والأكسجين بلا كسور:
# دقّة كاذبة على شاشة مستخدم توحي بيقين غير موجود، خصوصًا وSpO2 من المعصم مؤشّر اتجاه.
DECIMALS: Dict[str, int] = {"heart_rate": 0, "spo2": 0, "skin_temp": 1}

# لا اعتماد على اللون وحده: أيقونة **و** نص مع كل حالة (عمى ألوان + ضعف بصر).
ICON_WARN = "⚠"
ICON_INVALID = "⛔"
ICON_NO_CONTACT = "✋"
ICON_MOVE = "🔔"
ICON_LIFT_WRIST = "🤚"
ICON_FAULT = "🛠"
ICON_SILENT = "🔇"

MSG_WARN = "Outside expected range"
MSG_INVALID = "Reading unavailable"
MSG_NO_CONTACT = "Rest your wrist on the armrest"
MSG_MOVE = "Time to move"
MSG_LIFT_WRIST = "Lift your wrist off the armrest"
# النصّان أدناه يخاطبان المرافق لا المستخدم.
MSG_SENSOR_FAULT = "Device fault — needs checking"
MSG_SILENT = "Monitoring stopped — no readings"

# نائب القيمة الغائبة. مقصود ألا يشبه رقمًا بأي حال.
NO_VALUE_TEXT = "—"


class Severity(str, Enum):
    """شدّة العرض — يترجمها الراسم إلى لون/حجم، ولا يقرّر بنفسه شيئًا."""

    NORMAL = "NORMAL"    # رقم سليم
    WARN = "WARN"        # رقم يُعرض مع تمييز
    ALERT = "ALERT"      # إجراء مطلوب من المستخدم الآن
    BLOCKED = "BLOCKED"  # لا رقم — القراءة محجوبة


@dataclass(frozen=True)
class Tile:
    """بطاقة قياس واحدة على الشاشة. value_text=None يعني: **لا تطبع رقمًا**."""

    name: str
    label: str
    unit: str
    value_text: Optional[str]
    icon: str
    message: Optional[str]
    severity: Severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "unit": self.unit,
            "value_text": self.value_text,
            "icon": self.icon,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class Banner:
    """شريط عرضي فوق البطاقات: إجراء مطلوب أو سبب حجب عام."""

    kind: str          # "movement" أو "contact"
    icon: str
    text: str
    severity: Severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "icon": self.icon,
            "text": self.text,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class ScreenModel:
    """
    وصف كامل لِما يجب أن يظهر على الشاشة في لحظة واحدة.

    الراسم ينفّذ هذا الوصف حرفيًا ولا يضيف عليه ولا يستنتج.
    """

    t: float
    clock: str
    contact: bool
    tiles: List[Tile]
    banners: List[Banner]
    needs_sound: bool  # هل هذه اللحظة تستدعي تنبيهًا صوتيًا (لا بصريًا فقط)؟

    def tile(self, name: str) -> Tile:
        for tile in self.tiles:
            if tile.name == name:
                return tile
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "clock": self.clock,
            "contact": self.contact,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "banners": [banner.to_dict() for banner in self.banners],
            "needs_sound": self.needs_sound,
        }


def format_clock(seconds: float) -> str:
    """زمن الجلسة الافتراضي. الساعات تظهر فقط عند تجاوزها — تقليلًا للضجيج البصري."""
    total = int(max(0.0, seconds))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_value(name: str, value: float) -> str:
    return f"{value:.{DECIMALS.get(name, 1)}f}"


def _build_tile(name: str, result: ValidationResult) -> Tile:
    field_result = result.fields[name]
    label = LABELS[name]
    unit = FIELD_SPECS[name].unit
    status = field_result.status
    value = field_result.value

    # حارس دفاعي: حالة تدّعي وجود رقم بلا رقم = خلل في المصدر.
    # الأأمن أن تُعامل كقراءة متعذّرة، لا أن يُطبع فراغ مكان الرقم.
    if status in (Status.VALID, Status.WARN) and value is None:
        status = Status.INVALID

    if status is Status.VALID:
        return Tile(name, label, unit, _format_value(name, float(value)),
                    "", None, Severity.NORMAL)

    if status is Status.WARN:
        # الرقم يُعرض — لأنه قياس حقيقي — لكن مع أيقونة ونص، لا لون وحده.
        return Tile(name, label, unit, _format_value(name, float(value)),
                    ICON_WARN, MSG_WARN, Severity.WARN)

    if status is Status.NO_CONTACT:
        return Tile(name, label, unit, None,
                    ICON_NO_CONTACT, MSG_NO_CONTACT, Severity.BLOCKED)

    # INVALID: السبب التقني (NaN/تجمّد/قفزة) لا يعني المستخدم — رسالة واحدة واضحة.
    # التفصيل مكانه سجلّ التدقيق، لا شاشة المستخدم.
    return Tile(name, label, unit, None, ICON_INVALID, MSG_INVALID, Severity.BLOCKED)


def build_screen(result: ValidationResult) -> ScreenModel:
    """
    ValidationResult ⟵ المصدر الوحيد. لا مدخل آخر لهذه الدالة، ولا ذاكرة بين اللحظات:
    الشاشة انعكاس للحظة الحالية فقط، فلا يمكن أن يبقى رقم قديم معروضًا بالخطأ.
    """
    tiles = [_build_tile(name, result) for name in PATIENT_FIELDS]

    banners: List[Banner] = []

    # الترتيب مقصود ومرتّب بالخطر، لا بترتيب الاكتشاف في الكود:
    # خطر التقرّح (الجسم ثم المعصم) ⟵ ثم تعطّل المراقبة ⟵ ثم فقد التلامس العادي.
    if ALERT_NEEDS_MOVEMENT in result.alerts:
        banners.append(Banner("movement", ICON_MOVE, MSG_MOVE, Severity.ALERT))

    if ALERT_LIFT_WRIST in result.alerts:
        banners.append(Banner("lift_wrist", ICON_LIFT_WRIST, MSG_LIFT_WRIST, Severity.ALERT))

    if ALERT_SENSOR_FAULT in result.alerts:
        banners.append(Banner("sensor_fault", ICON_FAULT, MSG_SENSOR_FAULT, Severity.ALERT))

    if ALERT_MEASUREMENT_SILENT in result.alerts:
        banners.append(Banner("silent", ICON_SILENT, MSG_SILENT, Severity.ALERT))

    # رسالة تلامس واحدة عامة بدل تكرارها في ثلاث بطاقات — سبب واحد ونداء واحد.
    # تُحجب عند وجود عطب: "ضع معصمك" نداء لا يُصلح جهازًا معطوبًا، وتكراره
    # يدفع المستخدم لضغط معصمه أكثر بلا فائدة.
    if not result.contact and ALERT_SENSOR_FAULT not in result.alerts:
        banners.append(Banner("contact", ICON_NO_CONTACT, MSG_NO_CONTACT, Severity.BLOCKED))

    # التنبيه البصري وحده لا يكفي: المستخدم قد يكون غير ناظر للشاشة،
    # والمرافق قد يكون في غرفة أخرى. كل إنذار يستدعي صوتًا.
    needs_sound = any(b.severity is Severity.ALERT for b in banners)

    model = ScreenModel(
        t=result.t,
        clock=format_clock(result.t),
        contact=result.contact,
        tiles=tiles,
        banners=banners,
        needs_sound=needs_sound,
    )

    # ثابتة السلامة، مؤكَّدة عند التوليد لا عند الرسم فقط:
    # بطاقة محجوبة لا تحمل رقمًا بأي حال من الأحوال.
    for tile in model.tiles:
        assert not (tile.severity is Severity.BLOCKED and tile.value_text is not None), (
            f"خرق قاعدة سلامة: بطاقة محجوبة تحمل رقمًا ({tile.name})"
        )
    return model


@dataclass(frozen=True)
class CarerModel:
    """
    شاشة المرافق — **ملخّص ما يحتاج تدخّلًا**، لا نسخة ثانية من شاشة المستخدم.

    الفرق مقصود: المرافق قد يكون في غرفة أخرى ولا يتابع الأرقام لحظة بلحظة،
    فما يفيده هو "هل يوجد ما يستدعي التدخّل الآن، ومنذ متى".

    ⚠️ تُبنى على نفس الحدود السريرية **غير المعايرة**، فترث القيد نفسه.
    """

    clock: str
    attention: bool                 # هل يوجد ما يستدعي تدخّلًا الآن؟
    monitoring: bool                # هل يصل أي رقم أصلًا؟
    alerts: List[Banner]            # إجراءات مطلوبة
    abnormal: List[Tile]            # قراءات شاذّة معروضة (WARN)
    blocked: List[Tile]             # قراءات متعذّرة
    normal: List[Tile]              # قراءات سليمة — للطمأنة لا للمتابعة
    immobility_s: float
    wrist_rest_s: float
    silence_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clock": self.clock,
            "attention": self.attention,
            "monitoring": self.monitoring,
            "alerts": [a.to_dict() for a in self.alerts],
            "abnormal": [t.to_dict() for t in self.abnormal],
            "blocked": [t.to_dict() for t in self.blocked],
            "normal": [t.to_dict() for t in self.normal],
            "immobility_s": round(self.immobility_s, 1),
            "wrist_rest_s": round(self.wrist_rest_s, 1),
            "silence_s": round(self.silence_s, 1),
        }


def build_carer_screen(result: ValidationResult) -> CarerModel:
    """
    تُشتقّ من نفس ScreenModel — فقاعدة "متى يُعرض رقم" واحدة على الشاشتين.
    شاشة المرافق **لا تكشف رقمًا حجبته شاشة المستخدم**: القراءة المرفوضة مرفوضة
    للطرفين، وكونه مرافقًا لا يجعل الرقم التالف صالحًا.
    """
    model = build_screen(result)
    # "لا يوجد ما يستدعي التدخّل" بينما لا يصل رقم واحد = طمأنة كاذبة.
    # هي بالضبط الحالة التي بُني لأجلها إنذار الصمت، لكن قبل بلوغ حدّه الزمني.
    monitoring = any(t.value_text is not None for t in model.tiles)
    return CarerModel(
        clock=model.clock,
        attention=bool(model.banners) or any(
            t.severity is Severity.WARN for t in model.tiles),
        monitoring=monitoring,
        alerts=[b for b in model.banners if b.severity is Severity.ALERT],
        abnormal=[t for t in model.tiles if t.severity is Severity.WARN],
        blocked=[t for t in model.tiles if t.severity is Severity.BLOCKED],
        normal=[t for t in model.tiles if t.severity is Severity.NORMAL],
        immobility_s=result.immobility_s,
        wrist_rest_s=result.wrist_rest_s,
        silence_s=result.silence_s,
    )


def render_line(result: ValidationResult) -> str:
    """
    عرض نصّي لسطر واحد (الطرفية/السجلّ). يستهلك **نفس** ScreenModel،
    حتى لا تتفرّع قاعدة العرض إلى نسختين تتباعدان مع الوقت.
    """
    model = build_screen(result)
    parts = []
    for tile in model.tiles:
        if tile.value_text is None:
            parts.append(f"{tile.label} {tile.icon} {tile.message}")
        elif tile.message:
            parts.append(f"{tile.label} {tile.value_text}{tile.unit} {tile.icon} {tile.message}")
        else:
            parts.append(f"{tile.label} {tile.value_text}{tile.unit}")
    line = " | ".join(parts)
    for banner in model.banners:
        if banner.severity is Severity.ALERT:
            line += f"   {banner.icon} {banner.text}"
    return line

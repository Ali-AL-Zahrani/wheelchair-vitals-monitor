"""
محرّك التدقيق — قلب النظام.

ثلاثة أنظمة **مستقلة** تمامًا:
  1) بوابة التلامس  : لا معصم على المسند ⇒ تُصفَّر كل قيم المعصم بغضّ النظر عن منطقيّتها.
  2) الفحوصات الحيوية: SANITY (مستحيل) ⇒ INVALID وتُرمى القيمة.
                       CLINICAL (ممكن لكن شاذّ) ⇒ WARN وتُحفظ القيمة مع تعليم.
                       + فحصان ذوا ذاكرة: stuck (تجمّد الحسّاس) و artifact (قفزة مفاجئة).
  3) watchdog الخمول : مستقل عن المعصم كليًا؛ يعتمد على مصدر الحركة المنفصل.

المخرجات تفصل flags (جودة بيانات) عن alerts (إجراء مطلوب):
النبض قد يكون VALID وإنذار الحركة شغّال في نفس اللحظة.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from interface import VitalSample

# ═══════════════════════════════════════════════════════════════════════════
#  عتبات النظام
# ═══════════════════════════════════════════════════════════════════════════

# عتبة التلامس بنطاق المعصم. ir_dc من المعصم أقل من راحة اليد (تروية شعرية أقل
# والعظم أقرب تحت الجلد)، وعتبة مرتفعة زيادة تقرأ NO_CONTACT ومعصم المستخدم مستند.
# مشدودة مع مستوى ir_dc في mock_sensor.py — أي تغيير في أحدهما يوجب الآخر.
CONTACT_IR_THRESHOLD = 20_000.0

# حدّ معقولية لا عتبة سريرية: مدى ADC في MAX30102 ثمانية عشر بتًا (0..262143).
# قيمة فوقه أو سالبة لا يمكن أن تأتي من الحسّاس، بل من درايفر/ناقل I2C معطوب —
# وبوابة التلامس تُبنى عليها، فقيمة تالفة قد تفتح البوابة وتُمرّر ضوضاء كقراءات.
CONTACT_IR_SANITY_MAX = 262_143.0

# ✅ معتمدة من الفريق الطبي (12 أغسطس 2026): تحويل الوزن كل 15 دقيقة.
IMMOBILITY_LIMIT_S = 15 * 60.0

# مدّة الاستناد المتصل على المعصم قبل التذكير برفعه — مساوية لحدّ الخمول.
# مؤقّت مستقل عن حركة الجسم عمدًا: لو رُبط بحدث التحريك، لبقي معصم من لم يحرّكه
# أحد ساعاتٍ بلا تذكير — أي يغيب التنبيه في اللحظة التي يبلغ فيها الخطر ذروته.
WRIST_REST_LIMIT_S = 15 * 60.0

# ✅ معتمدة من الفريق الطبي (12 أغسطس 2026): 15 دقيقة.
# طولها مقصود: رقم قصير يولّد إنذارات مزعجة عند كل رفع معصم عادي (أكل، انتقال)،
# والإنذار المزعج يُطفأ الصوت بسببه ⟵ فيُفقد معه إنذار عطب الحسّاس.
SILENCE_LIMIT_S = 15 * 60.0

# مصدر الحركة: IMU (قرار الفريق الطبي 12 أغسطس). القيمة بوحدات مؤشّر حركة مجرّد.
MOVEMENT_THRESHOLD = 0.15

# عدد العيّنات المتطابقة تمامًا التي تعني "الحسّاس متجمّد" (≈5 دقائق عند 30 ث/عيّنة).
STUCK_REPEAT_LIMIT = 10

# بعد هذا العدد من الـ artifacts المتتالية نعتبر التغيّر حقيقيًا ونعيد ضبط الأساس،
# وإلا بقي الحقل مرفوضًا للأبد بعد أي تغيّر فسيولوجي سريع حقيقي.
ARTIFACT_RESYNC_N = 3


class Status(str, Enum):
    VALID = "VALID"            # اعرض الرقم عاديًا
    WARN = "WARN"              # اعرض الرقم + تمييز بصري
    INVALID = "INVALID"        # لا رقم ⟵ "تعذّرت القراءة"
    NO_CONTACT = "NO_CONTACT"  # لا رقم ⟵ "ضع يدك على المسند"


# ── أعلام جودة البيانات (flags) ──
FLAG_SANITY_MISSING = "SANITY_MISSING"          # القيمة None
FLAG_SANITY_TYPE = "SANITY_TYPE"                # نوع غير رقمي من الدرايفر
FLAG_SANITY_NAN = "SANITY_NAN"                  # NaN / inf
FLAG_SANITY_RANGE = "SANITY_RANGE"              # خارج الممكن فيزيائيًا
FLAG_CLINICAL_LOW = "CLINICAL_LOW"              # ممكن لكن تحت الحد السريري
FLAG_CLINICAL_HIGH = "CLINICAL_HIGH"            # ممكن لكن فوق الحد السريري
FLAG_STUCK = "STUCK"                            # الحسّاس متجمّد على نفس القيمة
FLAG_ARTIFACT = "ARTIFACT"                      # قفزة مفاجئة = رعشة/حركة
FLAG_ARTIFACT_RESYNC = "ARTIFACT_RESYNC"        # أُعيد ضبط الأساس بعد قفزات متتالية
FLAG_NO_CONTACT = "NO_CONTACT"                  # لا يد على المسند
FLAG_IR_IMPLAUSIBLE = "IR_IMPLAUSIBLE"          # ir_dc خارج مدى الحسّاس ⇒ عطب مصدر لا رفع معصم
FLAG_MOVEMENT_UNVERIFIED = "MOVEMENT_UNVERIFIED"  # تعذّر التحقق من الحركة
FLAG_TIME_BACKWARD = "TIME_BACKWARD"            # ساعة الحسّاس رجعت للخلف

# ── إنذارات (alerts) = إجراء مطلوب من المستخدم ──
ALERT_NEEDS_MOVEMENT = "NEEDS_MOVEMENT"      # تحويل وزن الجسم عن نقاط الضغط
ALERT_LIFT_WRIST = "LIFT_WRIST"              # رفع المعصم عن المسند (ضغط موضعي مطوّل)
ALERT_SENSOR_FAULT = "SENSOR_FAULT"          # عطب في الحسّاس/الناقل — يستدعي صيانة
ALERT_MEASUREMENT_SILENT = "MEASUREMENT_SILENT"  # لا قياس يصل منذ مدّة


@dataclass(frozen=True)
class FieldSpec:
    """حدود حقل واحد. الفصل بين sanity و clinical مقصود — خلطهما خطأ طبي."""

    name: str
    unit: str
    sanity_min: float                       # تحت هذا = مستحيل فيزيائيًا ⇒ ترمى
    sanity_max: float                       # فوق هذا = مستحيل فيزيائيًا ⇒ ترمى
    clinical_min: Optional[float]           # تحت هذا = شاذّ لكنه ممكن ⇒ WARN
    clinical_max: Optional[float]           # فوق هذا = شاذّ لكنه ممكن ⇒ WARN
    artifact_max_delta: Optional[float]     # أقصى تغيّر مقبول بين عيّنتين متتاليتين
    wrist_derived: bool                     # هل مصدره المعصم المستند على المسند؟


# ═══════════════════════════════════════════════════════════════════════════
#  حدود الحقول. تُبدَّل من هذا المكان وحده، وتُسجَّل لقطة منها في بداية كل جلسة.
#
#  ملاحظة على artifact_max_delta: مضبوطة على معدّل العيّنة الحالي (≈30 ث).
#  أي تغيير في معدّل العيّنة يوجب إعادة معايرتها، وإلا صارت إمّا عمياء أو حسّاسة زيادة.
# ═══════════════════════════════════════════════════════════════════════════
FIELD_SPECS: Dict[str, FieldSpec] = {
    "heart_rate": FieldSpec(
        name="heart_rate", unit="bpm",
        sanity_min=20.0, sanity_max=250.0,      # خارج هذا لا يمثّل قلبًا نابضًا
        clinical_min=50.0, clinical_max=120.0,
        artifact_max_delta=30.0, wrist_derived=True,
    ),
    "spo2": FieldSpec(
        name="spo2", unit="%",
        sanity_min=50.0, sanity_max=100.0,      # >100 مستحيل، <50 قراءة حسّاس لا إنسان
        clinical_min=94.0, clinical_max=None,
        artifact_max_delta=8.0, wrist_derived=True,
    ),
    "skin_temp": FieldSpec(
        name="skin_temp", unit="°C",
        sanity_min=10.0, sanity_max=45.0,       # حرارة جلد المعصم، لا حرارة جسم
        clinical_min=30.0, clinical_max=37.0,   # نطاق "متوقّع" فقط — ممنوع بناء منطق حمّى عليه
        artifact_max_delta=2.0, wrist_derived=True,
    ),
    "movement": FieldSpec(
        name="movement", unit="—",
        sanity_min=0.0, sanity_max=float("inf"),  # الوحدات غير محسومة حتى يُحسم مصدر الحركة
        clinical_min=None, clinical_max=None,     # الخمول ليس شذوذًا في القيمة، بل في الزمن ⟵ watchdog
        artifact_max_delta=None, wrist_derived=False,  # الحركة يُفترض أنها تقفز — لا فحص artifact
    ),
}


# الحقول المأخوذة من المعصم — تُشتقّ من FIELD_SPECS ولا تُكرَّر يدويًا،
# وإلا انفصلت القائمتان عند إضافة حقل جديد.
_WRIST_FIELD_NAMES = tuple(n for n, s in FIELD_SPECS.items() if s.wrist_derived)


@dataclass
class FieldResult:
    """نتيجة حقل واحد. value = None يعني: **لا تعرض رقمًا إطلاقًا**."""

    name: str
    value: Optional[float]
    status: Status
    flags: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """
    مخرج المدقّق لعيّنة واحدة — وهو **المصدر الوحيد** الذي تقرأ منه الشاشة.

    flags  = جودة البيانات (لماذا رُفضت/عُلّمت قراءة).
    alerts = إجراء مطلوب من المستخدم، مستقل تمامًا عن جودة القراءات.
    """

    t: float
    contact: bool
    ir_dc: Optional[float]
    fields: Dict[str, FieldResult]
    flags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    immobility_s: float = 0.0     # منذ آخر تحويل وزن مؤكَّد
    wrist_rest_s: float = 0.0     # استناد متصل للمعصم على المسند
    silence_s: float = 0.0        # منذ آخر قراءة معصم صالحة للعرض

    def status_of(self, name: str) -> Status:
        return self.fields[name].status

    def value_of(self, name: str) -> Optional[float]:
        return self.fields[name].value

    def flags_of(self, name: str) -> List[str]:
        return self.fields[name].flags


class Validator:
    """
    مدقّق ذو ذاكرة عبر العيّنات. نسخة واحدة لكل مستخدم/جلسة.

    مبدأ ثابت: عند الشك تُرفض القراءة. "لا قراءة" أصدق من قراءة مغلوطة.
    """

    def __init__(
        self,
        contact_ir_threshold: float = CONTACT_IR_THRESHOLD,
        immobility_limit_s: float = IMMOBILITY_LIMIT_S,
        movement_threshold: float = MOVEMENT_THRESHOLD,
        stuck_repeat_limit: int = STUCK_REPEAT_LIMIT,
        wrist_rest_limit_s: float = WRIST_REST_LIMIT_S,
        silence_limit_s: float = SILENCE_LIMIT_S,
    ) -> None:
        self.contact_ir_threshold = contact_ir_threshold
        self.immobility_limit_s = immobility_limit_s
        self.movement_threshold = movement_threshold
        self.stuck_repeat_limit = stuck_repeat_limit
        self.wrist_rest_limit_s = wrist_rest_limit_s
        self.silence_limit_s = silence_limit_s
        self.reset()

    def reset(self) -> None:
        """تصفير الذاكرة — تُستدعى عند بدء جلسة جديدة."""
        self._last_accepted: Dict[str, float] = {}   # أساس فحص artifact
        self._last_seen: Dict[str, float] = {}       # آخر قيمة اجتازت sanity (لفحص stuck)
        self._repeat: Dict[str, int] = {}
        self._artifact_streak: Dict[str, int] = {}
        self._last_t: Optional[float] = None
        # كل المؤقّتات **مُراكِمة**، لا مشتقّة من طوابع زمنية محفوظة. السبب في validate().
        self._immobility_s: float = 0.0
        self._wrist_rest_s: float = 0.0
        self._silence_s: float = 0.0

    # ── واجهة الاستعمال الوحيدة ──
    def validate(self, sample: VitalSample) -> ValidationResult:
        sample_flags: List[str] = []

        # ساعة الحسّاس هي المرجع، لا ساعة النظام — حتى تعمل المحاكاة المعجَّلة بنفس المنطق.
        #
        # الخمول يُراكَم بفروق زمنية موجبة فقط، ولا يُشتقّ من "آخر لحظة حركة" محفوظة.
        # الفرق ليس أسلوبيًا: ساعة راجعة للخلف (إعادة تشغيل المتحكّم مثلًا) كانت
        # ستُصفّر الطابع المحفوظ ⇒ تضيع المدّة المتراكمة ويُكتم إنذار قائم، فيقضي
        # المستخدم ضعف المدّة بلا تحويل وزن. المُراكِم يحتفظ بها: العيّنة المشبوهة
        # وحدها تُهمل (dt=0) ولا يُفقد ما قبلها.
        dt = 0.0
        if self._last_t is not None:
            if sample.t < self._last_t:
                sample_flags.append(FLAG_TIME_BACKWARD)
            else:
                dt = sample.t - self._last_t
        self._last_t = sample.t

        # ── 1) بوابة التلامس ──
        # حسّاس بلا معصم يطلّع ضوضاء قد تشبه قراءة حقيقية، لذا البوابة تسبق كل فحص آخر.
        # رفع المعصم وإرجاعه حدث متكرر جدًا، لا حالة استثنائية.
        contact = self._has_contact(sample.ir_dc)
        if not contact:
            sample_flags.append(FLAG_NO_CONTACT)
            # تمييز مقصود في السجلّ: "المعصم مرفوع" حدث طبيعي متكرر،
            # أما "ir_dc خارج مدى الحسّاس" فعطب عتاد يستدعي صيانة لا تنبيه مستخدم.
            if _is_number(sample.ir_dc) and not _ir_in_range(float(sample.ir_dc)):
                sample_flags.append(FLAG_IR_IMPLAUSIBLE)

        # ── 2) الفحوصات الحيوية ──
        results: Dict[str, FieldResult] = {}
        for name, spec in FIELD_SPECS.items():
            raw = getattr(sample, name)
            results[name] = self._check_field(spec, raw, contact)

        # ── 3) watchdog الخمول — مستقل عن المعصم ──
        mov = results["movement"]
        movement_confirmed = (
            mov.status in (Status.VALID, Status.WARN)
            and mov.value is not None
            and mov.value >= self.movement_threshold
        )
        if mov.status not in (Status.VALID, Status.WARN) or mov.value is None:
            # لا نستطيع تأكيد الحركة ⇒ المؤقّت يستمر (fail-loud).
            # تنبيه المستخدم للحركة بلا داعٍ غير ضار، أما كتم الإنذار فخطر تقرّحات.
            sample_flags.append(FLAG_MOVEMENT_UNVERIFIED)

        # حركة مؤكَّدة وحدها تصفّر المُراكِم. أي شيء آخر — بما فيه خلل الساعة — يُبقيه.
        if movement_confirmed:
            self._immobility_s = 0.0
        else:
            self._immobility_s += dt

        alerts: List[str] = []
        if self._immobility_s >= self.immobility_limit_s:
            alerts.append(ALERT_NEEDS_MOVEMENT)

        # ── 4) watchdog ضغط المعصم — مستقل عن حركة الجسم ──
        # جلد المعصم رقيق فوق عظم مباشرة، وقد يكون الإحساس ضعيفًا فلا ينبّه الألم.
        # رفع المعصم وحده يصفّر العدّاد؛ تحريك الجسم لا يرفع الضغط عن المعصم.
        if contact:
            self._wrist_rest_s += dt
        else:
            self._wrist_rest_s = 0.0
        if self._wrist_rest_s >= self.wrist_rest_limit_s:
            alerts.append(ALERT_LIFT_WRIST)

        # ── 5) عطب الحسّاس ── قيمة خارج مدى العتاد ⇒ صيانة، لا إجراء من المستخدم.
        if FLAG_IR_IMPLAUSIBLE in sample_flags:
            alerts.append(ALERT_SENSOR_FAULT)

        # ── 6) صمت القياس ──
        # المراقبة متوقّفة فعليًا حين لا يصل أي رقم صالح للعرض، مهما كان السبب.
        # شاشة صامتة تُقرأ كـ"كل شيء على ما يرام"، وهي أخطر من شاشة تصرخ.
        if any(results[name].value is not None for name in _WRIST_FIELD_NAMES):
            self._silence_s = 0.0
        else:
            self._silence_s += dt
        if self._silence_s >= self.silence_limit_s:
            alerts.append(ALERT_MEASUREMENT_SILENT)

        return ValidationResult(
            t=sample.t,
            contact=contact,
            ir_dc=sample.ir_dc,
            fields=results,
            flags=sample_flags,
            alerts=alerts,
            immobility_s=self._immobility_s,
            wrist_rest_s=self._wrist_rest_s,
            silence_s=self._silence_s,
        )

    # ── داخلي ──
    def _has_contact(self, ir_dc: Optional[float]) -> bool:
        """
        ir_dc مفقود أو NaN أو خارج مدى الحسّاس ⇒ نفترض عدم التلامس (الافتراض الآمن).

        البوابة تُغلق عند الشك: بوابة مفتوحة بقيمة تالفة تمرّر ضوضاء إلى عين المستخدم،
        وبوابة مغلقة بالخطأ تُظهر "ضع معصمك على المسند" — والثانية غلط غير مؤذٍ.
        """
        if not _is_number(ir_dc):
            return False
        v = float(ir_dc)  # type: ignore[arg-type]
        if not _ir_in_range(v):
            return False
        return v >= self.contact_ir_threshold

    def _check_field(
        self, spec: FieldSpec, raw: object, contact: bool
    ) -> FieldResult:
        # (1) بوابة التلامس تسبق كل شيء: لا معصم ⇒ لا رقم، مهما بدت الأرقام منطقية.
        if spec.wrist_derived and not contact:
            self._forget(spec.name)  # لئلا تُقارَن قراءة ما بعد العودة بأساس قديم
            return FieldResult(spec.name, None, Status.NO_CONTACT, [FLAG_NO_CONTACT])

        # (2) SANITY: غياب/نوع/NaN/استحالة فيزيائية ⇒ ترمى القيمة.
        if raw is None:
            return FieldResult(spec.name, None, Status.INVALID, [FLAG_SANITY_MISSING])
        if not _is_number(raw):
            return FieldResult(spec.name, None, Status.INVALID, [FLAG_SANITY_TYPE])
        v = float(raw)  # type: ignore[arg-type]
        if math.isnan(v) or math.isinf(v):
            return FieldResult(spec.name, None, Status.INVALID, [FLAG_SANITY_NAN])
        if not (spec.sanity_min <= v <= spec.sanity_max):
            return FieldResult(spec.name, None, Status.INVALID, [FLAG_SANITY_RANGE])

        flags: List[str] = []

        # (3) فحص التجمّد: قيمة متطابقة بتّيًا مرارًا = حسّاس معلّق، لا حالة مستقرة.
        last_seen = self._last_seen.get(spec.name)
        if last_seen is not None and v == last_seen:
            self._repeat[spec.name] = self._repeat.get(spec.name, 1) + 1
        else:
            self._repeat[spec.name] = 1
        self._last_seen[spec.name] = v
        stuck = self._repeat[spec.name] >= self.stuck_repeat_limit

        # (4) فحص القفزة: تغيّر أكبر من الممكن فسيولوجيًا بين عيّنتين = رعشة/حركة.
        artifact = False
        baseline = self._last_accepted.get(spec.name)
        if spec.artifact_max_delta is not None and baseline is not None:
            if abs(v - baseline) > spec.artifact_max_delta:
                artifact = True
                streak = self._artifact_streak.get(spec.name, 0) + 1
                self._artifact_streak[spec.name] = streak
                if streak >= ARTIFACT_RESYNC_N:
                    # قفزات متتالية = تغيّر حقيقي مستمر ⟵ نتبنّى الأساس الجديد.
                    # العيّنة الحالية تبقى مرفوضة، والتالية تُقبل.
                    self._last_accepted[spec.name] = v
                    self._artifact_streak[spec.name] = 0
                    flags.append(FLAG_ARTIFACT_RESYNC)
            else:
                self._artifact_streak[spec.name] = 0
                self._last_accepted[spec.name] = v
        else:
            self._last_accepted[spec.name] = v

        if stuck:
            flags.append(FLAG_STUCK)
        if artifact:
            flags.append(FLAG_ARTIFACT)
        if stuck or artifact:
            # القيمة ليست قياسًا حقيقيًا ⇒ لا تصل عين المستخدم.
            return FieldResult(spec.name, None, Status.INVALID, flags)

        # (5) CLINICAL: ممكنة لكن شاذّة ⇒ تُحفظ القيمة مع تعليم.
        if spec.clinical_min is not None and v < spec.clinical_min:
            flags.append(FLAG_CLINICAL_LOW)
        if spec.clinical_max is not None and v > spec.clinical_max:
            flags.append(FLAG_CLINICAL_HIGH)

        status = Status.WARN if flags else Status.VALID
        return FieldResult(spec.name, v, status, flags)

    def _forget(self, name: str) -> None:
        """نسيان ذاكرة حقل — بعد فقد التلامس لا معنى لمقارنة القراءة الجديدة بالقديمة."""
        self._last_accepted.pop(name, None)
        self._last_seen.pop(name, None)
        self._repeat.pop(name, None)
        self._artifact_streak.pop(name, None)


def _ir_in_range(v: float) -> bool:
    """قيمة ir_dc محتملة من الحسّاس فعلًا (لا NaN ولا inf ولا خارج مدى الـ ADC)."""
    if math.isnan(v) or math.isinf(v):
        return False
    return 0.0 <= v <= CONTACT_IR_SANITY_MAX


def _is_number(v: object) -> bool:
    """bool هو int في بايثون — نستبعده صراحةً لئلا يمرّ True كقراءة."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)

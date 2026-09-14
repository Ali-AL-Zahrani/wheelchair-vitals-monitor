"""
حسّاس وهمي: قراءات واقعية + حقن أعطال مقصودة.

الهدف ليس "بيانات جميلة" بل **إجهاد المدقّق**: انقطاع تلامس، NaN، قيم مستحيلة،
حسّاس متجمّد، وقفزات artifact، إضافة إلى فترات خمول طويلة لاختبار watchdog.

الساعة افتراضية (30 ث/عيّنة افتراضيًا) حتى نختبر منطق الخمول في ثوانٍ
بدل الانتظار الحقيقي 20 دقيقة.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from interface import SensorInterface, VitalSample

# أسماء الأعطال المدعومة (تُسنَد لفهرس العيّنة):
#   "no_contact"    ⟵ المعصم رُفع: ir_dc ينهار والقيم تصير ضوضاء تشبه قراءة حقيقية
#   "frozen"        ⟵ حسّاس المعصم معلّق على نفس البِتّات (movement يبقى حيًّا: مصدره جهاز آخر)
#   "nan_spo2"      ⟵ NaN من قناة الأكسجين
#   "none_hr"       ⟵ الدرايفر يرجّع None
#   "impossible_hr" ⟵ قيمة مستحيلة فيزيائيًا
#   "artifact_hr"   ⟵ قفزة مفاجئة (رعشة/حركة يد)
#   "warn_spo2"     ⟵ قيمة ممكنة لكن تحت الحد السريري
#   "ir_broken"     ⟵ ir_dc خارج مدى الـ ADC: عطب درايفر/ناقل، لا رفع معصم
#   "still"         ⟵ المستخدم ساكن تمامًا (لاختبار إنذار الحركة)

_WRIST_FIELDS = ("ir_dc", "heart_rate", "spo2", "skin_temp")


class MockSensor(SensorInterface):
    def __init__(
        self,
        seed: int = 42,
        sample_period_s: float = 30.0,
        faults: Optional[Dict[int, str]] = None,
        start_t: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._period = sample_period_s
        self._faults: Dict[int, str] = dict(faults or {})
        self._t = start_t
        self._i = -1
        self._last: Optional[VitalSample] = None

    @property
    def index(self) -> int:
        """فهرس آخر عيّنة أُرجعت (للتشخيص في demo)."""
        return self._i

    def read(self) -> VitalSample:
        self._i += 1
        if self._i > 0:
            self._t += self._period
        fault = self._faults.get(self._i)

        # قاعدة واقعية: بالغ مستريح، معصمه مستند على المسند.
        # ir_dc بنطاق معصم. يبقى متسقًا مع CONTACT_IR_THRESHOLD في validator.py؛
        # أي تغيير في أحدهما يوجب الآخر.
        ir_dc = self._rng.gauss(38_000, 3_500)
        heart_rate = self._rng.gauss(74, 3)
        spo2 = min(100.0, self._rng.gauss(97.0, 0.8))
        skin_temp = self._rng.gauss(33.4, 0.25)
        movement = self._movement()

        if fault == "no_contact":
            # لا معصم: الحسّاس ما زال يطلّع أرقامًا — وهذه بالضبط الخطورة.
            ir_dc = self._rng.gauss(1_800, 500)
            heart_rate = self._rng.uniform(35, 180)
            spo2 = self._rng.uniform(70, 100)
            skin_temp = self._rng.uniform(20, 29)
        elif fault == "frozen" and self._last is not None:
            ir_dc = self._last.ir_dc
            heart_rate = self._last.heart_rate
            spo2 = self._last.spo2
            skin_temp = self._last.skin_temp
        elif fault == "nan_spo2":
            spo2 = float("nan")
        elif fault == "none_hr":
            heart_rate = None
        elif fault == "impossible_hr":
            heart_rate = 320.0
        elif fault == "artifact_hr":
            heart_rate = heart_rate + 70.0
        elif fault == "warn_spo2":
            spo2 = self._rng.uniform(88.0, 91.0)
        elif fault == "ir_broken":
            # قيمة فوق مدى ADC (18-بت): لا يمكن أن تأتي من الحسّاس نفسه.
            ir_dc = self._rng.uniform(5e5, 9e6)

        elif fault == "still":
            movement = self._rng.uniform(0.0, 0.02)

        sample = VitalSample(
            t=self._t,
            ir_dc=ir_dc,
            heart_rate=heart_rate,
            spo2=spo2,
            skin_temp=skin_temp,
            movement=movement,
        )
        self._last = sample
        return sample

    def _movement(self) -> float:
        """ضوضاء منخفضة مع نوبات حركة متفرّقة — مصدرها جهاز منفصل عن المسند."""
        if self._rng.random() < 0.25:
            return self._rng.uniform(0.25, 0.9)
        return self._rng.uniform(0.0, 0.05)


def default_scenario() -> Dict[int, str]:
    """
    سيناريو العرض: 120 عيّنة × 30 ث = ساعة افتراضية.

    يمرّ على كل حالة يجب أن يمسكها المدقّق، وينتهي بخمول طويل
    يتجاوز IMMOBILITY_LIMIT_S ليُطلق إنذار الحركة.
    """
    faults: Dict[int, str] = {}
    for i in range(10, 14):          # رفع المعصم عن المسند
        faults[i] = "no_contact"
    faults[16] = "artifact_hr"       # رعشة/حركة يد
    faults[18] = "impossible_hr"     # قيمة مستحيلة
    faults[20] = "nan_spo2"
    faults[22] = "none_hr"
    for i in range(24, 26):          # أكسجين منخفض لكنه ممكن ⇒ WARN
        faults[i] = "warn_spo2"
    # التجمّد يحتاج 10 عيّنات متطابقة قبل أن يُرصد، ثم 30 عيّنة أخرى بلا رقم صالح
    # ليُطلق إنذار الصمت (15 دقيقة ÷ 30 ث) — فالنافذة 40 عيّنة، لا أقل.
    for i in range(28, 68):          # حسّاس متجمّد ⇒ تجمّد ثم صمت قياس
        faults[i] = "frozen"
    for i in range(69, 72):          # عطب في الناقل ⇒ إنذار صيانة
        faults[i] = "ir_broken"
    for i in range(75, 115):         # سكون تام ⇒ إنذار الحركة (وطول الاستناد ⇒ إنذار المعصم)
        faults[i] = "still"
    return faults

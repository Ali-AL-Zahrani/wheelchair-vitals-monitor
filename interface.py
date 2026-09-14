"""
نموذج العيّنة الحيوية + الواجهة المجرّدة للحسّاس.

هذا الملف هو **نقطة تبديل المحاكاة بالعتاد**: أي حسّاس فعلي (I2CSensor)
يرث SensorInterface ويطبّق read() فقط، دون لمس المدقّق ولا الشاشة.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VitalSample:
    """
    عيّنة خام واحدة كما تخرج من الحسّاس — **قبل** أي تدقيق.

    كل الحقول Optional لأن الحسّاس قد يفشل في أي قناة على حدة،
    وتمثيل الفشل بـ None أصدق من تمثيله بصفر (الصفر يشبه قراءة حقيقية).

    الأسماء تعكس ما يُقاس فعلًا:
      - skin_temp = حرارة جلد المعصم، **ليست** حرارة الجسم المركزية.
      - spo2 من وضع reflectance في المعصم = مؤشّر اتجاه، لا قيمة سريرية مطلقة.
        إشارة PPG من المعصم أضعف من راحة اليد وأضعف بكثير من مشبك الإصبع.
      - movement مصدره **منفصل** عن المسند (FSR أو IMU — لم يُحسم بعد).
    """

    t: float                        # زمن العيّنة بالثواني من بدء الجلسة (ساعة الحسّاس، لا ساعة النظام)
    ir_dc: Optional[float] = None   # المكوّن المستمر لقناة IR في MAX30102 — الدليل الوحيد على استناد المعصم
    heart_rate: Optional[float] = None   # نبضة/دقيقة (PPG)
    spo2: Optional[float] = None         # تشبّع الأكسجين %
    skin_temp: Optional[float] = None    # حرارة جلد المعصم °C (MAX30205)
    movement: Optional[float] = None     # مؤشّر حركة عام من مصدر منفصل عن المسند


class SensorInterface(ABC):
    """
    العقد الذي يلتزم به أي مصدر بيانات: المحاكاة اليوم، العتاد لاحقًا.

    الشرط الوحيد: read() تُعيد VitalSample واحدة أو ترفع استثناء.
    الحسّاس **لا يدقّق ولا يصحّح ولا يخفي** قراءة سيئة — التدقيق مسؤولية Validator وحده.
    """

    @abstractmethod
    def read(self) -> VitalSample:
        """تُعيد العيّنة التالية كما هي، بلا تنقية."""
        raise NotImplementedError

    def start(self) -> None:
        """تهيئة العتاد (I2C, تشغيل الـ LED...). لا شيء في المحاكاة."""

    def stop(self) -> None:
        """إغلاق نظيف للعتاد. لا شيء في المحاكاة."""

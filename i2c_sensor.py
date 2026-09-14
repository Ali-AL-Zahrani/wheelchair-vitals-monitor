"""
طبقة العتاد — MAX30102 + MAX30205 + MPU-6050 على ناقل I2C.

⚠️ **لم تُختبر على عتاد فعلي.** اختباراتها الآلية تُثبت فكّ السجلات وحساب الإشارة
بناقل وهمي، لا صحّة التوصيل ولا معايرة القيم. تُجرَّب على الجهاز قبل أي استخدام.

هذا الملف هو **الوحيد** الذي يُبدَّل عند الانتقال من المحاكاة إلى العتاد:
يرث `SensorInterface` ويطبّق `read()`، فلا يُلمس المدقّق ولا الشاشة ولا السجلّ.

────────────────────────────────────────────────────────────────────────────
حقيقة يجب أن تُفهم قبل قراءة الكود:
  **MAX30102 لا يُخرج "نبضة/دقيقة".** يُخرج عيّنات ضوء خام (أحمر + تحت أحمر)
  بتردد 100 هرتز عبر ذاكرة FIFO عمقها 32 عيّنة فقط. النبض والأكسجين
  **يُحسبان هنا** من نافذة زمنية من تلك العيّنات:
    - `ir_dc`      = المستوى المستمر لقناة IR ⟵ دليل وجود المعصم
    - `heart_rate` = كشف قمم النبض في المركّبة المتغيّرة
    - `spo2`       = نسبة النسب (ratio-of-ratios) بين القناتين
  وعمق الـFIFO 32 عيّنة يعني امتلاءه خلال ~0.3 ثانية عند 100 هرتز، فلا بدّ
  من تفريغه في **خيط مستقل** وإلا ضاعت عيّنات بصمت.
────────────────────────────────────────────────────────────────────────────

قاعدة الطبقة: **تُعيد ما قاسته، ولا تصحّح ولا تُخفي.** تعذّر الحساب ⇒ `None`،
والمدقّق هو من يقرّر ما يُعرض. القيمة المشبوهة تُمرَّر كما هي ليرفضها المدقّق،
لا تُقصّ هنا — التدقيق مسؤولية طبقة واحدة، لا اثنتين.

التشغيل يتطلّب: pip install smbus2
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence, Tuple

from interface import SensorInterface, VitalSample

# ═══════════════════════════════════════════════════════════════════════════
#  عناوين الأجهزة على الناقل
# ═══════════════════════════════════════════════════════════════════════════
ADDR_MAX30102 = 0x57
ADDR_MAX30205 = 0x48      # يتغيّر بأرجل A0–A2؛ تحقّق بـ i2cdetect
ADDR_MPU6050 = 0x68       # 0x69 إذا رُفعت رجل AD0

# ── سجلات MAX30102 ──
_M102_INT_STATUS_1 = 0x00
_M102_FIFO_WR_PTR = 0x04
_M102_OVF_COUNTER = 0x05
_M102_FIFO_RD_PTR = 0x06
_M102_FIFO_DATA = 0x07
_M102_FIFO_CONFIG = 0x08
_M102_MODE_CONFIG = 0x09
_M102_SPO2_CONFIG = 0x0A
_M102_LED1_PA = 0x0C      # الأحمر
_M102_LED2_PA = 0x0D      # تحت الأحمر
_M102_PART_ID = 0xFF
_M102_EXPECTED_PART_ID = 0x15

# ── سجلات MAX30205 ──
_M205_TEMPERATURE = 0x00
_M205_LSB_C = 1.0 / 256.0     # 0.00390625 °م لكل خطوة

# ── سجلات MPU-6050 ──
_MPU_PWR_MGMT_1 = 0x6B
_MPU_ACCEL_XOUT_H = 0x3B
_MPU_LSB_PER_G = 16384.0      # المدى الافتراضي ±2g

# ═══════════════════════════════════════════════════════════════════════════
#  إعدادات القياس
# ═══════════════════════════════════════════════════════════════════════════

PPG_SAMPLE_RATE_HZ = 100.0    # يجب أن يطابق إعداد _M102_SPO2_CONFIG أدناه
PPG_WINDOW_S = 8.0            # نافذة الحساب: ~8–10 نبضات، تكفي لقمم مستقرة

# شدّة تيار الـLED. الرفع الزائد يُشبع المستشعر ويُسطّح الإشارة.
LED_RED_CURRENT = 0x24        # ≈7.2 مللي أمبير
LED_IR_CURRENT = 0x24

# معادلة الأكسجين: SpO2 ≈ A − B·R (نسبة النسب).
SPO2_A = 110.0
SPO2_B = 25.0


# ═══════════════════════════════════════════════════════════════════════════
#  حسابات الإشارة — دوال خالصة، تُختبر بلا عتاد
# ═══════════════════════════════════════════════════════════════════════════

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _moving_average(xs: Sequence[float], window: int) -> List[float]:
    """متوسط متحرّك متمركز — يُستخرج به خطّ الأساس لفصل المركّبة المتغيّرة."""
    if window < 2 or len(xs) < window:
        avg = _mean(xs)
        return [avg] * len(xs)
    half = window // 2
    out: List[float] = []
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def estimate_heart_rate(ir: Sequence[float], fs: float = PPG_SAMPLE_RATE_HZ) -> Optional[float]:
    """
    نبضة/دقيقة من قمم موجة الـPPG، أو None إذا تعذّر الحساب.

    None هنا **ليست فشلًا صامتًا**: المدقّق يترجمها إلى INVALID فلا يُعرض رقم.
    وهذا أصحّ من إرجاع تخمين ضعيف — إشارة المعصم أضعف، ونافذة بلا قمم واضحة
    تعني غالبًا حركة أو تلامسًا رديئًا، لا قلبًا متوقفًا.
    """
    if fs <= 0 or len(ir) < int(fs * 4):      # أقلّ من 4 ثوانٍ لا تكفي
        return None

    # فصل المركّبة المتغيّرة: نافذة 0.75 ث أطول من نبضة واحدة، فتزيل خطّ الأساس
    # (تنفّس، انزياح الحسّاس) وتُبقي النبض.
    baseline = _moving_average(ir, max(3, int(fs * 0.75)))
    ac = [x - b for x, b in zip(ir, baseline)]

    positives = sorted(v for v in ac if v > 0)
    if len(positives) < 4:
        return None
    # عتبة القمّة عند المئين 70 للقيم الموجبة: أمتن من نصف الحدّ الأقصى،
    # فلا تختطفها قفزة حركة واحدة.
    threshold = positives[int(len(positives) * 0.70)]
    if threshold <= 0:
        return None

    refractory = max(1, int(fs * 0.30))       # سقف 200 نبضة/دقيقة
    peaks: List[int] = []
    i = 1
    while i < len(ac) - 1:
        if ac[i] > threshold and ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]:
            peaks.append(i)
            i += refractory
            continue
        i += 1

    if len(peaks) < 3:
        return None

    intervals = sorted((peaks[k + 1] - peaks[k]) / fs for k in range(len(peaks) - 1))
    median = intervals[len(intervals) // 2]   # الوسيط: نبضة ضائعة لا تُفسد الناتج
    if median <= 0:
        return None
    return 60.0 / median


def estimate_spo2(red: Sequence[float], ir: Sequence[float]) -> Optional[float]:
    """
    تشبّع الأكسجين التقريبي من نسبة النسب، أو None إذا تعذّر.

    ⚠️ الناتج **غير معاير** (انظر SPO2_A / SPO2_B). ولا يُقصّ هنا إلى مدى منطقي:
    القيمة المستحيلة تُمرَّر كما هي ليرفضها المدقّق. التدقيق في طبقة واحدة لا اثنتين.
    """
    if len(red) != len(ir) or len(red) < 8:
        return None
    dc_red, dc_ir = _mean(red), _mean(ir)
    if dc_red <= 0 or dc_ir <= 0:
        return None

    def _rms(xs: Sequence[float], dc: float) -> float:
        return (sum((x - dc) ** 2 for x in xs) / len(xs)) ** 0.5

    ac_red, ac_ir = _rms(red, dc_red), _rms(ir, dc_ir)
    if ac_ir <= 0:
        return None

    ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
    return SPO2_A - SPO2_B * ratio


def movement_index(accel_g: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    """
    مؤشّر حركة عام: متوسط انحراف مقدار التسارع عن متوسطه، بوحدة g.

    ⚠️ يقيس **حركة عامة لا تحويل وزن**. وطرح المتوسط
    يلغي الجاذبية تلقائيًا، فلا يحتاج معايرة اتجاه الكرسي.
    """
    if len(accel_g) < 2:
        return None
    magnitudes = [(x * x + y * y + z * z) ** 0.5 for x, y, z in accel_g]
    avg = _mean(magnitudes)
    return _mean([abs(m - avg) for m in magnitudes])


def _twos_complement_16(high: int, low: int) -> int:
    value = (high << 8) | low
    return value - 0x10000 if value & 0x8000 else value


# ═══════════════════════════════════════════════════════════════════════════
#  الحسّاس
# ═══════════════════════════════════════════════════════════════════════════

class I2CSensor(SensorInterface):
    """
    مصدر بيانات فعلي بنفس عقد `MockSensor` — يُبدَّل مكانه بلا تغيير آخر.

    `bus` أي كائن يوفّر واجهة smbus2 (`read_byte_data` / `write_byte_data` /
    `read_i2c_block_data`). حقنه من الخارج يجعل الطبقة قابلة للاختبار بناقل وهمي.
    """

    def __init__(
        self,
        bus: object,
        window_s: float = PPG_WINDOW_S,
        sample_rate_hz: float = PPG_SAMPLE_RATE_HZ,
        addr_ppg: int = ADDR_MAX30102,
        addr_temp: int = ADDR_MAX30205,
        addr_imu: int = ADDR_MPU6050,
        drain_interval_s: float = 0.02,
    ) -> None:
        self._bus = bus
        self._window_s = window_s
        self._fs = sample_rate_hz
        self._addr_ppg = addr_ppg
        self._addr_temp = addr_temp
        self._addr_imu = addr_imu
        self._drain_interval = drain_interval_s

        self._capacity = max(16, int(window_s * sample_rate_hz))
        self._red: List[float] = []
        self._ir: List[float] = []
        self._accel: List[Tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self.lost_samples = 0      # عدّاد فيض الـFIFO — عيّنات ضاعت فعلًا

    # ── دورة الحياة ──
    def start(self) -> None:
        """
        تهيئة الأجهزة الثلاثة وبدء تفريغ الـFIFO.

        يتحقّق من هوية MAX30102 أولًا: توصيل خاطئ يجب أن **يفشل بصوت عالٍ** هنا،
        لا أن ينتج أرقامًا عشوائية تبدو قراءات.
        """
        part_id = self._bus.read_byte_data(self._addr_ppg, _M102_PART_ID)
        if part_id != _M102_EXPECTED_PART_ID:
            raise RuntimeError(
                f"MAX30102 غير موجود على العنوان {self._addr_ppg:#04x} "
                f"(هوية القطعة {part_id:#04x} بدل {_M102_EXPECTED_PART_ID:#04x}). "
                "تحقّق من التوصيل بـ i2cdetect."
            )

        self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x40)   # إعادة ضبط
        time.sleep(0.05)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_WR_PTR, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_OVF_COUNTER, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_RD_PTR, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_CONFIG, 0x4F)   # متوسط 4 عيّنات
        self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x03)   # وضع SpO2
        # مدى ADC 4096nA + 100 عيّنة/ث + عرض نبضة 411µs (دقّة 18-بت)
        self._bus.write_byte_data(self._addr_ppg, _M102_SPO2_CONFIG, 0x27)
        self._bus.write_byte_data(self._addr_ppg, _M102_LED1_PA, LED_RED_CURRENT)
        self._bus.write_byte_data(self._addr_ppg, _M102_LED2_PA, LED_IR_CURRENT)

        self._bus.write_byte_data(self._addr_imu, _MPU_PWR_MGMT_1, 0x00)     # إيقاظ

        self._t0 = time.monotonic()
        self._stop.clear()
        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        try:
            self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x80)  # إسبات
        except Exception:
            pass   # الإغلاق لا يُفشل الجلسة

    # ── القراءة ──
    def read(self) -> VitalSample:
        """
        عيّنة واحدة محسوبة من نافذة الـPPG الحالية.

        الزمن من ساعة أحادية الاتجاه (monotonic) لا من ساعة النظام: تعديل وقت
        الجهاز أو الانتقال الصيفي يجب ألّا يُربك مؤقّتات المدقّق.
        """
        with self._lock:
            red = list(self._red)
            ir = list(self._ir)
            accel = list(self._accel)

        t = (time.monotonic() - self._t0) if self._t0 is not None else 0.0

        return VitalSample(
            t=t,
            ir_dc=_mean(ir) if ir else None,          # الدليل الوحيد على وجود المعصم
            heart_rate=estimate_heart_rate(ir, self._fs),
            spo2=estimate_spo2(red, ir),
            skin_temp=self._read_skin_temp(),
            movement=movement_index(accel),
        )

    # ── داخلي ──
    def _drain_loop(self) -> None:
        """
        تفريغ مستمر للـFIFO. عمق الـFIFO 32 عيّنة فقط ⟵ يمتلئ خلال ~0.3 ث
        عند 100 هرتز، والتأخر عنه يفقد عيّنات بلا إشعار.
        """
        while not self._stop.is_set():
            try:
                self._drain_once()
                self._sample_accel()
            except OSError:
                # خلل ناقل عابر: لا نُسقط الجلسة. غياب العيّنات سيظهر للمدقّق
                # كقراءة متعذّرة، وهو المسار الصحيح للتعامل معه.
                pass
            self._stop.wait(self._drain_interval)

    def _drain_once(self) -> None:
        wr = self._bus.read_byte_data(self._addr_ppg, _M102_FIFO_WR_PTR)
        rd = self._bus.read_byte_data(self._addr_ppg, _M102_FIFO_RD_PTR)
        overflow = self._bus.read_byte_data(self._addr_ppg, _M102_OVF_COUNTER)
        if overflow:
            self.lost_samples += overflow      # يُعدّ ولا يُبتلع

        pending = (wr - rd) % 32
        if pending == 0:
            return

        samples: List[Tuple[float, float]] = []
        remaining = pending
        while remaining > 0:
            chunk = min(remaining, 5)          # 5 عيّنات × 6 بايت = 30 ≤ حدّ الكتلة 32
            raw = self._bus.read_i2c_block_data(self._addr_ppg, _M102_FIFO_DATA, chunk * 6)
            for k in range(chunk):
                base = k * 6
                red = ((raw[base] << 16) | (raw[base + 1] << 8) | raw[base + 2]) & 0x03FFFF
                ir = ((raw[base + 3] << 16) | (raw[base + 4] << 8) | raw[base + 5]) & 0x03FFFF
                samples.append((float(red), float(ir)))
            remaining -= chunk

        with self._lock:
            for red, ir in samples:
                self._red.append(red)
                self._ir.append(ir)
            del self._red[:-self._capacity]
            del self._ir[:-self._capacity]

    def _sample_accel(self) -> None:
        raw = self._bus.read_i2c_block_data(self._addr_imu, _MPU_ACCEL_XOUT_H, 6)
        axes = tuple(
            _twos_complement_16(raw[i], raw[i + 1]) / _MPU_LSB_PER_G for i in (0, 2, 4)
        )
        with self._lock:
            self._accel.append(axes)            # type: ignore[arg-type]
            del self._accel[:-self._capacity]

    def _read_skin_temp(self) -> Optional[float]:
        """حرارة **جلد المعصم** — ليست حرارة الجسم."""
        try:
            raw = self._bus.read_i2c_block_data(self._addr_temp, _M205_TEMPERATURE, 2)
        except OSError:
            return None
        return _twos_complement_16(raw[0], raw[1]) * _M205_LSB_C


def open_default_bus(bus_number: int = 1):
    """
    ناقل I2C الفعلي على Raspberry Pi (المنفذ 1).

    الاستيراد داخل الدالة عمدًا: بقية المشروع تعمل بمكتبة قياسية فقط،
    و`smbus2` تلزم عند التشغيل على العتاد وحده.
    """
    try:
        from smbus2 import SMBus
    except ImportError as exc:
        raise RuntimeError(
            "طبقة العتاد تحتاج smbus2 — نصّبها بـ: pip install smbus2"
        ) from exc
    return SMBus(bus_number)

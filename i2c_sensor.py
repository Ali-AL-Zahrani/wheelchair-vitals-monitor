"""
Hardware layer — MAX30102 + MAX30205 + MPU-6050 over the I2C bus.

This is the **only** file that changes when moving from simulation to hardware:
it inherits `SensorInterface` and implements `read()`, so the validator, the
screen and the log are untouched.

────────────────────────────────────────────────────────────────────────────
A fact to understand before reading the code:
  **The MAX30102 does not output "beats per minute".** It outputs raw light
  samples (red + infrared) at 100 Hz through a FIFO only 32 samples deep.
  Heart rate and oxygen are **computed here** from a time window of those samples:
    - `ir_dc`      = the DC level of the IR channel ⟵ evidence a wrist is present
    - `heart_rate` = peak detection on the AC component
    - `spo2`       = ratio-of-ratios between the two channels
  A 32-sample FIFO fills in ~0.3 s at 100 Hz, so it must be drained in a
  **separate thread** or samples are silently lost.
────────────────────────────────────────────────────────────────────────────

Layer rule: **return what was measured; never correct or hide.** A failed
computation ⇒ `None`, and the validator decides what is displayed. A suspicious
value is passed through as-is for the validator to reject — it is not clipped
here. Validation belongs to one layer, not two.

Running requires: pip install smbus2
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence, Tuple

from interface import SensorInterface, VitalSample

# ═══════════════════════════════════════════════════════════════════════════
#  Device addresses on the bus
# ═══════════════════════════════════════════════════════════════════════════
ADDR_MAX30102 = 0x57
ADDR_MAX30205 = 0x48      # depends on pins A0–A2; verify with i2cdetect
ADDR_MPU6050 = 0x68       # 0x69 if the AD0 pin is pulled high

# ── MAX30102 registers ──
_M102_INT_STATUS_1 = 0x00
_M102_FIFO_WR_PTR = 0x04
_M102_OVF_COUNTER = 0x05
_M102_FIFO_RD_PTR = 0x06
_M102_FIFO_DATA = 0x07
_M102_FIFO_CONFIG = 0x08
_M102_MODE_CONFIG = 0x09
_M102_SPO2_CONFIG = 0x0A
_M102_LED1_PA = 0x0C      # red
_M102_LED2_PA = 0x0D      # infrared
_M102_PART_ID = 0xFF
_M102_EXPECTED_PART_ID = 0x15

# ── MAX30205 registers ──
_M205_TEMPERATURE = 0x00
_M205_LSB_C = 1.0 / 256.0     # 0.00390625 °C per step

# ── MPU-6050 registers ──
_MPU_PWR_MGMT_1 = 0x6B
_MPU_ACCEL_XOUT_H = 0x3B
_MPU_LSB_PER_G = 16384.0      # default ±2g range

# ═══════════════════════════════════════════════════════════════════════════
#  Measurement settings
# ═══════════════════════════════════════════════════════════════════════════

PPG_SAMPLE_RATE_HZ = 100.0    # must match the _M102_SPO2_CONFIG setting below
PPG_WINDOW_S = 8.0            # computation window: ~8–10 beats, enough for stable peaks

# LED drive current. Too much current saturates the photodetector and flattens the signal.
LED_RED_CURRENT = 0x24        # ≈7.2 mA
LED_IR_CURRENT = 0x24

# Oxygen equation: SpO2 ≈ A − B·R (ratio-of-ratios).
SPO2_A = 110.0
SPO2_B = 25.0


# ═══════════════════════════════════════════════════════════════════════════
#  Signal maths — pure functions, testable without hardware
# ═══════════════════════════════════════════════════════════════════════════

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _moving_average(xs: Sequence[float], window: int) -> List[float]:
    """Centred moving average — extracts the baseline so the AC component can be isolated."""
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
    Beats per minute from the peaks of the PPG waveform, or None if it cannot be computed.

    None here is **not a silent failure**: the validator turns it into INVALID so no
    number is displayed. That is more honest than returning a weak guess — the
    wrist signal is weak, and a window with no clear peaks usually means movement
    or poor contact, not a stopped heart.
    """
    if fs <= 0 or len(ir) < int(fs * 4):      # fewer than 4 seconds is not enough
        return None

    # Isolate the AC component: a 0.75 s window is longer than one beat, so it
    # removes the baseline (breathing, sensor drift) and keeps the pulse.
    baseline = _moving_average(ir, max(3, int(fs * 0.75)))
    ac = [x - b for x, b in zip(ir, baseline)]

    positives = sorted(v for v in ac if v > 0)
    if len(positives) < 4:
        return None
    # Peak threshold at the 70th percentile of positive values: more robust than
    # half the maximum, so a single motion spike cannot hijack it.
    threshold = positives[int(len(positives) * 0.70)]
    if threshold <= 0:
        return None

    refractory = max(1, int(fs * 0.30))       # 200 bpm ceiling
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
    median = intervals[len(intervals) // 2]   # the median: one missed beat does not corrupt the result
    if median <= 0:
        return None
    return 60.0 / median


def estimate_spo2(red: Sequence[float], ir: Sequence[float]) -> Optional[float]:
    """
    Approximate oxygen saturation from the ratio-of-ratios, or None if it cannot be computed.

    The result is not clipped to a plausible range here: an impossible value is
    passed through as-is for the validator to reject. Validation lives in one layer, not two.
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
    General movement index: mean deviation of acceleration magnitude from its mean, in g.

    Measures **general movement, not weight shift**. Subtracting the mean cancels
    gravity automatically, so no chair-orientation calibration is needed.
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
#  The sensor
# ═══════════════════════════════════════════════════════════════════════════

class I2CSensor(SensorInterface):
    """
    A physical data source with the same contract as `MockSensor` — a drop-in replacement.

    `bus` is any object exposing the smbus2 interface (`read_byte_data` /
    `write_byte_data` / `read_i2c_block_data`). Injecting it from outside makes
    the layer testable with a fake bus.
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
        self.lost_samples = 0      # FIFO overflow counter — samples that were genuinely lost

    # ── Lifecycle ──
    def start(self) -> None:
        """
        Initialise the three devices and start draining the FIFO.

        Checks the MAX30102 part ID first: bad wiring must **fail loudly** here,
        not produce random numbers that look like readings.
        """
        part_id = self._bus.read_byte_data(self._addr_ppg, _M102_PART_ID)
        if part_id != _M102_EXPECTED_PART_ID:
            raise RuntimeError(
                f"MAX30102 not found at address {self._addr_ppg:#04x} "
                f"(part ID {part_id:#04x} instead of {_M102_EXPECTED_PART_ID:#04x}). "
                "Check the wiring with i2cdetect."
            )

        self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x40)   # reset
        time.sleep(0.05)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_WR_PTR, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_OVF_COUNTER, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_RD_PTR, 0x00)
        self._bus.write_byte_data(self._addr_ppg, _M102_FIFO_CONFIG, 0x4F)   # 4-sample averaging
        self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x03)   # SpO2 mode
        # ADC range 4096 nA + 100 samples/s + 411 µs pulse width (18-bit resolution)
        self._bus.write_byte_data(self._addr_ppg, _M102_SPO2_CONFIG, 0x27)
        self._bus.write_byte_data(self._addr_ppg, _M102_LED1_PA, LED_RED_CURRENT)
        self._bus.write_byte_data(self._addr_ppg, _M102_LED2_PA, LED_IR_CURRENT)

        self._bus.write_byte_data(self._addr_imu, _MPU_PWR_MGMT_1, 0x00)     # wake up

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
            self._bus.write_byte_data(self._addr_ppg, _M102_MODE_CONFIG, 0x80)  # shutdown
        except Exception:
            pass   # shutdown must never fail the session

    # ── Reading ──
    def read(self) -> VitalSample:
        """
        One sample computed from the current PPG window.

        Time comes from a monotonic clock, not the system clock: changing the
        device time or a DST shift must not confuse the validator's timers.
        """
        with self._lock:
            red = list(self._red)
            ir = list(self._ir)
            accel = list(self._accel)

        t = (time.monotonic() - self._t0) if self._t0 is not None else 0.0

        return VitalSample(
            t=t,
            ir_dc=_mean(ir) if ir else None,          # the only evidence a wrist is present
            heart_rate=estimate_heart_rate(ir, self._fs),
            spo2=estimate_spo2(red, ir),
            skin_temp=self._read_skin_temp(),
            movement=movement_index(accel),
        )

    # ── Internal ──
    def _drain_loop(self) -> None:
        """
        Continuous FIFO drain. The FIFO holds only 32 samples ⟵ it fills in ~0.3 s
        at 100 Hz, and falling behind loses samples with no notice.
        """
        while not self._stop.is_set():
            try:
                self._drain_once()
                self._sample_accel()
            except OSError:
                # Transient bus error: do not bring the session down. Missing
                # samples will surface at the validator as an unavailable
                # reading — the correct path for handling it.
                pass
            self._stop.wait(self._drain_interval)

    def _drain_once(self) -> None:
        wr = self._bus.read_byte_data(self._addr_ppg, _M102_FIFO_WR_PTR)
        rd = self._bus.read_byte_data(self._addr_ppg, _M102_FIFO_RD_PTR)
        overflow = self._bus.read_byte_data(self._addr_ppg, _M102_OVF_COUNTER)
        if overflow:
            self.lost_samples += overflow      # counted, never swallowed

        pending = (wr - rd) % 32
        if pending == 0:
            return

        samples: List[Tuple[float, float]] = []
        remaining = pending
        while remaining > 0:
            chunk = min(remaining, 5)          # 5 samples × 6 bytes = 30 ≤ the 32-byte block limit
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
        """**Wrist skin** temperature — not body temperature."""
        try:
            raw = self._bus.read_i2c_block_data(self._addr_temp, _M205_TEMPERATURE, 2)
        except OSError:
            return None
        return _twos_complement_16(raw[0], raw[1]) * _M205_LSB_C


def open_default_bus(bus_number: int = 1):
    """
    The physical I2C bus on a Raspberry Pi (bus 1).

    The import is inside the function on purpose: the rest of the project runs on
    the standard library alone, and `smbus2` is needed only when running on hardware.
    """
    try:
        from smbus2 import SMBus
    except ImportError as exc:
        raise RuntimeError(
            "The hardware layer needs smbus2 — install it with: pip install smbus2"
        ) from exc
    return SMBus(bus_number)

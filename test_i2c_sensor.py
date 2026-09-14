"""
Hardware-layer tests — against a fake I2C bus.

⚠️ **What they prove:** register decoding for the three devices, the signal
maths, and failure behaviour.
⚠️ **What they do not prove:** correctness of the physical wiring, or value
calibration on a real wrist. Passing these tests **does not mean** the device
reads correctly — it means the logic is sound.

The fake bus plays the role of MAX30102, MAX30205 and MPU-6050 together, so we
can inject a PPG waveform of known frequency and check the extracted heart rate matches.
"""

from __future__ import annotations

import math

import pytest

from i2c_sensor import (
    ADDR_MAX30102,
    ADDR_MAX30205,
    ADDR_MPU6050,
    I2CSensor,
    estimate_heart_rate,
    estimate_spo2,
    movement_index,
)
from validator import Status, Validator


# ═══════════════════════════════════════════════════════════════════════════
#  Fake bus
# ═══════════════════════════════════════════════════════════════════════════

class FakeBus:
    """
    Simulates the three devices. Holds PPG samples in an internal FIFO with the
    same semantics as the hardware: write/read pointers modulo 32, and an overflow counter.
    """

    def __init__(self, ppg_samples=None, temp_raw=0x2180, accel_raw=None,
                 part_id=0x15, overflow=0):
        self.ppg = list(ppg_samples or [])       # [(red, ir), ...]
        self.temp_raw = temp_raw                 # 0x2180 = 33.5 °C
        self.accel_raw = accel_raw or [0x40, 0x00, 0x00, 0x00, 0x00, 0x00]
        self.part_id = part_id
        self.overflow = overflow
        self.writes = []
        self._rd = 0

    # ── smbus2 interface ──
    def read_byte_data(self, addr, reg):
        if addr == ADDR_MAX30102:
            if reg == 0xFF:
                return self.part_id
            if reg == 0x04:
                # The real FIFO holds only 32 samples, so at most 31 are pending
                # no matter how long the recording. Simulating it without this cap
                # hides the fact that a single drain cannot fill the measurement window.
                pending = min(31, len(self.ppg) - self._rd)
                return (self._rd + pending) % 32
            if reg == 0x06:
                return self._rd % 32
            if reg == 0x05:
                return self.overflow
        return 0

    def write_byte_data(self, addr, reg, value):
        self.writes.append((addr, reg, value))

    def read_i2c_block_data(self, addr, reg, length):
        if addr == ADDR_MAX30205:
            return [(self.temp_raw >> 8) & 0xFF, self.temp_raw & 0xFF]
        if addr == ADDR_MPU6050:
            return list(self.accel_raw)
        if addr == ADDR_MAX30102 and reg == 0x07:
            out = []
            for _ in range(length // 6):
                red, ir = self.ppg[self._rd] if self._rd < len(self.ppg) else (0, 0)
                self._rd += 1
                for value in (int(red), int(ir)):
                    out += [(value >> 16) & 0x03, (value >> 8) & 0xFF, value & 0xFF]
            return out
        return [0] * length


def fill_window(sensor, bus):
    """
    Drain repeatedly until the measurement window is full — as the drain thread does on hardware.
    One FIFO read is not enough: its capacity is 32 and the window is 800.
    """
    while bus._rd < len(bus.ppg):
        sensor._drain_once()
        sensor._sample_accel()      # the drain thread reads both on every pass


def ppg_wave(bpm=72.0, seconds=8.0, fs=100.0, dc_ir=40_000.0, ac_ir=800.0,
             dc_red=38_000.0, ac_red=600.0):
    """A synthetic PPG waveform with a known heart rate — the reference we measure extraction accuracy against."""
    n = int(seconds * fs)
    freq = bpm / 60.0
    out = []
    for i in range(n):
        phase = 2.0 * math.pi * freq * (i / fs)
        out.append((dc_red + ac_red * math.sin(phase),
                    dc_ir + ac_ir * math.sin(phase)))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Heart-rate extraction
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bpm", [48.0, 60.0, 72.0, 95.0, 130.0])
def test_heart_rate_matches_the_injected_waveform(bpm):
    ir = [s[1] for s in ppg_wave(bpm=bpm)]
    estimated = estimate_heart_rate(ir)
    assert estimated is not None
    assert abs(estimated - bpm) < 3.0, f"large deviation at {bpm}: {estimated}"


def test_heart_rate_survives_a_drifting_baseline():
    """Sensor drift and breathing shift the baseline — they must not corrupt the heart rate."""
    ir = [s[1] + 900.0 * math.sin(2 * math.pi * 0.15 * (i / 100.0))
          for i, s in enumerate(ppg_wave(bpm=72.0))]
    estimated = estimate_heart_rate(ir)
    assert estimated is not None and abs(estimated - 72.0) < 4.0


def test_flat_signal_returns_none_not_a_guess():
    """
    A lifted wrist or poor contact ⟵ a flat signal.
    None is correct here: the validator turns it into INVALID so no number is displayed.
    """
    assert estimate_heart_rate([40_000.0] * 800) is None


def test_too_short_a_window_returns_none():
    assert estimate_heart_rate([s[1] for s in ppg_wave(seconds=2.0)]) is None


# ═══════════════════════════════════════════════════════════════════════════
#  Oxygen
# ═══════════════════════════════════════════════════════════════════════════

def test_spo2_falls_when_the_red_to_ir_ratio_rises():
    """The direction of the relationship is what we guarantee — not the absolute value."""
    wave = ppg_wave()
    red, ir = [s[0] for s in wave], [s[1] for s in wave]
    high = estimate_spo2(red, ir)

    wave2 = ppg_wave(ac_red=1_500.0)      # higher ratio ⟵ lower saturation
    low = estimate_spo2([s[0] for s in wave2], [s[1] for s in wave2])
    assert high is not None and low is not None and low < high


def test_spo2_is_not_clamped_here_but_left_to_the_validator():
    """
    An impossible value is passed through as-is for the validator to reject — validation in one layer, not two.
    Clipping it here would hide a genuine sensor fault.
    """
    wave = ppg_wave(ac_red=20_000.0)      # extreme ratio ⟵ negative result
    value = estimate_spo2([s[0] for s in wave], [s[1] for s in wave])
    assert value is not None and value < 50.0
    assert Validator().validate(
        __import__("interface").VitalSample(t=0.0, ir_dc=40_000.0, spo2=value)
    ).status_of("spo2") is Status.INVALID


def test_zero_signal_gives_none():
    assert estimate_spo2([0.0] * 100, [0.0] * 100) is None


# ═══════════════════════════════════════════════════════════════════════════
#  Movement
# ═══════════════════════════════════════════════════════════════════════════

def test_still_chair_reads_near_zero_movement():
    assert movement_index([(0.0, 0.0, 1.0)] * 50) < 0.01


def test_shaking_reads_higher_than_stillness():
    still = movement_index([(0.0, 0.0, 1.0)] * 50)
    moving = movement_index([(0.0, 0.0, 1.0 + 0.4 * (-1) ** i) for i in range(50)])
    assert moving > still + 0.1


def test_gravity_alone_is_not_read_as_movement():
    """The IMU mounting orientation must not read as movement — subtracting the mean cancels gravity."""
    assert movement_index([(0.0, 1.0, 0.0)] * 50) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Bus integration
# ═══════════════════════════════════════════════════════════════════════════

def test_wrong_device_fails_loudly_at_startup():
    """Bad wiring must stop here, not produce random numbers that look like readings."""
    sensor = I2CSensor(FakeBus(part_id=0x00))
    with pytest.raises(RuntimeError, match="MAX30102"):
        sensor.start()


def test_configuration_reaches_the_device():
    bus = FakeBus()
    sensor = I2CSensor(bus)
    sensor.start()
    sensor.stop()
    registers = [reg for addr, reg, _ in bus.writes if addr == ADDR_MAX30102]
    assert 0x09 in registers and 0x0A in registers        # mode and configuration
    assert 0x0C in registers and 0x0D in registers        # both LED currents
    assert (ADDR_MPU6050, 0x6B, 0x00) in bus.writes       # IMU wake-up


def test_full_sample_reads_every_channel():
    bus = FakeBus(ppg_samples=ppg_wave(bpm=72.0))
    sensor = I2CSensor(bus)
    fill_window(sensor, bus)
    sensor._t0 = 0.0
    sample = sensor.read()

    assert sample.ir_dc is not None and 35_000 < sample.ir_dc < 45_000
    assert sample.heart_rate is not None and abs(sample.heart_rate - 72.0) < 3.0
    assert sample.spo2 is not None
    assert sample.skin_temp == pytest.approx(33.5, abs=0.01)
    assert sample.movement is not None


def test_sample_passes_the_validator_end_to_end():
    """The real contract: the hardware layer's output is accepted by the validator just like the simulator's."""
    bus = FakeBus(ppg_samples=ppg_wave(bpm=72.0))
    sensor = I2CSensor(bus)
    fill_window(sensor, bus)
    sensor._t0 = 0.0

    result = Validator().validate(sensor.read())
    assert result.contact is True
    assert result.status_of("heart_rate") is Status.VALID
    assert result.value_of("heart_rate") is not None


def test_negative_temperature_is_decoded_correctly():
    """Two's complement: a sign error silently turns 25° into a negative value."""
    bus = FakeBus(temp_raw=0xF000)        # −16 °C
    sensor = I2CSensor(bus)
    assert sensor._read_skin_temp() == pytest.approx(-16.0, abs=0.01)


def test_fifo_overflow_is_counted_not_swallowed():
    """Genuinely lost samples must be counted — silent overflow hides performance degradation."""
    bus = FakeBus(ppg_samples=ppg_wave(seconds=1.0), overflow=7)
    sensor = I2CSensor(bus)
    sensor._drain_once()
    assert sensor.lost_samples == 7


def test_bus_error_does_not_kill_the_session():
    """A transient bus error surfaces at the validator as an unavailable reading, not as a system crash."""
    class BrokenBus(FakeBus):
        def read_i2c_block_data(self, addr, reg, length):
            if addr == ADDR_MAX30205:
                raise OSError("I2C read failed")
            return super().read_i2c_block_data(addr, reg, length)

    bus = BrokenBus(ppg_samples=ppg_wave())
    sensor = I2CSensor(bus)
    fill_window(sensor, bus)
    sensor._t0 = 0.0
    sample = sensor.read()
    assert sample.skin_temp is None          # only the faulty channel drops out
    assert sample.heart_rate is not None      # the other channels carry on

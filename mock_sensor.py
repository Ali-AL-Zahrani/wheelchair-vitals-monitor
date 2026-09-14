"""
Mock sensor: realistic readings + deliberate fault injection.

The goal is not "pretty data" but **stressing the validator**: contact loss,
NaN, impossible values, a frozen sensor, artifact jumps, plus long immobility
stretches to exercise the watchdog.

The clock is virtual (30 s/sample by default) so immobility logic can be
tested in seconds instead of waiting a real 20 minutes.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from interface import SensorInterface, VitalSample

# Supported fault names (assigned to a sample index):
#   "no_contact"    ⟵ wrist lifted: ir_dc collapses and the values become noise that looks like a real reading
#   "frozen"        ⟵ wrist sensor stuck on the same bits (movement stays live: it comes from another device)
#   "nan_spo2"      ⟵ NaN from the oxygen channel
#   "none_hr"       ⟵ driver returns None
#   "impossible_hr" ⟵ physically impossible value
#   "artifact_hr"   ⟵ sudden jump (tremor / hand movement)
#   "warn_spo2"     ⟵ possible value but below the clinical bound
#   "ir_broken"     ⟵ ir_dc outside ADC range: driver/bus fault, not a lifted wrist
#   "still"         ⟵ user completely still (exercises the movement alert)

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
        """Index of the last sample returned (for diagnostics in demo)."""
        return self._i

    def read(self) -> VitalSample:
        self._i += 1
        if self._i > 0:
            self._t += self._period
        fault = self._faults.get(self._i)

        # Realistic baseline: a resting adult with the wrist on the armrest.
        # ir_dc in the wrist range. Kept consistent with CONTACT_IR_THRESHOLD in
        # validator.py; changing one requires the other.
        ir_dc = self._rng.gauss(38_000, 3_500)
        heart_rate = self._rng.gauss(74, 3)
        spo2 = min(100.0, self._rng.gauss(97.0, 0.8))
        skin_temp = self._rng.gauss(33.4, 0.25)
        movement = self._movement()

        if fault == "no_contact":
            # No wrist: the sensor still outputs numbers — that is exactly the danger.
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
            # Above the 18-bit ADC range: cannot come from the sensor itself.
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
        """Low noise with occasional bursts of movement — from a device separate from the armrest."""
        if self._rng.random() < 0.25:
            return self._rng.uniform(0.25, 0.9)
        return self._rng.uniform(0.0, 0.05)


def default_scenario() -> Dict[int, str]:
    """
    Demo scenario: 120 samples × 30 s = one virtual hour.

    Passes through every state the validator must catch, and ends with a long
    immobility stretch that exceeds IMMOBILITY_LIMIT_S to trigger the movement alert.
    """
    faults: Dict[int, str] = {}
    for i in range(10, 14):          # wrist lifted off the armrest
        faults[i] = "no_contact"
    faults[16] = "artifact_hr"       # tremor / hand movement
    faults[18] = "impossible_hr"     # impossible value
    faults[20] = "nan_spo2"
    faults[22] = "none_hr"
    for i in range(24, 26):          # low but possible oxygen ⇒ WARN
        faults[i] = "warn_spo2"
    # The stuck check needs 10 identical samples before it fires, then 30 more
    # samples with no valid number to raise the silence alert (15 min ÷ 30 s) —
    # so the window is 40 samples, no fewer.
    for i in range(28, 68):          # frozen sensor ⇒ stuck, then measurement silence
        faults[i] = "frozen"
    for i in range(69, 72):          # bus fault ⇒ maintenance alert
        faults[i] = "ir_broken"
    for i in range(75, 115):         # completely still ⇒ movement alert (and long rest ⇒ wrist alert)
        faults[i] = "still"
    return faults

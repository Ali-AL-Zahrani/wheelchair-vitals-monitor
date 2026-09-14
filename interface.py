"""
Vital-sample model + abstract sensor interface.

This file is the **swap point between simulation and hardware**: any physical
sensor (I2CSensor) inherits SensorInterface and implements read() only,
without touching the validator or the screen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VitalSample:
    """
    One raw sample exactly as it leaves the sensor — **before** any validation.

    Every field is Optional because the sensor may fail on any single channel,
    and representing failure as None is more honest than a zero (a zero looks
    like a real reading).

    Names describe what is actually measured:
      - skin_temp = wrist skin temperature, **not** core body temperature.
      - spo2 from reflectance mode at the wrist = a trend indicator, not an
        absolute clinical value. The PPG signal at the wrist is weaker than the
        palm and far weaker than a fingertip clip.
      - movement comes from a source **separate** from the armrest (IMU).
    """

    t: float                        # sample time in seconds since session start (sensor clock, not system clock)
    ir_dc: Optional[float] = None   # DC component of the MAX30102 IR channel — the only evidence a wrist is resting
    heart_rate: Optional[float] = None   # beats per minute (PPG)
    spo2: Optional[float] = None         # oxygen saturation %
    skin_temp: Optional[float] = None    # wrist skin temperature °C (MAX30205)
    movement: Optional[float] = None     # general movement index from a source separate from the armrest


class SensorInterface(ABC):
    """
    The contract every data source honours: simulation today, hardware later.

    The only requirement: read() returns one VitalSample or raises.
    The sensor **never validates, corrects or hides** a bad reading — validation
    is the Validator's responsibility alone.
    """

    @abstractmethod
    def read(self) -> VitalSample:
        """Return the next sample as-is, with no filtering."""
        raise NotImplementedError

    def start(self) -> None:
        """Hardware initialisation (I2C, LED power-on, ...). No-op in simulation."""

    def stop(self) -> None:
        """Clean hardware shutdown. No-op in simulation."""

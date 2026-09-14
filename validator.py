"""
Validation engine — the core of the system.

Three fully **independent** subsystems:
  1) Contact gate      : no wrist on the armrest ⇒ every wrist value is blanked,
                         however plausible the numbers look.
  2) Vital-sign checks : SANITY (impossible) ⇒ INVALID, value discarded.
                         CLINICAL (possible but abnormal) ⇒ WARN, value kept and flagged.
                         + two checks with memory: stuck (frozen sensor) and
                           artifact (sudden jump).
  3) Temporal monitors : fully independent of the wrist; driven by the separate
                         movement source.

The output separates flags (data quality) from alerts (action required):
heart rate can be VALID while the movement alert is active at the same moment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from interface import VitalSample

# ═══════════════════════════════════════════════════════════════════════════
#  System thresholds
# ═══════════════════════════════════════════════════════════════════════════

# Contact threshold in the wrist range. ir_dc at the wrist is lower than at the
# palm (less capillary perfusion, bone closer to the skin); a threshold set too
# high reads NO_CONTACT while the user's wrist is actually resting.
# Coupled with the ir_dc level in mock_sensor.py — changing one requires the other.
CONTACT_IR_THRESHOLD = 20_000.0

# Plausibility bound, not a clinical threshold: the MAX30102 ADC is 18-bit
# (0..262143). A value above it, or negative, cannot come from the sensor — only
# from a broken driver/I2C bus. The contact gate is built on ir_dc, so a corrupt
# value could open the gate and pass noise through as readings.
CONTACT_IR_SANITY_MAX = 262_143.0

# ✅ Approved by the medical team (12 Aug 2026): weight shift every 15 minutes.
IMMOBILITY_LIMIT_S = 15 * 60.0

# Continuous wrist rest before reminding the user to lift it — same as the
# immobility limit. Deliberately independent of body movement: if it were tied
# to the movement event, a wrist nobody moved for hours would never be reminded —
# the alert would vanish exactly when the risk peaks.
WRIST_REST_LIMIT_S = 15 * 60.0

# ✅ Approved by the medical team (12 Aug 2026): 15 minutes.
# Deliberately long: a short value produces nuisance alarms on every ordinary
# wrist lift (eating, transfers); a nuisance alarm gets the sound switched off ⟵
# and the sensor-fault alarm is lost with it.
SILENCE_LIMIT_S = 15 * 60.0

# Movement source: IMU (medical team decision, 12 Aug). Value in abstract
# movement-index units.
MOVEMENT_THRESHOLD = 0.15

# Number of bit-identical consecutive samples that means "frozen sensor"
# (≈5 minutes at 30 s/sample).
STUCK_REPEAT_LIMIT = 10

# After this many consecutive artifacts the change is treated as real and the
# baseline is re-synced; otherwise a field would stay rejected forever after any
# genuine rapid physiological change.
ARTIFACT_RESYNC_N = 3


class Status(str, Enum):
    VALID = "VALID"            # show the number normally
    WARN = "WARN"              # show the number + visual emphasis
    INVALID = "INVALID"        # no number ⟵ "Reading unavailable"
    NO_CONTACT = "NO_CONTACT"  # no number ⟵ "Rest your wrist on the armrest"


# ── Data-quality flags ──
FLAG_SANITY_MISSING = "SANITY_MISSING"          # value is None
FLAG_SANITY_TYPE = "SANITY_TYPE"                # non-numeric type from the driver
FLAG_SANITY_NAN = "SANITY_NAN"                  # NaN / inf
FLAG_SANITY_RANGE = "SANITY_RANGE"              # outside what is physically possible
FLAG_CLINICAL_LOW = "CLINICAL_LOW"              # possible but below the clinical bound
FLAG_CLINICAL_HIGH = "CLINICAL_HIGH"            # possible but above the clinical bound
FLAG_STUCK = "STUCK"                            # sensor frozen on the same value
FLAG_ARTIFACT = "ARTIFACT"                      # sudden jump = tremor / movement
FLAG_ARTIFACT_RESYNC = "ARTIFACT_RESYNC"        # baseline re-synced after consecutive jumps
FLAG_NO_CONTACT = "NO_CONTACT"                  # no wrist on the armrest
FLAG_IR_IMPLAUSIBLE = "IR_IMPLAUSIBLE"          # ir_dc outside sensor range ⇒ source fault, not a lifted wrist
FLAG_MOVEMENT_UNVERIFIED = "MOVEMENT_UNVERIFIED"  # movement could not be verified
FLAG_TIME_BACKWARD = "TIME_BACKWARD"            # sensor clock went backwards

# ── Alerts = action required from the user ──
ALERT_NEEDS_MOVEMENT = "NEEDS_MOVEMENT"      # shift body weight off pressure points
ALERT_LIFT_WRIST = "LIFT_WRIST"              # lift the wrist off the armrest (prolonged local pressure)
ALERT_SENSOR_FAULT = "SENSOR_FAULT"          # sensor/bus fault — needs maintenance
ALERT_MEASUREMENT_SILENT = "MEASUREMENT_SILENT"  # no measurement has arrived for a while


@dataclass(frozen=True)
class FieldSpec:
    """Bounds for one field. The sanity/clinical split is deliberate — mixing them is a medical error."""

    name: str
    unit: str
    sanity_min: float                       # below this = physically impossible ⇒ discarded
    sanity_max: float                       # above this = physically impossible ⇒ discarded
    clinical_min: Optional[float]           # below this = abnormal but possible ⇒ WARN
    clinical_max: Optional[float]           # above this = abnormal but possible ⇒ WARN
    artifact_max_delta: Optional[float]     # max accepted change between consecutive samples
    wrist_derived: bool                     # does it come from the wrist resting on the armrest?


# ═══════════════════════════════════════════════════════════════════════════
#  Field bounds. Changed here only; a snapshot is logged at the start of every session.
#
#  Note on artifact_max_delta: tuned to the current sample rate (≈30 s).
#  Any change to the sample rate requires re-tuning it, or it becomes either
#  blind or over-sensitive.
# ═══════════════════════════════════════════════════════════════════════════
FIELD_SPECS: Dict[str, FieldSpec] = {
    "heart_rate": FieldSpec(
        name="heart_rate", unit="bpm",
        sanity_min=20.0, sanity_max=250.0,      # outside this is not a beating heart
        clinical_min=50.0, clinical_max=120.0,
        artifact_max_delta=30.0, wrist_derived=True,
    ),
    "spo2": FieldSpec(
        name="spo2", unit="%",
        sanity_min=50.0, sanity_max=100.0,      # >100 impossible; <50 is a sensor reading, not a person
        clinical_min=94.0, clinical_max=None,
        artifact_max_delta=8.0, wrist_derived=True,
    ),
    "skin_temp": FieldSpec(
        name="skin_temp", unit="°C",
        sanity_min=10.0, sanity_max=45.0,       # wrist skin temperature, not body temperature
        clinical_min=30.0, clinical_max=37.0,   # an "expected" range only — no fever logic may be built on it
        artifact_max_delta=2.0, wrist_derived=True,
    ),
    "movement": FieldSpec(
        name="movement", unit="—",
        sanity_min=0.0, sanity_max=float("inf"),  # units depend on the movement source
        clinical_min=None, clinical_max=None,     # immobility is a time anomaly, not a value anomaly ⟵ watchdog
        artifact_max_delta=None, wrist_derived=False,  # movement is expected to jump — no artifact check
    ),
}


# Wrist-derived fields — derived from FIELD_SPECS rather than duplicated by hand,
# otherwise the two lists drift apart when a field is added.
_WRIST_FIELD_NAMES = tuple(n for n, s in FIELD_SPECS.items() if s.wrist_derived)


@dataclass
class FieldResult:
    """Result for one field. value = None means: **display no number at all**."""

    name: str
    value: Optional[float]
    status: Status
    flags: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """
    Validator output for one sample — the **only source** the screen reads from.

    flags  = data quality (why a reading was rejected/flagged).
    alerts = action required from the user, fully independent of reading quality.
    """

    t: float
    contact: bool
    ir_dc: Optional[float]
    fields: Dict[str, FieldResult]
    flags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    immobility_s: float = 0.0     # since the last confirmed weight shift
    wrist_rest_s: float = 0.0     # continuous wrist rest on the armrest
    silence_s: float = 0.0        # since the last displayable wrist reading

    def status_of(self, name: str) -> Status:
        return self.fields[name].status

    def value_of(self, name: str) -> Optional[float]:
        return self.fields[name].value

    def flags_of(self, name: str) -> List[str]:
        return self.fields[name].flags


class Validator:
    """
    Validator with memory across samples. One instance per user/session.

    Fixed principle: when in doubt, reject the reading. "No reading" is more
    honest than a wrong reading.
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
        """Clear memory — called at the start of a new session."""
        self._last_accepted: Dict[str, float] = {}   # artifact-check baseline
        self._last_seen: Dict[str, float] = {}       # last value that passed sanity (stuck check)
        self._repeat: Dict[str, int] = {}
        self._artifact_streak: Dict[str, int] = {}
        self._last_t: Optional[float] = None
        # All timers are **accumulators**, not derived from stored timestamps. See validate().
        self._immobility_s: float = 0.0
        self._wrist_rest_s: float = 0.0
        self._silence_s: float = 0.0

    # ── The single public entry point ──
    def validate(self, sample: VitalSample) -> ValidationResult:
        sample_flags: List[str] = []

        # The sensor clock is the reference, not the system clock — so accelerated
        # simulation runs on exactly the same logic.
        #
        # Immobility is accumulated from positive time deltas only, never derived
        # from a stored "last movement" timestamp. The difference is not stylistic:
        # a clock that jumps backwards (e.g. an MCU reboot) would reset the stored
        # timestamp ⇒ the accumulated time is lost and an active alert is silenced,
        # and the user goes twice as long without a weight shift. The accumulator
        # keeps it: only the suspicious sample is ignored (dt=0); nothing before it
        # is lost.
        dt = 0.0
        if self._last_t is not None:
            if sample.t < self._last_t:
                sample_flags.append(FLAG_TIME_BACKWARD)
            else:
                dt = sample.t - self._last_t
        self._last_t = sample.t

        # ── 1) Contact gate ──
        # A sensor with no wrist on it outputs noise that can look like a real
        # reading, so the gate runs before every other check. Lifting and
        # replacing the wrist is a very frequent event, not an exception.
        contact = self._has_contact(sample.ir_dc)
        if not contact:
            sample_flags.append(FLAG_NO_CONTACT)
            # Deliberate distinction in the log: "wrist lifted" is a normal,
            # frequent event; "ir_dc outside sensor range" is a hardware fault
            # that needs maintenance, not a user prompt.
            if _is_number(sample.ir_dc) and not _ir_in_range(float(sample.ir_dc)):
                sample_flags.append(FLAG_IR_IMPLAUSIBLE)

        # ── 2) Vital-sign checks ──
        results: Dict[str, FieldResult] = {}
        for name, spec in FIELD_SPECS.items():
            raw = getattr(sample, name)
            results[name] = self._check_field(spec, raw, contact)

        # ── 3) Immobility watchdog — independent of the wrist ──
        mov = results["movement"]
        movement_confirmed = (
            mov.status in (Status.VALID, Status.WARN)
            and mov.value is not None
            and mov.value >= self.movement_threshold
        )
        if mov.status not in (Status.VALID, Status.WARN) or mov.value is None:
            # Movement cannot be confirmed ⇒ the timer keeps running (fail-loud).
            # Prompting the user to move unnecessarily is harmless; silencing the
            # alert risks pressure injuries.
            sample_flags.append(FLAG_MOVEMENT_UNVERIFIED)

        # Only confirmed movement resets the accumulator. Anything else —
        # including a clock fault — leaves it running.
        if movement_confirmed:
            self._immobility_s = 0.0
        else:
            self._immobility_s += dt

        alerts: List[str] = []
        if self._immobility_s >= self.immobility_limit_s:
            alerts.append(ALERT_NEEDS_MOVEMENT)

        # ── 4) Wrist-pressure watchdog — independent of body movement ──
        # Wrist skin is thin over bone, and sensation may be impaired so pain
        # does not warn the user. Only lifting the wrist resets the counter;
        # moving the body does not relieve pressure on the wrist.
        if contact:
            self._wrist_rest_s += dt
        else:
            self._wrist_rest_s = 0.0
        if self._wrist_rest_s >= self.wrist_rest_limit_s:
            alerts.append(ALERT_LIFT_WRIST)

        # ── 5) Sensor fault ── value outside hardware range ⇒ maintenance, not a user action.
        if FLAG_IR_IMPLAUSIBLE in sample_flags:
            alerts.append(ALERT_SENSOR_FAULT)

        # ── 6) Measurement silence ──
        # Monitoring has effectively stopped when no displayable number arrives,
        # whatever the reason. A silent screen reads as "all is well", which is
        # more dangerous than a screen that shouts.
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

    # ── Internal ──
    def _has_contact(self, ir_dc: Optional[float]) -> bool:
        """
        ir_dc missing, NaN or outside the sensor range ⇒ assume no contact (the safe default).

        The gate closes on doubt: a gate opened by a corrupt value passes noise to
        the user's eyes; a gate closed by mistake shows "Rest your wrist on the
        armrest" — the second error is harmless.
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
        # (1) The contact gate precedes everything: no wrist ⇒ no number, however plausible.
        if spec.wrist_derived and not contact:
            self._forget(spec.name)  # so a post-return reading is not compared to a stale baseline
            return FieldResult(spec.name, None, Status.NO_CONTACT, [FLAG_NO_CONTACT])

        # (2) SANITY: missing / wrong type / NaN / physically impossible ⇒ value discarded.
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

        # (3) Stuck check: a bit-identical value repeated = a hung sensor, not a stable state.
        last_seen = self._last_seen.get(spec.name)
        if last_seen is not None and v == last_seen:
            self._repeat[spec.name] = self._repeat.get(spec.name, 1) + 1
        else:
            self._repeat[spec.name] = 1
        self._last_seen[spec.name] = v
        stuck = self._repeat[spec.name] >= self.stuck_repeat_limit

        # (4) Jump check: a change larger than physiologically possible between two samples = tremor/movement.
        artifact = False
        baseline = self._last_accepted.get(spec.name)
        if spec.artifact_max_delta is not None and baseline is not None:
            if abs(v - baseline) > spec.artifact_max_delta:
                artifact = True
                streak = self._artifact_streak.get(spec.name, 0) + 1
                self._artifact_streak[spec.name] = streak
                if streak >= ARTIFACT_RESYNC_N:
                    # Consecutive jumps = a sustained real change ⟵ adopt the new baseline.
                    # The current sample stays rejected; the next one is accepted.
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
            # Not a real measurement ⇒ it never reaches the user's eyes.
            return FieldResult(spec.name, None, Status.INVALID, flags)

        # (5) CLINICAL: possible but abnormal ⇒ value kept and flagged.
        if spec.clinical_min is not None and v < spec.clinical_min:
            flags.append(FLAG_CLINICAL_LOW)
        if spec.clinical_max is not None and v > spec.clinical_max:
            flags.append(FLAG_CLINICAL_HIGH)

        status = Status.WARN if flags else Status.VALID
        return FieldResult(spec.name, v, status, flags)

    def _forget(self, name: str) -> None:
        """Forget a field's memory — after contact loss, comparing the new reading to the old one is meaningless."""
        self._last_accepted.pop(name, None)
        self._last_seen.pop(name, None)
        self._repeat.pop(name, None)
        self._artifact_streak.pop(name, None)


def _ir_in_range(v: float) -> bool:
    """An ir_dc value the sensor could actually produce (not NaN, not inf, within ADC range)."""
    if math.isnan(v) or math.isinf(v):
        return False
    return 0.0 <= v <= CONTACT_IR_SANITY_MAX


def _is_number(v: object) -> bool:
    """bool is an int in Python — excluded explicitly so True never passes as a reading."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)

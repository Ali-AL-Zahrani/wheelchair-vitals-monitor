"""
Automated proof for every edge case in the validator.

Every test here guards a medical decision, not a programming detail:
no unverified number reaches the user's eyes, and no movement alert is silenced.
"""

from __future__ import annotations

import math

import pytest

from interface import VitalSample
from validator import (
    ALERT_LIFT_WRIST,
    ALERT_MEASUREMENT_SILENT,
    ALERT_NEEDS_MOVEMENT,
    ALERT_SENSOR_FAULT,
    ARTIFACT_RESYNC_N,
    FLAG_ARTIFACT,
    FLAG_CLINICAL_HIGH,
    FLAG_CLINICAL_LOW,
    FLAG_IR_IMPLAUSIBLE,
    FLAG_MOVEMENT_UNVERIFIED,
    FLAG_NO_CONTACT,
    FLAG_SANITY_MISSING,
    FLAG_SANITY_NAN,
    FLAG_SANITY_RANGE,
    FLAG_SANITY_TYPE,
    FLAG_STUCK,
    FLAG_TIME_BACKWARD,
    Status,
    Validator,
)

WRIST_FIELDS = ("heart_rate", "spo2", "skin_temp")


def sample(t=0.0, ir=38_000.0, hr=74.0, spo2=97.0, temp=33.4, mov=0.5) -> VitalSample:
    return VitalSample(t=t, ir_dc=ir, heart_rate=hr, spo2=spo2, skin_temp=temp, movement=mov)


@pytest.fixture
def v() -> Validator:
    return Validator()


# ── Normal case ──
def test_valid_sample_passes_clean(v):
    r = v.validate(sample())
    assert r.contact is True
    for name in WRIST_FIELDS:
        assert r.status_of(name) is Status.VALID
        assert r.flags_of(name) == []
    assert r.alerts == []


# ── (1) Contact gate ──
def test_no_contact_blanks_wrist_fields(v):
    r = v.validate(sample(ir=2_000.0))
    for name in WRIST_FIELDS:
        assert r.status_of(name) is Status.NO_CONTACT
        assert r.value_of(name) is None
    assert FLAG_NO_CONTACT in r.flags


def test_no_contact_blanks_even_perfectly_plausible_numbers(v):
    """The most dangerous case: a sensor with no wrist outputs perfect numbers. They must never be shown."""
    r = v.validate(sample(ir=1_500.0, hr=72.0, spo2=98.0, temp=33.0))
    for name in WRIST_FIELDS:
        assert r.value_of(name) is None


def test_missing_or_nan_ir_means_no_contact(v):
    assert v.validate(sample(ir=None)).contact is False
    assert v.validate(sample(ir=float("nan"))).contact is False


def test_movement_is_validated_even_without_contact(v):
    """The movement source is separate from the armrest — lifting the wrist does not disable it."""
    r = v.validate(sample(ir=1_000.0, mov=0.6))
    assert r.status_of("movement") is Status.VALID
    assert r.value_of("movement") == 0.6


# ── (2) SANITY ⇒ INVALID, value discarded ──
def test_none_value_is_invalid(v):
    r = v.validate(sample(hr=None))
    assert r.status_of("heart_rate") is Status.INVALID
    assert r.value_of("heart_rate") is None
    assert FLAG_SANITY_MISSING in r.flags_of("heart_rate")


def test_nan_value_is_invalid(v):
    r = v.validate(sample(spo2=float("nan")))
    assert r.status_of("spo2") is Status.INVALID
    assert FLAG_SANITY_NAN in r.flags_of("spo2")


def test_impossible_heart_rate_is_dropped(v):
    r = v.validate(sample(hr=320.0))
    assert r.status_of("heart_rate") is Status.INVALID
    assert r.value_of("heart_rate") is None
    assert FLAG_SANITY_RANGE in r.flags_of("heart_rate")


def test_spo2_above_100_is_dropped(v):
    r = v.validate(sample(spo2=104.0))
    assert r.status_of("spo2") is Status.INVALID
    assert FLAG_SANITY_RANGE in r.flags_of("spo2")


def test_non_numeric_value_is_invalid(v):
    r = v.validate(sample(hr="74"))
    assert r.status_of("heart_rate") is Status.INVALID
    assert FLAG_SANITY_TYPE in r.flags_of("heart_rate")


def test_boolean_is_not_accepted_as_number(v):
    r = v.validate(sample(hr=True))
    assert r.status_of("heart_rate") is Status.INVALID


def test_negative_movement_is_invalid(v):
    r = v.validate(sample(mov=-1.0))
    assert r.status_of("movement") is Status.INVALID


# ── (2) CLINICAL ⇒ WARN, value kept ──
def test_low_heart_rate_warns_but_keeps_value(v):
    r = v.validate(sample(hr=45.0))
    assert r.status_of("heart_rate") is Status.WARN
    assert r.value_of("heart_rate") == 45.0
    assert FLAG_CLINICAL_LOW in r.flags_of("heart_rate")


def test_high_heart_rate_warns(v):
    r = v.validate(sample(hr=140.0))
    assert r.status_of("heart_rate") is Status.WARN
    assert FLAG_CLINICAL_HIGH in r.flags_of("heart_rate")


def test_low_spo2_warns_but_keeps_value(v):
    r = v.validate(sample(spo2=90.0))
    assert r.status_of("spo2") is Status.WARN
    assert r.value_of("spo2") == 90.0


def test_cold_wrist_warns_without_raising_any_alert(v):
    """
    A cold wrist is very common in this population (poor peripheral circulation).
    It must only be flagged — and must never raise a hypothermia alert:
    skin_temp is not core body temperature.
    """
    r = v.validate(sample(temp=26.0))
    assert r.status_of("skin_temp") is Status.WARN
    assert r.value_of("skin_temp") == 26.0
    assert r.alerts == []


def test_sanity_and_clinical_are_not_confused(v):
    """A heart rate of 45 is kept (abnormal but possible); 5 is discarded (impossible). Mixing them is a medical error."""
    assert v.validate(sample(t=0, hr=45.0)).value_of("heart_rate") == 45.0
    assert Validator().validate(sample(t=0, hr=5.0)).value_of("heart_rate") is None


# ── (2) Checks with memory ──
def test_frozen_sensor_is_rejected_after_limit(v):
    """A bit-identical value repeated = a hung sensor; showing it implies a live reading."""
    limit = v.stuck_repeat_limit
    r = None
    for i in range(limit):
        r = v.validate(sample(t=i * 30.0, hr=74.0, spo2=97.0 + i * 0.01,
                              temp=33.4 + i * 0.01, mov=0.5 + i * 0.001))
    assert FLAG_STUCK in r.flags_of("heart_rate")
    assert r.status_of("heart_rate") is Status.INVALID
    assert r.value_of("heart_rate") is None


def test_stable_but_not_identical_readings_are_not_stuck(v):
    r = None
    for i in range(v.stuck_repeat_limit + 5):
        r = v.validate(sample(t=i * 30.0, hr=74.0 + (i % 3) * 0.1,
                              spo2=97.0 + i * 0.01, temp=33.4 + i * 0.01,
                              mov=0.5 + i * 0.001))
    assert FLAG_STUCK not in r.flags_of("heart_rate")
    assert r.status_of("heart_rate") is Status.VALID


def test_sudden_jump_is_rejected_as_artifact(v):
    v.validate(sample(t=0.0, hr=72.0))
    r = v.validate(sample(t=30.0, hr=150.0))
    assert FLAG_ARTIFACT in r.flags_of("heart_rate")
    assert r.status_of("heart_rate") is Status.INVALID
    assert r.value_of("heart_rate") is None


def test_artifact_resyncs_after_sustained_change(v):
    """
    A sustained real change must not lock the field forever:
    after ARTIFACT_RESYNC_N consecutive jumps the new baseline is adopted.
    """
    v.validate(sample(t=0.0, hr=72.0))
    r = None
    for i in range(1, ARTIFACT_RESYNC_N + 1):
        r = v.validate(sample(t=i * 30.0, hr=150.0 + i * 0.5))
    assert r.status_of("heart_rate") is Status.INVALID
    # The next sample is accepted because the baseline was re-synced
    nxt = v.validate(sample(t=999.0, hr=151.0))
    assert FLAG_ARTIFACT not in nxt.flags_of("heart_rate")
    assert nxt.value_of("heart_rate") == 151.0  # WARN because above the clinical bound, but displayed


def test_contact_loss_clears_memory_so_reattach_is_not_an_artifact(v):
    """After lifting and replacing the wrist, the new reading is not compared to an unrelated old baseline."""
    v.validate(sample(t=0.0, hr=72.0))
    v.validate(sample(t=30.0, ir=1_000.0))
    r = v.validate(sample(t=60.0, hr=115.0))
    assert FLAG_ARTIFACT not in r.flags_of("heart_rate")
    assert r.value_of("heart_rate") == 115.0


# ── (3) Immobility watchdog ──
def test_immobility_raises_movement_alert(v):
    r = None
    t = 0.0
    while t <= v.immobility_limit_s:
        r = v.validate(sample(t=t, hr=74.0 + (t % 7) * 0.1, mov=0.01))
        t += 30.0
    assert ALERT_NEEDS_MOVEMENT in r.alerts
    assert r.immobility_s >= v.immobility_limit_s


def test_movement_resets_the_timer(v):
    t = 0.0
    while t < v.immobility_limit_s - 60.0:
        v.validate(sample(t=t, hr=74.0 + (t % 7) * 0.1, mov=0.01))
        t += 30.0
    r = v.validate(sample(t=t, mov=0.8))          # clear movement
    assert r.alerts == []
    r = v.validate(sample(t=t + 30.0, mov=0.01))
    assert r.alerts == []
    assert r.immobility_s == 30.0


def test_alert_fires_even_without_wrist_on_armrest(v):
    """Immobility is a pressure-injury risk independent of contact — lifting the wrist does not silence the alert."""
    r = None
    t = 0.0
    while t <= v.immobility_limit_s:
        r = v.validate(sample(t=t, ir=1_000.0, mov=0.01))
        t += 30.0
    assert ALERT_NEEDS_MOVEMENT in r.alerts


def test_unverifiable_movement_keeps_timer_running(v):
    """A broken movement sensor ⇒ movement cannot be confirmed ⇒ the alert stays possible (fail-loud)."""
    r = None
    t = 0.0
    while t <= v.immobility_limit_s:
        r = v.validate(sample(t=t, hr=74.0 + (t % 7) * 0.1, mov=None))
        t += 30.0
    assert FLAG_MOVEMENT_UNVERIFIED in r.flags
    assert ALERT_NEEDS_MOVEMENT in r.alerts


def test_valid_vitals_and_movement_alert_coexist(v):
    """flags and alerts are separate systems: heart rate valid and the alert active at the same time."""
    r = None
    t = 0.0
    while t <= v.immobility_limit_s:
        r = v.validate(sample(t=t, hr=74.0 + (t % 7) * 0.1,
                              spo2=97.0, temp=33.4 + (t % 5) * 0.01, mov=0.01))
        t += 30.0
    assert r.status_of("heart_rate") is Status.VALID
    assert ALERT_NEEDS_MOVEMENT in r.alerts


def test_backward_clock_does_not_hide_alert(v):
    v.validate(sample(t=1000.0, mov=0.01))
    r = v.validate(sample(t=10.0, mov=0.01))
    assert FLAG_TIME_BACKWARD in r.flags
    assert r.immobility_s == 0.0
    assert not math.isnan(r.immobility_s)


def test_backward_clock_does_not_clear_an_active_alert(v):
    """
    An MCU reboot during a long immobility stretch: the clock returns to zero.
    The accumulated time must not be lost — otherwise an active alert is silenced
    and the user goes twice as long without a weight shift. Only confirmed
    movement resets the counter.
    """
    t = 0.0
    while t <= v.immobility_limit_s:
        r = v.validate(sample(t=t, hr=74.0 + (t % 7) * 0.1, mov=0.01))
        t += 30.0
    assert ALERT_NEEDS_MOVEMENT in r.alerts        # alert active before the reboot

    after_reset = v.validate(sample(t=0.0, hr=73.0, mov=0.01))
    assert FLAG_TIME_BACKWARD in after_reset.flags
    assert ALERT_NEEDS_MOVEMENT in after_reset.alerts, "a clock fault silenced an active alert"


def test_immobility_accumulates_and_is_only_cleared_by_confirmed_movement(v):
    """The counter accumulates rather than differencing timestamps: a suspicious sample is ignored and nothing before it is lost."""
    v.validate(sample(t=0.0, mov=0.01))
    v.validate(sample(t=30.0, mov=0.01))
    assert v.validate(sample(t=60.0, mov=0.01)).immobility_s == 60.0
    # Backward clock: the sample neither adds nor resets
    assert v.validate(sample(t=5.0, mov=0.01)).immobility_s == 60.0
    # Confirmed movement ⟵ reset
    assert v.validate(sample(t=35.0, mov=0.9)).immobility_s == 0.0


def test_movement_exactly_at_threshold_counts_as_movement(v):
    v.validate(sample(t=0.0, mov=0.0))
    r = v.validate(sample(t=30.0, mov=v.movement_threshold))
    assert r.immobility_s == 0.0


def test_forward_clock_jump_errs_towards_alerting(v):
    """
    A forward time jump (faulty clock) accumulates a large duration ⇒ a possibly
    unnecessary alert. The direction is deliberate: an extra movement prompt is
    harmless; a silenced alert is a pressure-injury risk.
    """
    v.validate(sample(t=0.0, mov=0.01))
    r = v.validate(sample(t=99_999.0, mov=0.01))
    assert ALERT_NEEDS_MOVEMENT in r.alerts


# ── (1) Contact gate: bounds and plausibility ──
def test_ir_exactly_at_threshold_counts_as_contact(v):
    assert v.validate(sample(ir=v.contact_ir_threshold)).contact is True
    assert v.validate(sample(ir=v.contact_ir_threshold - 0.001)).contact is False


def test_implausible_ir_closes_the_gate_instead_of_opening_it(v):
    """
    A value above the ADC range does not come from a sensor but from a broken driver/bus.
    If the gate opened on it, noise would pass to the user's eyes as readings.
    """
    r = v.validate(sample(ir=9_999_999.0, hr=72.0, spo2=98.0))
    assert r.contact is False
    assert FLAG_IR_IMPLAUSIBLE in r.flags          # hardware fault, not a lifted wrist
    for name in WRIST_FIELDS:
        assert r.value_of(name) is None


def test_negative_ir_is_implausible_not_contact(v):
    r = v.validate(sample(ir=-500.0))
    assert r.contact is False
    assert FLAG_IR_IMPLAUSIBLE in r.flags


def test_lifted_wrist_is_not_flagged_as_hardware_fault(v):
    """Lifting the wrist is a normal, frequent event — never confused with a hardware fault in the log."""
    r = v.validate(sample(ir=1_500.0))
    assert FLAG_NO_CONTACT in r.flags
    assert FLAG_IR_IMPLAUSIBLE not in r.flags


# ── Check bounds: inclusiveness is deliberate and proven ──
def test_clinical_bounds_are_inclusive(v):
    """A value exactly on the bound is not abnormal — otherwise the screen fills with false warnings."""
    assert v.validate(sample(t=0.0, hr=50.0)).status_of("heart_rate") is Status.VALID
    assert Validator().validate(sample(hr=120.0)).status_of("heart_rate") is Status.VALID
    assert Validator().validate(sample(spo2=94.0)).status_of("spo2") is Status.VALID


def test_sanity_bounds_are_inclusive_and_keep_the_value(v):
    """The physical bound itself is possible: it is clinically flagged, not discarded."""
    r = v.validate(sample(hr=20.0))
    assert r.status_of("heart_rate") is Status.WARN
    assert r.value_of("heart_rate") == 20.0
    r2 = Validator().validate(sample(hr=250.0))
    assert r2.status_of("heart_rate") is Status.WARN
    assert r2.value_of("heart_rate") == 250.0


def test_just_outside_sanity_is_dropped(v):
    assert v.validate(sample(hr=19.9)).status_of("heart_rate") is Status.INVALID
    assert Validator().validate(sample(hr=250.1)).status_of("heart_rate") is Status.INVALID


def test_infinity_is_rejected_like_nan(v):
    r = v.validate(sample(hr=float("inf")))
    assert r.status_of("heart_rate") is Status.INVALID
    assert FLAG_SANITY_NAN in r.flags_of("heart_rate")


def test_artifact_delta_exactly_at_limit_is_accepted(v):
    """The limit itself is accepted — rejecting it would reject a legitimate physiological change."""
    v.validate(sample(t=0.0, hr=72.0))
    r = v.validate(sample(t=30.0, hr=102.0))      # delta = artifact_max_delta exactly
    assert FLAG_ARTIFACT not in r.flags_of("heart_rate")
    assert r.value_of("heart_rate") == 102.0


def test_frozen_sensor_keeps_being_rejected_while_frozen(v):
    """Freezing is not a one-off event: it stays rejected as long as the value does not change."""
    r = None
    for i in range(v.stuck_repeat_limit + 6):
        r = v.validate(sample(t=i * 30.0, hr=74.0, spo2=97.0 + i * 0.01,
                              temp=33.4 + i * 0.01, mov=0.5 + i * 0.001))
    assert FLAG_STUCK in r.flags_of("heart_rate")
    assert r.value_of("heart_rate") is None


def test_recovered_sensor_is_accepted_again_after_being_stuck(v):
    """If the sensor starts changing again, the field is not locked forever."""
    for i in range(v.stuck_repeat_limit + 2):
        v.validate(sample(t=i * 30.0, hr=74.0, spo2=97.0 + i * 0.01,
                          temp=33.4 + i * 0.01, mov=0.5 + i * 0.001))
    r = v.validate(sample(t=999.0, hr=76.0, spo2=97.5, temp=33.5, mov=0.6))
    assert FLAG_STUCK not in r.flags_of("heart_rate")
    assert r.value_of("heart_rate") == 76.0


# ── (4) Wrist-pressure watchdog — medical team decision, 12 Aug ──
def test_approved_immobility_limit_is_fifteen_minutes():
    assert Validator().immobility_limit_s == 15 * 60.0


def test_wrist_reminder_fires_even_while_the_body_keeps_moving():
    """
    The most dangerous confusion in this alert: moving the body **does not relieve pressure on the wrist**.
    If the wrist reminder were tied to body movement it would fall silent exactly when the rest goes on too long.
    """
    v = Validator(wrist_rest_limit_s=60.0)
    r = None
    for i in range(5):
        r = v.validate(sample(t=i * 30.0, mov=0.9))     # clear, continuous body movement
    assert ALERT_NEEDS_MOVEMENT not in r.alerts          # the body is actually moving
    assert ALERT_LIFT_WRIST in r.alerts                  # and the wrist has been pressed for 120 s


def test_lifting_the_wrist_resets_its_own_timer(v):
    v = Validator(wrist_rest_limit_s=60.0)
    for i in range(4):
        v.validate(sample(t=i * 30.0))
    assert ALERT_LIFT_WRIST in v.validate(sample(t=120.0)).alerts

    lifted = v.validate(sample(t=150.0, ir=1_000.0))     # wrist lifted
    assert lifted.wrist_rest_s == 0.0
    assert ALERT_LIFT_WRIST not in lifted.alerts


def test_body_movement_does_not_clear_the_wrist_timer():
    v = Validator(wrist_rest_limit_s=90.0)
    v.validate(sample(t=0.0, mov=0.01))
    r = v.validate(sample(t=30.0, mov=0.95))            # confirmed weight shift
    assert r.immobility_s == 0.0                        # body counter reset
    assert r.wrist_rest_s == 30.0                       # wrist counter untouched


# ── (5) Sensor fault ──
def test_hardware_fault_raises_its_own_alert(v):
    r = v.validate(sample(ir=9_999_999.0))
    assert ALERT_SENSOR_FAULT in r.alerts
    assert FLAG_IR_IMPLAUSIBLE in r.flags


def test_lifted_wrist_does_not_raise_a_fault_alert(v):
    """Lifting the wrist is normal behaviour — a maintenance alert on it breeds alarm fatigue."""
    assert ALERT_SENSOR_FAULT not in v.validate(sample(ir=1_200.0)).alerts


# ── (6) Measurement silence ──
def test_silence_alert_fires_when_no_reading_is_displayable():
    """A silent screen reads as 'all is well' — while monitoring has effectively stopped."""
    v = Validator(silence_limit_s=60.0)
    r = None
    for i in range(4):
        r = v.validate(sample(t=i * 30.0, ir=900.0))     # no contact ⇒ no numbers
    assert ALERT_MEASUREMENT_SILENT in r.alerts


def test_silence_alert_also_fires_when_the_sensor_returns_garbage():
    """Silence is measured by the absence of a valid number, not by its cause — contact present, every reading corrupt."""
    v = Validator(silence_limit_s=60.0)
    r = None
    for i in range(4):
        r = v.validate(sample(t=i * 30.0, hr=None, spo2=float("nan"), temp=999.0))
    assert r.contact is True
    assert ALERT_MEASUREMENT_SILENT in r.alerts


def test_one_valid_reading_clears_the_silence_timer():
    v = Validator(silence_limit_s=60.0)
    for i in range(3):
        v.validate(sample(t=i * 30.0, ir=900.0))
    r = v.validate(sample(t=90.0))                       # the reading is back
    assert r.silence_s == 0.0
    assert ALERT_MEASUREMENT_SILENT not in r.alerts


def test_alerts_are_independent_of_each_other():
    """Four independent systems: they can coincide in one sample without hiding one another."""
    v = Validator(immobility_limit_s=60.0, wrist_rest_limit_s=60.0)
    r = None
    for i in range(4):
        r = v.validate(sample(t=i * 30.0, mov=0.0))      # still + wrist resting
    assert ALERT_NEEDS_MOVEMENT in r.alerts
    assert ALERT_LIFT_WRIST in r.alerts


# ── Integration with the mock sensor ──
def test_mock_scenario_exercises_every_failure_mode():
    from mock_sensor import MockSensor, default_scenario

    sensor = MockSensor(seed=42, sample_period_s=30.0, faults=default_scenario())
    validator = Validator()
    seen_flags = set()
    seen_alerts = set()
    for _ in range(120):
        r = validator.validate(sensor.read())
        seen_flags.update(r.flags)
        seen_alerts.update(r.alerts)
        for name in r.fields:
            seen_flags.update(r.flags_of(name))
            # The fixed rule: any state other than VALID/WARN carries no number
            if r.status_of(name) in (Status.INVALID, Status.NO_CONTACT):
                assert r.value_of(name) is None

    for expected in (FLAG_NO_CONTACT, FLAG_SANITY_NAN, FLAG_SANITY_MISSING,
                     FLAG_SANITY_RANGE, FLAG_ARTIFACT, FLAG_STUCK, FLAG_CLINICAL_LOW,
                     FLAG_IR_IMPLAUSIBLE):
        assert expected in seen_flags, f"the scenario did not trigger {expected}"
    for expected in (ALERT_NEEDS_MOVEMENT, ALERT_LIFT_WRIST,
                     ALERT_SENSOR_FAULT, ALERT_MEASUREMENT_SILENT):
        assert expected in seen_alerts, f"the scenario did not raise {expected}"

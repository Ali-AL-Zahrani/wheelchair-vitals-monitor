"""
Automated proof for the display-layer rules.

The tests go through the **real validator**, not fabricated results, because the
actual risk lies in the composition: a sound validator + a renderer that
misreads it = a wrong number in front of the user.

The central rule proven here: a number is displayed only if it is a measurement
that passed validation.
"""

from __future__ import annotations

import json

import pytest

from display import (
    MSG_INVALID,
    MSG_LIFT_WRIST,
    MSG_MOVE,
    MSG_NO_CONTACT,
    MSG_SENSOR_FAULT,
    MSG_WARN,
    PATIENT_FIELDS,
    Severity,
    Tile,
    build_carer_screen,
    build_screen,
    format_clock,
    render_line,
)
from interface import VitalSample
from validator import (
    CONTACT_IR_THRESHOLD,
    Status,
    Validator,
)


def _sample(t=0.0, ir_dc=38_000.0, hr=74.0, spo2=97.0, temp=33.4, movement=0.5):
    return VitalSample(t=t, ir_dc=ir_dc, heart_rate=hr, spo2=spo2,
                       skin_temp=temp, movement=movement)


def _screen(sample, validator=None):
    validator = validator or Validator()
    return build_screen(validator.validate(sample))


# ── Valid state ──

def test_valid_shows_number_without_message():
    model = _screen(_sample())
    tile = model.tile("heart_rate")
    assert tile.severity is Severity.NORMAL
    assert tile.value_text == "74"      # no decimals: false precision implies certainty that does not exist
    assert tile.message is None
    assert model.banners == []


def test_skin_temp_label_never_claims_body_temperature():
    """A 'Body Temp' label on a user screen is a medical error — the measurement is wrist skin."""
    tile = _screen(_sample()).tile("skin_temp")
    assert "Body" not in tile.label
    assert "Wrist" in tile.label
    assert tile.value_text == "33.4"


def test_movement_is_never_displayed_to_patient():
    """Movement is an internal indicator that drives the alert, not a number to be read."""
    assert "movement" not in PATIENT_FIELDS
    assert all(tile.name != "movement" for tile in _screen(_sample()).tiles)


# ── Blocked states: no number ──

def test_no_contact_blocks_every_wrist_number():
    """Wrist lifted: the sensor still outputs reading-like numbers — all of them are withheld."""
    model = _screen(_sample(ir_dc=CONTACT_IR_THRESHOLD - 1, hr=88.0, spo2=96.0, temp=32.0))
    assert model.contact is False
    for tile in model.tiles:
        assert tile.value_text is None, f"{tile.name}: a number appeared without contact"
        assert tile.severity is Severity.BLOCKED
        assert tile.message == MSG_NO_CONTACT
    assert [b.kind for b in model.banners] == ["contact"]


def test_invalid_reading_shows_message_not_number():
    model = _screen(_sample(hr=320.0))          # physically impossible
    tile = model.tile("heart_rate")
    assert tile.value_text is None
    assert tile.message == MSG_INVALID
    assert tile.severity is Severity.BLOCKED
    # The other fields are valid ⟵ blanking applies to the faulty field only
    assert model.tile("spo2").value_text is not None


def test_nan_is_blocked_not_printed():
    tile = _screen(_sample(spo2=float("nan"))).tile("spo2")
    assert tile.value_text is None
    assert tile.severity is Severity.BLOCKED


def test_missing_channel_is_blocked():
    tile = _screen(_sample(hr=None)).tile("heart_rate")
    assert tile.value_text is None
    assert tile.severity is Severity.BLOCKED


def test_stale_value_never_survives_contact_loss():
    """
    The most dangerous screen scenario: a valid reading, then the wrist is lifted.
    The old number must vanish immediately — leaving it implies a live reading.
    """
    validator = Validator()
    first = _screen(_sample(t=0.0), validator)
    assert first.tile("heart_rate").value_text == "74"

    second = _screen(_sample(t=30.0, ir_dc=1_500.0), validator)
    assert second.tile("heart_rate").value_text is None


# ── Abnormal-but-possible: the number is shown with emphasis ──

def test_warn_keeps_the_number_and_marks_it():
    """An abnormal but real reading ⇒ displayed. Withholding it hides important clinical information."""
    tile = _screen(_sample(spo2=89.0)).tile("spo2")
    assert tile.value_text == "89"
    assert tile.severity is Severity.WARN
    assert tile.message == MSG_WARN


def test_warning_does_not_rely_on_colour_alone():
    """Colour blindness + low vision: an icon **and** text are required with every non-normal state."""
    for sample in (_sample(spo2=89.0), _sample(hr=None), _sample(ir_dc=900.0)):
        for tile in _screen(sample).tiles:
            if tile.severity is not Severity.NORMAL:
                assert tile.icon, f"{tile.name}: emphasis without an icon"
                assert tile.message, f"{tile.name}: emphasis without text"


# ── The alert is independent of reading quality ──

def test_movement_alert_fires_while_readings_stay_valid():
    """flags (data quality) and alerts (action required) are separate tracks."""
    validator = Validator(immobility_limit_s=60.0)
    model = None
    for i in range(5):                       # completely still, beyond the limit
        model = _screen(_sample(t=i * 30.0, movement=0.0), validator)

    assert model.tile("heart_rate").severity in (Severity.NORMAL, Severity.WARN)
    assert model.tile("heart_rate").value_text is not None
    assert [b.kind for b in model.banners] == ["movement"]
    assert model.banners[0].text == MSG_MOVE
    assert model.needs_sound is True         # a visual alert alone is not enough


def test_movement_alert_outranks_contact_banner():
    """Pressure-injury risk outranks a data problem."""
    validator = Validator(immobility_limit_s=60.0)
    model = None
    for i in range(5):
        model = _screen(_sample(t=i * 30.0, ir_dc=1_200.0, movement=0.0), validator)
    assert [b.kind for b in model.banners] == ["movement", "contact"]


def test_no_alert_means_no_sound():
    assert _screen(_sample()).needs_sound is False


# ── The newer alerts (medical team decisions, 12 Aug) ──

def test_wrist_reminder_has_its_own_banner_and_sound():
    validator = Validator(wrist_rest_limit_s=60.0)
    model = None
    for i in range(5):
        model = _screen(_sample(t=i * 30.0, movement=0.9), validator)
    kinds = [b.kind for b in model.banners]
    assert "lift_wrist" in kinds
    assert model.banners[kinds.index("lift_wrist")].text == MSG_LIFT_WRIST
    assert model.needs_sound is True


def test_sensor_fault_shows_a_maintenance_banner_with_sound():
    model = _screen(_sample(ir_dc=9_999_999.0))
    assert [b.kind for b in model.banners] == ["sensor_fault"]
    assert model.banners[0].text == MSG_SENSOR_FAULT
    assert model.needs_sound is True


def test_sensor_fault_suppresses_the_put_your_wrist_message():
    """
    "Rest your wrist" cannot fix a broken device, and repeating it pushes the user
    to press the wrist harder for nothing — and impaired sensation may hide the harm.
    """
    model = _screen(_sample(ir_dc=9_999_999.0))
    assert model.contact is False
    assert "contact" not in [b.kind for b in model.banners]
    # And still no number is shown: a fault does not open the door to an unverified reading
    for tile in model.tiles:
        assert tile.value_text is None


def test_lifted_wrist_still_shows_the_ordinary_contact_message():
    model = _screen(_sample(ir_dc=1_200.0))
    assert [b.kind for b in model.banners] == ["contact"]
    assert model.needs_sound is False        # normal behaviour does not deserve a sound


def test_measurement_silence_announces_that_monitoring_stopped():
    validator = Validator(silence_limit_s=60.0)
    model = None
    for i in range(4):
        model = _screen(_sample(t=i * 30.0, ir_dc=900.0), validator)
    kinds = [b.kind for b in model.banners]
    assert "silent" in kinds
    assert model.needs_sound is True


def test_alert_banners_are_ordered_by_risk():
    """Pressure-injury risk first, then monitoring failure, then ordinary contact loss."""
    validator = Validator(immobility_limit_s=60.0, wrist_rest_limit_s=60.0,
                          silence_limit_s=60.0)
    model = None
    for i in range(4):
        model = _screen(_sample(t=i * 30.0, movement=0.0), validator)
    kinds = [b.kind for b in model.banners]
    assert kinds.index("movement") < kinds.index("lift_wrist")


# ── General invariants ──

@pytest.mark.parametrize("sample", [
    _sample(),
    _sample(ir_dc=500.0),
    _sample(hr=None, spo2=float("nan"), temp=99.0),
    _sample(spo2=89.0),
    _sample(hr=320.0),
])
def test_blocked_tile_never_carries_a_number(sample):
    """The invariant that protects the user's eyes — tested across every state, not just one."""
    for tile in _screen(sample).tiles:
        if tile.severity is Severity.BLOCKED:
            assert tile.value_text is None


def test_defensive_guard_blocks_value_less_valid_status():
    """
    A state the validator never produces today, but the screen does not trust it:
    a status claiming a number without one is treated as an unavailable reading,
    not printed as a blank.
    """
    from display import _build_tile
    from validator import FieldResult, ValidationResult

    broken = ValidationResult(
        t=0.0, contact=True, ir_dc=38_000.0,
        fields={"heart_rate": FieldResult("heart_rate", None, Status.VALID, [])},
    )
    tile = _build_tile("heart_rate", broken)
    assert tile.severity is Severity.BLOCKED
    assert tile.value_text is None
    assert tile.message == MSG_INVALID


def test_screen_model_is_json_serialisable():
    """The renderer may live in another process (a web page) — the description must cross as JSON."""
    payload = json.dumps(_screen(_sample()).to_dict(), ensure_ascii=False)
    assert "heart_rate" in payload


def test_text_renderer_matches_screen_rules():
    """The terminal and the screen read from the same source — the rule never forks into two copies."""
    validator = Validator()
    result = validator.validate(_sample(ir_dc=800.0))
    line = render_line(result)
    assert MSG_NO_CONTACT in line
    assert "74" not in line                  # no number leaks without contact


# ── Caregiver screen ──

def test_carer_screen_hides_no_number_that_the_user_screen_hid():
    """
    One rule for both screens: a rejected reading is rejected for both parties.
    Being a caregiver does not make a corrupt number valid.
    """
    # Wrist lifted while the sensor outputs perfect numbers
    carer = build_carer_screen(Validator().validate(
        _sample(ir_dc=900.0, hr=72.0, spo2=98.0, temp=33.0)))
    assert carer.normal == [] and carer.abnormal == []
    assert len(carer.blocked) == 3
    for tile in carer.blocked:
        assert tile.value_text is None


def test_carer_screen_never_reassures_while_no_reading_arrives():
    """
    "Nothing needs attention" while all three readings are blocked = false reassurance —
    the very state the silence alert was built for, before its time limit is reached.
    """
    validator = Validator()
    frozen = None
    for i in range(validator.stuck_repeat_limit + 1):
        frozen = build_carer_screen(validator.validate(
            _sample(t=i * 30.0, hr=74.0, spo2=97.0, temp=33.4, movement=0.5 + i * 0.001)))
    assert len(frozen.blocked) == 3          # sensor frozen
    assert frozen.monitoring is False        # so the screen does not announce reassurance
    assert build_carer_screen(Validator().validate(_sample())).monitoring is True


def test_carer_screen_flags_attention_only_when_something_needs_it():
    quiet = build_carer_screen(Validator().validate(_sample()))
    assert quiet.attention is False
    assert quiet.monitoring is True
    assert quiet.alerts == [] and quiet.abnormal == []
    assert len(quiet.normal) == 3

    warned = build_carer_screen(Validator().validate(_sample(spo2=89.0)))
    assert warned.attention is True
    assert [t.name for t in warned.abnormal] == ["spo2"]


def test_carer_screen_separates_alerts_from_readings():
    """The caregiver needs 'what needs intervention' before the numbers."""
    validator = Validator(immobility_limit_s=60.0)
    carer = None
    for i in range(4):
        carer = build_carer_screen(validator.validate(_sample(t=i * 30.0, movement=0.0)))
    assert [a.kind for a in carer.alerts] == ["movement"]
    assert carer.attention is True
    assert carer.immobility_s >= 60.0          # and for how long — not merely "there is an alert"


def test_carer_model_is_json_serialisable():
    payload = json.dumps(build_carer_screen(Validator().validate(_sample())).to_dict(),
                         ensure_ascii=False)
    assert "immobility_s" in payload


def test_clock_formatting():
    assert format_clock(0) == "00:00"
    assert format_clock(90) == "01:30"
    assert format_clock(3_600) == "01:00:00"
    assert format_clock(-5) == "00:00"       # negative time is not shown as an odd value


def test_tile_is_immutable():
    """The description is not edited after creation — a renderer that edits a value bypasses a safety rule."""
    tile = _screen(_sample()).tile("heart_rate")
    with pytest.raises(Exception):
        tile.value_text = "999"  # type: ignore[misc]
    assert isinstance(tile, Tile)

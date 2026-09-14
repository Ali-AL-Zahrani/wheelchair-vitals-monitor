"""
Display layer — turns validator output into what the user's eyes see. **Pure logic, no rendering**.

Deliberately separated from the renderer: the rule "when a number may be shown
and when it may not" is a **safety** rule, not a design detail. It is written
once, tested automatically, and consumed by every renderer (the web page in
screen.py, the terminal in demo.py, and the hardware screen later).

Governing rule: **"no reading" is more honest than a wrong reading.**
Any state that is not VALID/WARN ⇒ no number at all: no stale number, no zero,
no dash that could be read as a value.

The only input to this layer is ValidationResult. It never touches a raw VitalSample.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from validator import (
    ALERT_LIFT_WRIST,
    ALERT_MEASUREMENT_SILENT,
    ALERT_NEEDS_MOVEMENT,
    ALERT_SENSOR_FAULT,
    FIELD_SPECS,
    Status,
    ValidationResult,
)

# What is shown to the user: wrist values only.
# movement is an internal indicator driving the immobility alert — a number that
# means nothing to the user, so it is never displayed.
PATIENT_FIELDS = ("heart_rate", "spo2", "skin_temp")

# Labels describe what is actually measured: "Wrist Skin Temp", not "Body Temp".
# A misleading label on a user screen is a medical error, not a wording choice.
LABELS: Dict[str, str] = {
    "heart_rate": "Heart Rate",
    "spo2": "Blood Oxygen",
    "skin_temp": "Wrist Skin Temp",   # not Body Temp — the label names the measurement site
    "movement": "Movement",
}

# Displayed decimal places. Heart rate and oxygen without decimals: false
# precision on a user screen implies certainty that does not exist, especially
# as wrist SpO2 is a trend indicator.
DECIMALS: Dict[str, int] = {"heart_rate": 0, "spo2": 0, "skin_temp": 1}

# Never colour alone: an icon **and** text with every state (colour blindness, low vision).
ICON_WARN = "⚠"
ICON_INVALID = "⛔"
ICON_NO_CONTACT = "✋"
ICON_MOVE = "🔔"
ICON_LIFT_WRIST = "🤚"
ICON_FAULT = "🛠"
ICON_SILENT = "🔇"

MSG_WARN = "Outside expected range"
MSG_INVALID = "Reading unavailable"
MSG_NO_CONTACT = "Rest your wrist on the armrest"
MSG_MOVE = "Time to move"
MSG_LIFT_WRIST = "Lift your wrist off the armrest"
# The two messages below address the caregiver, not the user.
MSG_SENSOR_FAULT = "Device fault — needs checking"
MSG_SILENT = "Monitoring stopped — no readings"

# Placeholder for an absent value. Deliberately nothing that resembles a number.
NO_VALUE_TEXT = "—"


class Severity(str, Enum):
    """Display severity — the renderer maps it to colour/size and decides nothing itself."""

    NORMAL = "NORMAL"    # a valid number
    WARN = "WARN"        # a number shown with emphasis
    ALERT = "ALERT"      # action required from the user now
    BLOCKED = "BLOCKED"  # no number — the reading is withheld


@dataclass(frozen=True)
class Tile:
    """One measurement card on the screen. value_text=None means: **print no number**."""

    name: str
    label: str
    unit: str
    value_text: Optional[str]
    icon: str
    message: Optional[str]
    severity: Severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "unit": self.unit,
            "value_text": self.value_text,
            "icon": self.icon,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class Banner:
    """A full-width strip above the cards: a required action or a general reason for blanking."""

    kind: str          # "movement" or "contact"
    icon: str
    text: str
    severity: Severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "icon": self.icon,
            "text": self.text,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class ScreenModel:
    """
    A complete description of what must appear on the screen at one instant.

    The renderer executes this description literally; it adds nothing and infers nothing.
    """

    t: float
    clock: str
    contact: bool
    tiles: List[Tile]
    banners: List[Banner]
    needs_sound: bool  # does this instant call for an audible alert (not only visual)?

    def tile(self, name: str) -> Tile:
        for tile in self.tiles:
            if tile.name == name:
                return tile
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "clock": self.clock,
            "contact": self.contact,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "banners": [banner.to_dict() for banner in self.banners],
            "needs_sound": self.needs_sound,
        }


def format_clock(seconds: float) -> str:
    """Virtual session time. Hours appear only once exceeded — less visual noise."""
    total = int(max(0.0, seconds))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_value(name: str, value: float) -> str:
    return f"{value:.{DECIMALS.get(name, 1)}f}"


def _build_tile(name: str, result: ValidationResult) -> Tile:
    field_result = result.fields[name]
    label = LABELS[name]
    unit = FIELD_SPECS[name].unit
    status = field_result.status
    value = field_result.value

    # Defensive guard: a status claiming a number without one = a fault upstream.
    # Safest to treat it as an unavailable reading, not to print a blank where the number was.
    if status in (Status.VALID, Status.WARN) and value is None:
        status = Status.INVALID

    if status is Status.VALID:
        return Tile(name, label, unit, _format_value(name, float(value)),
                    "", None, Severity.NORMAL)

    if status is Status.WARN:
        # The number is shown — it is a real measurement — but with icon and text, never colour alone.
        return Tile(name, label, unit, _format_value(name, float(value)),
                    ICON_WARN, MSG_WARN, Severity.WARN)

    if status is Status.NO_CONTACT:
        return Tile(name, label, unit, None,
                    ICON_NO_CONTACT, MSG_NO_CONTACT, Severity.BLOCKED)

    # INVALID: the technical reason (NaN / frozen / jump) means nothing to the user — one clear message.
    # The detail belongs in the audit log, not on the user screen.
    return Tile(name, label, unit, None, ICON_INVALID, MSG_INVALID, Severity.BLOCKED)


def build_screen(result: ValidationResult) -> ScreenModel:
    """
    ValidationResult ⟵ the only source. No other input to this function, and no
    memory between instants: the screen reflects the current instant only, so a
    stale number can never remain displayed by mistake.
    """
    tiles = [_build_tile(name, result) for name in PATIENT_FIELDS]

    banners: List[Banner] = []

    # Order is deliberate and ranked by risk, not by discovery order in the code:
    # pressure-injury risk (body, then wrist) ⟵ then monitoring failure ⟵ then ordinary contact loss.
    if ALERT_NEEDS_MOVEMENT in result.alerts:
        banners.append(Banner("movement", ICON_MOVE, MSG_MOVE, Severity.ALERT))

    if ALERT_LIFT_WRIST in result.alerts:
        banners.append(Banner("lift_wrist", ICON_LIFT_WRIST, MSG_LIFT_WRIST, Severity.ALERT))

    if ALERT_SENSOR_FAULT in result.alerts:
        banners.append(Banner("sensor_fault", ICON_FAULT, MSG_SENSOR_FAULT, Severity.ALERT))

    if ALERT_MEASUREMENT_SILENT in result.alerts:
        banners.append(Banner("silent", ICON_SILENT, MSG_SILENT, Severity.ALERT))

    # One general contact message instead of repeating it on three cards — one cause, one prompt.
    # Suppressed during a fault: "rest your wrist" cannot fix a broken device, and
    # repeating it pushes the user to press the wrist harder for nothing.
    if not result.contact and ALERT_SENSOR_FAULT not in result.alerts:
        banners.append(Banner("contact", ICON_NO_CONTACT, MSG_NO_CONTACT, Severity.BLOCKED))

    # A visual alert alone is not enough: the user may not be looking at the
    # screen, and the caregiver may be in another room. Every alert calls for sound.
    needs_sound = any(b.severity is Severity.ALERT for b in banners)

    model = ScreenModel(
        t=result.t,
        clock=format_clock(result.t),
        contact=result.contact,
        tiles=tiles,
        banners=banners,
        needs_sound=needs_sound,
    )

    # Safety invariant, asserted at construction, not only at render time:
    # a blocked card never carries a number under any circumstances.
    for tile in model.tiles:
        assert not (tile.severity is Severity.BLOCKED and tile.value_text is not None), (
            f"Safety rule violated: blocked tile carries a number ({tile.name})"
        )
    return model


@dataclass(frozen=True)
class CarerModel:
    """
    Caregiver screen — **a summary of what needs intervention**, not a second copy of the user screen.

    The difference is deliberate: the caregiver may be in another room and not
    following the numbers moment by moment; what helps them is "is there
    something that needs intervention now, and for how long".
    """

    clock: str
    attention: bool                 # is there anything that needs intervention now?
    monitoring: bool                # is any number arriving at all?
    alerts: List[Banner]            # required actions
    abnormal: List[Tile]            # abnormal readings that are displayed (WARN)
    blocked: List[Tile]             # unavailable readings
    normal: List[Tile]              # valid readings — for reassurance, not monitoring
    immobility_s: float
    wrist_rest_s: float
    silence_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clock": self.clock,
            "attention": self.attention,
            "monitoring": self.monitoring,
            "alerts": [a.to_dict() for a in self.alerts],
            "abnormal": [t.to_dict() for t in self.abnormal],
            "blocked": [t.to_dict() for t in self.blocked],
            "normal": [t.to_dict() for t in self.normal],
            "immobility_s": round(self.immobility_s, 1),
            "wrist_rest_s": round(self.wrist_rest_s, 1),
            "silence_s": round(self.silence_s, 1),
        }


def build_carer_screen(result: ValidationResult) -> CarerModel:
    """
    Derived from the same ScreenModel — so the "when a number is shown" rule is
    one rule on both screens. The caregiver screen **never reveals a number the
    user screen withheld**: a rejected reading is rejected for both parties;
    being a caregiver does not make a corrupt number valid.
    """
    model = build_screen(result)
    # "Nothing needs attention" while not a single number arrives = false reassurance.
    # It is exactly the state the silence alert was built for, before its time limit is reached.
    monitoring = any(t.value_text is not None for t in model.tiles)
    return CarerModel(
        clock=model.clock,
        attention=bool(model.banners) or any(
            t.severity is Severity.WARN for t in model.tiles),
        monitoring=monitoring,
        alerts=[b for b in model.banners if b.severity is Severity.ALERT],
        abnormal=[t for t in model.tiles if t.severity is Severity.WARN],
        blocked=[t for t in model.tiles if t.severity is Severity.BLOCKED],
        normal=[t for t in model.tiles if t.severity is Severity.NORMAL],
        immobility_s=result.immobility_s,
        wrist_rest_s=result.wrist_rest_s,
        silence_s=result.silence_s,
    )


def render_line(result: ValidationResult) -> str:
    """
    One-line text rendering (terminal / log). Consumes the **same** ScreenModel,
    so the display rule never forks into two copies that drift apart over time.
    """
    model = build_screen(result)
    parts = []
    for tile in model.tiles:
        if tile.value_text is None:
            parts.append(f"{tile.label} {tile.icon} {tile.message}")
        elif tile.message:
            parts.append(f"{tile.label} {tile.value_text}{tile.unit} {tile.icon} {tile.message}")
        else:
            parts.append(f"{tile.label} {tile.value_text}{tile.unit}")
    line = " | ".join(parts)
    for banner in model.banners:
        if banner.severity is Severity.ALERT:
            line += f"   {banner.icon} {banner.text}"
    return line

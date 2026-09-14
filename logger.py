"""
Audit trail — a record of every rejected reading and every alert, with time and reason.

This is not debug logging; it is a **medical audit log**: why the user did not
see a number at that moment, and on which threshold the decision was made.
Without it the system's behaviour cannot be reviewed after a session or
compared against reference data later.

Format: JSON Lines (one line = one event) — machine-readable without libraries,
and truncation-tolerant: a power cut loses only the last line, not the file.

**The raw rejected value is logged here** — it never reaches the screen, but a
medical audit needs to know *what* was rejected, not merely that it was.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, Optional, Set, TextIO

from interface import VitalSample
from validator import FIELD_SPECS, Status, ValidationResult, Validator

# Event types
EV_SESSION_START = "session_start"   # snapshot of the thresholds in force — every later decision is read against it
EV_REJECTED = "rejected"             # a reading discarded (INVALID) with its reason and raw value
EV_NO_CONTACT = "no_contact"         # wrist lifted — a sample-level event, not per field
EV_WARN = "warn"                     # a reading that was displayed but is outside the clinical range
EV_ALERT_RAISED = "alert_raised"
EV_ALERT_CLEARED = "alert_cleared"


class AuditLogger:
    """
    Called after every validate(). Decides nothing and changes nothing — it only records.

    Alerts are logged on **transition** (raised/cleared), not on every sample,
    or the log drowns in repetition with no information; the alert duration is
    computed and logged when it clears.
    """

    def __init__(
        self,
        path: Optional[str] = "audit_log.jsonl",
        stream: Optional[TextIO] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if stream is not None:
            self._stream = stream
            self._owns_stream = False
        else:
            # append: a previous session's log is never erased.
            self._stream = open(path, "a", encoding="utf-8")
            self._owns_stream = True
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._active_alerts: Set[str] = set()
        self._alert_started_t: Dict[str, float] = {}
        self.event_count = 0

    # ── Public interface ──
    def log_session_start(self, validator: Validator) -> None:
        """
        Snapshot of the thresholds in force.

        Mandatory: reading the log a month later without knowing which threshold
        rejected a reading = a log that cannot be audited.
        """
        self._write({
            "type": EV_SESSION_START,
            "t": 0.0,
            "thresholds": {
                "CONTACT_IR_THRESHOLD": validator.contact_ir_threshold,
                "IMMOBILITY_LIMIT_S": validator.immobility_limit_s,
                "MOVEMENT_THRESHOLD": validator.movement_threshold,
                "STUCK_REPEAT_LIMIT": validator.stuck_repeat_limit,
                "WRIST_REST_LIMIT_S": validator.wrist_rest_limit_s,
                "SILENCE_LIMIT_S": validator.silence_limit_s,
            },
            # The clinical bounds decide when the user is warned and when the
            # caregiver is notified, so a warning cannot be audited a month later
            # without the bound that triggered it.
            "field_limits": {
                name: {
                    "sanity_min": spec.sanity_min,
                    "sanity_max": _json_safe(spec.sanity_max),
                    "clinical_min": spec.clinical_min,
                    "clinical_max": spec.clinical_max,
                    "unit": spec.unit,
                }
                for name, spec in FIELD_SPECS.items()
            },
        })

    def log_result(self, sample: VitalSample, result: ValidationResult) -> int:
        """Log whatever in this sample deserves auditing; return the number of events written."""
        before = self.event_count

        if not result.contact:
            blanked = [n for n, fr in result.fields.items()
                       if fr.status is Status.NO_CONTACT]
            self._write({
                "type": EV_NO_CONTACT,
                "t": result.t,
                "ir_dc": _json_safe(result.ir_dc),
                "blanked_fields": blanked,
                # Sample flags distinguish a lifted wrist (normal, frequent) from
                # IR_IMPLAUSIBLE (a driver/bus fault needing maintenance).
                # Without them the two events look identical in the log.
                "flags": list(result.flags),
                "reason": "ir_dc outside the accepted contact range ⇒ no number displayed",
            })

        for name, fr in result.fields.items():
            if fr.status is Status.INVALID:
                self._write({
                    "type": EV_REJECTED,
                    "t": result.t,
                    "field": name,
                    "flags": list(fr.flags),
                    "raw": _json_safe(getattr(sample, name)),  # what was rejected, not just that it was
                    "contact": result.contact,
                })
            elif fr.status is Status.WARN:
                # Displayed to the user with emphasis — an abnormal value that was
                # shown deserves logging more, not less.
                self._write({
                    "type": EV_WARN,
                    "t": result.t,
                    "field": name,
                    "flags": list(fr.flags),
                    "value": _json_safe(fr.value),
                })

        self._log_alert_transitions(result)
        return self.event_count - before

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Internal ──
    def _log_alert_transitions(self, result: ValidationResult) -> None:
        current = set(result.alerts)
        for alert in sorted(current - self._active_alerts):
            self._alert_started_t[alert] = result.t
            self._write({
                "type": EV_ALERT_RAISED,
                "t": result.t,
                "alert": alert,
                "immobility_s": round(result.immobility_s, 1),
                "flags": list(result.flags),  # e.g. MOVEMENT_UNVERIFIED: raised without confirmed movement
            })
        for alert in sorted(self._active_alerts - current):
            started = self._alert_started_t.pop(alert, result.t)
            self._write({
                "type": EV_ALERT_CLEARED,
                "t": result.t,
                "alert": alert,
                "duration_s": round(result.t - started, 1),
            })
        self._active_alerts = current

    def _write(self, event: Dict[str, object]) -> None:
        event["session_id"] = self.session_id
        # Two clocks on purpose: t is the sensor clock (ties to the sample), ts is wall-clock (ties to reality).
        event["ts"] = datetime.now(timezone.utc).isoformat()
        # allow_nan=False prevents writing non-standard NaN that breaks any strict JSON reader.
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()  # medical log: never lose the latest events on a sudden cut
        self.event_count += 1


def _json_safe(v: object) -> object:
    """
    Convert a raw value into something JSON can hold without lying to the reader.

    NaN and inf are written as explicit text, not as null — the difference matters:
    null means "no value arrived"; "NaN" means "a corrupt value arrived". Two different reasons for rejection.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return repr(v)          # bool is not a numeric reading — logged as received for investigation
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return round(f, 4)
    return repr(v)              # an odd type from the driver: kept as text for investigation


EV_CORRUPT = "corrupt_line"   # a line that could not be parsed — counted, never swallowed


def summarize(path: str) -> Dict[str, int]:
    """
    Count events by type — for quick reports and for testing the log.

    **A corrupt line is skipped and counted; it does not bring down the whole file.**
    JSON Lines was chosen precisely because it tolerates truncation: a power cut
    or a concurrent write may corrupt one line, and rejecting the entire log for
    it means losing hundreds of valid events before it.

    Conversely, corruption is not hidden: it appears in the result under
    EV_CORRUPT, because a medical log that silently swallows a fault is worse
    than one that crashes — the former implies everything is fine.
    """
    counts: Dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                counts[EV_CORRUPT] = counts.get(EV_CORRUPT, 0) + 1
                continue
            key = str(ev.get("type", "?")) if isinstance(ev, dict) else "?"
            counts[key] = counts.get(key, 0) + 1
    return counts

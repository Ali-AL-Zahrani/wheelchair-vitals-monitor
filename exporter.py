"""
Measurement export — a data file for later analysis.

Fundamentally different from `logger.py`:
  - `audit_log.jsonl` is an **audit trail**: it records what was rejected and why, with the raw value.
  - `measurements.csv` is **measurement data**: one row per sample containing what passed validation.
The first answers "why did the user not see a number"; the second answers "what were the readings across the session".

The same safety rule applies here: **no number that failed validation is exported.**
A rejected field leaves its cell **empty**, never filled with a zero — a zero is a
measured value, an empty cell is the absence of one, and mixing them corrupts
any later analysis and implies a reading that never happened.

A companion `<name>.meta.json` is written with the thresholds in force:
**data without its thresholds cannot be interpreted** — a reading flagged WARN
without the bound that flagged it means nothing a month later.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TextIO

from validator import FIELD_SPECS, ValidationResult, Validator

# Unit suffix per field in the CSV header.
# Explicit rather than derived from `unit`: symbols like ° and % break some
# analysis tools when used in column names.
_UNIT_SUFFIX: Dict[str, str] = {
    "heart_rate": "bpm",
    "spo2": "pct",
    "skin_temp": "c",
    "movement": "idx",
}

_SEP = ";"   # separator inside the flags/alerts cell — the comma is reserved for CSV structure


def _column(name: str) -> str:
    return f"{name}_{_UNIT_SUFFIX.get(name, 'val')}"


def header_row() -> List[str]:
    cols = ["t_s", "contact"]
    for name in FIELD_SPECS:
        cols += [_column(name), f"{name}_status"]
    cols += ["immobility_s", "wrist_rest_s", "silence_s", "flags", "alerts"]
    return cols


class MeasurementExporter:
    """
    Called after every validate(), exactly like AuditLogger. Decides nothing, corrects nothing.

    The file is opened for writing, not appending: mixing two sessions with
    different thresholds in one file produces uninterpretable data. Each session
    gets its own file and its own threshold file.
    """

    def __init__(
        self,
        path: Optional[str] = "measurements.csv",
        stream: Optional[TextIO] = None,
        meta_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if stream is not None:
            self._stream = stream
            self._owns_stream = False
            self._meta_path = meta_path
        else:
            # newline="" is required by the csv module so line endings are not doubled on Windows.
            self._stream = open(path, "w", encoding="utf-8", newline="")
            self._owns_stream = True
            self._meta_path = meta_path or f"{path}.meta.json"

        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._writer = csv.writer(self._stream)
        self._writer.writerow(header_row())
        self.row_count = 0

    def write_meta(self, validator: Validator) -> Optional[Dict[str, Any]]:
        """
        Threshold snapshot accompanying the data. Called once at session start.

        Without this file the `*_status` column is meaningless: there is no way
        to know which bound classified a reading as WARN.
        """
        meta: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": {
                "empty_cell": "empty cell = no measurement (rejected or no contact), not zero",
                "rejected_values": "rejected values and their reasons are in audit_log.jsonl, not here",
            },
            "thresholds": {
                "CONTACT_IR_THRESHOLD": validator.contact_ir_threshold,
                "IMMOBILITY_LIMIT_S": validator.immobility_limit_s,
                "MOVEMENT_THRESHOLD": validator.movement_threshold,
                "STUCK_REPEAT_LIMIT": validator.stuck_repeat_limit,
                "WRIST_REST_LIMIT_S": validator.wrist_rest_limit_s,
                "SILENCE_LIMIT_S": validator.silence_limit_s,
            },
            "field_limits": {
                name: {
                    "unit": spec.unit,
                    "sanity_min": spec.sanity_min,
                    "sanity_max": (None if spec.sanity_max == float("inf")
                                   else spec.sanity_max),
                    "clinical_min": spec.clinical_min,
                    "clinical_max": spec.clinical_max,
                }
                for name, spec in FIELD_SPECS.items()
            },
        }
        if self._meta_path:
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        return meta

    def write(self, result: ValidationResult) -> None:
        """One row per sample — from validator output only."""
        row: List[Any] = [round(result.t, 3), 1 if result.contact else 0]
        for name in FIELD_SPECS:
            field = result.fields[name]
            # The empty cell is deliberate: absence of a measurement, not a measurement of zero.
            row.append("" if field.value is None else round(field.value, 3))
            row.append(field.status.value)
        row += [
            round(result.immobility_s, 1),
            round(result.wrist_rest_s, 1),
            round(result.silence_s, 1),
            _SEP.join(result.flags),
            _SEP.join(result.alerts),
        ]
        self._writer.writerow(row)
        self.row_count += 1

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> "MeasurementExporter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

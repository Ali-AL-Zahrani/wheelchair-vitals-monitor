"""
تصدير القياسات — ملف بيانات للتحليل اللاحق.

يختلف عن `logger.py` اختلافًا جوهريًا:
  - `audit_log.jsonl` **سجلّ تدقيق**: يوثّق ما رُفض ولماذا، بقيمته الخام.
  - `measurements.csv` **بيانات قياس**: صفّ لكل عيّنة بما اجتاز التدقيق.
الأول يجيب "لماذا لم يرَ المستخدم رقمًا"، والثاني يجيب "ما القراءات عبر الجلسة".

قاعدة السلامة نفسها سارية هنا: **لا يُصدَّر رقم لم يجتز المدقّق.** الحقل المرفوض
تُترك خانته **فارغة**، ولا تُملأ بصفر — الصفر قيمة قياس، والفراغ غياب قياس،
وخلطهما يفسد أي تحليل لاحق ويوهم بقراءة لم تحدث.

يُكتب معه ملف `<الاسم>.meta.json` يحمل العتبات السارية وحالة المعايرة:
**بيانات بلا عتباتها لا تُفسَّر** — قراءة مُعلَّمة WARN بلا معرفة الحدّ الذي علّمها
لا تعني شيئًا بعد شهر.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TextIO

from validator import FIELD_SPECS, ValidationResult, Validator

# لاحقة وحدة لكل حقل في ترويسة الـ CSV.
# صريحة لا مشتقّة من `unit`: رموز مثل ° و% تكسر بعض أدوات التحليل في أسماء الأعمدة.
_UNIT_SUFFIX: Dict[str, str] = {
    "heart_rate": "bpm",
    "spo2": "pct",
    "skin_temp": "c",
    "movement": "idx",
}

_SEP = ";"   # فاصل داخل خانة الأعلام/الإنذارات — الفاصلة محجوزة لبنية الـ CSV


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
    يُستدعى بعد كل validate()، مثل AuditLogger تمامًا. لا يقرّر ولا يصحّح.

    الملف يُفتح بالكتابة لا بالإلحاق: خلط جلستين بعتبتين مختلفتين في ملف واحد
    يُنتج بيانات لا تُفسَّر. كل جلسة ملفها وملف عتباتها.
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
            # newline="" شرط وحدة csv لئلا تتضاعف أسطر النهاية على ويندوز.
            self._stream = open(path, "w", encoding="utf-8", newline="")
            self._owns_stream = True
            self._meta_path = meta_path or f"{path}.meta.json"

        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._writer = csv.writer(self._stream)
        self._writer.writerow(header_row())
        self.row_count = 0

    def write_meta(self, validator: Validator) -> Optional[Dict[str, Any]]:
        """
        لقطة العتبات المصاحبة للبيانات. تُستدعى مرة عند بدء الجلسة.

        بلا هذا الملف، عمود `*_status` أرقام بلا معنى: لا يُعرف أي حدّ
        صنّف القراءة WARN، ولا هل كانت الحدود معتمدة طبيًا أصلًا.
        """
        meta: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "calibrated": False,
            "warning": (
                "بروتوتايب بحثي — العتبات غير معتمدة سريريًا ودقّة القراءات غير مُثبتة "
                "(لا بيانات مرجعية). لا يُتّخذ قرار طبي بناءً على هذه البيانات."
            ),
            "notes": {
                "empty_cell": "خانة فارغة = لا قياس (رُفض أو لا تلامس)، وليست صفرًا",
                "spo2": "reflectance من المعصم — مؤشّر اتجاه لا قيمة سريرية مطلقة",
                "skin_temp": "حرارة جلد المعصم، ليست حرارة الجسم المركزية",
                "movement": "مؤشّر حركة عام من IMU — لا يقيس تحويل الوزن",
                "rejected_values": "القيم المرفوضة وأسبابها في audit_log.jsonl لا هنا",
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
        """صفّ واحد لكل عيّنة — من مخرجات المدقّق وحدها."""
        row: List[Any] = [round(result.t, 3), 1 if result.contact else 0]
        for name in FIELD_SPECS:
            field = result.fields[name]
            # الفراغ مقصود: غياب قياس، لا قياس بقيمة صفر.
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

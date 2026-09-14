"""
طبقة التدقيق (audit trail) — سجلّ لكل قراءة مرفوضة وكل إنذار، بالوقت والسبب.

ليست logging للتشخيص البرمجي، بل **سجلّ طبي**: لماذا لم يرَ المستخدم رقمًا في تلك اللحظة،
وعلى أي عتبة اتُّخذ القرار. بدون هذا السجلّ لا يمكن مراجعة سلوك النظام بعد الجلسة
ولا مقارنته ببيانات مرجعية لاحقًا.

الصيغة JSON Lines (سطر = حدث) — قابلة للقراءة الآلية بلا مكتبات، وتحتمل الاقتطاع:
انقطاع الكهرباء يفقد السطر الأخير فقط، لا الملف كله.

**القيمة الخام المرفوضة تُسجَّل هنا** — لا تصل الشاشة أبدًا، لكن التدقيق الطبي يحتاج
معرفة *ما الذي رُفض*، لا مجرّد أنه رُفض.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, Optional, Set, TextIO

from interface import VitalSample
from validator import FIELD_SPECS, Status, ValidationResult, Validator

# أنواع الأحداث
EV_SESSION_START = "session_start"   # لقطة العتبات السارية — أي قرار لاحق يُفسَّر بها
EV_REJECTED = "rejected"             # قراءة رُميت (INVALID) مع سببها وقيمتها الخام
EV_NO_CONTACT = "no_contact"         # المعصم مرفوع — حدث على مستوى العيّنة لا الحقل
EV_WARN = "warn"                     # قراءة عُرضت لكنها خارج النطاق السريري
EV_ALERT_RAISED = "alert_raised"
EV_ALERT_CLEARED = "alert_cleared"


class AuditLogger:
    """
    يُستدعى بعد كل validate(). لا يقرّر شيئًا ولا يعدّل شيئًا — يسجّل فقط.

    الإنذار يُسجَّل عند **التحوّل** (بدأ/انتهى) لا في كل عيّنة، وإلا غرق السجلّ
    في تكرار بلا معلومة؛ ومدّة الإنذار تُحسب وتُسجَّل عند انتهائه.
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
            # append: لا نمحو سجلّ جلسة سابقة أبدًا.
            self._stream = open(path, "a", encoding="utf-8")
            self._owns_stream = True
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._active_alerts: Set[str] = set()
        self._alert_started_t: Dict[str, float] = {}
        self.event_count = 0

    # ── واجهة الاستعمال ──
    def log_session_start(self, validator: Validator) -> None:
        """
        لقطة العتبات السارية.

        إلزامي: العتبات كلها ما زالت غير معايرة، فقراءة السجلّ بعد شهر بلا معرفة
        العتبة التي رفضت القراءة = سجلّ لا يصلح للتدقيق.
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
            # الحدود السريرية تقرّر متى يُحذَّر المستخدم ومتى يُبلَّغ المرافق،
            # فلا يُدقَّق تحذير بعد شهر دون معرفة الحدّ الذي أطلقه.
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
            # صريح: الأرقام مرجعية عامة للبالغين، لا معايرة لهذه الفئة ولا لهذا العتاد.
            "calibrated": False,
        })

    def log_result(self, sample: VitalSample, result: ValidationResult) -> int:
        """يسجّل ما يستحق التدقيق في هذه العيّنة، ويُعيد عدد الأحداث المكتوبة."""
        before = self.event_count

        if not result.contact:
            blanked = [n for n, fr in result.fields.items()
                       if fr.status is Status.NO_CONTACT]
            self._write({
                "type": EV_NO_CONTACT,
                "t": result.t,
                "ir_dc": _json_safe(result.ir_dc),
                "blanked_fields": blanked,
                # أعلام العيّنة تميّز رفع المعصم (طبيعي متكرر) عن IR_IMPLAUSIBLE
                # (عطب درايفر/ناقل يستدعي صيانة). بدونها يتساوى الحدثان في السجلّ.
                "flags": list(result.flags),
                "reason": "ir_dc خارج نطاق التلامس المقبول ⇒ لا رقم يُعرض",
            })

        for name, fr in result.fields.items():
            if fr.status is Status.INVALID:
                self._write({
                    "type": EV_REJECTED,
                    "t": result.t,
                    "field": name,
                    "flags": list(fr.flags),
                    "raw": _json_safe(getattr(sample, name)),  # ما رُفض، لا "أنه رُفض" فقط
                    "contact": result.contact,
                })
            elif fr.status is Status.WARN:
                # عُرضت للمستخدم مع تمييز — وقيمة شاذّة معروضة تستحق التسجيل أكثر لا أقل.
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

    # ── داخلي ──
    def _log_alert_transitions(self, result: ValidationResult) -> None:
        current = set(result.alerts)
        for alert in sorted(current - self._active_alerts):
            self._alert_started_t[alert] = result.t
            self._write({
                "type": EV_ALERT_RAISED,
                "t": result.t,
                "alert": alert,
                "immobility_s": round(result.immobility_s, 1),
                "flags": list(result.flags),  # مثلًا MOVEMENT_UNVERIFIED: أُطلق بلا تأكيد حركة
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
        # ساعتان مقصودتان: t ساعة الحسّاس (تربط بالعيّنة)، ts ساعة الجدار (تربط بالواقع).
        event["ts"] = datetime.now(timezone.utc).isoformat()
        # allow_nan=False يمنع كتابة NaN غير القياسي الذي يكسر أي قارئ JSON صارم.
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()  # سجلّ طبي: لا نفقد آخر الأحداث عند انقطاع مفاجئ
        self.event_count += 1


def _json_safe(v: object) -> object:
    """
    تحويل القيمة الخام إلى شيء يُكتب في JSON دون كذب على القارئ.

    NaN و inf تُكتب كنصّ صريح لا كـ null — الفرق مهم:
    null تعني "لم تصل قيمة"، و"NaN" تعني "وصلت قيمة تالفة". سببان مختلفان للرفض.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return repr(v)          # bool ليس قراءة رقمية — يُسجَّل كما وصل ليُحقَّق فيه
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return round(f, 4)
    return repr(v)              # نوع غريب من الدرايفر: يُحفظ نصًّا للتحقيق


EV_CORRUPT = "corrupt_line"   # سطر تعذّرت قراءته — يُعدّ ولا يُبتلع


def summarize(path: str) -> Dict[str, int]:
    """
    عدّ الأحداث حسب النوع — للتقارير السريعة ولاختبار السجلّ.

    **السطر التالف يُتخطّى ويُعدّ، ولا يُسقط الملف كله.** صيغة JSON Lines اختيرت
    أصلًا لأنها تحتمل الاقتطاع: انقطاع كهرباء أو كتابة متزامنة قد تفسد سطرًا واحدًا،
    ورفض السجلّ كاملًا بسببه يعني فقدان مئات الأحداث السليمة قبله.

    وفي المقابل لا يُخفى الفساد: يظهر في النتيجة تحت EV_CORRUPT، لأن سجلًّا طبيًا
    يبتلع خللًا بصمت أسوأ من سجلّ ينهار — الأول يوهم بأن كل شيء سليم.
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

"""
User screen — the renderer. Displays **the display.py description literally** and decides nothing itself.

Why a local web page instead of a desktop window:
  - Large type, high contrast and RTL/LTR support are ready-made and reliable on the web.
  - It is the closest form to the real product: a tablet mounted on the chair.
  - stdlib only: no external library is added to the deliverable.

Architecture here:
    MockSensor ⟵ Validator ⟵ AuditLogger ⟵ build_screen()  (background thread)
                                    ↓ JSON
                            the page polls /state and renders

The page never sees a raw VitalSample; only what passed the validator reaches it.

Run:  python screen.py
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from display import build_carer_screen, build_screen
from exporter import MeasurementExporter
from logger import AuditLogger
from mock_sensor import MockSensor, default_scenario
from validator import Validator

# ═══════════════════════════════════════════════════════════════════════════
#  State shared between the sensor thread and the server threads
# ═══════════════════════════════════════════════════════════════════════════

_state_lock = threading.Lock()
_state: Dict[str, Any] = {"model": None, "carer": None}
_stop = threading.Event()


def _publish(model_dict: Optional[Dict[str, Any]],
             carer_dict: Optional[Dict[str, Any]]) -> None:
    with _state_lock:
        _state["model"] = model_dict
        _state["carer"] = carer_dict


def _snapshot(key: str) -> Optional[Dict[str, Any]]:
    with _state_lock:
        return _state[key]


# ═══════════════════════════════════════════════════════════════════════════
#  Sensor thread: exactly the same path as demo.py, at a configurable real-time rate
# ═══════════════════════════════════════════════════════════════════════════

def _sensor_loop(period_s: float, sample_period_s: float, seed: int,
                 log_path: str, cycle_samples: int, csv_path: Optional[str],
                 clean: bool = False) -> None:
    """
    The screen is a continuous display, but the scenario is only 120 samples.
    Once exhausted every reading stays valid forever ⟵ whoever opens the screen
    late never sees a single display state. So the scenario restarts with a new
    seed on a **continuous clock**, without resetting the validator: one session
    with unbroken memory; only the injected faults vary.
    """
    validator = Validator()
    audit = AuditLogger(path=log_path)
    audit.log_session_start(validator)      # threshold snapshot: a decision without its threshold cannot be audited
    export = MeasurementExporter(path=csv_path) if csv_path else None
    if export:
        export.write_meta(validator)

    cycle = 0
    sensor = MockSensor(seed=seed, sample_period_s=sample_period_s,
                        faults={} if clean else default_scenario(), start_t=0.0)
    sensor.start()
    try:
        while not _stop.is_set():
            sample = sensor.read()
            result = validator.validate(sample)
            audit.log_result(sample, result)
            if export:
                export.write(result)
            # Both screens are built from the **same** ValidationResult at the same
            # instant, so they can never contradict each other or lag behind one another.
            _publish(build_screen(result).to_dict(),
                     build_carer_screen(result).to_dict())

            if sensor.index >= cycle_samples - 1:
                # The clock resumes from the next sample, not the same instant,
                # otherwise one timestamp repeats and the immobility maths is confused.
                sensor.stop()
                cycle += 1
                sensor = MockSensor(seed=seed + cycle, sample_period_s=sample_period_s,
                                    faults={} if clean else default_scenario(),
                                    start_t=sample.t + sample_period_s)
                sensor.start()

            _stop.wait(period_s)
    finally:
        sensor.stop()
        audit.close()
        if export:
            export.close()


# ═══════════════════════════════════════════════════════════════════════════
#  The page
# ═══════════════════════════════════════════════════════════════════════════

# STALE_MS: if the state source stops, the numbers are wiped immediately and the
# screen announces the disconnection. A screen frozen on the last number is more
# dangerous than an empty one — the user reads it as a live measurement.
PAGE = """<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>User Screen</title>
<style>
  :root {
    --bg:#0A0E14; --tile:#151B24; --edge:#232C39;
    --ink:#EAF0F7; --muted:#93A1B2;
    --warn:#F0A93B; --warn-ink:#FFC96B;
    --alert:#D32F2F; --blocked:#6E7A8A;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    background:var(--bg); color:var(--ink); min-height:100vh; padding:2vh 2vw;
    font-family:"Segoe UI","Tahoma","Arial",sans-serif;
    display:flex; flex-direction:column; gap:2vh;
  }
  header { display:flex; justify-content:space-between; align-items:baseline;
           border-bottom:2px solid var(--edge); padding-bottom:1.2vh; }
  h1 { font-size:clamp(18px,2.2vw,28px); font-weight:600; letter-spacing:.5px; }
  #clock { font-size:clamp(18px,2.2vw,28px); color:var(--muted);
           font-variant-numeric:tabular-nums; }

  #banners { display:flex; flex-direction:column; gap:1.2vh; }
  .banner { display:flex; align-items:center; gap:16px; border-radius:14px;
            padding:2.2vh 2vw; font-size:clamp(24px,3.6vw,52px); font-weight:700; }
  .banner .icon { font-size:1.25em; line-height:1; }
  .banner.ALERT   { background:var(--alert); color:#fff; animation:pulse 1.1s infinite; }
  .banner.BLOCKED { background:var(--tile); color:var(--ink); border:3px solid var(--edge); }
  @keyframes pulse { 50% { opacity:.55; } }
  @media (prefers-reduced-motion:reduce) { .banner.ALERT { animation:none; } }

  #tiles { flex:1; display:grid; gap:2vh;
           grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
  .tile { background:var(--tile); border:3px solid var(--edge); border-radius:18px;
          padding:2.5vh 2vw; display:flex; flex-direction:column;
          justify-content:center; gap:1.2vh; }
  .tile.WARN { border-color:var(--warn); }
  .label { font-size:clamp(18px,2.2vw,30px); color:var(--muted); font-weight:600; }
  .value { font-size:clamp(58px,11vw,150px); font-weight:700; line-height:1;
           font-variant-numeric:tabular-nums; }
  .unit  { font-size:.32em; color:var(--muted); margin-inline-start:.15em; font-weight:600; }
  .tile.WARN    .value { color:var(--warn-ink); }
  .tile.BLOCKED .value { color:var(--blocked); }
  .msg { display:flex; align-items:center; gap:10px;
         font-size:clamp(17px,2vw,28px); font-weight:600; min-height:1.4em; }
  .tile.WARN    .msg { color:var(--warn-ink); }
  .tile.BLOCKED .msg { color:var(--muted); }

  footer { display:flex; justify-content:flex-end; align-items:center;
           gap:16px; color:var(--muted); font-size:clamp(13px,1.4vw,18px);
           border-top:2px solid var(--edge); padding-top:1.2vh; }
  #sound { background:var(--tile); color:var(--ink); border:2px solid var(--edge);
           border-radius:10px; padding:10px 18px; font-size:inherit;
           font-family:inherit; cursor:pointer; }
  #sound[data-on="1"] { border-color:var(--warn); color:var(--warn-ink); }
</style>
</head>
<body>
  <header>
    <h1>Vital Signs</h1>
    <div id="clock">--:--</div>
  </header>

  <div id="banners" role="status" aria-live="assertive"></div>
  <main id="tiles"></main>

  <footer>
    <button id="sound" data-on="0" type="button">🔇 Enable sound</button>
  </footer>

<script>
const STALE_MS = 3000;      // after this the state is considered lost and the numbers are wiped
const POLL_MS  = 250;
let lastOk = 0, alerting = false, audio = null, lastBeep = 0;

document.getElementById("sound").addEventListener("click", (e) => {
  // Browsers block audio before a user gesture — hence an explicit button.
  audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  audio.resume();
  const on = e.currentTarget.dataset.on === "1" ? "0" : "1";
  e.currentTarget.dataset.on = on;
  e.currentTarget.textContent = on === "1" ? "🔔 Sound on" : "🔇 Enable sound";
  if (on === "1") beep();
});

function beep() {
  const btn = document.getElementById("sound");
  if (!audio || btn.dataset.on !== "1") return;
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.frequency.value = 880; gain.gain.value = 0.12;
  osc.connect(gain).connect(audio.destination);
  osc.start(); osc.stop(audio.currentTime + 0.28);
}

function tile(t) {
  const el = document.createElement("section");
  el.className = "tile " + t.severity;
  const value = t.value_text === null ? "—" : t.value_text;
  const unit  = t.value_text === null || t.unit === "—"
              ? "" : `<span class="unit">${t.unit}</span>`;
  el.innerHTML =
    `<div class="label">${t.label}</div>` +
    `<div class="value">${value}${unit}</div>` +
    `<div class="msg">${t.message ? `<span>${t.icon}</span><span>${t.message}</span>` : ""}</div>`;
  return el;
}

function banner(b) {
  const el = document.createElement("div");
  el.className = "banner " + b.severity;
  el.innerHTML = `<span class="icon">${b.icon}</span><span>${b.text}</span>`;
  return el;
}

// The clock shows the device's real time, not the virtual session time.
// Deliberately independent of data arrival: the wall clock stays correct even if the sensor drops out.
function tickClock() {
  const d = new Date();
  document.getElementById("clock").textContent =
    String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}
tickClock(); setInterval(tickClock, 1000);

function draw(model) {
  const tilesEl = document.getElementById("tiles");
  tilesEl.replaceChildren(...model.tiles.map(tile));
  const bannersEl = document.getElementById("banners");
  bannersEl.replaceChildren(...model.banners.map(banner));

  // Repeat the beep every 8 s while the alert is active — a single call can be missed.
  const now = Date.now();
  if (model.needs_sound) {
    if (!alerting || now - lastBeep > 8000) { beep(); lastBeep = now; }
    alerting = true;
  } else { alerting = false; }
}

function drawDisconnected() {
  // Source lost: no number stays on screen. A frozen screen reads as a live measurement.
  document.getElementById("banners").replaceChildren(banner(
    { severity:"BLOCKED", icon:"⛔", text:"Connection lost — no readings" }));
  document.querySelectorAll(".tile").forEach(el => {
    el.className = "tile BLOCKED";
    el.querySelector(".value").textContent = "—";
    el.querySelector(".msg").innerHTML = "<span>⛔</span><span>Reading unavailable</span>";
  });
}

async function poll() {
  try {
    const res = await fetch("/state", { cache:"no-store" });
    const data = await res.json();
    if (data.model) { draw(data.model); lastOk = Date.now(); }
  } catch (e) { /* handled below by the staleness logic */ }
  if (Date.now() - lastOk > STALE_MS) drawDisconnected();
}
lastOk = Date.now();
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


# Caregiver screen: a summary of "what needs intervention", not a second copy of
# the user screen. The caregiver may be in another room, so what is needed is one
# state readable from a distance, plus how long it has been going on.
CARER_PAGE = """<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caregiver Screen</title>
<style>
  :root { --bg:#0A0E14; --card:#151B24; --edge:#232C39; --ink:#EAF0F7;
          --muted:#93A1B2; --ok:#3FA96A; --warn:#F0A93B; --alert:#D32F2F; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--ink); min-height:100vh; padding:2vh 3vw;
         font-family:"Segoe UI","Tahoma","Arial",sans-serif;
         display:flex; flex-direction:column; gap:2vh; }
  header { display:flex; justify-content:space-between; align-items:baseline;
           border-bottom:2px solid var(--edge); padding-bottom:1vh; }
  h1 { font-size:clamp(17px,2vw,26px); font-weight:600; }
  #clock { color:var(--muted); font-variant-numeric:tabular-nums;
           font-size:clamp(16px,1.8vw,22px); }
  #status { border-radius:16px; padding:3vh 2vw; text-align:center;
            font-size:clamp(26px,4vw,54px); font-weight:700; }
  #status.ok    { background:var(--card); border:3px solid var(--ok); color:var(--ok); }
  #status.busy  { background:var(--alert); color:#fff; }
  #status.dark  { background:var(--card); border:3px solid var(--warn); color:var(--warn); }
  .group { display:flex; flex-direction:column; gap:1vh; }
  .group h2 { font-size:clamp(15px,1.6vw,20px); color:var(--muted); font-weight:600; }
  /* The thick edge uses a logical property, not left/right — correct in both LTR and RTL. */
  .row { background:var(--card); border:2px solid var(--edge); border-inline-start-width:8px;
         border-radius:12px; padding:1.6vh 1.4vw; display:flex; justify-content:space-between;
         align-items:center; gap:14px; font-size:clamp(16px,1.9vw,26px); }
  .row.alert { border-inline-start-color:var(--alert); }
  .row.warn  { border-inline-start-color:var(--warn); }
  .row.blocked { border-inline-start-color:var(--muted); color:var(--muted); }
  .row .since { color:var(--muted); font-size:.8em; font-variant-numeric:tabular-nums; }
  .val { font-weight:700; font-variant-numeric:tabular-nums; }
</style>
</head>
<body>
  <header><h1>Caregiver Monitor</h1><div id="clock">--:--</div></header>
  <div id="status" class="ok" role="status" aria-live="assertive">—</div>
  <div id="groups" class="group"></div>
<script>
const STALE_MS = 4000; let lastOk = Date.now();

function row(cls, label, value, since) {
  const el = document.createElement("div");
  el.className = "row " + cls;
  el.innerHTML = `<span>${label}</span><span class="val">${value}</span>` +
                 (since ? `<span class="since">${since}</span>` : "");
  return el;
}
function mmss(s) {
  s = Math.max(0, Math.round(s));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" +
         String(s % 60).padStart(2, "0");
}
// Real device time — independent of data arrival (see the user screen).
function tickClock() {
  const d = new Date();
  document.getElementById("clock").textContent =
    String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}
tickClock(); setInterval(tickClock, 1000);

function draw(m) {
  const status = document.getElementById("status");
  // Three states, not two: "nothing needs attention" while no number arrives = false reassurance.
  if (m.attention)        { status.className = "busy"; status.textContent = "Needs your attention"; }
  else if (!m.monitoring) { status.className = "dark"; status.textContent = "No readings right now"; }
  else                    { status.className = "ok";   status.textContent = "Nothing needs attention"; }

  const g = document.getElementById("groups");
  g.replaceChildren();
  m.alerts.forEach(a => g.appendChild(row("alert", a.icon + " " + a.text, "",
    a.kind === "movement" ? "for " + mmss(m.immobility_s) :
    a.kind === "lift_wrist" ? "for " + mmss(m.wrist_rest_s) :
    a.kind === "silent" ? "for " + mmss(m.silence_s) : "")));
  m.abnormal.forEach(t => g.appendChild(row("warn",
    t.icon + " " + t.label, t.value_text + " " + t.unit, t.message)));
  m.blocked.forEach(t => g.appendChild(row("blocked", t.label, "—", t.message)));
  m.normal.forEach(t => g.appendChild(row("", t.label, t.value_text + " " + t.unit, "")));
}
function disconnected() {
  const status = document.getElementById("status");
  status.className = "busy";
  status.textContent = "⛔ Connection lost — monitoring stopped";
  document.getElementById("groups").replaceChildren();
}
async function poll() {
  try {
    const d = await (await fetch("/state/carer", { cache:"no-store" })).json();
    if (d.model) { draw(d.model); lastOk = Date.now(); }
  } catch (e) {}
  if (Date.now() - lastOk > STALE_MS) disconnected();
}
poll(); setInterval(poll, 400);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802  (signature imposed by the library)
        if self.path.startswith("/state/carer"):
            body = json.dumps({"model": _snapshot("carer")},
                              ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        elif self.path.startswith("/state"):
            body = json.dumps({"model": _snapshot("model")},
                              ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.rstrip("/") == "/family":
            self._send(CARER_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silence: the HTTP log is noise that buries the diagnostic output."""


def main() -> None:
    parser = argparse.ArgumentParser(description="User screen (prototype)")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--period", type=float, default=0.5,
                        help="real seconds between two samples on screen")
    parser.add_argument("--sample-period", type=float, default=30.0,
                        help="virtual seconds between two samples (sensor clock)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default="audit_log.jsonl")
    parser.add_argument("--csv", default=None,
                        help="measurement export path (without it no data file is written)")
    parser.add_argument("--cycle-samples", type=int, default=120,
                        help="scenario cycle length before faults are re-injected")
    parser.add_argument("--clean", action="store_true",
                        help="run without fault injection — continuous clean readings (for demos)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    worker = threading.Thread(
        target=_sensor_loop,
        args=(args.period, args.sample_period, args.seed, args.log,
              args.cycle_samples, args.csv, args.clean),
        daemon=True,
    )
    worker.start()

    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"User screen      ⟵ {url}")
    print(f"Caregiver screen ⟵ {url}family    (Ctrl+C to stop)")
    # At --period 0.5 and 30 s/sample the 15-minute immobility limit is reached in ~15 real seconds.
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        server.server_close()
        worker.join(timeout=2.0)
        print("\nScreen stopped.")


if __name__ == "__main__":
    main()

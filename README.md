# Wheelchair Armrest Vital-Signs Monitor — Software Prototype

Vital-signs monitoring from the **armrest of a wheelchair**: the user rests their wrist on the armrest and readings are taken continuously, with prompts to move (circulation and pressure-injury prevention).

---

## Where this fits in the full system

The complete project has three cooperating components. **This repository is the vital-signs monitoring subsystem** — the layer that turns raw sensor signals into validated readings and alerts.

```
Armrest sensors ──▶ Validation engine ──▶ Screen (live readings + alerts)
                           │
                           └──▶ Validated data ──▶ AI rehabilitation model
Chair camera ─────▶ Gaze direction mapping ──▶ Screen (gaze-driven interaction)
```

| Component | What it does | Where |
|---|---|---|
| **Vital-signs monitoring** | Reads heart rate, blood oxygen, wrist skin temperature and movement; validates every reading; raises the four safety alerts; logs and exports the data | **this repository** |
| **Gaze direction mapping** | A camera on the chair tracks where the user is looking and maps it to screen coordinates, so the screen can be used without hands | partner module |
| **AI rehabilitation** | Learns from the validated vital signs, movement patterns and response to alerts, and predicts a rehabilitation programme personalised to each user rather than a generic plan | this repository |

### The AI layer and why validation comes first

The rehabilitation model is only as good as the data it learns from. A sensor with no wrist on it still emits plausible-looking numbers; a frozen sensor repeats the same value; a hand tremor produces sudden jumps. If those reached the model, it would learn from noise and recommend with false confidence.

That is why the AI layer is designed never to read raw sensor output. It receives **only what passed the validation engine**, through two files this repository already produces:

- `measurements.csv` — one row per sample containing validated readings only, with an empty cell (never a zero) wherever a reading was withheld, so no artificial value can enter a training set.
- `audit_log.jsonl` — every rejected reading with its raw value and reason, every alert raised and cleared with its duration, and the thresholds in force for the session.

From these the model has what it needs to personalise: the user's own resting baselines, how long they stay immobile and how quickly they respond to a movement prompt, how often the wrist-pressure reminder fires, and how the readings trend across sessions. Because every row carries the threshold snapshot that produced it, the model can also be retrained correctly when thresholds are tuned on real hardware.

The three components share the armrest screen: validated readings and alerts are displayed there, and gaze mapping lets the user interact with it.

---

## Running

Requirements: **Python 3.8+**. No external libraries (stdlib only), except `pytest` for the tests.

```bash
python demo.py
```
Full run in the terminal: 120 samples (one virtual hour) passing through every fault state, then a statistical summary and an audit log.

```bash
python screen.py
```
Both screens on a local server:
- **User screen** ⟵ `http://127.0.0.1:8770/`
- **Caregiver screen** ⟵ `http://127.0.0.1:8770/family`

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8770 | server port |
| `--period` | 0.5 | **real** seconds between two samples on screen |
| `--sample-period` | 30 | **virtual** seconds between two samples (sensor clock) |
| `--cycle-samples` | 120 | scenario length before faults are re-injected |
| `--seed` | 42 | random seed |
| `--log` | `audit_log.jsonl` | audit log path |
| `--csv` | (off) | measurement export path |
| `--clean` | — | run without fault injection (continuous clean readings) |
| `--no-browser` | — | do not open the browser automatically |

At `--period 0.5` the virtual hour completes in one real minute, so every display state and alert appears in sequence, then the cycle repeats.

```bash
python -m pytest -q
```
**141 tests.** Each one guards a medical decision, not a programming detail.
(`pip install -r requirements.txt` — running the system itself needs no external libraries.)

---

## Architecture

```
Sensor ⟶ Validator ⟶ Display ⟶ Screen
              ↓
           Logger
```

**Fixed safety rule:** the screen reads from **validator output only**. No raw, unverified number reaches the user's eyes — no exceptions.

| File | Role |
|---|---|
| `interface.py` | `VitalSample` + `SensorInterface` — **the swap point between simulation and hardware** |
| `mock_sensor.py` | Mock sensor with realistic readings and deliberate fault injection |
| `validator.py` | **Validation engine — the core of the system** |
| `display.py` | Pure display logic (no rendering): what is shown and when |
| `screen.py` | Renderer: user screen and caregiver screen (local web page) |
| `logger.py` | Medical audit log (JSON Lines) |
| `exporter.py` | Measurement export for analysis (CSV + threshold snapshot) |
| `i2c_sensor.py` | Hardware layer: MAX30102 + MAX30205 + MPU-6050 over I²C (integration in progress) |
| `demo.py` | End-to-end terminal run + statistics |
| `test_*.py` | Automated proof |

---

## Validator logic

### Strict separation of two kinds of error
| Kind | Meaning | Result |
|---|---|---|
| **SANITY** | physically impossible / `None` / `NaN` | `INVALID` — **value discarded** |
| **CLINICAL** | possible but abnormal | `WARN` — **value kept and shown with emphasis** |

Mixing the two is a medical error: a heart rate of 45 is a real reading that deserves display and a warning; 320 is a sensor fault to be discarded.

Plus two checks with memory: `STUCK` (frozen sensor) and `ARTIFACT` (sudden jump = tremor or movement).

### Contact gate
`ir_dc` below the threshold ⟵ `NO_CONTACT` and **every** wrist value is blanked however plausible it looks — a sensor with no wrist on it outputs noise that resembles a real reading. The gate closes on doubt, and it distinguishes **a lifted wrist** (normal behaviour) from **a value outside the hardware range** (a fault needing maintenance).

### Alerts — four independent timers
| Alert | Fires on | Reset by |
|---|---|---|
| `NEEDS_MOVEMENT` | 15 min immobility | confirmed movement above the threshold |
| `LIFT_WRIST` | 15 min continuous wrist rest | **lifting the wrist only** |
| `SENSOR_FAULT` | `ir_dc` outside the hardware range | (immediate) |
| `MEASUREMENT_SILENT` | 15 min without a valid number | any valid reading |

All timers are **accumulators**, not derived from stored timestamps, so a sensor clock jumping backwards (an MCU reboot) neither loses accumulated time nor silences an active alert.

The output separates `flags` (data quality) from `alerts` (action required): heart rate can be `VALID` while an alert is active at the same instant.

---

## Display layer

| State | User screen |
|---|---|
| `VALID` | the number, normally |
| `WARN` | the number + **colour, icon and text** (never colour alone) |
| `NO_CONTACT` | **no number** ⟵ "Rest your wrist on the armrest" |
| `INVALID` | **no number** ⟵ "Reading unavailable" |
| alert | a clear banner + sound |

**"No reading" is more honest than a wrong reading.** A blocked card never carries a number: no stale value, no zero. Loss of the state source wipes every number immediately — a screen frozen on the last number reads as a live measurement, which is more dangerous than an empty one.

**The caregiver screen** is a summary of "what needs intervention", not a second copy: alerts and their duration ⟵ abnormal ⟵ unavailable ⟵ valid. It is derived from the same model, so it never reveals a number the user screen withheld. It has three states: **Needs your attention** / **No readings right now** / **Nothing needs attention** — announcing reassurance while no number arrives is false reassurance.

---

## Audit log

`audit_log.jsonl` — one line per event (JSON Lines), **appended, never overwritten**:

| Event | Content |
|---|---|
| `session_start` | **a snapshot of every threshold in force** — a decision without its threshold cannot be audited |
| `rejected` | the rejected reading **with its raw value** and the reason |
| `no_contact` | contact loss + flags distinguishing a lifted wrist from a hardware fault |
| `warn` | the abnormal reading that was displayed |
| `alert_raised` / `alert_cleared` | alert transitions and their duration |

`NaN` and `inf` are written as explicit text, not `null` — the difference is deliberate: `null` means "no value arrived", `"NaN"` means "a corrupt value arrived". Two different reasons for rejection.

---

## Measurement export

`measurements.csv` — one row per sample, written automatically by `demo.py` (and by `screen.py` with `--csv`).

```
t_s,contact,heart_rate_bpm,heart_rate_status,spo2_pct,spo2_status,...,flags,alerts
0.0,1,73.481,VALID,96.911,VALID,...,,
330.0,0,,NO_CONTACT,,NO_CONTACT,...,NO_CONTACT,
```

⚠️ **An empty cell means "no measurement", not "a measurement of zero".** A zero is a measured value and an empty cell is its absence; mixing them puts heart rate = 0 into an analysis mean and implies a reading that never happened. The `*_status` column gives the reason (`INVALID` / `NO_CONTACT`); the detail and raw value are in `audit_log.jsonl`.

A companion **`measurements.csv.meta.json`** is written with the thresholds in force. **Data without its thresholds cannot be interpreted** — a `status` column without the bound that produced it is meaningless a month later.

| File | Answers |
|---|---|
| `measurements.csv` | what were the readings across the session? |
| `audit_log.jsonl` | why did the user not see a number at that moment? |

> The final output format is not yet fixed; CSV is an initial choice that converts easily.

## Changing thresholds

Every threshold lives in `validator.py` **and is duplicated nowhere else**:

- **Timers and the contact gate:** constants at the top of the file (`IMMOBILITY_LIMIT_S`, `WRIST_REST_LIMIT_S`, `SILENCE_LIMIT_S`, `CONTACT_IR_THRESHOLD`, `MOVEMENT_THRESHOLD`).
- **Clinical and physical bounds:** `FIELD_SPECS`.

After changing them: `python -m pytest -q`, then `python demo.py` to review the effect on the acceptance rate. New thresholds are recorded automatically in the session-start snapshot.

---

## Swapping simulation for hardware

The system is **simulation-first**: all logic was verified before the hardware. The transition touches one file:

```python
from interface import SensorInterface, VitalSample

class I2CSensor(SensorInterface):
    def start(self) -> None:
        ...  # I2C init, LED power-on

    def read(self) -> VitalSample:
        # Return the sample **as-is** — no filtering, no correction, no hiding.
        # Validation is the Validator's responsibility alone.
        return VitalSample(t=..., ir_dc=..., heart_rate=...,
                           spo2=..., skin_temp=..., movement=...)

    def stop(self) -> None:
        ...  # clean shutdown
```

Then replace `MockSensor(...)` with `I2CSensor(...)` in `demo.py` and `screen.py`. **The validator and the screen are untouched.**

A ready implementation for MAX30102 + MAX30205 + MPU-6050 is in `i2c_sensor.py` (requires `pip install smbus2` on the Raspberry Pi). Afterwards `CONTACT_IR_THRESHOLD` and `MOVEMENT_THRESHOLD` need calibrating on the real hardware.

---

## Deliberate design decisions

Decisions taken on purpose that may look counter-intuitive:

- **`skin_temp`, not `body_temp`** — the name describes what is actually measured. A misleading name is a medical error.
- **"User", not "patient"** — the device is for people with disabilities, and a wheelchair user is not necessarily ill.
- **A web screen, not a desktop window** — reliable large type, high contrast and bidirectional text; closest to the real product (a tablet on the chair).
- **Unverified movement keeps the timer running** (fail-loud) — an extra movement prompt is harmless; a silenced alert is a pressure-injury risk.
- **Movement is never shown to the user** — an internal indicator that drives the alert, not a number that means anything to them.

---

## Status and what remains

**Complete:** the core, the audit log, measurement export, the user screen, the caregiver screen, the four alerts, 141 tests.

**Next phase:** hardware integration of `I2CSensor` on the chair, signal filtering, threshold tuning on real hardware, the escalation policy, and the AI rehabilitation model.

<div align="center">

# PCB Defect Inspector

**Automated PCB surface-defect detection web application**

`FastAPI` · `YOLO26` · `SAHI` · `OpenCV` · `SQLite` · `ReportLab`

Detect 6 classes of PCB manufacturing defects from a single image, with
automated image-quality gates, per-user history and downloadable PDF reports.

</div>

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Tech Stack](#2-tech-stack)
3. [Features](#3-features)
4. [Setup — Step by Step](#4-setup--step-by-step)
5. [Step-by-Step Commands (Copy/Paste)](#5-step-by-step-commands-copypaste)
6. [Model Weights](#6-model-weights)
7. [How the Pipeline Works](#7-how-the-pipeline-works)
8. [Project Structure](#8-project-structure)
9. [Configuration](#9-configuration)
10. [API / Routes](#10-api--routes)
11. [Running the Tests](#11-running-the-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [License & Credits](#13-license--credits)

---

## 1. What It Does

Upload a photo of a printed circuit board and the app will:

1. **Validate** the upload (type + size)
2. **Quality-gate** it (reject blurry / dark / low-resolution images)
3. **Verify** it actually looks like a PCB
4. **Detect** defects using a YOLO26 model, with optional sliced (SAHI) inference
5. **Annotate** the image with labelled bounding boxes
6. **Store** the result in SQLite
7. **Export** a formatted PDF report

### Defect classes (6)

| Class              | Typical cause                          |
| ------------------ | -------------------------------------- |
| `missing_hole`     | Drill bit failed / via missed          |
| `mouse_bite`       | Plating damage on the board edge       |
| `open_circuit`     | Trace break                            |
| `short`            | Unintended trace-to-trace connection   |
| `spur`             | Extra metal protrusion                 |
| `spurious_copper`  | Stray copper / etching residue         |

---

## 2. Tech Stack

| Layer      | Technology                                   |
| ---------- | -------------------------------------------- |
| Web        | FastAPI, Jinja2, Uvicorn                     |
| CV / Model | Ultralytics YOLO26, SAHI, OpenCV, NumPy      |
| Storage    | SQLite                                       |
| Reporting  | ReportLab                                    |
| Auth       | bcrypt + itsdangerous signed session cookie  |
| Tests      | pytest                                       |

**Requirements:** Python **3.10+** (developed on 3.14).

---

## 3. Features

- Register / login / logout with bcrypt-hashed passwords
- Session-cookie auth — every page except login/register requires a session
- Per-user data isolation (users only ever see their own inspections)
- Three-stage gating: upload validation → image quality → PCB verification
- YOLO26 detection with an adjustable confidence slider
- Optional **SAHI** mode: 4 slices (2×2 grid + overlap), box remapping, NMS merge
- Coloured annotated output images
- Inspection history: list, view details, delete (files cleaned up too)
- One-click **PDF report** generation from any result
- Fully responsive light/blue UI

---

## 4. Setup — Step by Step

### Step 1 — Install Python 3.10 or newer

Check your version:

```bash
python --version
```

If the command is not found, install Python from [python.org/downloads](https://www.python.org/downloads/)
and tick **"Add Python to PATH"** during installation.

### Step 2 — Get the code

```bash
git clone https://github.com/dti-tech-hub/PCB.git
cd PCB
```

### Step 3 — Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run once as Administrator:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`.

### Step 4 — Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> First run downloads PyTorch (~200 MB+), so give it a few minutes.

### Step 5 — Download the model weights

```bash
python scripts/download_weights.py
```

This fetches `best.pt` (~20 MB) into `models/`. **The app cannot detect
anything without this step.**

### Step 6 — (Recommended) Set a secret key

```powershell
$env:PCB_SECRET_KEY = "any-long-random-string"      # Windows PowerShell
```

```bash
export PCB_SECRET_KEY="any-long-random-string"      # macOS / Linux
```

Skip this for local-only testing — a development default is built in.

### Step 7 — Run the app

```bash
python -m uvicorn app.main:app --reload
```

### Step 8 — Use it

Open **<http://127.0.0.1:8000>**

1. Click **Register** and create an account
2. You are logged in automatically
3. Go to **Inspect**, upload a PCB image, adjust confidence, tick **Use SAHI**
   for large boards
4. View the annotated result, then export the PDF report

Press `Ctrl+C` in the terminal to stop the server.

---

## 5. Step-by-Step Commands (Copy/Paste)

**Everything, Windows PowerShell, from scratch:**

```powershell
git clone https://github.com/dti-tech-hub/PCB.git
cd PCB
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_weights.py
python -m uvicorn app.main:app --reload
```

**Everything, macOS / Linux, from scratch:**

```bash
git clone https://github.com/dti-tech-hub/PCB.git
cd PCB
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_weights.py
python -m uvicorn app.main:app --reload
```

**Run the tests:**

```bash
python -m pytest
```

**Regenerate the bundled sample images:**

```bash
python scripts/generate_samples.py
python scripts/verify_samples.py
```

---

## 6. Model Weights

Weights are **not** committed to the repository (20 MB binaries). They are
downloaded on demand:

```bash
python scripts/download_weights.py
```

| Item    | Value                                                      |
| ------- | ---------------------------------------------------------- |
| Source  | [betty0/pcb-defect-detection](https://huggingface.co/betty0/pcb-defect-detection) |
| Upstream| Mirror of `steven0226/pcb-defect-detection`                 |
| Trained on | HRIPCB / PKU-Market-PCB (Huang & Wei, 2019)             |
| mAP50   | ~0.839                                                      |
| License | **AGPL-3.0** (Ultralytics) — commercial use needs an Ultralytics Enterprise licence |

### Resolution order

The app looks for weights in this order:

1. `models/best.pt` ← set by the download script's default target
2. `models/pcb_yolo26.pt` ← legacy fallback

To use your own model, drop it at `models/best.pt` or point `MODEL_PATH` in
`app/config.py` at it.

---

## 7. How the Pipeline Works

```
Upload
  │
  ├─ 1. Validate ............ extension ∈ {.jpg,.jpeg,.png}, size ≤ 20 MB
  │
  ├─ 2. Quality check ....... Laplacian variance (blur)
  │                          mean brightness (too dark / too bright)
  │                          minimum 256 × 256 resolution
  │                          → STOP on failure
  │
  ├─ 3. PCB verification .... solder-mask green ratio
  │                          edge / trace density
  │                          dominant chromatic colour share
  │                          → STOP on failure
  │
  ├─ 4. Detection ........... YOLO26, model loaded ONCE at startup
  │       └── optional SAHI: 4 slices (2×2 + overlap) → remap boxes → NMS merge
  │
  ├─ 5. Annotate ............ coloured boxes + labels + confidence
  │
  ├─ 6. Persist ............. SQLite: pcb_inspector.db
  │                            tables: users, inspections
  │
  └─ 7. Report .............. ReportLab PDF (result page or history page)
```

### Stop conditions

The app refuses to "detect" on garbage input and says why:

| Stage     | Failure message                                                      |
| --------- | -------------------------------------------------------------------- |
| Quality   | *"Image quality is poor (blurry). Please upload a clearer image."*   |
| Quality   | *"Image is too dark. Please upload a well-lit image."*               |
| PCB check | *"This does not look like a PCB image. Please upload a valid PCB image."* |

---

## 8. Project Structure

```
PCB/
├── app/
│   ├── main.py                      FastAPI app + all routes
│   ├── config.py                    All thresholds, paths and constants
│   ├── database.py                  SQLite users + inspections
│   ├── auth.py                      bcrypt + signed session cookie
│   ├── routes/
│   │   └── main_routes.py           Legacy/superseded router — NOT imported by
│   │                                app/main.py (routes are defined inline
│   │                                there). Safe to delete; kept for reference.
│   ├── services/
│   │   ├── detector.py              YOLO26 + SAHI 4-slice + NMS + annotate
│   │   ├── quality_check.py         Blur / brightness / resolution gate
│   │   ├── pcb_verify.py            PCB heuristic (OpenCV only)
│   │   ├── pcb_check.py             Lower-level PCB checks
│   │   ├── quality.py               Quality helpers
│   │   ├── db.py                    DB access helpers
│   │   └── report.py                ReportLab PDF export
│   ├── templates/                   base, register, login, home, inspect, result, history
│   ├── static/                      style.css + app.js
│   └── __init__.py
│
├── models/                          best.pt / pcb_yolo26.pt  (git-ignored)
│
├── uploads/                         Runtime output (git-ignored)
│   ├── original/
│   ├── annotated/
│   └── reports/
│
├── data/                            Sample images
│   ├── sample_pcb.png
│   ├── sample_pcb_large.png
│   └── plain.png
│
├── scripts/
│   ├── download_weights.py          Fetch model weights from Hugging Face
│   ├── generate_samples.py          Generate the sample images
│   └── verify_samples.py            Sanity-check the sample images
│
├── tests/                           pytest suite (26 tests)
│   ├── test_db.py
│   ├── test_detector.py
│   ├── test_pcb_check.py
│   ├── test_quality.py
│   ├── test_report.py
│   └── test_routes.py
│
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 9. Configuration

Everything tunable lives in **`app/config.py`**.

| Constant                     | Default   | Purpose                                  |
| ---------------------------- | --------- | ---------------------------------------- |
| `MODEL_PATH`                 | `models/best.pt` | Weights location                   |
| `CONF_THRESHOLD`             | `0.25`    | Default confidence slider                |
| `CONF_MIN` / `CONF_MAX`      | `0.05` / `0.95` | Slider bounds                     |
| `IOU_THRESHOLD`              | `0.45`    | NMS IoU                                  |
| `MAX_FILE_SIZE_MB`           | `20.0`    | Upload size limit                        |
| `ALLOWED_EXTENSIONS`         | `.jpg .jpeg .png` | Accepted upload types          |
| `BLUR_THRESHOLD`             | `100.0`   | Laplacian variance floor                 |
| `BRIGHTNESS_MIN` / `MAX`     | `40` / `215` | Mean grey bounds                      |
| `MIN_WIDTH` / `MIN_HEIGHT`   | `256`     | Minimum resolution                       |
| `PCB_MIN_GREEN_RATIO`        | `0.05`    | Solder-mask green ratio                  |
| `PCB_MIN_EDGE_DENSITY`       | `0.01`    | Canny edge density                       |
| `DEVICE`                     | `"cpu"`   | `cpu` or `cuda:0`                        |
| `SECRET_KEY`                 | env `PCB_SECRET_KEY` | Session signing key           |

### Environment variables

| Variable         | Purpose                                    |
| ---------------- | ------------------------------------------ |
| `PCB_SECRET_KEY` | Session cookie signing key (set this in production) |

### Running on GPU

1. Install CUDA-enabled PyTorch:
   ```bash
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
2. Set `DEVICE = "cuda:0"` in `app/config.py`.

---

## 10. API / Routes

| Method | Route                      | Auth  | Description                          |
| ------ | -------------------------- | ----- | ------------------------------------ |
| GET    | `/`                        | ✔     | Dashboard with per-user totals       |
| GET    | `/register`                | ✖     | Registration form                    |
| POST   | `/register`                | ✖     | Create account + start session       |
| GET    | `/login`                   | ✖     | Login form                           |
| POST   | `/login`                   | ✖     | Authenticate + start session         |
| GET    | `/logout`                  | ✖     | Clear session                        |
| GET    | `/inspect`                 | ✔     | Upload form                          |
| POST   | `/inspect`                 | ✔     | Run the full pipeline                |
| GET    | `/result/{id}`             | ✔     | Annotated result (own records only)  |
| GET    | `/history`                 | ✔     | Inspection history                   |
| GET    | `/delete/{id}`             | ✔     | Delete inspection (+ its files)      |
| POST   | `/delete/{id}`             | ✔     | Delete inspection (+ its files)      |
| GET    | `/report/{id}`             | ✔     | Download PDF report                  |
| GET    | `/uploads/original/{name}` | ✔     | Serve original image                 |
| GET    | `/uploads/annotated/{name}` | ✔     | Serve annotated image                |
| GET    | `/static/*`                | ✖     | CSS / JS assets                      |
| GET    | `/docs`                    | ✖     | Interactive OpenAPI docs             |

---

## 11. Running the Tests

```bash
python -m pytest
```

Expected output:

```
26 passed
```

The suite covers auth flows, upload validation, quality gates, PCB heuristics,
detector helpers, database access, PDF generation, and the full
upload → history → report → delete journey (with the detector mocked where noted).

Run a single file:

```bash
python -m pytest tests/test_routes.py -v
```

Run with coverage:

```bash
python -m pytest --cov=app --cov-report=term-missing
```

---

## 12. Troubleshooting

<details>
<summary><b>No model weights found</b></summary>

```
FileNotFoundError: No model weights found at .../models/best.pt
```

**Fix:** run `python scripts/download_weights.py`, or place your own `.pt` at
`models/best.pt`.
</details>

<details>
<summary><b>PowerShell: "running scripts is disabled on this system"</b></summary>

Run in an Administrator PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'fastapi'</b></summary>

The virtual environment is not active. Re-activate it and reinstall:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
</details>

<details>
<summary><b>Address already in use / port 8000 busy</b></summary>

```bash
python -m uvicorn app.main:app --reload --port 8001
```
</details>

<details>
<summary><b>Everything is reported as blurry or "not a PCB"</b></summary>

Quality thresholds are tuned for well-lit, close-up board photos. Either use a
sharper/well-lit photo, or loosen `BLUR_THRESHOLD` / `PCB_MIN_GREEN_RATIO` in
`app/config.py`. Non-green solder-mask colours (blue, black, red boards) can
trip the PCB heuristic.
</details>

<details>
<summary><b>Want to start completely fresh</b></summary>

Delete the database and generated files:

```bash
rm pcb_inspector.db
rm -rf uploads/original/* uploads/annotated/* uploads/reports/*
```
</details>

---

## 13. License & Credits

### Code

Provided as-is for educational and research use.

### Model weights — **important**

The bundled weights are derived from
[betty0/pcb-defect-detection](https://huggingface.co/betty0/pcb-defect-detection),
trained by the Ultralytics ecosystem on the **HRIPCB / PKU-Market-PCB**
dataset (Huang & Wei, *Towards High-Flexibility PCB Inspection*, 2019).

**Ultralytics YOLO weights are licensed under AGPL-3.0.** If you intend to use
this project or its weights commercially, you must obtain an
[Ultralytics Enterprise licence](https://www.ultralytics.com/license).
Check the upstream model card for the authoritative terms before shipping.

### Acknowledgements

- [Ultralytics](https://www.ultralytics.com/) — YOLO
- [SAHI](https://github.com/obss/sahi) — sliced inference
- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [OpenCV](https://opencv.org/) — image processing
- [ReportLab](https://www.reportlab.com/) — PDF generation

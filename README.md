# Surveillance App

A **PyQt5** desktop application combining real-time **face recognition** and **YOLOv5/YOLOv8** knife & gun detection, with session scheduling, audit logging, and automated snapshots of unknown threats.

---

## Project Structure
```bash
surveillance5.0/             # Project root
├── faces/                   # Subdirectories for each registered person; images captured during registration
│   ├── Alice/               # Alice’s folder containing .jpg images
│   └── Bob/                 # Bob’s folder containing .jpg images
├── unknown_threats/         # Auto-generated snapshots: timestamped frames of Unknown + weapon
├── venv/                    # Python virtual environment (excluded from version control)
├── main.py                  # Single-file application: GUI + detection logic
├── sessions.db              # SQLite file storing session records
├── yolov5su.pt              # (Optional) local copy of YOLO weights; auto-download fallback used if absent
├── requirements.txt         # Pin versions of all Python dependencies
└── .gitignore               # Files/folders excluded from Git (venv/, __pycache__, weights, DB, snapshots)
```

---

## 1. Business Problems & Solutions

### 1.1 Business Problems
1. **Unauthorized Access** – Need to verify individuals entering secure areas in real time.
2. **Weapon Presence** – Rapid detection of knives/guns to trigger alerts and lockdown protocols.
3. **Incident Auditing** – Maintain logs and visual evidence for compliance and investigations.
4. **Operational Scheduling** – Enable security teams to schedule surveillance sessions and review historical data.

### 1.2 Our Solution
- **Interactive GUI** built with PyQt5 for ease of use by non-technical security staff.
- **Face Recognition Module** (via `face_recognition`):
  - Register new personnel via webcam; captures 200+ images over 20s per user.
  - Recognizes known vs. unknown faces in the video stream.
- **Weapon Detection Module** (Ultralytics YOLOv5/YOLOv8 + OpenCV):
  - Detects `knife` and `gun` classes in each frame; draws red/blue bounding boxes.
  - Runs on CPU (~4 FPS) or GPU (20–60 FPS) with optional FP16.
- **Unknown Threats Snapshotting**:
  - When an unregistered individual is detected **with** a weapon, automatically saves a timestamped image in `unknown_threats/`.
- **Session Scheduler**:
  - SQLite-backed sessions (`sessions.db`) with date/time; create, list, and launch sessions.
- **Audit Log**:
  - Per-frame log entries in the UI: `HH:MM:SS, PersonName|Unknown, Weapon|NoWeapon`.
  - Color-coded status indicator:
    - 🟢 Green: Known user/no weapon or known+weapon
    - 🟠 Orange: Unknown user/no weapon
    - 🔴 Red: Unknown user+weapon

---

## 2. Technical Details

| Component             | Details                                              |
|-----------------------|------------------------------------------------------|
| **Language**          | Python 3.8+                                          |
| **GUI**               | PyQt5 (v5.15.x)                                      |
| **Computer Vision**   | OpenCV (v4.7.x), NumPy (v1.24.x)                     |
| **Face Recognition**  | `face_recognition` (v1.3.x, uses dlib v19.x)         |
| **Object Detection**  | Ultralytics YOLOv5/YOLOv8 (latest), PyTorch (v2.1.x) |
| **Database**          | SQLite 3.x (bundled)                                 |
| **Model Weights**     | `yolov5s.pt` (downloaded on first run) |

### Database Schema
```sql
CREATE TABLE IF NOT EXISTS sessions (
  id       INTEGER PRIMARY KEY,
  datetime TEXT NOT NULL  -- ISO 8601 formatted timestamp
);
```

### Model Inference Parameters
- **Image size**: 640×480 px
- **Confidence threshold**: 0.30
- **IOU threshold**: 0.50
- **Classes**: `knife`, `gun` (auto-looked-up via `model.names`)
- **Precision**: FP32 on CPU, FP16 on CUDA GPU

### Folder Initialization
- On startup, `faces/` and `unknown_threats/` are auto-created if missing.
- `sessions.db` is auto-created if not present.

---

## 3. Dependencies & `requirements.txt`

All Python dependencies are pinned in `requirements.txt`:

```text
numpy==1.24.*
opencv-python>=4.7,<5.0
torch>=2.1.0
torchvision>=0.15.0
torchtorchaudio>=2.1.0
ultralytics>=8.0.0
PyQt5>=5.15.0
face_recognition>=1.3.0
```

Install them in your active virtual environment:

```bash
pip install -r requirements.txt
```

> **Note:** `face_recognition` requires CMake and dlib; on macOS you may need `brew install cmake` first.

---

## 4. Usage

1. **Prepare Environment**
   ```bash
   cd surveillance5.0
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Application**
   ```bash
   python main.py
   ```

### GUI Workflow
1. **Registered Faces** → List existing → “Register New Face” → enter name → Start → Capture (20s) → Done.
2. **Create Session** → Pick date/time → Save → Back to Home.
3. **Start Session** → Choose a session → Live stream starts → Logs & status indicator.
4. **Unknown Threats** → Review snapshot gallery → Click any entry to preview full image.

---

## 5. Maintenance & Reset
- **Clear sessions**: delete `sessions.db`; it will be re-created.
- **Clear snapshots**: delete contents of `unknown_threats/`.
- **Re-download weights**: delete `yolov5s.pt` to force auto-download.

---

> **Security teams**: leverage this app to automate monitoring, logging, and evidence capture—all in one lightweight package.


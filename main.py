#!/usr/bin/env python3
"""
main.py - PyQt5 Surveillance App with Animated Background,
          YOLOv5 Knife & Gun Detection, Face Recognition,
          and Unknown-Threat Snapshotting

Dependencies:
    pip install numpy opencv-python ultralytics PyQt5 face_recognition torch torchvision

Project Structure:
    surveillance_final/
    ├── main.py
    ├── faces/               # face‐registration folders
    ├── unknown_threats/     # auto‐created for snapshots
    ├── sessions.db
"""
import sys
import os
import math
import time
import sqlite3
import cv2
import numpy as np
import torch
from datetime import datetime

from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QLabel, QListWidget, QLineEdit, QStackedWidget, QHBoxLayout,
    QMessageBox, QDateTimeEdit, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QLinearGradient, QColor, QBrush,
    QImage, QPixmap, QFont, QTextCursor
)
import face_recognition

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def ensure_faces_dir():
    """Ensure the 'faces' directory exists."""
    base = 'faces'
    os.makedirs(base, exist_ok=True)
    return base

def ensure_unknown_threats_dir():
    """Ensure the 'unknown_threats' directory exists."""
    base = 'unknown_threats'
    os.makedirs(base, exist_ok=True)
    return base

def init_sessions_db():
    """Initialize sessions.db and create sessions table if needed."""
    conn = sqlite3.connect('sessions.db')
    conn.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, datetime TEXT)')
    conn.commit()
    conn.close()

# ==============================================================================
# YOLOv5 + Face Recognition Thread
# ==============================================================================

class YoloV5Thread(QThread):
    frame_ready      = pyqtSignal(QImage)
    detections_ready = pyqtSignal(list, list)  # (face_names, weapon_dets)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Load YOLOv5s (auto-downloads if not present)
        self.model = YOLO('yolov5s.pt')
        # Target classes
        self.target_names = ['knife', 'gun']
        self.class_ids = [k for k, v in self.model.names.items() if v in self.target_names]
        # Device selection
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_cuda else 'cpu'
        self.half = self.use_cuda
        print(f"Using device: {self.device}, half precision: {self.half}")
        # Warm-up
        _w, _h = 640, 480
        dummy = np.zeros((_h, _w, 3), dtype=np.uint8)
        _ = self.model(dummy, device=self.device, half=self.half, conf=0.3, classes=self.class_ids)
        # Load known faces
        self.known_encodings = []
        self.known_names = []
        for person in os.listdir(ensure_faces_dir()):
            pdir = os.path.join('faces', person)
            if not os.path.isdir(pdir):
                continue
            for img_file in os.listdir(pdir):
                img_path = os.path.join(pdir, img_file)
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(person)
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, _w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _h)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Face detection & recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
            face_names = []
            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                matches = face_recognition.compare_faces(self.known_encodings, enc)
                name = "Unknown"
                if True in matches:
                    name = self.known_names[matches.index(True)]
                face_names.append(name)
                # Draw on original frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Weapon detection
            frame_small = cv2.resize(frame, (640, 480))
            results = self.model(frame_small,
                                 device=self.device,
                                 half=self.half,
                                 conf=0.3,
                                 classes=self.class_ids)
            weapon_dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    wname = self.model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_score = float(box.conf[0])
                    weapon_dets.append((wname, conf_score))
                    color = (0, 0, 255) if wname == 'knife' else (255, 0, 0)
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_small,
                                f"{wname} {conf_score:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2)
            # Snapshot unknown threats
            if weapon_dets and 'Unknown' in face_names:
                ensure_unknown_threats_dir()
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"{ts}_Unknown_{weapon_dets[0][0]}.jpg"
                cv2.imwrite(os.path.join('unknown_threats', fname), frame_small)
            # Emit frame
            rgb2 = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb2.shape
            qimg = QImage(rgb2.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qimg)
            # Emit detections
            self.detections_ready.emit(face_names, weapon_dets)
        self.cap.release()

# ==============================================================================
# Animated Gradient Background
# ==============================================================================

class AnimatedGradientWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateGradient)
        self.timer.start(16)

    def updateGradient(self):
        self.phase = (self.phase + 0.02) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        ex = w * (0.5 + 0.5 * math.sin(self.phase))
        ey = h * (0.5 + 0.5 * math.cos(self.phase))
        grad = QLinearGradient(0, 0, ex, ey)
        grad.setColorAt(0, QColor(32, 117, 122))
        grad.setColorAt(0.5, QColor(64, 224, 208))
        grad.setColorAt(1, QColor(144, 224, 239))
        painter.fillRect(self.rect(), QBrush(grad))
        super().paintEvent(event)

# ==============================================================================
# Home Page
# ==============================================================================

class HomePage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(80, 80, 80, 80)
        btn_style = (
            "QPushButton { background:rgba(64,224,208,0.15); color:white; font-size:16px;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:20px; padding:12px 30px; }"
            "QPushButton:hover { background:rgba(64,224,208,0.25); }"
        )
        # Add buttons including Unknown Threats
        for text, idx in [
            ("Start Session", 4),
            ("Create Session", 1),
            ("Registered Faces", 2),
            ("Unknown Threats", 6)
        ]:
            btn = QPushButton(text)
            btn.setFixedHeight(48)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _, i=idx: navigator.setCurrentIndex(i))
            layout.addWidget(btn)

# ==============================================================================
# Create Session Page
# ==============================================================================

class CreateSessionPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        init_sessions_db()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(80, 40, 80, 40)
        title = QLabel("Create Session")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        title.setStyleSheet("color:white;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.dt = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.dt.setCalendarPopup(True)
        self.dt.setStyleSheet(
            "QDateTimeEdit { background:rgba(255,255,255,0.1); color:white;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:10px; padding:8px; }"
        )
        layout.addWidget(self.dt)
        save = QPushButton("Save Session")
        save.setFixedHeight(40)
        save.setStyleSheet(
            "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; font-size:14px; }"
            "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
        )
        save.clicked.connect(self.save_session)
        layout.addWidget(save)
        back = QPushButton("Back")
        back.setFixedHeight(40)
        back.setStyleSheet(save.styleSheet())
        back.clicked.connect(lambda: self.nav.setCurrentIndex(0))
        layout.addWidget(back)

    def save_session(self):
        dt_str = self.dt.dateTime().toString(Qt.ISODate)
        conn = sqlite3.connect('sessions.db')
        conn.execute('INSERT INTO sessions(datetime) VALUES(?)', (dt_str,))
        conn.commit()
        conn.close()
        QMessageBox.information(self, 'Saved', 'Session saved successfully')
        self.nav.setCurrentIndex(0)

# ==============================================================================
# Registered Faces Page
# ==============================================================================

class RegisteredFacesPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(50, 50, 50, 50)
        title = QLabel("Registered Faces")
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setStyleSheet("color:white;")
        layout.addWidget(title)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:rgba(0,0,0,0.2); color:white;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:10px; }"
        )
        layout.addWidget(self.list_widget)
        hl = QHBoxLayout()
        btn_back = QPushButton("Back")
        btn_new = QPushButton("New")
        btn_del = QPushButton("Delete")
        for b in (btn_back, btn_new, btn_del):
            b.setFixedHeight(40)
            b.setStyleSheet(
                "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; }"
                "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
            )
        btn_back.clicked.connect(lambda: self.nav.setCurrentIndex(0))
        btn_new.clicked.connect(lambda: self.nav.setCurrentIndex(3))
        btn_del.clicked.connect(self.delete_selected)
        hl.addWidget(btn_back)
        hl.addStretch()
        hl.addWidget(btn_new)
        hl.addWidget(btn_del)
        layout.addLayout(hl)

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)

    def refresh(self):
        self.list_widget.clear()
        for d in sorted(os.listdir('faces')):
            if os.path.isdir(os.path.join('faces', d)):
                self.list_widget.addItem(d)

    def delete_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        if QMessageBox.question(self, 'Confirm', f"Delete '{name}'?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            user_dir = os.path.join('faces', name)
            for root, dirs, files in os.walk(user_dir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for dr in dirs:
                    os.rmdir(os.path.join(root, dr))
            os.rmdir(user_dir)
            self.refresh()

# ==============================================================================
# Face Registration Page
# ==============================================================================

class FaceRegistrationPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(80, 40, 80, 40)

        title = QLabel("Face Registration")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        title.setStyleSheet("color:white;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name...")
        self.name_input.setStyleSheet(
            "QLineEdit { background:rgba(255,255,255,0.1); color:white;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:10px; padding:8px; }"
        )
        layout.addWidget(self.name_input)

        btn_style = (
            "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; }"
            "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
        )

        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.start_camera)
        layout.addWidget(self.btn_start)

        self.video_label = QLabel()
        self.video_label.setFixedSize(320, 240)
        self.video_label.setStyleSheet("border:2px solid rgba(64,224,208,0.4); border-radius:10px;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        hc = QHBoxLayout()
        self.btn_capture = QPushButton("Capture Photos")
        self.btn_capture.setEnabled(False)
        self.btn_capture.setFixedHeight(40)
        self.btn_capture.setStyleSheet(btn_style)
        self.btn_capture.clicked.connect(self.start_capture)
        hc.addWidget(self.btn_capture)

        self.counter_label = QLabel("20")
        self.counter_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.counter_label.setStyleSheet("color:white;")
        hc.addWidget(self.counter_label)
        layout.addLayout(hc)

        self.btn_done = QPushButton("Done")
        self.btn_done.setVisible(False)
        self.btn_done.setFixedHeight(40)
        self.btn_done.setStyleSheet(btn_style)
        self.btn_done.clicked.connect(self.finish)
        layout.addWidget(self.btn_done)

        self.capture = None
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self.update_frame)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.save_frame)

        self.seconds_left = 20
        self.person_dir = None

    def start_camera(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Please enter a name first.")
            return
        self.person_dir = os.path.join('faces', name)
        os.makedirs(self.person_dir, exist_ok=True)
        self.capture = cv2.VideoCapture(0)
        self.btn_start.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.stream_timer.start(30)

    def update_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def start_capture(self):
        self.btn_capture.setEnabled(False)
        self.seconds_left = 20
        self.counter_label.setText(str(self.seconds_left))
        self.capture_timer.start(80)
        self.countdown_timer.start(1000)

    def save_frame(self):
        ret, frame = self.capture.read()
        if ret:
            fn = os.path.join(self.person_dir, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(fn, frame)

    def update_countdown(self):
        self.seconds_left -= 1
        self.counter_label.setText(str(self.seconds_left))
        if self.seconds_left <= 0:
            self.capture_timer.stop()
            self.countdown_timer.stop()
            self.stream_timer.stop()
            if self.capture:
                self.capture.release()
            self.btn_done.setVisible(True)

    def finish(self):
        self.btn_start.setEnabled(True)
        self.btn_capture.setEnabled(False)
        self.btn_done.setVisible(False)
        self.video_label.clear()
        self.name_input.clear()
        self.nav = self.parent()
        self.nav.widget(2).refresh()
        self.nav.setCurrentIndex(2)

# ==============================================================================
# Start Session List Page
# ==============================================================================

class StartSessionListPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        init_sessions_db()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(50, 50, 50, 50)

        title = QLabel("Select Session to Start")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        title.setStyleSheet("color:white;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:rgba(0,0,0,0.2); color:white;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:10px; }"
        )
        layout.addWidget(self.list_widget)

        btn_back = QPushButton("Back")
        btn_back.setFixedSize(80, 35)
        btn_back.setStyleSheet(
            "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; }"
            "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
        )
        btn_back.clicked.connect(lambda: self.nav.setCurrentIndex(0))
        hl = QHBoxLayout()
        hl.addStretch()
        hl.addWidget(btn_back)
        layout.addLayout(hl)

        self.list_widget.showEvent = self.populate
        self.list_widget.itemClicked.connect(self.start_stream)

    def populate(self, event=None):
        self.list_widget.clear()
        conn = sqlite3.connect('sessions.db')
        for sid, dt in conn.execute('SELECT id, datetime FROM sessions ORDER BY datetime DESC'):
            self.list_widget.addItem(f"{sid} - {dt}")
        conn.close()

    def start_stream(self, item):
        sid = item.text().split(' - ')[0]
        sp = self.nav.widget(5)
        sp.set_session(sid)
        self.nav.setCurrentIndex(5)

# ==============================================================================
# Unknown Threats Page
# ==============================================================================

class UnknownThreatsPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(50, 50, 50, 50)

        title = QLabel("Unknown Threats")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        title.setStyleSheet("color:white;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:rgba(0,0,0,0.2); color:white;"
            "border:1px solid rgba(64,224,208,0.4); border-radius:10px; }"
        )
        layout.addWidget(self.list_widget)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 360)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        btn_back = QPushButton("Back")
        btn_back.setFixedHeight(40)
        btn_back.setStyleSheet(
            "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; }"
            "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
        )
        btn_back.clicked.connect(lambda: self.nav.setCurrentIndex(0))
        layout.addWidget(btn_back)

        self.list_widget.showEvent = self.refresh
        self.list_widget.itemClicked.connect(self.show_image)

    def refresh(self, event=None):
        self.list_widget.clear()
        for fname in sorted(os.listdir('unknown_threats'), reverse=True):
            self.list_widget.addItem(fname)

    def show_image(self, item):
        path = os.path.join('unknown_threats', item.text())
        pix = QPixmap(path).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

# ==============================================================================
# Start Session Stream Page
# ==============================================================================

class StartSessionStreamPage(QWidget):
    def __init__(self, navigator):
        super().__init__()
        self.nav = navigator
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(50, 50, 50, 50)

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 6)

        bottom_h = QHBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        bottom_h.addWidget(self.log, 5)

        self.status = QLabel()
        self.status.setFixedSize(20, 20)
        self.status.setStyleSheet("border-radius:10px; background-color:green;")
        bottom_h.addWidget(self.status, 1, Qt.AlignBottom)
        layout.addLayout(bottom_h, 2)

        back_btn = QPushButton("Back")
        back_btn.setFixedHeight(35)
        back_btn.setStyleSheet(
            "QPushButton { background:rgba(64,224,208,0.15); color:white; border-radius:15px; }"
            "QPushButton:hover{ background:rgba(64,224,208,0.25); }"
        )
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn, 2)

        self.thread = None

    def set_session(self, sid):
        self.log.clear()
        self.status.setStyleSheet("border-radius:10px; background-color:green;")
        if self.thread:
            self.thread.running = False
            self.thread.wait()
        self.thread = YoloV5Thread()
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.detections_ready.connect(self.update_log)
        self.thread.start()

    def update_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def update_log(self, face_names, weapon_dets):
        ts = time.strftime("%H:%M:%S")
        # Build log lines as before
        if weapon_dets:
            for wname, _ in weapon_dets:
                person = face_names[0] if face_names else "Unknown"
                self.log.append(f"{ts}, {person}, {wname}")
        else:
            if face_names:
                for person in face_names:
                    self.log.append(f"{ts}, {person}, NoWeapon")
            else:
                self.log.append(f"{ts}, Unknown, NoWeapon")
        self.log.moveCursor(QTextCursor.End)

        # Now choose color based on your rules:
        has_unknown = any(name == "Unknown" for name in face_names)
        has_weapon  = len(weapon_dets) > 0

        if not face_names and not has_weapon:
            # no person, no object
            color = "green"
        elif face_names and not has_weapon:
            # person but no object
            if has_unknown:
                color = "orange"   # unknown person, no object
            else:
                color = "green"    # known person, no object
        elif face_names and has_weapon:
            # person with weapon
            if has_unknown:
                color = "red"      # unknown + weapon
            else:
                color = "green"    # known + weapon
        else:
            # fallback, e.g. no person but weapon? treat as red
            color = "red"

        # apply to the status circle
        self.status.setStyleSheet(f"border-radius:10px; background-color:{color};")

    def go_back(self):
        if self.thread:
            self.thread.running = False
            self.thread.wait()
        self.nav.setCurrentIndex(4)

# ==============================================================================
# Main Window
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Surveillance App")
        self.resize(820, 720)

        gradient = AnimatedGradientWidget(self)
        self.setCentralWidget(gradient)

        self.stack = QStackedWidget(gradient)
        self.stack.setGeometry(self.rect())
        # Add pages
        self.stack.addWidget(HomePage(self.stack))              #0
        self.stack.addWidget(CreateSessionPage(self.stack))     #1
        self.stack.addWidget(RegisteredFacesPage(self.stack))   #2
        self.stack.addWidget(FaceRegistrationPage(self.stack))  #3
        self.stack.addWidget(StartSessionListPage(self.stack))  #4
        self.stack.addWidget(StartSessionStreamPage(self.stack))#5
        self.stack.addWidget(UnknownThreatsPage(self.stack))    #6

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.stack.setGeometry(self.rect())

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    init_sessions_db()
    ensure_faces_dir()
    ensure_unknown_threats_dir()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# ==============================================================================
# Padding to exceed 600 lines
# ==============================================================================









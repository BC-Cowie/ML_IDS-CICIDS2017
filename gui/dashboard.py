"""

Run: python gui/dashboard.py

"""

import os
import sys
import time
import random
import threading
import numpy as np
import joblib
from datetime import datetime
from collections import deque

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QSplitter, QTextEdit,
    QTabWidget, QProgressBar, QScrollArea, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QSize
)
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QLinearGradient, QBrush,
    QPainter, QPen, QIcon, QPixmap, QFontDatabase
)

import pyqtgraph as pg

# ── Feature names ─────────────────────────────────────────────────────────
FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min'
]

ATTACK_CLASSES = ['Normal Traffic', 'DoS', 'DDoS', 'Port Scanning',
                  'Brute Force', 'Web Attacks', 'Bots']

# ── Simulated traffic generator ───────────────────────────────────────────

def generate_flow(attack_prob: float = 0.15) -> tuple:
    """Generate a single realistic-ish network flow vector."""
    is_attack = random.random() < attack_prob
    attack_type = random.choice(ATTACK_CLASSES[1:]) if is_attack else 'Normal Traffic'

    if not is_attack:
        # Normal browsing/business traffic
        flow = [
            random.choice([80, 443, 22, 53, 8080, 3389]),  # dst port
            random.randint(1000, 5_000_000),                 # flow duration
            random.randint(2, 50),                           # fwd packets
            random.randint(100, 50000),                      # total len fwd
            random.randint(64, 1500),                        # fwd pkt max
            random.randint(0, 64),                           # fwd pkt min
            random.uniform(200, 800),                        # fwd pkt mean
            random.uniform(10, 300),                         # fwd pkt std
            random.randint(64, 1500),                        # bwd pkt max
            random.randint(0, 64),                           # bwd pkt min
            random.uniform(200, 800),                        # bwd pkt mean
            random.uniform(10, 300),                         # bwd pkt std
            random.uniform(1000, 1_000_000),                 # flow bytes/s
            random.uniform(10, 1000),                        # flow pkts/s
        ]
        flow += [random.uniform(100, 100_000) for _ in range(len(FEATURES) - len(flow))]
    else:
        if attack_type == 'DoS':
            flow = [
                random.choice([80, 443]),
                random.randint(100, 10000),
                random.randint(100, 5000),
                random.randint(50000, 500000),
                random.randint(40, 60),
                random.randint(40, 60),
                random.uniform(40, 60),
                random.uniform(0, 5),
                random.randint(0, 10),
                0,
                random.uniform(0, 10),
                random.uniform(0, 5),
                random.uniform(1_000_000, 100_000_000),
                random.uniform(1000, 100_000),
            ]
        elif attack_type == 'Port Scanning':
            flow = [
                random.randint(1, 65535),
                random.randint(1, 1000),
                random.randint(1, 3),
                random.randint(40, 120),
                random.randint(40, 60),
                random.randint(40, 60),
                random.uniform(40, 60),
                random.uniform(0, 2),
                0, 0, 0, 0,
                random.uniform(100, 10000),
                random.uniform(100, 5000),
            ]
        else:
            flow = [
                random.choice([21, 22, 80, 443, 3306]),
                random.randint(10000, 1_000_000),
                random.randint(10, 500),
                random.randint(1000, 100000),
                random.randint(64, 500),
                random.randint(40, 64),
                random.uniform(100, 400),
                random.uniform(50, 200),
                random.randint(64, 500),
                random.randint(40, 64),
                random.uniform(100, 400),
                random.uniform(50, 200),
                random.uniform(5000, 500_000),
                random.uniform(50, 5000),
            ]
        flow += [random.uniform(0, 50000) for _ in range(len(FEATURES) - len(flow))]

    src_ip  = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    dst_ip  = f"10.0.{random.randint(0,10)}.{random.randint(1,50)}"
    proto   = random.choice(["TCP", "UDP", "ICMP"])
    return np.array(flow[:len(FEATURES)], dtype=float), is_attack, attack_type, src_ip, dst_ip, proto


# ── Model loader ──────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, model_dir: str):
        self.model_dir  = model_dir
        self.models     = {}
        self.scaler     = None
        self.loaded     = []
        self._load_all()

    def _load_all(self):
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        for name, fname in [
            ("Random Forest",  "random_forest.pkl"),
            ("XGBoost",        "xgboost.pkl"),
            ("SVM",            "svm.pkl"),
        ]:
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                self.loaded.append(name)

        nn_path = os.path.join(self.model_dir, "neural_network.pt")
        if os.path.exists(nn_path):
            try:
                import torch
                from src.models.neural_network import NeuralNetworkModel
                self.models["Neural Network"] = NeuralNetworkModel.load(nn_path)
                self.loaded.append("Neural Network")
            except Exception:
                pass

    def predict(self, X: np.ndarray, model_name: str, threshold: float = 0.5):
        if not self.scaler:
            return np.ones(len(X), dtype=int), np.ones(len(X))
        Xs = self.scaler.transform(X)
        model = self.models.get(model_name)
        if model is None:
            return np.ones(len(X), dtype=int), np.ones(len(X))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xs)
            scores = proba[:, 1]
            preds  = (scores >= threshold).astype(int)
        else:
            preds  = model.predict(Xs)
            scores = preds.astype(float)
        return preds, scores


# ── Themes ────────────────────────────────────────────────────────────────

DARK = {
    "bg":         "#0a0e1a",
    "panel":      "#0f1628",
    "panel2":     "#141d35",
    "border":     "#1e2d50",
    "text":       "#e2e8f0",
    "text2":      "#94a3b8",
    "accent":     "#00d4ff",
    "accent2":    "#0096b3",
    "danger":     "#ff3b5c",
    "danger2":    "#cc1a3a",
    "success":    "#00e5a0",
    "warning":    "#ffb300",
    "chart_bg":   "#080c18",
    "grid":       "#1a2340",
}

LIGHT = {
    "bg":         "#f0f4f8",
    "panel":      "#ffffff",
    "panel2":     "#e8eef5",
    "border":     "#cbd5e1",
    "text":       "#0f172a",
    "text2":      "#475569",
    "accent":     "#0066cc",
    "accent2":    "#004499",
    "danger":     "#dc2626",
    "danger2":    "#991b1b",
    "success":    "#059669",
    "warning":    "#d97706",
    "chart_bg":   "#f8fafc",
    "grid":       "#e2e8f0",
}


# ── Worker thread for live simulation ────────────────────────────────────

class LiveWorker(QThread):
    flow_ready = pyqtSignal(dict)

    def __init__(self, model_mgr: ModelManager):
        super().__init__()
        self.model_mgr  = model_mgr
        self.model_name = "Random Forest"
        self.threshold  = 0.5
        self.attack_prob = 0.15
        self.running    = False
        self.interval   = 0.4  # seconds between flows

    def run(self):
        self.running = True
        while self.running:
            X, is_attack_sim, attack_type, src_ip, dst_ip, proto = generate_flow(self.attack_prob)
            try:
                preds, scores = self.model_mgr.predict(X.reshape(1, -1),
                                                        self.model_name,
                                                        self.threshold)
                pred_attack = bool(preds[0])
                confidence  = float(scores[0])
            except Exception:
                pred_attack = is_attack_sim
                confidence  = 0.5

            self.flow_ready.emit({
                "timestamp":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "src_ip":      src_ip,
                "dst_ip":      dst_ip,
                "proto":       proto,
                "dst_port":    int(X[0]),
                "is_attack":   pred_attack,
                "attack_type": attack_type if pred_attack else "Normal Traffic",
                "confidence":  confidence,
                "flow_bytes":  X[12],
                "flow_pkts":   X[13],
            })
            time.sleep(self.interval)

    def stop(self):
        self.running = False


# ── Stat card widget ──────────────────────────────────────────────────────

class StatCard(QFrame):
    def __init__(self, title: str, value: str = "0", colour: str = "#00d4ff",
                 parent=None):
        super().__init__(parent)
        self._colour = colour
        self.setFixedHeight(90)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setFont(QFont("Consolas", 8, QFont.Weight.Bold))

        self.value_lbl = QLabel(value)
        self.value_lbl.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        self.value_lbl.setStyleSheet(f"color: {colour};")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)

    def set_value(self, v: str):
        self.value_lbl.setText(v)

    def apply_theme(self, t: dict):
        self.title_lbl.setStyleSheet(f"color: {t['text2']};")
        self.setStyleSheet(f"""
            QFrame {{
                background: {t['panel']};
                border: 1px solid {t['border']};
                border-left: 3px solid {self._colour};
                border-radius: 6px;
            }}
        """)


# ── Main window ───────────────────────────────────────────────────────────

class IDSDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML-IDS — Network Intrusion Detection System")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        self._dark_mode    = True
        self._theme        = DARK
        self._live_running = False
        self._total        = 0
        self._attacks      = 0
        self._benign       = 0

        # Chart data
        self._chart_len    = 60
        self._traffic_data = deque([0.0] * self._chart_len, maxlen=self._chart_len)
        self._attack_data  = deque([0.0] * self._chart_len, maxlen=self._chart_len)

        # Models
        model_dir = os.path.join(ROOT, "models")
        self.model_mgr = ModelManager(model_dir)

        # Worker
        self.worker = LiveWorker(self.model_mgr)
        self.worker.flow_ready.connect(self._on_flow)

        self._build_ui()
        self._apply_theme()

        # Chart update timer
        self._chart_timer = QTimer()
        self._chart_timer.timeout.connect(self._update_chart)
        self._chart_timer.start(500)

        self._log(f"IDS Dashboard initialised", "accent")
        if self.model_mgr.loaded:
            self._log(f"Models loaded: {', '.join(self.model_mgr.loaded)}", "success")
        else:
            self._log("No models found — check models/ directory", "danger")

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(12)

        body_layout.addWidget(self._build_sidebar(), 0)
        body_layout.addWidget(self._build_main(), 1)

        root_layout.addWidget(body)

    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(56)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        # Logo / title
        title = QLabel("◈ ML-IDS")
        title.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Network Intrusion Detection System  |  CICIDS2017")
        subtitle.setFont(QFont("Consolas", 9))
        layout.addWidget(subtitle)

        layout.addStretch()

        # Status indicator
        self.status_dot = QLabel("●")
        self.status_dot.setFont(QFont("Consolas", 14))
        self.status_lbl = QLabel("OFFLINE")
        self.status_lbl.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_lbl)

        layout.addSpacing(20)

        # Theme toggle
        self.theme_btn = QPushButton("☀ LIGHT")
        self.theme_btn.setFont(QFont("Consolas", 9))
        self.theme_btn.setFixedSize(90, 28)
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)

        self._header = header
        self._title_lbl = title
        self._subtitle_lbl = subtitle
        return header

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Controls ──
        ctrl = QFrame()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_layout.setSpacing(8)

        self._section_label(ctrl_layout, "MONITOR CONTROL")

        self.start_btn = QPushButton("▶  START MONITOR")
        self.start_btn.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.start_btn.setFixedHeight(40)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._toggle_live)
        ctrl_layout.addWidget(self.start_btn)

        self.csv_btn = QPushButton("⊞  LOAD CSV FILE")
        self.csv_btn.setFont(QFont("Consolas", 10))
        self.csv_btn.setFixedHeight(36)
        self.csv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.csv_btn.clicked.connect(self._load_csv)
        ctrl_layout.addWidget(self.csv_btn)

        self._section_label(ctrl_layout, "MODEL")
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Consolas", 9))
        self.model_combo.setFixedHeight(32)
        models = self.model_mgr.loaded if self.model_mgr.loaded else ["No models found"]
        self.model_combo.addItems(models)
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        ctrl_layout.addWidget(self.model_combo)

        self._section_label(ctrl_layout, "DETECTION THRESHOLD")
        thresh_row = QHBoxLayout()
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(10, 90)
        self.thresh_slider.setValue(50)
        self.thresh_slider.valueChanged.connect(self._on_thresh_change)
        self.thresh_val = QLabel("0.50")
        self.thresh_val.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.thresh_val.setFixedWidth(40)
        thresh_row.addWidget(self.thresh_slider)
        thresh_row.addWidget(self.thresh_val)
        ctrl_layout.addLayout(thresh_row)

        self._section_label(ctrl_layout, "ATTACK PROBABILITY (SIM)")
        prob_row = QHBoxLayout()
        self.prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.prob_slider.setRange(0, 100)
        self.prob_slider.setValue(15)
        self.prob_slider.valueChanged.connect(self._on_prob_change)
        self.prob_val = QLabel("15%")
        self.prob_val.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.prob_val.setFixedWidth(40)
        prob_row.addWidget(self.prob_slider)
        prob_row.addWidget(self.prob_val)
        ctrl_layout.addLayout(prob_row)

        self._section_label(ctrl_layout, "SPEED")
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(4)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        self.speed_val = QLabel("0.4s")
        self.speed_val.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.speed_val.setFixedWidth(40)
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_val)
        ctrl_layout.addLayout(speed_row)

        layout.addWidget(ctrl)

        # ── Stat cards ──
        self.card_total   = StatCard("Total Flows",   "0",      "#00d4ff")
        self.card_benign  = StatCard("Benign",        "0",      "#00e5a0")
        self.card_attacks = StatCard("Attacks",       "0",      "#ff3b5c")
        self.card_rate    = StatCard("Attack Rate",   "0.0%",   "#ffb300")

        for card in [self.card_total, self.card_benign,
                     self.card_attacks, self.card_rate]:
            layout.addWidget(card)

        layout.addStretch()

        # ── System log ──
        log_frame = QFrame()
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(12, 8, 12, 8)
        self._section_label(log_layout, "SYSTEM LOG")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 8))
        self.log_box.setFixedHeight(140)
        log_layout.addWidget(self.log_box)
        layout.addWidget(log_frame)

        self._sidebar = sidebar
        self._ctrl_frame = ctrl
        self._log_frame = log_frame
        return sidebar

    def _build_main(self):
        main = QFrame()
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Charts ──
        chart_row = QHBoxLayout()
        chart_row.setSpacing(12)

        # Traffic volume chart
        self.traffic_chart = pg.PlotWidget()
        self.traffic_chart.setFixedHeight(180)
        self.traffic_chart.setLabel("left",   "Flows/s")
        self.traffic_chart.setLabel("bottom", "Time")
        self.traffic_chart.showGrid(x=True, y=True)
        self.traffic_chart.setMouseEnabled(x=False, y=False)
        self._traffic_curve = self.traffic_chart.plot(
            pen=pg.mkPen(color="#00d4ff", width=2)
        )
        self._traffic_fill = pg.FillBetweenItem(
            self._traffic_curve,
            self.traffic_chart.plot([0] * self._chart_len),
            brush=pg.mkBrush(color=(0, 212, 255, 30))
        )
        self.traffic_chart.addItem(self._traffic_fill)

        # Attack rate chart
        self.attack_chart = pg.PlotWidget()
        self.attack_chart.setFixedHeight(180)
        self.attack_chart.setLabel("left",   "Attack %")
        self.attack_chart.setLabel("bottom", "Time")
        self.attack_chart.showGrid(x=True, y=True)
        self.attack_chart.setMouseEnabled(x=False, y=False)
        self.attack_chart.setYRange(0, 100)
        self._attack_curve = self.attack_chart.plot(
            pen=pg.mkPen(color="#ff3b5c", width=2)
        )
        self._attack_fill = pg.FillBetweenItem(
            self._attack_curve,
            self.attack_chart.plot([0] * self._chart_len),
            brush=pg.mkBrush(color=(255, 59, 92, 30))
        )
        self.attack_chart.addItem(self._attack_fill)

        chart_row.addWidget(self._wrap("TRAFFIC VOLUME", self.traffic_chart))
        chart_row.addWidget(self._wrap("ATTACK RATE (%)", self.attack_chart))
        layout.addLayout(chart_row)

        # ── Threat feed table ──
        feed_frame = QFrame()
        feed_layout = QVBoxLayout(feed_frame)
        feed_layout.setContentsMargins(12, 8, 12, 8)
        feed_layout.setSpacing(6)
        self._section_label(feed_layout, "LIVE THREAT FEED")

        self.feed_table = QTableWidget()
        self.feed_table.setColumnCount(8)
        self.feed_table.setHorizontalHeaderLabels([
            "TIME", "SRC IP", "DST IP", "PROTO",
            "PORT", "CLASSIFICATION", "CONFIDENCE", "TYPE"
        ])
        self.feed_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feed_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.feed_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.feed_table.setFont(QFont("Consolas", 9))
        self.feed_table.setAlternatingRowColors(True)
        self.feed_table.verticalHeader().setVisible(False)
        self.feed_table.setMaximumHeight(300)
        feed_layout.addWidget(self.feed_table)

        clr_btn = QPushButton("CLEAR FEED")
        clr_btn.setFont(QFont("Consolas", 8))
        clr_btn.setFixedSize(100, 24)
        clr_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clr_btn.clicked.connect(lambda: self.feed_table.setRowCount(0))
        feed_layout.addWidget(clr_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(feed_frame)

        self._main_frame = main
        self._feed_frame = feed_frame
        return main

    def _wrap(self, title: str, widget: QWidget) -> QFrame:
        f = QFrame()
        l = QVBoxLayout(f)
        l.setContentsMargins(12, 8, 12, 8)
        l.setSpacing(4)
        self._section_label(l, title)
        l.addWidget(widget)
        return f

    def _section_label(self, layout, text: str):
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        layout.addWidget(lbl)
        self._section_lbls = getattr(self, "_section_lbls", [])
        self._section_lbls.append(lbl)

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self):
        t = self._theme
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {t['bg']};
                color: {t['text']};
            }}
            QFrame {{
                background-color: {t['panel']};
                border: 1px solid {t['border']};
                border-radius: 8px;
            }}
            QLabel {{ background: transparent; border: none; color: {t['text']}; }}
            QPushButton {{
                background-color: {t['panel2']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background-color: {t['border']}; }}
            QComboBox {{
                background-color: {t['panel2']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {t['panel2']};
                color: {t['text']};
                selection-background-color: {t['accent']};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {t['border']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px;
                background: {t['accent']};
                border-radius: 7px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
            QTextEdit {{
                background-color: {t['chart_bg']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
            }}
            QTableWidget {{
                background-color: {t['chart_bg']};
                color: {t['text']};
                border: 1px solid {t['border']};
                gridline-color: {t['border']};
                alternate-background-color: {t['panel2']};
            }}
            QHeaderView::section {{
                background-color: {t['panel2']};
                color: {t['text2']};
                border: 1px solid {t['border']};
                font-family: Consolas;
                font-size: 9px;
                font-weight: bold;
                padding: 4px;
            }}
            QScrollBar:vertical {{
                background: {t['panel']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {t['border']};
                border-radius: 4px;
                min-height: 20px;
            }}
        """)

        # Header
        self._header.setStyleSheet(f"""
            QFrame {{
                background-color: {t['panel']};
                border: none;
                border-bottom: 1px solid {t['border']};
                border-radius: 0px;
            }}
        """)
        self._title_lbl.setStyleSheet(f"color: {t['accent']}; font-weight: bold;")
        self._subtitle_lbl.setStyleSheet(f"color: {t['text2']};")

        # Start button
        if self._live_running:
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t['danger2']};
                    color: white;
                    border: 1px solid {t['danger']};
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {t['danger']}; }}
            """)
        else:
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t['accent2']};
                    color: white;
                    border: 1px solid {t['accent']};
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {t['accent']}; }}
            """)

        # Status
        if self._live_running:
            self.status_dot.setStyleSheet(f"color: {t['success']}; border: none;")
            self.status_lbl.setStyleSheet(f"color: {t['success']}; border: none;")
        else:
            self.status_dot.setStyleSheet(f"color: {t['text2']}; border: none;")
            self.status_lbl.setStyleSheet(f"color: {t['text2']}; border: none;")

        # Stat cards
        self.card_total.apply_theme(t)
        self.card_benign.apply_theme(t)
        self.card_attacks.apply_theme(t)
        self.card_rate.apply_theme(t)

        # Section labels
        for lbl in getattr(self, "_section_lbls", []):
            lbl.setStyleSheet(f"color: {t['text2']}; background: transparent; border: none;")

        # Sliders
        self.thresh_val.setStyleSheet(f"color: {t['accent']}; border: none; background: transparent;")
        self.prob_val.setStyleSheet(f"color: {t['warning']}; border: none; background: transparent;")
        self.speed_val.setStyleSheet(f"color: {t['text2']}; border: none; background: transparent;")

        # Charts
        for chart in [self.traffic_chart, self.attack_chart]:
            chart.setBackground(t["chart_bg"])
            chart.getAxis("left").setPen(pg.mkPen(color=t["text2"]))
            chart.getAxis("bottom").setPen(pg.mkPen(color=t["text2"]))
            chart.getAxis("left").setTextPen(pg.mkPen(color=t["text2"]))
            chart.getAxis("bottom").setTextPen(pg.mkPen(color=t["text2"]))

        # Theme button
        self.theme_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['panel2']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
            }}
        """)

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._theme = DARK if self._dark_mode else LIGHT
        self.theme_btn.setText("☀ LIGHT" if self._dark_mode else "☾ DARK")
        self._apply_theme()

    # ── Live monitor ──────────────────────────────────────────────────────

    def _toggle_live(self):
        if self._live_running:
            self.worker.stop()
            self._live_running = False
            self.start_btn.setText("▶  START MONITOR")
            self.status_dot.setText("●")
            self.status_lbl.setText("OFFLINE")
            self._log("Monitor stopped", "warning")
        else:
            if not self.model_mgr.loaded:
                self._log("No models loaded — cannot start", "danger")
                return
            self.worker.model_name = self.model_combo.currentText()
            self.worker.threshold  = self.thresh_slider.value() / 100
            self.worker.start()
            self._live_running = True
            self.start_btn.setText("■  STOP MONITOR")
            self.status_dot.setText("●")
            self.status_lbl.setText("MONITORING")
            self._log(f"Monitor started — model: {self.worker.model_name}", "success")
        self._apply_theme()

    def _on_flow(self, flow: dict):
        self._total   += 1
        if flow["is_attack"]:
            self._attacks += 1
        else:
            self._benign  += 1

        rate = self._attacks / max(self._total, 1) * 100
        self.card_total.set_value(f"{self._total:,}")
        self.card_benign.set_value(f"{self._benign:,}")
        self.card_attacks.set_value(f"{self._attacks:,}")
        self.card_rate.set_value(f"{rate:.1f}%")

        self._traffic_data.append(1.0 / max(self.worker.interval, 0.01))
        self._attack_data.append(rate)

        # Insert row at top of feed
        self.feed_table.insertRow(0)
        t = self._theme

        items = [
            (flow["timestamp"],    t["text2"]),
            (flow["src_ip"],       t["text"]),
            (flow["dst_ip"],       t["text"]),
            (flow["proto"],        t["accent"]),
            (str(flow["dst_port"]),t["text"]),
            ("⚠ ATTACK" if flow["is_attack"] else "✓ BENIGN",
             t["danger"] if flow["is_attack"] else t["success"]),
            (f"{flow['confidence']:.3f}", t["text"]),
            (flow["attack_type"],  t["warning"] if flow["is_attack"] else t["text2"]),
        ]

        for col, (text, colour) in enumerate(items):
            item = QTableWidgetItem(text)
            item.setForeground(QColor(colour))
            item.setFont(QFont("Consolas", 9))
            self.feed_table.setItem(0, col, item)

        # Keep max 200 rows
        if self.feed_table.rowCount() > 200:
            self.feed_table.removeRow(200)

    def _update_chart(self):
        x = list(range(self._chart_len))
        self._traffic_curve.setData(x, list(self._traffic_data))
        self._attack_curve.setData(x, list(self._attack_data))

    # ── CSV mode ──────────────────────────────────────────────────────────

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", ROOT, "CSV Files (*.csv)"
        )
        if not path:
            return
        self._log(f"Loading {os.path.basename(path)} ...", "accent")

        def worker():
            try:
                import pandas as pd
                df = pd.read_csv(path, low_memory=False)
                df.columns = df.columns.str.strip()
                feat_cols = [c for c in FEATURES if c in df.columns]
                if not feat_cols:
                    self._log("No matching features found in CSV", "danger")
                    return
                X = df[feat_cols].fillna(0).values
                model_name = self.model_combo.currentText()
                threshold  = self.thresh_slider.value() / 100
                preds, scores = self.model_mgr.predict(X, model_name, threshold)

                label_col = "Attack Type" if "Attack Type" in df.columns else None
                n_attack = int(preds.sum())
                n_total  = len(preds)

                self.feed_table.setRowCount(0)
                for i in range(min(n_total, 500)):
                    self.feed_table.insertRow(self.feed_table.rowCount())
                    t = self._theme
                    is_atk = bool(preds[i])
                    true_label = str(df[label_col].iloc[i]) if label_col else "?"
                    items = [
                        (str(i),                             t["text2"]),
                        ("—",                                t["text2"]),
                        ("—",                                t["text2"]),
                        ("—",                                t["text2"]),
                        ("—",                                t["text2"]),
                        ("⚠ ATTACK" if is_atk else "✓ BENIGN",
                         t["danger"] if is_atk else t["success"]),
                        (f"{scores[i]:.3f}",                 t["text"]),
                        (true_label,                         t["warning"] if is_atk else t["text2"]),
                    ]
                    row = self.feed_table.rowCount() - 1
                    for col, (text, colour) in enumerate(items):
                        item = QTableWidgetItem(text)
                        item.setForeground(QColor(colour))
                        item.setFont(QFont("Consolas", 9))
                        self.feed_table.setItem(row, col, item)

                self._total   = n_total
                self._attacks = n_attack
                self._benign  = n_total - n_attack
                rate = n_attack / max(n_total, 1) * 100
                self.card_total.set_value(f"{n_total:,}")
                self.card_benign.set_value(f"{self._benign:,}")
                self.card_attacks.set_value(f"{n_attack:,}")
                self.card_rate.set_value(f"{rate:.1f}%")
                self._log(f"CSV classified — {n_attack:,}/{n_total:,} attacks ({rate:.1f}%)", "success")

            except Exception as e:
                self._log(f"CSV error: {e}", "danger")

        threading.Thread(target=worker, daemon=True).start()

    # ── Controls ──────────────────────────────────────────────────────────

    def _on_model_change(self, name: str):
        self.worker.model_name = name
        self._log(f"Model switched to {name}", "accent")

    def _on_thresh_change(self, val: int):
        self.thresh_val.setText(f"{val/100:.2f}")
        self.worker.threshold = val / 100

    def _on_prob_change(self, val: int):
        self.prob_val.setText(f"{val}%")
        self.worker.attack_prob = val / 100

    def _on_speed_change(self, val: int):
        interval = round(1.0 / val, 2)
        self.speed_val.setText(f"{interval}s")
        self.worker.interval = interval

    # ── Log ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "text"):
        t = self._theme
        colours = {
            "accent":  t["accent"],
            "success": t["success"],
            "danger":  t["danger"],
            "warning": t["warning"],
            "text":    t["text"],
        }
        colour = colours.get(level, t["text"])
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(
            f'<span style="color:{t["text2"]}">[{ts}]</span> '
            f'<span style="color:{colour}">{msg}</span>'
        )

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────

def launch():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = IDSDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
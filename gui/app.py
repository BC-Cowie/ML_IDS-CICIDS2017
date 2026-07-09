"""
gui/app.py
Tkinter GUI for the ML IDS.
- Load CSV → run binary prediction → show results table
- Switch to multiclass mode for attack type breakdown
- Manual single-flow input for live demo
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.predict import predict_binary, predict_multiclass
from src.preprocess import load_scaler, get_features, clean
from src.models import random_forest


# ── Colour palette ─────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
TEXT     = "#c9d1d9"
ACCENT   = "#3b82f6"
DANGER   = "#ef4444"
SUCCESS  = "#10b981"
WARNING  = "#f59e0b"
BTN_BG   = "#21262d"
BTN_HOV  = "#30363d"

FONT_MONO = ("Courier New", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_HEAD = ("Segoe UI", 12, "bold")
FONT_TITLE= ("Segoe UI", 16, "bold")


class IDSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Intrusion Detection System — CICIDS2017")
        self.geometry("1100x720")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._df          = None
        self._results     = None
        self._scaler      = None
        self._feature_names = None
        self._label_encoder = None
        self._model_name  = tk.StringVar(value="random_forest")
        self._mode        = tk.StringVar(value="binary")
        self._threshold   = tk.DoubleVar(value=0.5)
        self._strategy    = tk.StringVar(value="majority")

        self._build_ui()
        self._try_load_scaler()

    # ── Scaler ──────────────────────────────────────────────────────────────

    def _try_load_scaler(self):
        try:
            self._scaler = load_scaler()
            self._log("Scaler loaded from disk.", colour=SUCCESS)
        except Exception:
            self._log("No saved scaler found — train models first.", colour=WARNING)

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=PANEL, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="ML Intrusion Detection System",
                 font=FONT_TITLE, bg=PANEL, fg=TEXT).pack(side="left", padx=20)
        tk.Label(header, text="CICIDS2017",
                 font=FONT_BODY, bg=PANEL, fg=ACCENT).pack(side="left")

        # Main body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left  = tk.Frame(body, bg=BG, width=280)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_results(right)

    def _build_controls(self, parent):
        self._section(parent, "Data")
        self._btn(parent, "Load CSV file", self._load_csv)
        self._btn(parent, "Run prediction", self._run_prediction)

        self._section(parent, "Settings")

        tk.Label(parent, text="Model", bg=BG, fg=TEXT,
                 font=FONT_BODY).pack(anchor="w", pady=(4, 0))
        model_opts = ["random_forest", "xgboost", "svm", "neural_network", "ensemble"]
        ttk.Combobox(parent, textvariable=self._model_name,
                     values=model_opts, state="readonly").pack(fill="x", pady=(0, 8))

        tk.Label(parent, text="Mode", bg=BG, fg=TEXT,
                 font=FONT_BODY).pack(anchor="w")
        for val, txt in [("binary", "Binary (BENIGN / ATTACK)"),
                          ("multiclass", "Multiclass (attack type)")]:
            tk.Radiobutton(parent, text=txt, variable=self._mode,
                           value=val, bg=BG, fg=TEXT, selectcolor=PANEL,
                           activebackground=BG, activeforeground=ACCENT,
                           font=FONT_BODY).pack(anchor="w")

        tk.Label(parent, text=f"Recall threshold: {self._threshold.get():.2f}",
                 bg=BG, fg=TEXT, font=FONT_BODY).pack(anchor="w", pady=(8, 0))
        thresh_lbl = tk.Label(parent, text=f"{self._threshold.get():.2f}",
                               bg=BG, fg=ACCENT, font=FONT_BODY)
        thresh_lbl.pack(anchor="e")

        def on_thresh(val):
            thresh_lbl.config(text=f"{float(val):.2f}")

        tk.Scale(parent, variable=self._threshold,
                 from_=0.1, to=0.9, resolution=0.05,
                 orient="horizontal", bg=BG, fg=TEXT,
                 troughcolor=PANEL, highlightthickness=0,
                 command=on_thresh).pack(fill="x")

        tk.Label(parent, text="Ensemble strategy", bg=BG, fg=TEXT,
                 font=FONT_BODY).pack(anchor="w", pady=(8, 0))
        for val, txt in [("majority", "Majority vote"),
                          ("any", "Any (max recall)"),
                          ("all", "All agree (max precision)")]:
            tk.Radiobutton(parent, text=txt, variable=self._strategy,
                           value=val, bg=BG, fg=TEXT, selectcolor=PANEL,
                           activebackground=BG, activeforeground=ACCENT,
                           font=FONT_BODY).pack(anchor="w")

        self._section(parent, "Manual flow")
        self._btn(parent, "Enter single flow", self._single_flow_dialog)

        self._section(parent, "Log")
        self._log_box = tk.Text(parent, height=8, bg=PANEL, fg=TEXT,
                                 font=FONT_MONO, relief="flat",
                                 state="disabled", wrap="word")
        self._log_box.pack(fill="x")
        self._log_box.tag_config("SUCCESS", foreground=SUCCESS)
        self._log_box.tag_config("DANGER",  foreground=DANGER)
        self._log_box.tag_config("WARNING", foreground=WARNING)
        self._log_box.tag_config("ACCENT",  foreground=ACCENT)

    def _build_results(self, parent):
        # Stats bar
        self._stats_frame = tk.Frame(parent, bg=PANEL, pady=8)
        self._stats_frame.pack(fill="x", pady=(0, 12))
        self._stat_labels = {}
        for key in ["Total", "BENIGN", "ATTACK", "Attack rate"]:
            f = tk.Frame(self._stats_frame, bg=PANEL, padx=16)
            f.pack(side="left")
            tk.Label(f, text=key, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 9)).pack()
            lbl = tk.Label(f, text="—", bg=PANEL, fg=ACCENT,
                           font=("Segoe UI", 14, "bold"))
            lbl.pack()
            self._stat_labels[key] = lbl

        # Results table
        cols = ("Index", "Label", "Confidence", "Details")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings",
                                   selectmode="browse")
        for col in cols:
            self._tree.heading(col, text=col)
        self._tree.column("Index",      width=60,  anchor="center")
        self._tree.column("Label",      width=120, anchor="center")
        self._tree.column("Confidence", width=100, anchor="center")
        self._tree.column("Details",    width=300)

        self._tree.tag_configure("ATTACK",  background="#3b1010", foreground=DANGER)
        self._tree.tag_configure("BENIGN",  background="#0f2818", foreground=SUCCESS)
        self._tree.tag_configure("UNKNOWN", background=PANEL,     foreground=TEXT)

        scroll = ttk.Scrollbar(parent, orient="vertical",
                                command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._style_treeview()

    def _style_treeview(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
                         background=PANEL, foreground=TEXT,
                         fieldbackground=PANEL, rowheight=28,
                         font=FONT_BODY)
        style.configure("Treeview.Heading",
                         background=BTN_BG, foreground=TEXT,
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

    # ── Actions ─────────────────────────────────────────────────────────────

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self._df = pd.read_csv(path, low_memory=False)
            self._df.columns = self._df.columns.str.strip()
            if config.LABEL_COL in self._df.columns:
                self._df[config.LABEL_COL] = self._df[config.LABEL_COL].str.strip()
            self._feature_names = get_features(self._df)
            self._log(f"Loaded {len(self._df):,} rows from {os.path.basename(path)}", "SUCCESS")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self._log(f"Error: {e}", "DANGER")

    def _run_prediction(self):
        if self._df is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return
        if self._scaler is None:
            messagebox.showwarning("No scaler", "Train models first (python main.py).")
            return

        self._log("Running prediction ...", "ACCENT")
        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            X_raw = self._df[self._feature_names].values
            mode  = self._mode.get()

            if self._model_name.get() == "ensemble":
                from src.predict import ensemble_predict
                result = ensemble_predict(X_raw, strategy=self._strategy.get(),
                                          threshold=self._threshold.get())
                labels = result["labels"]
                confs  = result["confidence"]
                details = [f"Strategy: {self._strategy.get()}"] * len(labels)

            elif mode == "binary":
                result = predict_binary(X_raw, self._model_name.get(),
                                         threshold=self._threshold.get())
                labels  = result["labels"]
                confs   = result["confidence"]
                details = [""] * len(labels)

            else:
                result = predict_multiclass(X_raw, self._model_name.get(),
                                             self._label_encoder)
                labels  = ["ATTACK" if c != config.BENIGN_LABEL else "BENIGN"
                           for c in result["class_names"]]
                confs   = [1.0] * len(labels)
                details = list(result["class_names"])

            self.after(0, self._update_results, labels, confs, details)

        except Exception as e:
            self.after(0, self._log, f"Prediction error: {e}", "DANGER")

    def _update_results(self, labels, confs, details):
        # Clear table
        for row in self._tree.get_children():
            self._tree.delete(row)

        n_attack = sum(1 for l in labels if l == "ATTACK")
        n_benign = len(labels) - n_attack

        self._stat_labels["Total"].config(text=f"{len(labels):,}")
        self._stat_labels["BENIGN"].config(text=f"{n_benign:,}", fg=SUCCESS)
        self._stat_labels["ATTACK"].config(text=f"{n_attack:,}", fg=DANGER)
        self._stat_labels["Attack rate"].config(
            text=f"{n_attack/max(len(labels),1):.1%}",
            fg=DANGER if n_attack > 0 else SUCCESS
        )

        for i, (label, conf, detail) in enumerate(zip(labels, confs, details)):
            tag = "ATTACK" if label == "ATTACK" else "BENIGN"
            self._tree.insert("", "end",
                              values=(i, label, f"{conf:.3f}", detail),
                              tags=(tag,))

        self._log(f"Done — {n_attack:,} attacks detected ({n_attack/max(len(labels),1):.1%})",
                  "SUCCESS" if n_attack == 0 else "DANGER")

    def _single_flow_dialog(self):
        if not self._feature_names:
            messagebox.showwarning("No features", "Load a CSV first to get feature names.")
            return

        dialog = tk.Toplevel(self, bg=BG)
        dialog.title("Manual flow entry")
        dialog.geometry("500x500")

        tk.Label(dialog, text="Enter feature values (leave blank = 0)",
                 bg=BG, fg=TEXT, font=FONT_HEAD).pack(pady=8)

        canvas = tk.Canvas(dialog, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        frame  = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        entries = {}
        for feat in self._feature_names[:30]:  # show first 30
            row = tk.Frame(frame, bg=BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=feat, bg=BG, fg=TEXT, font=FONT_MONO,
                     width=30, anchor="w").pack(side="left")
            e = tk.Entry(row, bg=PANEL, fg=TEXT, font=FONT_MONO,
                         insertbackground=TEXT, width=12)
            e.pack(side="left")
            entries[feat] = e

        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        def submit():
            flow = {f: float(e.get() or 0) for f, e in entries.items()}
            try:
                from src.predict import predict_single_flow
                result = predict_single_flow(
                    flow, self._feature_names,
                    model_name=self._model_name.get(),
                    threshold=self._threshold.get()
                )
                label = result["labels"][0]
                conf  = result["confidence"][0]
                colour = DANGER if label == "ATTACK" else SUCCESS
                messagebox.showinfo(
                    "Prediction",
                    f"Result: {label}\nConfidence: {conf:.4f}"
                )
                dialog.destroy()
            except Exception as ex:
                messagebox.showerror("Error", str(ex))

        self._btn(dialog, "Predict", submit)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(12, 4))
        tk.Label(f, text=text.upper(), bg=BG, fg=BORDER,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=8)

    def _btn(self, parent, text, command):
        b = tk.Button(
            parent, text=text, command=command,
            bg=BTN_BG, fg=TEXT, font=FONT_BODY,
            relief="flat", padx=10, pady=6,
            activebackground=BTN_HOV, activeforeground=TEXT,
            cursor="hand2",
        )
        b.pack(fill="x", pady=2)
        return b

    def _log(self, msg: str, colour: str = None):
        self._log_box.config(state="normal")
        tag = None
        if colour == SUCCESS: tag = "SUCCESS"
        elif colour == DANGER: tag = "DANGER"
        elif colour == WARNING: tag = "WARNING"
        elif colour == ACCENT: tag = "ACCENT"
        self._log_box.insert("end", f"» {msg}\n", tag or "")
        self._log_box.see("end")
        self._log_box.config(state="disabled")


def launch():
    app = IDSApp()
    app.mainloop()


if __name__ == "__main__":
    launch()

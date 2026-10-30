import sys
import time
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from collections import deque

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout,
    QGridLayout, QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

FINGER_NAMES = ["Thumb","Index","Middle","Ring","Pinky"]
FINGER_TIPS = [4,8,12,16,20]

NUM_FINGERS = 5
FPS_ASSUMED = 30
ROLLING_WINDOW = 60
SIGNAL_BUFFER_SIZE = 300


class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Tracking")
        self.resize(1400,900)
        layout = QVBoxLayout()
        self.video_label = QLabel("Camera Feed (embedded backup)")
        self.video_label.setMinimumSize(1280,720)
        layout.addWidget(self.video_label)
        self.status_label = QLabel("Real-time monitoring running")
        layout.addWidget(self.status_label)
        self.setLayout(layout)


class TremorGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tremor Detection Dashboard v2")
        self.resize(1900,1100)

        self.cap = cv2.VideoCapture(2)
        
        self.alpha = 0.5
        self.mm_per_pixel = 0.2

        self.prev_xy = [None]*NUM_FINGERS
        self.prev_filtered = [0]*NUM_FINGERS

        self.start_time = time.time()

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.signal_buffers = [
            deque(maxlen=SIGNAL_BUFFER_SIZE)
            for _ in range(NUM_FINGERS)
        ]

        self.time_history = [[] for _ in range(NUM_FINGERS)]
        self.mm_history = [[] for _ in range(NUM_FINGERS)]
        self.hz_history = [[] for _ in range(NUM_FINGERS)]

        self.classification_start_time = time.time()
        self.class_mm_history = [[] for _ in range(NUM_FINGERS)]
        self.class_hz_history = [[] for _ in range(NUM_FINGERS)]

        self.camera_window = CameraWindow()
        self.camera_window.show()

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):

        root = QVBoxLayout()

        self.status_label = QLabel("Real-time monitoring running")
        root.addWidget(self.status_label)
        self.metrics_label = QLabel("Classification: Normal | Mean Freq: 0 | Peak Freq: 0 | RMS: 0 | Dominant Finger: -")
        root.addWidget(self.metrics_label)

        grid = QGridLayout()

        self.mm_curves = []
        self.hz_curves = []
        self.mm_plots = []
        self.hz_plots = []

        for i,name in enumerate(FINGER_NAMES):

            p1 = pg.PlotWidget(title=f"{name} - mm vs Time")
            p2 = pg.PlotWidget(title=f"{name} - Hz vs Time")

            p1.setLabel("left","mm")
            p1.setLabel("bottom","Time (s)")

            p2.setLabel("left","Hz")
            p2.setLabel("bottom","Time (s)")

            c1 = p1.plot()
            c2 = p2.plot()

            self.mm_curves.append(c1)
            self.hz_curves.append(c2)

            self.mm_plots.append(p1)
            self.hz_plots.append(p2)

            grid.addWidget(p1,i,0)
            grid.addWidget(p2,i,1)

        root.addLayout(grid)

        btn_layout = QHBoxLayout()

        self.btn_calibrate = QPushButton("Calibrate ATM Card")
        self.btn_export = QPushButton("Export CSV")
        self.btn_png = QPushButton("Export PNG")
        self.btn_pdf = QPushButton("Export PDF")
        self.btn_quit = QPushButton("Quit")

        self.btn_calibrate.clicked.connect(self.calibrate_scale)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_png.clicked.connect(self.export_png)
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_quit.clicked.connect(self.close)

        for b in [
            self.btn_calibrate,
            self.btn_export,
            self.btn_png,
            self.btn_pdf,
            self.btn_quit
        ]:
            btn_layout.addWidget(b)

        root.addLayout(btn_layout)
        self.setLayout(root)
    def compute_fft(self, signal):

        if len(signal) < 30:
            return np.nan

        sig = np.array(signal)
        sig -= np.mean(sig)

        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1/FPS_ASSUMED)

        mask = ((freqs >= 2) & (freqs <= 12))

        if not np.any(mask):
            return np.nan

        return freqs[mask][np.argmax(fft_vals[mask])]

    def update_tracking(self, frame):

        values = [0]*NUM_FINGERS

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return values

        h,w,_ = frame.shape

        hand = result.multi_hand_landmarks[0]

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand,
            self.mp_hands.HAND_CONNECTIONS
        )

        wrist = hand.landmark[0]

        for i,tip_id in enumerate(FINGER_TIPS):

            lm = hand.landmark[tip_id]

            x_mm = ((lm.x-wrist.x)*w)*self.mm_per_pixel
            y_mm = ((lm.y-wrist.y)*h)*self.mm_per_pixel

            if self.prev_xy[i] is not None:

                dx = x_mm-self.prev_xy[i][0]
                dy = y_mm-self.prev_xy[i][1]

                mag = np.sqrt(dx**2+dy**2)

                filt = (
                    self.alpha*self.prev_filtered[i]
                    +(1-self.alpha)*mag
                )

                self.prev_filtered[i] = filt
                values[i] = filt

            self.prev_xy[i] = (x_mm,y_mm)

        return values

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return


        mm_values = self.update_tracking(frame)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        h,w,ch = rgb.shape

        img = QImage(
            rgb.data,
            w,h,
            ch*w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(img)
        self.camera_window.video_label.setPixmap(pix)

        t = time.time() - self.start_time

        for i in range(NUM_FINGERS):

            mm = mm_values[i]

            self.signal_buffers[i].append(mm)

            hz = self.compute_fft(self.signal_buffers[i])

            self.time_history[i].append(t)
            self.mm_history[i].append(mm)
            self.hz_history[i].append(hz)

            self.class_mm_history[i].append(mm)
            self.class_hz_history[i].append(hz)

            while (
                len(self.time_history[i]) > 0 and
                t - self.time_history[i][0] > ROLLING_WINDOW
            ):
                self.time_history[i].pop(0)
                self.mm_history[i].pop(0)
                self.hz_history[i].pop(0)

            self.mm_curves[i].setData(
                self.time_history[i],
                self.mm_history[i]
            )

            self.hz_curves[i].setData(
                self.time_history[i],
                self.hz_history[i]
            )

            self.mm_plots[i].setXRange(
                max(0,t-ROLLING_WINDOW),
                t,
                padding=0
            )

            self.hz_plots[i].setXRange(
                max(0,t-ROLLING_WINDOW),
                t,
                padding=0
            )



        mean_freqs = [np.nanmean(h) if len(h) else np.nan for h in self.class_hz_history]
        peak_freqs = [np.nanmax(h) if len(h) else np.nan for h in self.class_hz_history]
        rms_vals = [np.sqrt(np.mean(np.square(m))) if len(m) else 0 for m in self.class_mm_history]

        valid_mean = np.nanmean(mean_freqs) if np.any(~np.isnan(mean_freqs)) else 0
        valid_peak = np.nanmax(peak_freqs) if np.any(~np.isnan(peak_freqs)) else 0
        valid_rms = max(rms_vals) if len(rms_vals) else 0

        dominant_idx = int(np.argmax(rms_vals)) if len(rms_vals) else 0
        dominant_finger = FINGER_NAMES[dominant_idx]

        tremor_class = self.classify_tremor(valid_peak)

        self.metrics_label.setText(
            f"Classification: {tremor_class} | "
            f"Mean Frequency: {valid_mean:.2f} Hz | "
            f"Peak Frequency: {valid_peak:.2f} Hz | "
            f"RMS Displacement: {valid_rms:.3f} mm | "
            f"Dominant Finger: {dominant_finger}"
        )

        if time.time() - self.classification_start_time >= 2:
            self.classification_start_time = time.time()
            self.class_mm_history = [[] for _ in range(NUM_FINGERS)]
            self.class_hz_history = [[] for _ in range(NUM_FINGERS)]

    def classify_tremor(self, freq):
        if np.isnan(freq) or freq < 2:
            return "Normal"
        elif freq < 4:
            return "Mild Tremor"
        elif freq < 8:
            return "Moderate Tremor"
        else:
            return "Severe Tremor"

    def calibrate_scale(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        pts=[]

        def click(event,x,y,flags,param):
            if event==cv2.EVENT_LBUTTONDOWN:
                pts.append((x,y))

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration",click)

        while True:

            temp = frame.copy()

            for p in pts:
                cv2.circle(temp,p,5,(0,0,255),-1)

            cv2.imshow("Calibration",temp)

            if len(pts)==2:
                break

            if cv2.waitKey(1)==ord("q"):
                break

        cv2.destroyAllWindows()

        if len(pts)==2:

            dist = np.linalg.norm(
                np.array(pts[0]) - np.array(pts[1])
            )

            self.mm_per_pixel = 85.6/dist

    def export_csv(self):

        path,_ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "tremor.csv",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        df = pd.DataFrame()

        for i,name in enumerate(FINGER_NAMES):

            df[f"{name}_time"] = pd.Series(self.time_history[i])
            df[f"{name}_mm"] = pd.Series(self.mm_history[i])
            df[f"{name}_hz"] = pd.Series(self.hz_history[i])

        df.to_csv(path,index=False)

    def export_png(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder"
        )

        if not folder:
            return

        for i,name in enumerate(FINGER_NAMES):

            ImageExporter(
                self.mm_plots[i].plotItem
            ).export(
                os.path.join(folder,f"{name}_mm.png")
            )

            ImageExporter(
                self.hz_plots[i].plotItem
            ).export(
                os.path.join(folder,f"{name}_hz.png")
            )

        QMessageBox.information(
            self,
            "Done",
            "10 graph PNG files exported."
        )

    def export_pdf(self):

        path,_ = QFileDialog.getSaveFileName(
            self,
            "Save PDF",
            "report.pdf",
            "PDF Files (*.pdf)"
        )

        if not path:
            return

        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()

        story = [
            Paragraph(
                "Tremor Analysis Report",
                styles["Title"]
            )
        ]

        for i,name in enumerate(FINGER_NAMES):

            hz = np.nanmean(self.hz_history[i])

            rms = np.sqrt(
                np.mean(
                    np.square(self.mm_history[i])
                )
            ) if len(self.mm_history[i]) else 0

            story.append(
                Paragraph(
                    f"{name}: Mean Hz={hz:.2f}, RMS={rms:.3f} mm",
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1,12))

        doc.build(story)

    def closeEvent(self,event):

        self.cap.release()
        self.hands.close()

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    pg.setConfigOptions(
        antialias=True
    )

    win = TremorGUI()
    win.show()

    sys.exit(app.exec_())

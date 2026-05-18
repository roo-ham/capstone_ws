#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
area_multicamera_hybrid_pi_v1.py

Hybrid Raspberry Pi version:
- MIPI/CSI cameras are opened with Picamera2/libcamera.
- USB cameras are opened with OpenCV/V4L2.
- Area-only multi-circle optical tactile tracker.
- Works with 1, 2, or 3 sensors.
- Each sensor has separate trapezoid ROI, zero, threshold, and calibration.
- Config save/load included.

Typical Raspberry Pi 5 setup:
    python3 area_multicamera_hybrid_pi_v1.py --sources mipi0 mipi1 usb16 --width 640 --height 480 --fps 60

USB only:
    python3 area_multicamera_hybrid_pi_v1.py --sources usb16 --width 640 --height 480 --fps 60

MIPI only:
    python3 area_multicamera_hybrid_pi_v1.py --sources mipi0 mipi1 --width 640 --height 480 --fps 60

Source format:
    mipi0, mipi1, mipi2 ...   -> Picamera2 camera_num
    usb16, usb0, usb2 ...     -> /dev/video16, /dev/video0, /dev/video2 via OpenCV

Keys:
    1 / 2 / 3 : select active sensor
    t         : set trapezoid ROI for active sensor, click LT -> RT -> RB -> LB
    z         : zero active sensor
    Z         : zero all sensors
    a         : add calibration sample for active sensor using current calib weight
    n         : next calibration weight
    p         : previous calibration weight
    f         : fit active sensor force calibration
    F         : fit all sensor force calibrations
    s         : save config
    l         : load config
    d         : debug binary view on/off for active sensor
    c         : clear active sensor ROI/zero
    q         : quit

Install:
    sudo apt update
    sudo apt install -y python3-opencv python3-numpy python3-picamera2 v4l-utils
"""

import argparse
import csv
import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False


@dataclass
class AreaResult:
    valid: bool
    n_detected: int
    total_area: float
    mean_area: float
    max_area: float
    delta_total_area: float
    delta_mean_area: float
    delta_max_area: float
    raw_score: float
    score: float
    force_gf: float
    markers: list
    binary: np.ndarray


class HybridCamera:
    def __init__(self, source: str, width: int, height: int, fps: float):
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.kind, self.index = self.parse_source(source)
        self.cap = None
        self.picam = None

    @staticmethod
    def parse_source(source):
        s = source.strip().lower()
        if s.startswith("mipi"):
            return "mipi", int(s.replace("mipi", ""))
        if s.startswith("csi"):
            return "mipi", int(s.replace("csi", ""))
        if s.startswith("usb"):
            return "usb", int(s.replace("usb", ""))
        if s.startswith("video"):
            return "usb", int(s.replace("video", ""))
        # plain number = USB/OpenCV fallback
        return "usb", int(s)

    def open(self):
        if self.kind == "mipi":
            return self._open_mipi()
        return self._open_usb()

    def _open_mipi(self):
        if not PICAMERA2_AVAILABLE:
            print("[ERROR] Picamera2 not available. Install: sudo apt install python3-picamera2")
            return False

        try:
            self.picam = Picamera2(camera_num=self.index)
            config = self.picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps},
            )
            self.picam.configure(config)
            self.picam.start()
            time.sleep(0.4)

            frame = self.picam.capture_array()
            if frame is None or frame.size == 0:
                print(f"[WARN] MIPI camera {self.index} opened but no frame")
                self.close()
                return False

            print(f"[CAM] opened {self.source}: Picamera2 {self.width}x{self.height}@{self.fps}")
            return True

        except Exception as e:
            print(f"[WARN] failed to open {self.source} with Picamera2: {e}")
            self.close()
            return False

    def _open_usb(self):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        fourcc_list = ["MJPG", "YUYV", ""]
        fps_list = []
        for f in [self.fps, 60, 30, 15]:
            if f not in fps_list:
                fps_list.append(f)
        res_list = []
        for wh in [(self.width, self.height), (640, 480), (320, 240)]:
            if wh not in res_list:
                res_list.append(wh)

        for backend in backends:
            for fourcc in fourcc_list:
                for width, height in res_list:
                    for fps in fps_list:
                        cap = cv2.VideoCapture(self.index, backend)
                        if not cap.isOpened():
                            cap.release()
                            continue

                        if fourcc:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                        cap.set(cv2.CAP_PROP_FPS, float(fps))
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        ok = False
                        frame = None
                        for _ in range(10):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                ok = True
                                break

                        if ok:
                            self.cap = cap
                            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                            print(
                                f"[CAM] opened {self.source}: OpenCV/V4L2 "
                                f"req={width}x{height}@{fps} fourcc={fourcc or 'default'} "
                                f"actual={actual_w:.0f}x{actual_h:.0f}@{actual_fps:.1f} {fourcc_str}"
                            )
                            return True

                        cap.release()

        print(f"[WARN] failed to read USB camera /dev/video{self.index}")
        return False

    def read(self):
        if self.kind == "mipi":
            if self.picam is None:
                return False, None
            try:
                frame = self.picam.capture_array()
                if frame is None or frame.size == 0:
                    return False, None
                # Picamera2 config is RGB888. Convert to BGR for OpenCV consistency.
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception:
                return False, None

        if self.cap is None:
            return False, None
        return self.cap.read()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.picam is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
            try:
                self.picam.close()
            except Exception:
                pass
            self.picam = None


class AreaSensor:
    def __init__(self, sensor_id, source, args):
        self.sensor_id = sensor_id
        self.source = source
        self.args = args
        self.camera = HybridCamera(source, args.width, args.height, args.fps)

        self.trap_src = None
        self.trap_points = []
        self.select_trap = False
        self.M_warp = None
        self.warp_size = tuple(args.warp_size)

        self.threshold = args.threshold
        self.min_area = args.min_area
        self.max_area = args.max_area
        self.open_iter = args.open_iter
        self.close_iter = args.close_iter
        self.blur_ksize = args.blur_ksize
        self.smooth_alpha = args.smooth_alpha

        self.total_area_s = None
        self.mean_area_s = None
        self.max_area_s = None
        self.score_s = 0.0

        self.zero_total_area = None
        self.zero_mean_area = None
        self.zero_max_area = None
        self.zero_score = None

        self.cal_k = args.default_cal_k
        self.cal_b = args.default_cal_b
        self.cal_samples = []

        self.last_result = None
        self.last_frame_gray = None
        self.last_roi_gray = None
        self.last_fps = 0.0
        self._read_fail_count = 0

    def open_camera(self):
        return self.camera.open()

    def release(self):
        self.camera.close()

    def compute_warp(self):
        W, H = self.warp_size
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        self.M_warp = cv2.getPerspectiveTransform(self.trap_src.astype(np.float32), dst)
        print(f"[S{self.sensor_id}] [WARP] trapezoid -> {W}x{H}")

    def get_roi(self, gray):
        if self.M_warp is not None:
            W, H = self.warp_size
            return cv2.warpPerspective(gray, self.M_warp, (W, H))
        return gray

    def mouse_callback(self, event, x, y, flags, param):
        if not self.select_trap:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            labels = ["LT", "RT", "RB", "LB"]
            idx = len(self.trap_points)
            if idx < 4:
                self.trap_points.append((x, y))
                print(f"[S{self.sensor_id}] [TRAP] {labels[idx]} = ({x}, {y})")
            if len(self.trap_points) == 4:
                self.trap_src = np.array(self.trap_points, dtype=np.float32)
                self.compute_warp()
                self.select_trap = False
                self.trap_points = []
                self.reset_state(clear_roi=False)
                print(f"[S{self.sensor_id}] [TRAP] done. Press z at unloaded/preload state.")

    def reset_state(self, clear_roi=False):
        self.total_area_s = None
        self.mean_area_s = None
        self.max_area_s = None
        self.score_s = 0.0
        self.zero_total_area = None
        self.zero_mean_area = None
        self.zero_max_area = None
        self.zero_score = None
        self.last_result = None

        if clear_roi:
            self.trap_src = None
            self.M_warp = None
            self.trap_points = []
            self.select_trap = False

    def zero_current(self):
        if self.last_result is None or not self.last_result.valid:
            print(f"[S{self.sensor_id}] [ZERO] failed: no valid marker")
            return

        self.zero_total_area = self.last_result.total_area
        self.zero_mean_area = self.last_result.mean_area
        self.zero_max_area = self.last_result.max_area
        self.zero_score = self.last_result.raw_score
        self.score_s = 0.0

        print(
            f"[S{self.sensor_id}] [ZERO] total={self.zero_total_area:.2f}, "
            f"mean={self.zero_mean_area:.2f}, max={self.zero_max_area:.2f}, "
            f"score={self.zero_score:.2f}"
        )

    def score_to_force_gf(self, score):
        return float(self.cal_k * score + self.cal_b)

    def add_calibration_sample(self, force_gf):
        if self.last_result is None or not self.last_result.valid:
            print(f"[S{self.sensor_id}] [CAL] failed: no valid score")
            return
        sample = {
            "time_s": time.time(),
            "score": float(self.last_result.score),
            "force_gf": float(force_gf),
        }
        self.cal_samples.append(sample)
        print(
            f"[S{self.sensor_id}] [CAL] sample #{len(self.cal_samples)}: "
            f"score={sample['score']:.3f}, force={sample['force_gf']:.1f} gf"
        )

    def fit_force_calibration(self):
        if len(self.cal_samples) < 2:
            print(f"[S{self.sensor_id}] [CAL] need at least 2 samples")
            return False

        x = np.array([s["score"] for s in self.cal_samples], dtype=np.float64)
        y = np.array([s["force_gf"] for s in self.cal_samples], dtype=np.float64)

        A = np.vstack([x, np.ones_like(x)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        self.cal_k = float(k)
        self.cal_b = float(b)

        pred = self.cal_k * x + self.cal_b
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        print(
            f"[S{self.sensor_id}] [CAL] force_gf = {self.cal_k:.8g} * score + "
            f"{self.cal_b:.4f}, R2={r2:.4f}, n={len(self.cal_samples)}"
        )
        return True

    def to_config(self):
        return {
            "sensor_id": self.sensor_id,
            "source": self.source,
            "trap_src": None if self.trap_src is None else self.trap_src.tolist(),
            "warp_size": list(self.warp_size),
            "threshold": int(self.threshold),
            "min_area": int(self.min_area),
            "max_area": int(self.max_area),
            "open_iter": int(self.open_iter),
            "close_iter": int(self.close_iter),
            "blur_ksize": int(self.blur_ksize),
            "smooth_alpha": float(self.smooth_alpha),
            "zero_total_area": self.zero_total_area,
            "zero_mean_area": self.zero_mean_area,
            "zero_max_area": self.zero_max_area,
            "zero_score": self.zero_score,
            "cal_k": float(self.cal_k),
            "cal_b": float(self.cal_b),
            "cal_samples": self.cal_samples,
        }

    def apply_config(self, cfg):
        if cfg.get("trap_src") is not None:
            self.trap_src = np.array(cfg["trap_src"], dtype=np.float32)
            ws = cfg.get("warp_size", list(self.warp_size))
            if isinstance(ws, list) and len(ws) == 2:
                self.warp_size = (int(ws[0]), int(ws[1]))
            self.compute_warp()

        self.threshold = int(cfg.get("threshold", self.threshold))
        self.min_area = int(cfg.get("min_area", self.min_area))
        self.max_area = int(cfg.get("max_area", self.max_area))
        self.open_iter = int(cfg.get("open_iter", self.open_iter))
        self.close_iter = int(cfg.get("close_iter", self.close_iter))
        self.blur_ksize = int(cfg.get("blur_ksize", self.blur_ksize))
        self.smooth_alpha = float(cfg.get("smooth_alpha", self.smooth_alpha))

        self.zero_total_area = cfg.get("zero_total_area", self.zero_total_area)
        self.zero_mean_area = cfg.get("zero_mean_area", self.zero_mean_area)
        self.zero_max_area = cfg.get("zero_max_area", self.zero_max_area)
        self.zero_score = cfg.get("zero_score", self.zero_score)

        self.cal_k = float(cfg.get("cal_k", self.cal_k))
        self.cal_b = float(cfg.get("cal_b", self.cal_b))
        self.cal_samples = list(cfg.get("cal_samples", self.cal_samples))

    def binarize(self, gray):
        if self.blur_ksize > 1:
            blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        else:
            blur = gray

        typ = cv2.THRESH_BINARY_INV if self.args.dark_markers else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blur, self.threshold, 255, typ)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if self.open_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=self.open_iter)
        if self.close_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=self.close_iter)

        return binary

    def detect_markers(self, roi_gray):
        binary = self.binarize(roi_gray)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if not (self.min_area <= area <= self.max_area):
                continue
            M = cv2.moments(cnt)
            if abs(M["m00"]) < 1e-9:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            peri = float(cv2.arcLength(cnt, True))
            circularity = 4.0 * math.pi * area / (peri * peri + 1e-9)
            if circularity < self.args.min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            diam = 2.0 * math.sqrt(area / math.pi)
            markers.append({
                "cnt": cnt,
                "area": area,
                "cx": cx,
                "cy": cy,
                "circularity": circularity,
                "bbox": (x, y, w, h),
                "diam": diam,
            })

        markers.sort(key=lambda m: m["area"], reverse=True)
        if self.args.keep_largest > 0:
            markers = markers[:self.args.keep_largest]
        return markers, binary

    def update_from_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        self.last_frame_gray = gray
        roi = self.get_roi(gray)
        self.last_roi_gray = roi

        markers, binary = self.detect_markers(roi)
        n = len(markers)

        if n < self.args.min_points:
            result = AreaResult(False, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, markers, binary)
            self.last_result = result
            return result

        total_area = float(sum(m["area"] for m in markers))
        mean_area = float(total_area / n)
        max_area = float(markers[0]["area"])

        if self.total_area_s is None:
            self.total_area_s = total_area
            self.mean_area_s = mean_area
            self.max_area_s = max_area
        else:
            a = self.smooth_alpha
            self.total_area_s = a * total_area + (1.0 - a) * self.total_area_s
            self.mean_area_s = a * mean_area + (1.0 - a) * self.mean_area_s
            self.max_area_s = a * max_area + (1.0 - a) * self.max_area_s

        if self.args.area_mode == "total":
            raw_score = self.total_area_s
        elif self.args.area_mode == "mean":
            raw_score = self.mean_area_s
        else:
            raw_score = self.max_area_s

        if self.zero_total_area is None:
            d_total = 0.0
            d_mean = 0.0
            d_max = 0.0
            score = 0.0
        else:
            d_total = self.total_area_s - self.zero_total_area
            d_mean = self.mean_area_s - self.zero_mean_area
            d_max = self.max_area_s - self.zero_max_area
            score = raw_score - self.zero_score

        if self.args.invert_score:
            score = -score
        if abs(score) < self.args.deadband:
            score = 0.0

        self.score_s = self.args.score_alpha * score + (1.0 - self.args.score_alpha) * self.score_s
        force_gf = self.score_to_force_gf(self.score_s)

        result = AreaResult(
            True, n,
            float(self.total_area_s), float(self.mean_area_s), float(self.max_area_s),
            float(d_total), float(d_mean), float(d_max),
            float(raw_score), float(self.score_s), float(force_gf),
            markers, binary,
        )
        self.last_result = result
        return result

    def draw_original(self, active=False):
        if self.last_frame_gray is None:
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

        vis = cv2.cvtColor(self.last_frame_gray, cv2.COLOR_GRAY2BGR)

        if self.trap_src is not None:
            pts = self.trap_src.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            for p, lab in zip(self.trap_src.astype(int), ["LT", "RT", "RB", "LB"]):
                cv2.circle(vis, tuple(p), 5, (0, 0, 255), -1)
                cv2.putText(vis, lab, (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if self.select_trap:
            labels = ["LT", "RT", "RB", "LB"]
            for p in self.trap_points:
                cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"S{self.sensor_id}: CLICK {labels[len(self.trap_points)]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            prefix = "ACTIVE " if active else ""
            cv2.putText(vis, f"{prefix}S{self.sensor_id} {self.source}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if active else (220, 220, 220), 2)
        return vis

    def draw_roi_card(self, active=False):
        if self.last_roi_gray is None:
            return np.zeros((self.warp_size[1], self.warp_size[0], 3), dtype=np.uint8)

        vis = cv2.cvtColor(self.last_roi_gray, cv2.COLOR_GRAY2BGR)
        result = self.last_result

        if result is not None:
            for m in result.markers:
                cv2.drawContours(vis, [m["cnt"]], -1, (0, 255, 255), 1)
                cx, cy = int(round(m["cx"])), int(round(m["cy"]))
                cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

        color = (0, 255, 255) if active else (100, 100, 100)
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 2)

        if result is None:
            lines = [f"S{self.sensor_id} {self.source}", "no data"]
        else:
            lines = [
                f"S{self.sensor_id} {self.source} {'ACTIVE' if active else ''}",
                f"valid={result.valid} n={result.n_detected} FPS={self.last_fps:.1f}",
                f"score={result.score:+.1f}",
                f"force={result.force_gf:+.1f} gf",
                f"T={result.total_area:.0f} dT={result.delta_total_area:+.0f}",
            ]
        self.draw_panel(vis, lines)
        return vis

    def draw_panel(self, img, lines):
        fs = 0.43
        thick = 1
        margin = 6
        line_h = 17
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0][0] for t in lines]
        pw = max(widths) + 2 * margin
        ph = len(lines) * line_h + 2 * margin
        x0, y0 = 6, 6
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        y = y0 + margin + line_h - 4
        for line in lines:
            cv2.putText(img, line, (x0 + margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), thick, cv2.LINE_AA)
            y += line_h


class HybridAreaDemo:
    def __init__(self, args):
        self.args = args
        self.sensors = [AreaSensor(i + 1, src, args) for i, src in enumerate(args.sources[:3])]
        self.active_index = 0
        self.debug = args.debug
        self.calib_weights = self.parse_weights(args.calib_weights_g)
        self.calib_weight_index = 0

        self.udp_sock = None
        if args.udp_port is not None:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[UDP] sending to {args.udp_host}:{args.udp_port}")

        self.csv_fp = None
        self.csv_writer = None

    @staticmethod
    def parse_weights(s):
        if not s:
            return []
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    @property
    def active(self):
        return self.sensors[self.active_index]

    def current_calib_weight(self):
        if not self.calib_weights:
            return self.args.calib_weight_g
        idx = max(0, min(self.calib_weight_index, len(self.calib_weights) - 1))
        return self.calib_weights[idx]

    def next_weight(self):
        if not self.calib_weights:
            print(f"[CAL] current weight = {self.args.calib_weight_g:.1f} gf (--calib_weight_g)")
            return
        self.calib_weight_index = min(self.calib_weight_index + 1, len(self.calib_weights) - 1)
        print(f"[CAL] current weight = {self.current_calib_weight():.1f} gf")

    def prev_weight(self):
        if not self.calib_weights:
            print(f"[CAL] current weight = {self.args.calib_weight_g:.1f} gf (--calib_weight_g)")
            return
        self.calib_weight_index = max(self.calib_weight_index - 1, 0)
        print(f"[CAL] current weight = {self.current_calib_weight():.1f} gf")

    def mouse_callback(self, event, x, y, flags, param):
        self.active.mouse_callback(event, x, y, flags, param)

    def create_trackbars(self, ctrl_name):
        cv2.createTrackbar("Threshold", ctrl_name, self.active.threshold, 255, self.on_threshold)
        cv2.createTrackbar("Min Area", ctrl_name, int(self.active.min_area), 50000, self.on_min_area)
        cv2.createTrackbar("Max Area", ctrl_name, int(self.active.max_area), 120000, self.on_max_area)
        cv2.createTrackbar("Open iter", ctrl_name, int(self.active.open_iter), 15, self.on_open_iter)
        cv2.createTrackbar("Close iter", ctrl_name, int(self.active.close_iter), 15, self.on_close_iter)
        cv2.createTrackbar("Blur ksize", ctrl_name, int(self.active.blur_ksize), 31, self.on_blur)
        cv2.createTrackbar("Smooth x0.01", ctrl_name, int(self.active.smooth_alpha * 100), 100, self.on_smooth)

    def sync_trackbars_to_active(self, ctrl_name):
        f = self.active
        cv2.setTrackbarPos("Threshold", ctrl_name, int(f.threshold))
        cv2.setTrackbarPos("Min Area", ctrl_name, int(f.min_area))
        cv2.setTrackbarPos("Max Area", ctrl_name, int(f.max_area))
        cv2.setTrackbarPos("Open iter", ctrl_name, int(f.open_iter))
        cv2.setTrackbarPos("Close iter", ctrl_name, int(f.close_iter))
        cv2.setTrackbarPos("Blur ksize", ctrl_name, int(f.blur_ksize))
        cv2.setTrackbarPos("Smooth x0.01", ctrl_name, int(f.smooth_alpha * 100))

    def on_threshold(self, val):
        self.active.threshold = val

    def on_min_area(self, val):
        self.active.min_area = max(0, val)

    def on_max_area(self, val):
        self.active.max_area = max(val, self.active.min_area + 1)

    def on_open_iter(self, val):
        self.active.open_iter = val

    def on_close_iter(self, val):
        self.active.close_iter = val

    def on_blur(self, val):
        val = max(1, val)
        if val % 2 == 0:
            val += 1
        self.active.blur_ksize = val

    def on_smooth(self, val):
        self.active.smooth_alpha = max(1, val) / 100.0

    def save_config(self):
        path = Path(self.args.config)
        data = {
            "version": "area_multicamera_hybrid_pi_v1",
            "sources": self.args.sources,
            "global": {
                "area_mode": self.args.area_mode,
                "dark_markers": self.args.dark_markers,
                "min_circularity": self.args.min_circularity,
                "keep_largest": self.args.keep_largest,
                "min_points": self.args.min_points,
                "score_alpha": self.args.score_alpha,
                "invert_score": self.args.invert_score,
                "deadband": self.args.deadband,
            },
            "sensors": [s.to_config() for s in self.sensors],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CONFIG] saved: {path}")

    def load_config(self):
        path = Path(self.args.config)
        if not path.exists():
            print(f"[CONFIG] not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg_sensors = data.get("sensors", data.get("fingers", []))

        by_sid = {int(c.get("sensor_id", c.get("finger_id", -1))): c for c in cfg_sensors}
        by_source = {str(c.get("source", "")): c for c in cfg_sensors}

        for idx, s in enumerate(self.sensors):
            cfg = by_sid.get(s.sensor_id) or by_source.get(s.source)
            if cfg is None and idx < len(cfg_sensors):
                cfg = cfg_sensors[idx]
            if cfg is not None:
                s.apply_config(cfg)

        print(f"[CONFIG] loaded: {path}")

    def make_dashboard(self):
        card_w, card_h = 420, 320
        cards = []
        for i, s in enumerate(self.sensors):
            card = s.draw_roi_card(active=(i == self.active_index))
            cards.append(cv2.resize(card, (card_w, card_h)))

        while len(cards) < 3:
            blank = np.zeros((card_h, card_w, 3), dtype=np.uint8)
            cv2.putText(blank, "No camera", (30, card_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
            cards.append(blank)

        dashboard = np.hstack(cards)
        H, W = dashboard.shape[:2]
        footer_h = 42
        out = np.zeros((H + footer_h, W, 3), dtype=np.uint8)
        out[:H, :] = dashboard

        forces = []
        for s in self.sensors:
            if s.last_result is not None and s.last_result.valid:
                forces.append(s.last_result.force_gf)
            else:
                forces.append(0.0)

        text = (
            f"Active: S{self.active.sensor_id} | "
            f"1/2/3 select, t ROI, z zero, Z all, a sample({self.current_calib_weight():.1f}gf), "
            f"n/p weight, f fit, s save, l load, d debug, q quit | "
            f"gf={['%.1f' % v for v in forces]}"
        )
        cv2.putText(out, text, (10, H + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return out

    def send_udp(self, t_now):
        if self.udp_sock is None:
            return

        payload = {
            "time_s": t_now,
            "area_mode": self.args.area_mode,
            "sensors": [],
        }
        for s in self.sensors:
            r = s.last_result
            if r is None:
                payload["sensors"].append({
                    "sensor_id": s.sensor_id,
                    "source": s.source,
                    "valid": False,
                    "score": 0.0,
                    "force_gf": 0.0,
                })
            else:
                payload["sensors"].append({
                    "sensor_id": s.sensor_id,
                    "source": s.source,
                    "valid": r.valid,
                    "n_detected": r.n_detected,
                    "total_area": r.total_area,
                    "mean_area": r.mean_area,
                    "max_area": r.max_area,
                    "delta_total_area": r.delta_total_area,
                    "delta_mean_area": r.delta_mean_area,
                    "delta_max_area": r.delta_max_area,
                    "score": r.score,
                    "force_gf": r.force_gf,
                    "cal_k": s.cal_k,
                    "cal_b": s.cal_b,
                })
        self.udp_sock.sendto((json.dumps(payload) + "\n").encode("utf-8"),
                             (self.args.udp_host, self.args.udp_port))

    def write_csv(self, t_now):
        if self.csv_writer is None:
            return
        for s in self.sensors:
            r = s.last_result
            if r is None:
                continue
            self.csv_writer.writerow([
                f"{t_now:.6f}", s.sensor_id, s.source, int(r.valid), r.n_detected,
                f"{r.total_area:.6f}", f"{r.mean_area:.6f}", f"{r.max_area:.6f}",
                f"{r.delta_total_area:.6f}", f"{r.delta_mean_area:.6f}", f"{r.delta_max_area:.6f}",
                f"{r.score:.6f}", f"{r.force_gf:.6f}", f"{s.cal_k:.9g}", f"{s.cal_b:.9g}",
            ])

    def run(self):
        opened = []
        for s in self.sensors:
            if s.open_camera():
                opened.append(s)

        self.sensors = opened
        if not self.sensors:
            raise RuntimeError("No cameras opened. Check --sources.")

        self.active_index = 0

        if self.args.save_csv:
            self.csv_fp = open(self.args.save_csv, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_fp)
            self.csv_writer.writerow([
                "time_s", "sensor_id", "source", "valid", "n_detected",
                "total_area", "mean_area", "max_area",
                "delta_total_area", "delta_mean_area", "delta_max_area",
                "score", "force_gf", "cal_k", "cal_b",
            ])

        dashboard_name = "Hybrid Area Dashboard"
        active_name = "Active Original - click trapezoid here"
        ctrl_name = "Active Sensor Controls"

        cv2.namedWindow(dashboard_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dashboard_name, 1260, 400)

        cv2.namedWindow(active_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(active_name, self.mouse_callback)

        cv2.namedWindow(ctrl_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctrl_name, 520, 360)
        self.create_trackbars(ctrl_name)

        if self.args.load_config:
            self.load_config()

        self.sync_trackbars_to_active(ctrl_name)

        print("=" * 90)
        print("Hybrid Area Demo Pi v1")
        print(f"Opened sources: {[s.source for s in self.sensors]}")
        print("Keys: 1/2/3 select | t ROI | z zero | Z all | a sample | n/p weight | f fit | F all fit | s save | l load | d debug | c clear | q quit")
        print("=" * 90)

        t0 = time.time()
        fps_t = [time.time() for _ in self.sensors]
        fps_n = [0 for _ in self.sensors]

        while True:
            t_now = time.time() - t0

            for i, s in enumerate(self.sensors):
                ret, frame = s.camera.read()
                if not ret:
                    s._read_fail_count += 1
                    if s._read_fail_count % 60 == 1:
                        print(f"[WARN] camera read failed: S{s.sensor_id} {s.source} count={s._read_fail_count}")
                    continue
                s._read_fail_count = 0

                fps_n[i] += 1
                now = time.time()
                if now - fps_t[i] >= 1.0:
                    s.last_fps = fps_n[i] / (now - fps_t[i])
                    fps_n[i] = 0
                    fps_t[i] = now

                s.update_from_frame(frame)

            dashboard = self.make_dashboard()
            active_full = self.active.draw_original(active=True)

            cv2.imshow(dashboard_name, dashboard)
            cv2.imshow(active_name, active_full)

            if self.debug and self.active.last_result is not None:
                bin_vis = cv2.cvtColor(self.active.last_result.binary, cv2.COLOR_GRAY2BGR)
                cv2.putText(bin_vis,
                            f"S{self.active.sensor_id} thr={self.active.threshold} area[{self.active.min_area},{self.active.max_area}]",
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow("Active Binary Debug", bin_vis)
            else:
                try:
                    cv2.destroyWindow("Active Binary Debug")
                except cv2.error:
                    pass

            self.send_udp(t_now)
            self.write_csv(t_now)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in [ord("1"), ord("2"), ord("3")]:
                idx = key - ord("1")
                if idx < len(self.sensors):
                    self.active_index = idx
                    for s in self.sensors:
                        s.select_trap = False
                        s.trap_points = []
                    self.sync_trackbars_to_active(ctrl_name)
                    print(f"[ACTIVE] S{self.active.sensor_id} {self.active.source}")
                else:
                    print(f"[ACTIVE] sensor {idx + 1} not available")
            elif key == ord("t"):
                for s in self.sensors:
                    s.select_trap = False
                    s.trap_points = []
                self.active.select_trap = True
                print(f"[S{self.active.sensor_id}] [TRAP] click LT -> RT -> RB -> LB")
            elif key == ord("z"):
                self.active.zero_current()
            elif key == ord("Z"):
                for s in self.sensors:
                    s.zero_current()
            elif key == ord("a"):
                self.active.add_calibration_sample(self.current_calib_weight())
            elif key == ord("n"):
                self.next_weight()
            elif key == ord("p"):
                self.prev_weight()
            elif key == ord("f"):
                self.active.fit_force_calibration()
            elif key == ord("F"):
                for s in self.sensors:
                    s.fit_force_calibration()
            elif key == ord("s"):
                self.save_config()
            elif key == ord("l"):
                self.load_config()
                self.sync_trackbars_to_active(ctrl_name)
            elif key == ord("d"):
                self.debug = not self.debug
                print(f"[DEBUG] {self.debug}")
            elif key == ord("c"):
                self.active.reset_state(clear_roi=True)
                print(f"[S{self.active.sensor_id}] [CLEAR] ROI/state cleared")

        for s in self.sensors:
            s.release()

        if self.csv_fp:
            self.csv_fp.close()
        if self.udp_sock:
            self.udp_sock.close()

        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--sources", type=str, nargs="+", default=["mipi0"],
                   help="Camera sources. Examples: --sources mipi0 mipi1 usb16")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=60)

    p.add_argument("--warp_size", type=int, nargs=2, default=[420, 320])

    p.add_argument("--threshold", type=int, default=120)
    p.add_argument("--dark_markers", action="store_true")

    p.add_argument("--min_area", type=int, default=20)
    p.add_argument("--max_area", type=int, default=30000)
    p.add_argument("--min_points", type=int, default=1)
    p.add_argument("--min_circularity", type=float, default=0.05)
    p.add_argument("--keep_largest", type=int, default=0)

    p.add_argument("--open_iter", type=int, default=1)
    p.add_argument("--close_iter", type=int, default=1)
    p.add_argument("--blur_ksize", type=int, default=5)
    p.add_argument("--smooth_alpha", type=float, default=0.35)

    p.add_argument("--area_mode", choices=["total", "mean", "max"], default="total")
    p.add_argument("--score_alpha", type=float, default=0.35)
    p.add_argument("--invert_score", action="store_true")
    p.add_argument("--deadband", type=float, default=0.0)

    p.add_argument("--config", type=str, default="area_hybrid_config.json")
    p.add_argument("--load_config", action="store_true")
    p.add_argument("--calib_weights_g", type=str, default="130,160,200,240,280")
    p.add_argument("--calib_weight_g", type=float, default=130.0)
    p.add_argument("--default_cal_k", type=float, default=1.0)
    p.add_argument("--default_cal_b", type=float, default=0.0)

    p.add_argument("--debug", action="store_true")
    p.add_argument("--udp_host", type=str, default="127.0.0.1")
    p.add_argument("--udp_port", type=int, default=None)
    p.add_argument("--save_csv", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    HybridAreaDemo(parse_args()).run()

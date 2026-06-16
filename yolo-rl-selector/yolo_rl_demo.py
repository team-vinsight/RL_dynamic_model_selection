import time
import cv2
import numpy as np

import psutil
from pynvml import *

from ultralytics import YOLO
from stable_baselines3 import DQN


# =========================
# GPU INIT
# =========================
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)


# =========================
# LOAD YOLO MODELS
# =========================
models = {
    0: YOLO("yolov8s.pt").to("cuda"),
    1: YOLO("yolov8m.pt").to("cuda"),
    2: YOLO("yolov8l.pt").to("cuda")
}

model_names = {
    0: "YOLOv8-S",
    1: "YOLOv8-M",
    2: "YOLOv8-L"
}


# =========================
# LOAD RL MODEL
# =========================
rl_model = DQN.load("yolo_dynamic_selector" \
"" \
"" \
"" \
"" \
"" \
"" \
"" \
"" \
"" \
"")


# =========================
# VIDEO SOURCE
# =========================
cap = cv2.VideoCapture("1.mp4")


# =========================
# STATE FUNCTION
# (same logic as training)
# =========================
def get_state(fps, inference_time, obj_count):

    battery = psutil.sensors_battery()
    battery_percent = battery.percent if battery else 100

    cpu_usage = psutil.cpu_percent()

    gpu_util = nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_mem = nvmlDeviceGetMemoryInfo(gpu_handle)

    return np.array([
        battery_percent,
        cpu_usage,
        gpu_util.gpu,
        gpu_mem.used / gpu_mem.total * 100,
        fps,
        inference_time,
        obj_count
    ], dtype=np.float32)


# =========================
# RUN DEMO
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # dummy metrics (updated after inference)
    fps = 0
    inference_time = 0
    obj_count = 0

    # temporary state
    state = get_state(fps, inference_time, obj_count)

    # RL chooses model
    action, _ = rl_model.predict(state, deterministic=True)

    action = int(action)   

    model = models[action]

    start = time.time()

    results = model.predict(
        frame,
        device="cuda",
        verbose=False
    )

    end = time.time()

    inference_time = (end - start) * 1000
    fps = 1000 / (inference_time + 1e-6)
    obj_count = len(results[0].boxes)

    # rebuild state with real metrics (optional for display)
    state = get_state(fps, inference_time, obj_count)

    # draw results
    annotated = results[0].plot()

    # overlay info
    cv2.putText(
        annotated,
        f"RL Selected: {model_names[action]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Objects: {obj_count}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("RL Dynamic YOLO Selector", annotated)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
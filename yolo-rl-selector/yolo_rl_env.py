import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import psutil
from pynvml import *

from ultralytics import YOLO
from stable_baselines3 import DQN


# =========================
# GPU MONITOR INIT
# =========================
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)


# =========================
# ENVIRONMENT
# =========================
class YoloSelectionEnv(gym.Env):

    def __init__(self, video_source=0):

        super().__init__()

        # 3 actions = 3 YOLO models
        self.action_space = spaces.Discrete(3)

        # state vector:
        # battery, cpu, gpu_util, gpu_mem, fps, latency, objects
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(7,),
            dtype=np.float32
        )

        # Load YOLO models (GPU)
        self.models = {
            0: YOLO("yolov8s.pt").to("cuda"),
            1: YOLO("yolov8m.pt").to("cuda"),
            2: YOLO("yolov8l.pt").to("cuda")
        }

        self.cap = cv2.VideoCapture(video_source)

        self.current_fps = 0
        self.current_inference_time = 0
        self.current_object_count = 0
        self.batery_percent = 100

        self.step_count = 0
        self.max_steps = 200

    # -------------------------
    def reset(self, seed=None, options=None):

        self.step_count = 0

        return self.get_state(), {}

    # -------------------------
    def step(self, action):

        metrics = self.run_selected_model(action)

        reward = self.calculate_reward(metrics)

        self.step_count += 1

        terminated = self.step_count >= self.max_steps
        truncated = False

        return self.get_state(), reward, terminated, truncated, {}

    # -------------------------
    def get_state(self):

        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else 100
        self.batery_percent = battery_percent
        cpu_usage = psutil.cpu_percent()

        gpu_util = nvmlDeviceGetUtilizationRates(gpu_handle)
        gpu_mem = nvmlDeviceGetMemoryInfo(gpu_handle)

        return np.array([
            battery_percent,
            cpu_usage,
            gpu_util.gpu,
            gpu_mem.used / gpu_mem.total * 100,
            self.current_fps,
            self.current_inference_time,
            self.current_object_count
        ], dtype=np.float32)

    # -------------------------
    def run_selected_model(self, action):

        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)

        model = self.models[action]

        start = time.time()

        results = model.predict(
            frame,
            device="cuda",
            verbose=False
        )

        end = time.time()

        inference_time = (end - start) * 1000
        fps = 1000 / (inference_time + 1e-6)

        object_count = len(results[0].boxes)

        self.current_fps = fps
        self.current_inference_time = inference_time
        self.current_object_count = object_count

        return {
            "fps": fps,
            "inference_time": inference_time,
            "object_count": object_count,
            "action": action,
        }

    # -------------------------
    def calculate_reward(self, metrics):

        # reward = performance - cost
        # reward = (
        #     0.05 * (50 + metrics["fps"])
        #     + 0.01 * (metrics["object_count"] / 10.0)
        #     - 0.4 * (metrics["inference_time"] / 100.0)
        # )
        reward =(
            0.1*(metrics["fps"]*3)
            + 0.1*(metrics["action"]-3)*self.batery_percent
        )

        return reward


# =========================
# TRAINING
# =========================
if __name__ == "__main__":

    env = YoloSelectionEnv(video_source="1.mp4")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=20000,
        batch_size=32,
        verbose=1
    )

    model.learn(total_timesteps=50000)

    model.save("yolo_dynamic_selector")

    print("Training complete. Model saved.")
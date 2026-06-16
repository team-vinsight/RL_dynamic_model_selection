# Reinforcement Learning-Based Dynamic YOLO Model Selection

## Overview

This project demonstrates the application of **Reinforcement Learning (RL)** for **Dynamic Algorithm Selection (DAS)** in real-time object detection. Inspired by research on deep reinforcement learning for dynamic algorithm selection, the system dynamically chooses between three YOLOv8 object detection models based on current system conditions and inference performance.

The objective is to balance detection quality and computational efficiency by allowing an RL agent to decide which model should be used at runtime instead of relying on fixed rules or manually defined thresholds.

---

## Motivation

Modern object detection systems often face a trade-off between accuracy and computational cost.

| Model    | Accuracy | Speed    | Resource Usage |
| -------- | -------- | -------- | -------------- |
| YOLOv8-S | Lower    | Fastest  | Lowest         |
| YOLOv8-M | Medium   | Moderate | Moderate       |
| YOLOv8-L | Highest  | Slowest  | Highest        |

In traditional deployments, a single model is selected and used throughout execution. However, runtime conditions such as scene complexity, GPU load, and inference latency continuously change.

This project explores whether a Reinforcement Learning agent can learn to dynamically switch between these models to achieve a better balance between performance and resource consumption.

---

## Project Architecture

```text
                    ┌──────────────────┐
                    │   Video Stream   │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │  RL Agent (DQN)      │
                 └────────┬─────────────┘
                          │
             Select Model │
                          ▼
        ┌─────────┬─────────┬─────────┐
        │ YOLOv8S │ YOLOv8M │ YOLOv8L │
        └────┬────┴────┬────┴────┬────┘
             │         │         │
             └─────────┴─────────┘
                       │
                       ▼
               Detection Results
                       │
                       ▼
            System Metrics & Reward
                       │
                       └────► RL Agent
```

---

## State Representation

The RL agent observes a state vector containing runtime system and inference metrics:

- Battery Percentage
- CPU Utilization
- GPU Utilization
- GPU Memory Usage
- Current FPS
- Inference Latency
- Number of Detected Objects

These metrics provide information about both system load and scene complexity.

---

## Action Space

The agent selects one of three available YOLO models:

| Action | Model    |
| ------ | -------- |
| 0      | YOLOv8-S |
| 1      | YOLOv8-M |
| 2      | YOLOv8-L |

---

## Reward Function

The reward function encourages the agent to maximize detection performance while minimizing computational cost.

Example formulation:

```python
reward = (
    0.5 * (fps / 60.0)
    + 0.5 * (object_count / 10.0)
    - 0.3 * (inference_time / 100.0)
)
```

Positive rewards are given for:

- Higher FPS
- More detections

Negative rewards are given for:

- Higher inference latency

---

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- Stable-Baselines3
- Gymnasium
- OpenCV
- NVIDIA NVML
- CUDA

---

## Training Process

The system is trained using a Deep Q-Network (DQN) agent.

### Training Loop

1. Observe current state
2. Select a YOLO model
3. Run object detection
4. Measure system metrics
5. Compute reward
6. Store experience
7. Update policy network

The agent gradually learns which model provides the best trade-off under different runtime conditions.

---

## Project Structure

```text
.
├── yolo_rl_train.py          # RL training script
├── yolo_rl_demo.py           # Demonstration script
├── yolo_dynamic_selector.zip # Trained RL model
├── 1.mp4                     # Training/demo video
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/team-vinsight/RL_dynamic_model_selection.git
cd yolo-rl-selector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# For CUDA support, install PyTorch separately according to your CUDA version rather than relying on requirements.txt.

## Requirements

```text
ultralytics
torch
torchvision
stable-baselines3
gymnasium
opencv-python
numpy
psutil
nvidia-ml-py3
```

---

## Training

Run the training script:

```bash
python yolo_rl_train.py
```

After training completes, a trained model will be saved:

```text
yolo_dynamic_selector
```

---

## Demonstration

Run the demonstration script:

```bash
python yolo_rl_demo.py
```

The system will:

- Load the trained DQN model
- Process video frames
- Dynamically select YOLO models
- Display:
  - Selected model
  - FPS
  - Object count
  - Detection output

---

## Experimental Observations

During testing, the RL agent frequently switched between:

- YOLOv8-S
- YOLOv8-L

while rarely selecting YOLOv8-M.

This behavior suggests that the learned policy favored extreme trade-offs between speed and detection capability.

Additionally, rapid switching between models was observed due to the absence of switching penalties in the reward function.

---

## Current Limitations

### Limited Training Data

The agent was trained using a single video source, limiting generalization to unseen environments.

### Simplified Reward Function

The reward uses proxy metrics rather than true detection accuracy metrics such as mAP.

### No Switching Penalty

The current implementation does not penalize frequent model changes, leading to unstable switching behavior.

### DQN Stability

DQN can be sensitive to noisy reward signals and may converge to suboptimal policies.

### No Temporal Awareness

The policy treats each state independently and does not consider historical context.

---

## Future Improvements

### Improved RL Algorithms

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

### Temporal Policies

- LSTM-based policies
- GRU-based policies

### Better Reward Design

Include:

- Detection accuracy
- Energy consumption
- Switching penalties

### Diverse Training Dataset

Train using:

- Traffic videos
- Crowded scenes
- Indoor environments
- Night-time footage

### Advanced Evaluation

Compare against:

- Fixed YOLOv8-S
- Fixed YOLOv8-M
- Fixed YOLOv8-L
- Rule-based switching
- RL-based switching

---

## Research Inspiration

This project was inspired by research on reinforcement learning-based dynamic algorithm selection:

**"Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution"**

The work adapts the dynamic algorithm selection concept from optimization algorithms to real-time computer vision systems.

---

## Disclaimer

This project is a proof-of-concept implementation intended for educational and research purposes. The current implementation demonstrates the feasibility of RL-driven model selection but is not optimized for production deployment.

---

## Author

Developed as a research and learning project exploring Reinforcement Learning, Dynamic Algorithm Selection, and Real-Time Computer Vision.

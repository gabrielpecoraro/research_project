# Autonomous Reinforcement Learning Pursuer

## Overview  
This project implements **Reinforcement Learning (RL)**, **Computer Vision (CV)**, and **Multi-Agent Systems** to develop an autonomous vehicle capable of pursuing and intercepting moving targets. The system combines **A* pathfinding**, **QMIX Multi-Agent RL**, and **real-time object detection** to create intelligent pursuit behaviors in dynamic environments.

### Key Features
- **Hybrid Pathfinding**: A* algorithms with Bézier curve smoothing for natural movement
- **Multi-Agent Coordination**: QMIX-based reinforcement learning for coordinated pursuit
- **Real-Time Detection**: YOLO-based target detection and tracking
- **JetBot Integration**: Designed for deployment on NVIDIA JetBot autonomous vehicles

## Why Reinforcement Learning?  
- **Adaptability & Generalization**: Learns from interactions, adapts to uncertain environments.  
- **Optimizes Long-Term Rewards**: Focuses on cumulative gains rather than immediate accuracy.  
- **Handles Complex Decision-Making**: Ideal for sequential tasks like autonomous navigation.  
- **Less Need for Labeled Data**: Learns through trial and error.  
- **Human-Level & Beyond Performance**: Proven effectiveness in AI-driven decision-making.  

## Technologies Used  
- **Languages**: Python  
- **Libraries**: TensorFlow, OpenCV, PyTorch, Gym, Ultralytics, FastAPI  
- **Hardware**: NVIDIA JetBot, Raspberry Pi, Cameras, Autonomous Vehicles  
- **Algorithms**: A* Pathfinding, QMIX, YOLO, Q-Learning, Double Q-Learning  

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- JetBot hardware (for physical deployment)

### Dependencies Installation



```bash
# Clone the repository
git clone https://github.com/your-username/autonomous-rl-pursuer.git
cd autonomous-rl-pursuer

# Install core dependencies
pip install torch torchvision matplotlib numpy scipy
pip install opencv-python pillow requests fastapi uvicorn
pip install ultralytics transformers

# For JetBot integration (on JetBot hardware)
pip install jetbot traitlets ipywidgets

# For multi-agent RL training
pip install gym stable-baselines3
```

or

```bash
pip3 install -r requirements.txt
```
# Running the A* Pathfinding System
## Basic Pathfinding Demo
```bash
cd RL_alternative_approach
python3 pathfinding.py
```

## Multi-Agent Pursuit Animation
'''bash
cd RL_alternative_approach/MARL
python pathfinding_marl.py
'''

## Features:

- Interactive visualization of agent pursuit
- Configurable environment obstacles
- Smooth trajectory generation with Bézier curves
- Real-time target fleeing behaviors

# Training the MARL Algorithm

## Start QMIX Training
```bash
cd RL_alternative_approach/MARL
python3 train_qmix.py
```

### Training Configuration
- Agent 1: Uses deterministic A* pathfinding (no learning)
- Agent 2: Learns coordination strategies via QMIX
- Hybrid Approach: Combines classical pathfinding with modern RL
- Training Episodes: 3000+ episodes across varied environments

## Visualize Trained Agents
```bash
python3 visualize_trained_agents.py
```

### Visualization Features:

- Real-time agent behavior analysis
- Performance metrics tracking
- Interactive controls (start/pause/step/reset)
- Multi-environment testing

# Running the Computer Vision System

## Real-Time Detection Server

```bash
cd WEB

# Start the web dashboard
python3 server_main.py

# In another terminal, start detection
python3 detection_sender.py
```

### Access the Dashboard
    Open your browser and navigate to: http://localhost:8000

### CV Features:

- Target Detection: YOLO-based object recognition
- Weapon Detection: Security-focused threat identification
- Real-Time Reporting: Automatic incident logging
- Web Dashboard: Live surveillance interface


# JetBot Integration

## JetBot Navigation System

```bash
cd RL_alternative_approach/jetbot_navigation/src
python3 main_navigation.py
```

## JetBot Features:

- Autonomous Navigation: GPS-free indoor navigation
- Obstacle Avoidance: RGB-based collision detection
- Target Pursuit: Real-time object tracking and following
- Path Planning: Integration of A* with motor control
## Hardware Requirements
- NVIDIA JetBot with camera module
- Raspberry Pi 4 (alternative platform)
- GPIO-compatible motors and sensors

## JetBot Setup
```bash
# On JetBot device
cd notebooks_jetbot/collision_avoidance
jupyter notebook data_collection.ipynb  # Collect training data
jupyter notebook train_model.ipynb      # Train collision avoidance
jupyter notebook live_demo.ipynb        # Deploy live system
```


# Project Structure
```bash
research_project/
├── RL_alternative_approach/         # Core RL algorithms
│   ├── MARL/                       # Multi-agent systems
│   │   ├── train_qmix.py          # QMIX training
│   │   ├── qmix_marl.py           # QMIX implementation
│   │   └── visualize_trained_agents.py
│   ├── jetbot_navigation/          # JetBot integration
│   └── pathfinding.py             # A* pathfinding
├── WEB/                           # Computer vision system
│   ├── server_main.py            # Web dashboard
│   └── detection_sender.py       # YOLO detection
└── notebooks_jetbot/             # JetBot Jupyter notebooks
    ├── collision_avoidance/
    └── basic_motion/

```

# Quick Start Guide

1. Test Pathfinding: python pathfinding.py
2. Train MARL: python train_qmix.py
3. Start CV System: python server_main.py & python detection_sender.py
4. Deploy on JetBot: Upload notebooks to JetBot and run via Jupyter

# Project Goals

## Done 
Debugged and compiled sample RL projects
Studied prior autonomous vehicle projects for reference
Implemented A pathfinding* with smooth trajectories
Developed QMIX multi-agent coordination system
Created YOLO-based computer vision detection pipeline
Built JetBot integration framework

## WIP
🔲 Deploy and test on physical JetBot hardware
🔲 Optimize real-time performance for embedded systems

# References & Resources
- Reinforcement Learning with TensorFlow
- Deep RL Bootcamp Lectures
- QMIX Paper
- JetBot Documentation
- RaspberryPi Robot Projects
# Future Development
- Advanced RL Algorithms: Integration of PPO and SAC
- Enhanced Computer Vision: Multi-object tracking and prediction
- Swarm Intelligence: Scaling to 5+ coordinated agents
- Real-World Testing: Field deployment and performance validation

# Contributors
Gabriel Pecoraro 
Maroua Oukrid 

# License
**This project is developed for research purposes at ECASP LABORATORY of the Illinois Institute of Technology**

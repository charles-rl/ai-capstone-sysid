# AI-Driven System Identification for Robotic Actuators
### NYCU Undergraduate AI Capstone - Project #1 (Spring 2026)

## 📌 Project Motivation
The motivation for this project stems from a previous research effort involving a **physical inverted pendulum** controlled via Reinforcement Learning (RL). While the RL policy was successfully trained in simulation and deployed to the real world using **Domain Randomization**. I realized that I was manually "guessing" the uncertainty ranges for physical parameters (friction, damping, armature inertia) based on what "looked realistic" in the simulator. This heuristic approach is unscientific and scales poorly to complex robots.

**The goal of this project is to bridge that gap.** Instead of guessing, can we pass real-world sensor trajectories through a neural network to infer the actual physical parameters? Even if the prediction isn't perfect, a **Bayesian Gaussian distribution** is significantly more useful for robust control than a human guess.

### 🎥 My Previous Work: Inverted Pendulum (Sim-to-Real)
You can see the hardware implementation and the RL policy that inspired this research [here](https://youtu.be/C4AyItrXxZA?si=I_kgxug3YxGwecLh).

---

## 🛠️ Project Scope
This project focuses on **Active System Identification** for a single-joint actuator (modeled as a single-link pendulum). 

1. **The Dataset:** 50,000 trajectories generated in **MuJoCo**, simulating a torque-impulse response across randomized physical parameters (Damping, Friction, Armature).
2. **The Challenge:** Real-world sensor data is noisy and quantized. The models must distinguish between physical dynamics and stochastic sensor interference.
3. **The Models:**
    * **Baseline:** A physics-informed Random Forest Regressor using manually extracted features (Settling time, overshoot, etc.).
    * **Deep Learning:** A hybrid **CNN + Bi-LSTM** architecture designed for multi-scale temporal feature extraction.
4. **Bayesian Output:** Unlike standard regression, this model outputs a **Gaussian distribution** ($\mu, \sigma$), allowing the system to quantify its own "confidence" when encountering Out-of-Distribution (OOD) hardware states.

---

## 📂 Repository Structure
* `/simulation`: Contains the `scene.xml` MuJoCo environment.
* `/data`: Dataset collection and preprocessing scripts (NPZ format).
* `/models`: PyTorch implementations of the CNN-LSTM and Baseline models.
* `environment.py`: Gymnasium-compatible wrapper for the actuator.
* `data_collection.py`: Multi-core parallel data generation script.

## 🚀 Getting Started
1. **Clone the repo:**
   ```bash
   git clone https://github.com/charles-rl/ai-capstone-sysid.git
   cd ai-capstone-sysid
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate the Dataset:**
   ```bash
   python data_collection.py
   ```
4. **Train the Model:**


---

### Author
查逸哲 Charles A. Sosmeña - National Yang Ming Chiao Tung University (NYCU).

---

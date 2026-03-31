# AI-Driven System Identification for Robotic Actuators
### NYCU Undergraduate AI Capstone - Project #1 (Spring 2026)

## 📌 Project Motivation
The motivation for this project stems from a previous research effort involving a **physical inverted pendulum** controlled via Reinforcement Learning (RL). While the RL policy was successfully trained in simulation and deployed to the real world using **Domain Randomization**. I realized that I was manually "guessing" the uncertainty ranges for physical parameters (friction, damping, armature inertia) based on what "looked realistic" in the simulator. This human heuristic approach is unscientific and scales poorly to complex robots.

**The goal of this project is to bridge that gap.** Instead of guessing, can we pass real-world sensor trajectories through a neural network to infer the actual physical parameters? Even if the prediction isn't perfect, a **Bayesian Gaussian distribution** is significantly more useful for robust control than a human guess.

### 🎥 My Previous Work: Inverted Pendulum (Sim-to-Real)
You can see the hardware implementation and the RL policy that inspired this research [here](https://youtu.be/C4AyItrXxZA?si=I_kgxug3YxGwecLh).

---

## 🛠️ Project Scope
This project focuses on **System Identification** for a single-joint actuator (modeled as a single-link pendulum). 

1. **The Dataset:** 50,000 trajectories generated in **MuJoCo**, simulating a torque-impulse response across randomized physical parameters: **Damping** ([0.5, 5.0]), **Friction** ([0.2, 2.0]), and **Armature** ([0.0, 1.0]).
2. **The Challenge:** Real-world sensor data is noisy and quantized. The models must distinguish between physical dynamics and stochastic sensor interference.
3. **The Models:**
    * **Baseline:** A Random Forest Regressor using manually extracted features (Settling time, overshoot, etc.).
    * **Deep Learning:** A hybrid **CNN + Bi-LSTM** architecture designed for multi-scale temporal feature extraction (Parallel 1D-CNNs for multi-frequency features followed by a 2-layer Bidirectional LSTM).
4. **Bayesian Output:** Unlike standard regression, the AI model outputs a **Gaussian distribution** ($\mu, \sigma$) for each parameter, allowing the system to quantify its own "confidence" when encountering Out-of-Distribution (OOD) hardware states.

---

## 📂 Repository Structure
* `/simulation`: MuJoCo `scene.xml` environment definition.
* `/data`: Dataset storage for raw and processed `.npz` files.
* `/models`: Saved PyTorch models (`.pth`) and Random Forest baselines (`.pkl`).
* `/figures`: Visualization outputs (Training curves, trajectory comparisons).
* `/src`:
    * `data_collection.py`: Multi-core parallel data generation script.
    * `preprocess_dataset.py`: Noise/quantization injection and feature engineering.
    * `data_analysis.py`: Dataset distribution visualization and trajectory plotting.
    * `training_models.py`: PyTorch model definitions (CNN + Bi-LSTM).
    * `train_ai.py` / `train_rf.py`: Training scripts for AI and Random Forest models.
    * `test.py`: Quantitative evaluation (MSE/NLL) on Test and OOD sets.
    * `sample_testing.py`: Visualizes "Uncertainty Clouds" by sampling from the predicted Gaussian distributions.
    * `environment.py`: Gymnasium-compatible wrapper for the actuator.

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
3. **Generate & Preprocess Data:**
   ```bash
   cd src

   # 1. Collect 50k trajectories (Multicore)
   python data_collection.py

   # 2. Add noise and split into ID/OOD sets
   python preprocess_dataset.py

   # 3. (Optional) Verify dataset distribution
   python data_analysis.py
   ```
4. **Train the Models:**
   ```bash
   # Train the Deep Learning model
   python train_ai.py

   # Train the Random Forest baseline
   python train_rf.py
   ```
5. **Evaluation & Sampling:**
   ```bash
   # Run quantitative tests
   python test.py

   # Qualitative check: Sample mu/sigma for "Uncertainty Clouds"
   python sample_testing.py
   ```

---

### Author
查逸哲 Charles A. Sosmeña - National Yang Ming Chiao Tung University (NYCU).

---

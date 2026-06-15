# 🏭 JSSP: Operations Research vs Reinforcement Learning

## 🎯 The Challenge: Optimality vs. Reactivity

In a factory, time is money. Mathematical solvers like Google OR-Tools are incredible: they can find the absolute perfect schedule (the optimal Makespan) for a static environment. But reality is rarely static. A machine breaks, an urgent order drops, and suddenly, you don't have the time to let a solver recompute everything from scratch.

This project started with a practical question: *Can we train a Reinforcement Learning (RL) agent to learn enough empirical rules to readjust a factory schedule in real-time?*

To test this, I pitted a standard RL architecture (Maskable PPO with a flat 1D MLP policy) against the absolute baseline of the Job Shop Scheduling Problem (JSSP): the infamous Fisher & Thompson 10x10 benchmark (FT10).

## 📄 Full Study (Report)

If you want to dive into the methodology, the reward shaping strategies, and why flat vectors are mathematically doomed to fail against graph structures, you can read my full study report (in French):

👉 **[Read the Full Report (PDF)](./study_report.pdf)**

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bobL59/jssp-rl-vs-or.git
cd jssp-rl-vs-or
pip install -r requirements.txt
```

## 🚀 Usage

The project is organized around two Jupyter notebooks. Run them from the `notebooks/` directory (or set the working directory accordingly so paths to `data/` resolve correctly).

### OR-Tools baseline on FT10

Open and run [notebooks/01_OR_solving.ipynb](./notebooks/01_OR_solving.ipynb):

- Parses the FT10 instance from `data/ft10.txt`
- Solves it with OR-Tools CP-SAT (`jssp_env/or_solvers.py`)
- Displays the optimal Gantt chart

```python
from jssp_env import parser, or_solvers, visualizer

instance = parser.parse_jssp_instance_from_file("../data/ft10.txt")
or_solvers.solve_jssp_exact(instance)
visualizer.visualize_gantt_chart(instance)
```

### Train and evaluate the Maskable PPO agent

Open and run [notebooks/02_rl_training.ipynb](./notebooks/02_rl_training.ipynb):

- Trains **Maskable PPO** (`MlpPolicy`) on random 10×10 instances via `FlatJSSPEnv`
- Saves checkpoints under `saved_models/`
- Evaluates each run on the fixed FT10 instance and aggregates makespan statistics

Training uses action masking, entropy regularization sweeps (`ent_coef` ∈ {0.0, 0.05}), and TensorBoard logs under `notebooks/tensorboard_logs/`.

## 📊 Key Findings on the FT10 Benchmark

| Method                        | Makespan (time units) | Inference Time    |
| ----------------------------- | --------------------- | ----------------- |
| **OR-Tools (Exact Solver)**   | **930** (Optimal)     | Seconds / Minutes |
| **RL Agent (No Entropy)**     | ~2629.60 ± 302.02     | **~0.68 ms**      |
| **RL Agent (Entropy = 0.05)** | ~2608.80 ± 162.77     | **~0.68 ms**      |

**What do these results tell us?**

1. **The promise of speed is kept:** The RL agent generates a valid schedule in less than a millisecond (0.68 ms).
2. **More stability:** By forcing entropy (0.05), we cut the variance in half, making the model much more robust.
3. **The limits of the MLP architecture:** The RL agent stagnates in a local minimum far from the 930 u.t. optimum. An MLP is essentially a "rigid box". It reads a flat list of numbers but natively lacks the relational logic to understand that *Task A physically blocks Task B*. Furthermore, its fixed input size strictly prevents any *Curriculum Learning* (e.g., training on 3x3 grids before moving to 10x10).

*Conclusion:* This study clearly maps the breaking point of classical models for the JSSP. The logical next step for future development is to abandon 1D vectors in favor of Disjunctive Graphs and Graph Neural Networks (GNN).

## 📁 Repository Structure

```text
jssp_solver/
├── jssp_env/              # Core library
│   ├── models.py          # Task, Job, JSSPInstance data model
│   ├── parser.py          # Benchmark file parser
│   ├── or_solvers.py      # OR-Tools CP-SAT exact solver
│   ├── rl_env_flat.py     # Gymnasium env (flat 1D observations)
│   ├── generator10x10.py  # Random 10×10 instance generator
│   └── visualizer.py      # Gantt chart plotting
├── notebooks/
│   ├── 01_OR_solving.ipynb    # OR-Tools baseline + visualization
│   └── 02_rl_training.ipynb   # Maskable PPO training & FT10 evaluation
├── data/
│   └── ft10.txt           # Fisher & Thompson 10×10 benchmark
├── saved_models/          # Trained PPO checkpoints (created at runtime)
└── requirements.txt
```

## 📚 References

- Fisher & Thompson benchmark instance: `data/ft10.txt`
- Scheduling theory background: Pinedo, *Scheduling: Theory, Algorithms, and Systems* (Springer, 2008)

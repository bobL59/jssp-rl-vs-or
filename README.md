# JSSP: Operations Research vs Reinforcement Learning

## Background

Exact solvers such as Google OR-Tools can compute optimal makespans for a fixed JSSP instance. In practice, schedules often need to be revised when a machine fails or a priority order arrives, without time to rerun a full optimization from scratch.

This project compares a reinforcement learning (RL) approach against that baseline: can an agent learn enough scheduling heuristics to adjust a plan quickly?

The RL setup uses Maskable PPO with a flat 1D MLP policy. The benchmark is the Fisher & Thompson 10×10 instance (FT10).

## Study Report

Methodology, reward shaping, and the limitations of flat vector observations versus graph-based representations are described in the full study (French):

**[Study report (PDF)](./study_report.pdf)**

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/bobL59/jssp-rl-vs-or.git
cd jssp-rl-vs-or
pip install -r requirements.txt
```

## Usage

The project is organized around two Jupyter notebooks. Run them from the `notebooks/` directory (or set the working directory so paths to `data/` resolve correctly).

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

- Trains Maskable PPO (`MlpPolicy`) on random 10×10 instances via `FlatJSSPEnv`
- Saves checkpoints under `saved_models/`
- Evaluates each run on the fixed FT10 instance and aggregates makespan statistics

Training uses action masking, entropy regularization sweeps (`ent_coef` ∈ {0.0, 0.05}), and TensorBoard logs under `notebooks/tensorboard_logs/`.

## Results on FT10

| Method                        | Makespan (time units) | Inference Time    |
| ----------------------------- | --------------------- | ----------------- |
| OR-Tools (exact solver)       | 930 (optimal)         | seconds / minutes |
| RL agent (no entropy)         | ~2629.60 ± 302.02     | ~0.68 ms          |
| RL agent (entropy = 0.05)     | ~2608.80 ± 162.77     | ~0.68 ms          |

Summary:

1. **Inference speed:** the RL agent returns a valid schedule in about 0.68 ms.
2. **Entropy regularization:** with `ent_coef = 0.05`, variance is roughly halved compared to `ent_coef = 0.0`.
3. **MLP limitations:** the agent stabilizes far from the optimal makespan of 930. A flat observation vector does not represent precedence and machine conflicts as explicitly as a graph structure, and the fixed input size prevents curriculum training (e.g. 3×3 instances before 10×10).

Graph-based models (disjunctive graphs, GNNs) are a more direct representation for JSSP than 1D vectors.

## Repository Structure

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

## References

- Fisher & Thompson benchmark instance: `data/ft10.txt`
- Scheduling theory background: Pinedo, *Scheduling: Theory, Algorithms, and Systems* (Springer, 2008)

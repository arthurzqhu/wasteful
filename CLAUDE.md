# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toy project to test which training data sampling strategy converges to a known target parameter set the fastest when using emulator-guided MCMC. A PPE (set of perturbed parameter ensemble training points) is generated each round, and the three strategies differ in how they select training data across rounds:

- **Global**: Keeps the prior bounds unchanged; uses all PPEs from all rounds
- **Buffered**: Uses the latest PPE + all previous PPEs that fall within the bounds of the latest PPE plus an x% buffer margin
- **Local**: Uses the latest PPE + all previous PPEs strictly within the bounds of the latest PPE (no buffer)

ML emulators (Gaussian Processes and Neural Networks) are trained on the selected PPEs each round and used to accelerate MCMC sampling in place of expensive direct simulator evaluations.

## How to Run

# Install dependencies (no requirements.txt)
pip install jax jaxopt flax optax tinygp blackjax matplotlib corner numpy

# Run main experiment
python main_experiment.py

## Architecture

**Core pipeline** (orchestrated by `main_experiment.py`):

1. **Simulator** (`simulator.py`): Base interface + power-law physical model generating synthetic observations
2. **Emulators** (`emulators.py`): GP (tinygp with ARD kernels) and NN (Flax, CRPS loss) emulators that learn the simulator's input-output mapping with uncertainty quantification
3. **MCMC** (`mcmc_sampler.py`): BlackJAX NUTS sampler with log-probability functions for both the true simulator and each emulator type

**Experiment phases**: Discovery (broad sampling) → Strategy initialization → Iterative refinement (rounds 2-5) → Analysis/plotting

**Three training data strategies** compared across rounds (see Project Overview)

## Key Conventions

- **JAX float64**: Enabled at module start via `jax.config.update("jax_enable_x64", True)`
- **Log-space modeling**: Outputs modeled in log-space, transformed back via Delta Method
- **Input standardization**: All emulator inputs standardized to ~N(0,1) before training
- **Uncertainty scaling**: Emulator uncertainty scaled by sqrt(N_samples) to prevent overconfidence as data accumulates
- **Posterior-weighted sampling**: New training points sampled from previous round's posterior
- **NN training**: Uses CRPS loss (not NLL) for robustness; early stopping with patience

## Configuration

Key parameters in `main_experiment.py` (top of file):
- `TRUE_THETA = [3, 2, 16]` — ground truth parameters
- `n_samples_R1 = 1000` — discovery phase training data size
- `n_samples_R2 = 500` — refinement round training data size
- `num_steps = 2500` — MCMC steps per round
- `MODELS_TO_RUN` / `STRATEGIES_TO_RUN` — control which strategies are evaluated

## Output

- `posteriors/` — `.npz` files with posterior samples per round/strategy
- `plots/` — PDF corner plots and performance comparisons

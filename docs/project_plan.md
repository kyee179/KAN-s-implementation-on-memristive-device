# Project Plan

This document collects project decisions, theory notes, and experiment milestones.

## Phase 1: Setup

- Local repository: `D:\UCL\course\research_project\KANExperiment`
- Conda environment: `D:\UCL\course\research_project\conda_envs\kan-memristor`
- Remote repository: `https://github.com/kyee179/KAN-s-implementation-on-memristive-device`

## Phase 2: Basic KAN Validation

Start with small supervised regression/classification tasks to confirm that KAN software experiments behave as expected before introducing hardware constraints.

Current software baselines include B-spline KAN, odd-polynomial KAN, and an MLP reference. See `docs/kan_baseline_experiments.md`.

## Phase 3: Physical Hardware Relationships

Behavioral models now cover:

- Dynamic memdiode/RRAM switching and conductance mapping.
- Gilbert multiplier blocks for x*x power generation.
- Odd-polynomial hardware edge simulation.

See `docs/hardware_physical_models.md`.

## Phase 4/5: Hardware KAN Structure and Learning

The first trainable physical KAN is implemented as a memristive odd-polynomial KAN. It maps ideal KAN coefficients onto differential RRAM conductance pairs, connects layers with a fixed current-to-voltage scaling parameter `k`, and fine-tunes conductances with accumulated SET/RESET pulse updates guided by KAN gradients.

See `docs/hardware_kan_training.md` and run `python -m kan_memristor.experiments.hardware_train`.

# KAN's Implementation on Memristive Devices

This repository contains experiments and implementation work for applying Kolmogorov-Arnold Networks (KANs) to memristive hardware.

## Initial Roadmap

1. Configure the repository and reproducible Python environment.
2. Validate basic KAN theory with software experiments.
3. Build theoretical physical relationships for the target hardware.
4. Map KAN structure onto the hardware model.
5. Implement the learning algorithm on the network.
6. Validate feasibility with two datasets.
7. Define evaluation methods and metrics.

## Environment

The initial environment is CPU-oriented so the baseline experiments can run on most machines.

Create the environment on drive D:

```powershell
conda env create -p D:\UCL\course\research_project\conda_envs\kan-memristor -f environment.yml
conda activate D:\UCL\course\research_project\conda_envs\kan-memristor
pip install -e D:\UCL\course\research_project\KANExperiment
python -m ipykernel install --user --name kan-memristor --display-name "Python (kan-memristor)"
```

For pip-only reproduction, use:

```powershell
pip install -r requirements.txt
```

## Repository Layout

```text
src/kan_memristor/      Python package for experiments and hardware mapping code
tests/                  Smoke tests and later unit tests
notebooks/              Exploratory notebooks
data/raw/               Local raw datasets, ignored by git except .gitkeep
data/processed/         Local processed datasets, ignored by git except .gitkeep
docs/                   Project notes and theory references
```
## Baseline KAN Tests

Run the first validation experiments with:

```powershell
conda activate D:\UCL\course\research_project\conda_envs\kan-memristor
python -m kan_memristor.experiments.baseline_kan
```

For details, see `docs/kan_baseline_experiments.md`.

## Hardware Physical Models

Run hardware block characterization with:

```powershell
python -m kan_memristor.experiments.hardware_characterization
```

For details, see `docs/hardware_physical_models.md`.

## Hardware KAN Training

Run the first memristive odd-polynomial KAN training experiment with:

```powershell
python -m kan_memristor.experiments.hardware_train
```

For the current method and results, see `docs/hardware_kan_training.md`.

## Complete Physical Simulation

Run the stricter physical path with Gilbert-generated power rows and DynamicMemdiode pulse programming:

```powershell
python -m kan_memristor.experiments.hardware_train --complete-physical
```

For results and the remaining physical-system checklist, see `docs/complete_physical_simulation.md`.

## Inter-layer Normalization

Run normalized complete physical experiments with:

```powershell
python -m kan_memristor.experiments.hardware_train --complete-physical --inter-layer-normalization tanh --normalization-gain 2.0
```

For results, see `docs/inter_layer_normalization_experiment.md`.

## Energy Estimation

Estimate physical KAN inference/programming energy and compare it with a digital MLP baseline:

```powershell
python -m kan_memristor.experiments.energy_estimation
```

For assumptions and results, see `docs/energy_estimation.md`.

## Expanded Polynomial Edges

Test full polynomial edge functions, including even powers from `x` to `x^7`:

```powershell
python -m kan_memristor.experiments.hardware_train --complete-physical --powers 1 2 3 4 5 6 7
```

For the hardware interpretation and results, see `docs/full_polynomial_edge_experiment.md`.

## Fixed GBF Basis KAN

Train a KAN whose edge functions use fixed generalized bell-shaped basis cells, then map the trained weights to a GBF/TIA/RRAM-crossbar physical model:

```powershell
python -m kan_memristor.experiments.gbf_kan
```

For the architecture and first results, see `docs/gbf_kan_experiment.md`.

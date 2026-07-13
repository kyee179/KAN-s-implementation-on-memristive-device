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

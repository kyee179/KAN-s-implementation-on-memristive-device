# KAN Baseline Experiments

This phase validates that a software cubic B-spline KAN can solve small nonlinear tasks before hardware constraints are introduced.

## Sources Used

- Daily Dose of Data Science KAN introduction: KANs move learnable nonlinear functions onto edges rather than using fixed node activations.
- Taglietti et al. (2026), "Learning Nonlinear Heterogeneity in Physical Kolmogorov-Arnold Networks": the paper evaluates physical KANs on nonlinear function regression, yin-yang classification, harder synthetic classifiers, and NASA Li-Ion battery aging prediction.

## Implemented Benchmarks

1. `complicated_function`: a deterministic 2D nonlinear regression problem mixing sinusoidal, product, polynomial, and localized Gaussian terms.
2. `taglietti_yinyang`: a deterministic procedural yin-yang binary classifier inspired by Taglietti et al.'s synthetic classification benchmark. The paper reports 8,000 training points, 10,000 test points, binary cross-entropy training, and a physical KAN structure of `[2,2,1]` with 8 SYNE devices per synapse. This repo uses a slightly wider software cubic B-spline KAN by default for robust CPU training.

## Running

```powershell
conda activate D:\UCL\course\research_project\conda_envs\kan-memristor
pip install -e D:\UCL\course\research_project\KANExperiment
python -m kan_memristor.experiments.baseline_kan
```

Fast smoke run:

```powershell
python -m kan_memristor.experiments.baseline_kan --epochs 5 --n-train 128 --n-test 128 --models kan
```

Outputs are written to `outputs/baseline_tests/` and are ignored by git.

## NASA Battery Dataset Note

The paper's real-world dataset is NASA Li-Ion battery degradation data for end-of-life prediction from multi-sensor measurements. This repo does not fabricate that data. The next data step should add a downloader or documented manual placement path under `data/raw/nasa_battery/`, then implement a loader with explicit preprocessing and split rules.
## Initial Results

Run date: 2026-07-14. Command:

```powershell
python -m kan_memristor.experiments.baseline_kan --epochs 150 --n-train 2048 --n-test 2048 --num-basis 13 --spline-degree 3 --output-dir outputs/baseline_tests
```

| Dataset | Model | Parameters | Test metric |
| --- | ---: | ---: | ---: |
| `complicated_function` | KAN `[2,16,16,1]` | 4,289 | MSE 0.00062 |
| `complicated_function` | MLP `[2,64,64,1]` | 4,417 | MSE 0.31035 |
| `taglietti_yinyang` | KAN `[2,12,1]` | 517 | Accuracy 99.27% |
| `taglietti_yinyang` | MLP `[2,64,64,1]` | 4,417 | Accuracy 94.58% |

These results are not a final benchmark claim. They are a sanity check showing that the cubic B-spline edge-function KAN implementation can learn both a smooth complicated regression target and a sharp nonlinear classification boundary before hardware constraints are introduced.

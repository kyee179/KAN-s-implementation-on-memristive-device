# Full Polynomial Edge Experiment

This experiment tests whether the physically motivated KAN edge function should remain odd-polynomial,

```text
phi(x) = c1 x + c3 x^3 + c5 x^5
```

or be expanded to a denser polynomial basis,

```text
phi(x) = c1 x + c2 x^2 + c3 x^3 + c4 x^4 + c5 x^5 + c6 x^6 + c7 x^7
```

The goal is to answer two questions:

1. Does the more complex edge function improve the final result?
2. Are even powers a hardware problem, and if so can they be solved?

## Hardware Interpretation

Even powers are possible in the current physical model. The original odd-only edge function has odd symmetry, so `phi(-x) = -phi(x)`. Adding `x^2`, `x^4`, and `x^6` removes this symmetry and allows an edge to represent asymmetric nonlinearities.

The hardware mapping is:

- Gilbert voltage multipliers generate the voltage rows `x`, `x^2`, ..., `x^7`.
- Each polynomial coefficient is stored by a differential RRAM pair, `G+ - G-`.
- The crossbar sums the currents from all polynomial rows and all input edges.
- The input voltage scale and each layer's current-to-voltage gain keep the next layer input inside the useful voltage range; inter-layer `tanh` normalization further limits the signal before the next polynomial expansion.

The main even-power issue is not sign representation. Although `x^2`, `x^4`, and `x^6` are always nonnegative, their coefficients can still be positive or negative because the coefficient is represented by the differential conductance pair. The real hardware concern is common-mode current: even-power rows can add a large always-positive current before differential subtraction. This may require careful current subtraction, bias handling, and ADC/common-mode range design.

The second cost is resource scaling. Expanding from three powers to seven powers increases the number of RRAM devices by `7/3`, and it increases the number of Gilbert multipliers more strongly because higher powers need repeated multiplications.

## Command

```powershell
python -m kan_memristor.experiments.hardware_train `
  --datasets complicated_function taglietti_yinyang `
  --powers 1 2 3 4 5 6 7 `
  --complete-physical `
  --inter-layer-normalization tanh `
  --normalization-gain 2.0 `
  --k-values 0.2 `
  --n-train 2048 `
  --n-test 2048 `
  --pretrain-epochs 150 `
  --pulse-epochs 0 `
  --batch-size 256 `
  --n-states 64 `
  --output-dir outputs/full_polynomial_physical_mapped
```

An energy estimate was also run with the same powers:

```powershell
python -m kan_memristor.experiments.energy_estimation `
  --datasets complicated_function taglietti_yinyang `
  --powers 1 2 3 4 5 6 7 `
  --k 0.2 `
  --inter-layer-normalization tanh `
  --normalization-gain 2.0 `
  --n-train 2048 `
  --n-test 2048 `
  --energy-samples 2048 `
  --pretrain-epochs 150 `
  --pulse-epochs 0 `
  --batch-size 256 `
  --n-states 64 `
  --read-time-s 1e-9 `
  --energy-per-mac-j 4.6e-12 `
  --output-dir outputs/full_polynomial_energy_estimation
```

## Results

Compared with the normalized odd-power result, the full polynomial basis is much better in ideal software on the complicated regression function. However, the mapped physical result does not improve for that dataset. This suggests the larger basis creates a better software model, but the extra coefficients are more sensitive to conductance quantization, current-to-voltage scaling, and multiplier-chain error.

| Dataset | Edge powers | Ideal software result | Complete physical mapped result | Memristors | Gilbert multipliers |
| --- | --- | ---: | ---: | ---: | ---: |
| Complicated function | `1,3,5` | MSE `0.07749` | MSE `0.41117` | `1824` | `204` |
| Complicated function | `1..7` | MSE `0.01065` | MSE `0.43608` | `4256` | `714` |
| Taglietti yin-yang | `1,3,5` | Accuracy `95.61%` | Accuracy `94.34%` | `216` | `84` |
| Taglietti yin-yang | `1..7` | Accuracy `95.85%` | Accuracy `94.68%` | `504` | `294` |

The classification task benefits slightly after physical mapping, from `94.34%` to `94.68%`. The regression task does not: the physical MSE worsens from `0.41117` to `0.43608`, despite the ideal software pretrain improving from `0.07749` to `0.01065`.

## Energy Impact

The expanded polynomial KAN remains lower-energy than the digital MLP estimate, but the advantage is smaller because the added multiplier chain dominates the physical inference energy.

| Dataset | Expanded KAN energy/sample | Digital MLP energy/sample | KAN / MLP |
| --- | ---: | ---: | ---: |
| Complicated function | `31.57 pJ` | `19.72 nJ` | `0.160%` |
| Taglietti yin-yang | `12.96 pJ` | `19.72 nJ` | `0.0657%` |

For comparison, the odd-power KAN previously used about `9.10 pJ/sample` on the complicated function and `3.72 pJ/sample` on yin-yang. So the expanded basis costs about `3.5x` more inference energy, mostly because the Gilbert multiplier count rises from `204` to `714` and from `84` to `294`.

## Interpretation

Even powers can be solved physically with the existing differential memristor representation. They are not mathematically or circuit-wise forbidden. The issue is that they increase the analog burden:

- More coefficient states must be programmed and preserved.
- More multiplier stages are needed for high powers.
- Even-power rows create nonnegative voltage/current components, increasing common-mode current.
- The richer software model can overuse high-order coefficients that are fragile after conductance quantization.

The current result supports a cautious conclusion: full polynomial edges are useful as a software upper bound and may help classification slightly, but they are not automatically better for the complete physical KAN. A good next experiment is a hybrid basis such as `(1, 2, 3, 5)` or `(1, 2, 3, 4, 5)`, combined with coefficient regularization that penalizes high powers. That would keep the asymmetric benefit of even powers while reducing multiplier cost and quantization sensitivity.

# Hardware-aware odd-polynomial KAN training

This step connects the odd-polynomial KAN edge function to the simulated physical hardware blocks:

```text
phi_oi(x_i) = c_oi,1*x_i + c_oi,3*x_i^3 + c_oi,5*x_i^5
```

## Hardware correspondence

Each KAN edge is represented by three voltage rows, one for each odd power. Gilbert multiplier stages generate the high-power rows, while a normalization stage keeps every row in the same safe voltage window:

```text
V_row,p = k * x^p, p in {1, 3, 5}
```

Differential RRAM pairs represent signed coefficients. For one edge, the physical node current is modeled as:

```text
I_oi = (G+_oi,1 - G-_oi,1)*k*x_i
     + (G+_oi,3 - G-_oi,3)*k*x_i^3
     + (G+_oi,5 - G-_oi,5)*k*x_i^5
```

Currents from all incoming edges sum at the node. A fixed current-to-voltage gain then creates the next layer signal:

```text
x_next,o = A_IV * sum_i(I_oi) + b_o
```

`k` is treated as a fixed hardware calibration parameter, following Mehonic et al. They convert software layer inputs to voltage with an input scaling factor `k` and use the same scaling concept to convert crossbar currents back to software outputs. In their device example, `k` is bounded by the available reset-voltage region and low values such as `0.1` or `0.2` preserve accuracy because the I/V curves are more linear at low voltage.

## Training flow

The implementation follows a two-stage workflow.

1. Train the ideal `OddPolynomialKAN` with normal KAN backpropagation. This gives the ideal coefficients already validated in the earlier experiments.
2. Map those coefficients to differential finite-state RRAM conductance pairs.
3. Fine-tune the physical model with KAN gradients, but use those gradients only to request SET/RESET pulses. Positive conductance updates correspond to SET-like pulses; negative updates correspond to RESET-like pulses.

The pulse optimizer accumulates requested conductance changes and fires a pulse only when the accumulated request exceeds one conductance level. This avoids pulsing every high-gradient device on every batch, which was too coarse for regression.

## Reproduce the current run

```bash
python -m kan_memristor.experiments.hardware_train \
  --datasets complicated_function taglietti_yinyang \
  --k-values 0.2 \
  --n-train 2048 \
  --n-test 2048 \
  --pretrain-epochs 150 \
  --pulse-epochs 10 \
  --batch-size 256 \
  --n-states 64 \
  --gradient-deadzone-quantile 0.75 \
  --conductance-learning-rate 1e-10 \
  --output-dir outputs/hardware_training
```

Outputs include `metrics.json`, prediction plots, and `.pt` checkpoints for the ideal and pulse-trained models.

## First results

With `k=0.2`, 64 conductance states, `R_LRS=10 kOhm`, and `R_HRS=1 MOhm`:

| Dataset | Stage | Test result | Memristors | Pulses |
| --- | --- | ---: | ---: | ---: |
| Complicated function | ideal pretrain | MSE 0.1416 | - | - |
| Complicated function | mapped quantized | MSE 0.1673 | 1824 | 0 |
| Complicated function | pulse trained | MSE 0.3140 | 1824 | 602 SET / 602 RESET |
| Taglietti yin-yang | ideal pretrain | Acc. 94.97% | - | - |
| Taglietti yin-yang | mapped quantized | Acc. 94.87% | 216 | 0 |
| Taglietti yin-yang | pulse trained | Acc. 94.97% | 216 | 1 SET / 1 RESET |

The mapping result is the key positive result so far: finite-state RRAM mapping keeps most of the ideal model accuracy, especially on the yin-yang classification task. The pulse fine-tuning rule is feasible and stable for classification, but it still damages the smooth regression task. That suggests the next research step should tune pulse scheduling per layer or per task, rather than assume one global pulse rule is enough.

# Energy estimation: physical KAN vs digital MLP

This experiment estimates the energy consumption of the normalized complete-physical KAN and compares it with the software MLP baseline. The estimate is paper-based and should be treated as an order-of-magnitude comparison, not a measured chip result.

## Energy model

The physical KAN inference energy is separated into RRAM read energy and Gilbert multiplier energy.

### RRAM read energy

For every physical KAN layer, each voltage row is applied to differential RRAM conductance pairs. The read energy is approximated as:

```text
E_RRAM = sum(V_row^2 * G_device * t_read)
```

The default read time is:

```text
t_read = 1 ns
```

This term uses the actual physical KAN voltage rows and conductance tensors after software pretraining and RRAM mapping.

### Gilbert multiplier energy

The Gilbert multiplier energy uses the Renduchintala et al. defaults encoded in `GilbertMultiplierParameters`:

```text
P = 440 uW
f = 10 GHz
E_multiplier = P / f = 44 fJ
```

The total Gilbert energy is:

```text
E_Gilbert = number_of_multipliers * 44 fJ
```

### Programming energy

For SET/RESET pulse programming, the DynamicMemdiode model is used:

```text
E_pulse = |V_pulse * I_memdiode(V_pulse, state)| * pulse_width
```

The default pulse settings are:

```text
SET voltage = 1.8 V
RESET voltage = -1.0 V
pulse width = 1 ns
```

The estimated average pulse energies are:

```text
SET pulse  = 24.60 pJ
RESET pulse = 6.45 pJ
```

### Digital MLP energy

The MLP baseline uses the existing architecture:

```text
[2, 64, 64, 1]
```

This requires:

```text
2*64 + 64*64 + 64*1 = 4288 MACs per inference
```

The default digital energy assumption is:

```text
E_MAC = 4.6 pJ
```

So the estimated MLP inference energy is:

```text
E_MLP = 4288 * 4.6 pJ = 19.72 nJ per sample
```

This MAC energy is configurable in the experiment and should be replaced by a measured accelerator value if available.

## Command

```bash
python -m kan_memristor.experiments.energy_estimation \
  --datasets complicated_function taglietti_yinyang \
  --k 0.2 \
  --inter-layer-normalization tanh \
  --normalization-gain 2.0 \
  --n-train 2048 \
  --n-test 2048 \
  --energy-samples 2048 \
  --pretrain-epochs 150 \
  --pulse-epochs 0 \
  --n-states 64 \
  --read-time-s 1e-9 \
  --energy-per-mac-j 4.6e-12 \
  --output-dir outputs/energy_estimation
```

A second run used `--pulse-epochs 10` with gentle pulse settings to estimate programming energy from simulated pulse counts.

## Inference energy results

Lower-bound estimate with no ADC/DAC/current-to-voltage peripheral energy added:

| Dataset | Physical result | Memristors | Gilbert multipliers | RRAM read | Gilbert | Physical total | MLP total | KAN / MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complicated function | MSE 0.4112 | 1824 | 204 | 0.126 pJ | 8.976 pJ | 9.102 pJ | 19.72 nJ | 0.046% |
| Taglietti yin-yang | Acc. 94.34% | 216 | 84 | 0.0198 pJ | 3.696 pJ | 3.716 pJ | 19.72 nJ | 0.0188% |

The physical KAN inference estimate is much smaller than the digital MLP estimate under these assumptions. The dominant KAN inference cost is the Gilbert multiplier energy, not the RRAM read energy.

## Programming energy results

With gentle DynamicMemdiode pulse training:

| Dataset | Pulse-trained result | SET pulses | RESET pulses | Programming energy |
| --- | ---: | ---: | ---: | ---: |
| Complicated function | MSE 0.7576 | 9 | 9 | 0.279 nJ |
| Taglietti yin-yang | Acc. 94.38% | 0 | 0 | 0 nJ |

For the regression task, the programming energy in this short pulse-training run is still small in absolute terms, but the accuracy became worse than the mapped model. This means the current pulse rule is not yet useful for regression, even though its energy cost is not the main issue.

## Important caveats

The physical KAN estimate currently excludes several peripheral costs unless they are manually supplied with `--peripheral-energy-j`:

- DAC/input driver energy
- current-to-voltage amplifier energy
- inter-layer normalization circuit energy
- ADC energy, if a mixed-signal interface is used
- clocking/control overhead
- wire/crossbar parasitic losses
- memory movement for the digital MLP baseline

Therefore, the reported physical KAN energy should be interpreted as a lower-bound compute-array estimate. It is still useful because it shows which physical block dominates: in this model, the Gilbert multipliers dominate inference energy.

## Conclusion

Under the current assumptions, the normalized complete-physical KAN has a very favorable inference energy estimate compared with the digital MLP baseline. However, a fair chip-level comparison needs peripheral energy estimates. The next useful energy experiment should sweep assumed peripheral energy, e.g. `1 pJ`, `10 pJ`, and `100 pJ` per sample, to see when the physical KAN advantage disappears.

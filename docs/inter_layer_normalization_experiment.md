# Inter-layer normalization experiment

The complete physical simulation showed that directly connecting KAN layers can overdrive the next layer's Gilbert multipliers and RRAM read path. To test whether this can be solved, the software odd-polynomial KAN and the physical KAN were both extended with a hardware-compatible inter-layer normalization block.

## Normalization method

After each hidden KAN layer, the layer output is transformed before it is sent to the next layer:

```text
x_next = tanh(gamma * y_layer)
```

where `gamma` is a calibration gain. This keeps the next layer input bounded in `[-1, 1]`, so the physical voltage entering the next layer remains:

```text
V_next = k * x_next
```

This is intended to represent a saturating current-to-voltage / gain-control block between KAN layers. The final output layer is not normalized, so regression outputs can still take continuous values.

The important point is that the same normalization is used during software pretraining and during complete physical simulation. Therefore the software model learns under a constraint that can also be implemented by hardware.

## Command

The best run used `tanh` normalization with gain `2.0`:

```bash
python -m kan_memristor.experiments.hardware_train \
  --datasets complicated_function taglietti_yinyang \
  --complete-physical \
  --inter-layer-normalization tanh \
  --normalization-gain 2.0 \
  --k-values 0.2 \
  --n-train 2048 \
  --n-test 2048 \
  --pretrain-epochs 150 \
  --pulse-epochs 0 \
  --batch-size 256 \
  --n-states 64 \
  --output-dir outputs/normalized_physical_training_gain2
```

A gentler pulse-training follow-up used:

```bash
--pulse-epochs 10 --conductance-learning-rate 1e-11 --max-pulses-per-update 1 --gradient-deadzone-quantile 0.9
```

## Results

Previous complete physical simulation without inter-layer normalization:

| Dataset | Stage | Result |
| --- | --- | ---: |
| Complicated function | ideal pretrain | MSE 0.1416 |
| Complicated function | complete physical mapped | MSE 1.7043 |
| Complicated function | complete physical pulse trained | MSE 1.6439 |
| Taglietti yin-yang | ideal pretrain | Acc. 94.97% |
| Taglietti yin-yang | complete physical mapped | Acc. 92.68% |
| Taglietti yin-yang | complete physical pulse trained | Acc. 93.95% |

With `tanh` inter-layer normalization, gain `2.0`:

| Dataset | Stage | Result |
| --- | --- | ---: |
| Complicated function | normalized ideal pretrain | MSE 0.0775 |
| Complicated function | normalized complete physical mapped | MSE 0.4112 |
| Complicated function | gentler pulse trained | MSE 0.7576 |
| Taglietti yin-yang | normalized ideal pretrain | Acc. 95.61% |
| Taglietti yin-yang | normalized complete physical mapped | Acc. 94.34% |
| Taglietti yin-yang | gentler pulse trained | Acc. 94.38% |

A smaller gain, `0.5`, was too restrictive. It reduced software expressiveness and gave worse physical mapping:

| Dataset | Gain 0.5 mapped result |
| --- | ---: |
| Complicated function | MSE 1.3257 |
| Taglietti yin-yang | Acc. 93.21% |

Gain `1.0` was better than no normalization but worse than gain `2.0` for regression:

| Dataset | Gain 1.0 mapped result |
| --- | ---: |
| Complicated function | MSE 0.6373 |
| Taglietti yin-yang | Acc. 93.99% |

## Interpretation

Inter-layer normalization is applicable and important. It reduced the complete-physical regression mapping error from `1.7043` to `0.4112`, and improved yin-yang classification from `92.68%` to `94.34%`. This strongly supports the hypothesis that the previous failure was mainly a signal-scaling problem rather than only a memristor-state precision problem.

However, pulse training is still not solved. Even with normalization, the regression task became worse after DynamicMemdiode pulse updates. The classification task was stable, but not significantly improved. This suggests the next training issue is not the forward voltage range; it is the pulse programming rule. A more realistic verify-write loop, smaller adaptive pulse requests, or layer-specific pulse calibration is needed before pulse training can reliably improve the mapped model.

## Conclusion

Inter-layer normalization should be included as both a software constraint and a hardware block. The best current design is:

```text
KAN layer current output
-> current-to-voltage conversion
-> tanh-like bounded gain block
-> next layer voltage input V = k*x
```

This makes the complete physical KAN much more feasible. The next priority is to improve physical pulse training, not to simply add more memristive states.

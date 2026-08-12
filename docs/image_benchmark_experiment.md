# MNIST and Fashion-MNIST Image Benchmark

This experiment adds MNIST and Fashion-MNIST as 10-class classification datasets and compares three routes:

- `mlp`: software dense MLP baseline.
- `gilbert_kan`: polynomial KAN with powers `(1, 3, 5)`, then physical mapping through Gilbert-generated voltage rows and RRAM differential pairs.
- `gbf_kan`: fixed generalized-bell basis KAN, then physical mapping through GBF cells, TIAs, and RRAM differential pairs.

The first run is a pilot subset benchmark, not a final large-scale image-recognition claim. Images are flattened from `28 x 28` to `784` features and scaled to `[-1, 1]`.

## Why This Evaluation

RBF-KAN and FastKAN-style papers motivate comparing KAN variants by accuracy and parameter count. For hardware KAN, we also need the physical resources:

- trainable parameter count;
- mapped memristor count;
- Gilbert multiplier count for polynomial KAN;
- GBF-cell and TIA count for GBF-KAN.

The runner supports both same-hidden-width and roughly matched-parameter comparisons.

## Commands

Same hidden width pilot:

```powershell
python -m kan_memristor.experiments.image_benchmark `
  --datasets mnist fashion_mnist `
  --models mlp gilbert_kan gbf_kan `
  --n-train 1024 `
  --n-test 512 `
  --hidden-width 16 `
  --mlp-hidden-width 64 `
  --epochs 8 `
  --output-dir outputs/image_benchmark_pilot
```

Roughly matched trainable-parameter pilot:

```powershell
python -m kan_memristor.experiments.image_benchmark `
  --datasets mnist fashion_mnist `
  --models mlp gilbert_kan gbf_kan `
  --n-train 1024 `
  --n-test 512 `
  --kan-hidden-width 16 `
  --gbf-hidden-width 5 `
  --mlp-hidden-width 48 `
  --epochs 8 `
  --output-dir outputs/image_benchmark_matched_params_pilot
```

## Matched-Parameter Pilot Results

| Dataset | Model | Stage | Accuracy | Trainable parameters | Physical resources |
| --- | --- | --- | ---: | ---: | --- |
| MNIST | MLP | Software | `87.50%` | `38,170` | - |
| MNIST | Gilbert-KAN | Software | `41.60%` | `38,138` | - |
| MNIST | Gilbert-KAN | Physical mapped | `36.91%` | `76,250` | `76,224` memristors, `4,800` Gilbert multipliers |
| MNIST | GBF-KAN | Software | `11.72%` | `35,745` | - |
| MNIST | GBF-KAN | Physical mapped | `11.72%` | `71,475` | `71,460` memristors, `7,101` GBF cells/TIAs |
| Fashion-MNIST | MLP | Software | `78.32%` | `38,170` | - |
| Fashion-MNIST | Gilbert-KAN | Software | `71.68%` | `38,138` | - |
| Fashion-MNIST | Gilbert-KAN | Physical mapped | `69.34%` | `76,250` | `76,224` memristors, `4,800` Gilbert multipliers |
| Fashion-MNIST | GBF-KAN | Software | `31.05%` | `35,745` | - |
| Fashion-MNIST | GBF-KAN | Physical mapped | `31.25%` | `71,475` | `71,460` memristors, `7,101` GBF cells/TIAs |

## Interpretation

The MLP is the strongest raw-pixel baseline in this pilot. The Gilbert-polynomial KAN is usable on Fashion-MNIST but much weaker on MNIST under the small subset and short training schedule. Its physical mapping loses additional accuracy, consistent with earlier findings that the soft-clipped Gilbert multiplier changes the effective basis.

The fixed GBF-KAN performs very well on low-dimensional continuous inputs, but it is weak on raw flattened images in this configuration. This is probably because fixed membership functions over individual pixel intensities do not capture spatial structure. Each pixel is treated independently on KAN edges, so the model has no convolutional locality or learned feature extractor.

This does not mean the GBF method is bad; it means the raw-pixel image setup is not the right showcase yet. Better next steps are:

- use PCA or patch features before the KAN layer;
- add a small convolutional or pooling front-end, then map the KAN classifier;
- sweep GBF centers/widths for pixel distributions while still keeping them fixed during training;
- train longer on larger subsets after the architecture is chosen.

For the current report, the honest conclusion is: GBF-KAN is excellent for the continuous 2D tasks tested earlier, while raw MNIST/FM​NIST requires a feature-extraction stage or basis redesign.

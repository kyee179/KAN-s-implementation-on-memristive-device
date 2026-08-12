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

The runner supports both same-hidden-width and roughly matched-parameter comparisons. For KAN-style models, same hidden-node count is often the more natural comparison because each edge has an internal basis expansion; equal node count does not imply equal trainable parameter count.

## Commands

Same hidden width pilot:

```powershell
python -m kan_memristor.experiments.image_benchmark `
  --datasets mnist fashion_mnist `
  --models mlp gilbert_kan gbf_kan `
  --n-train 1024 `
  --n-test 512 `
  --hidden-width 32 `
  --mlp-hidden-width 32 `
  --epochs 8 `
  --output-dir outputs/image_benchmark_same_nodes_pilot
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

## Same-Node Pilot Results

This run uses 32 hidden nodes for MLP, Gilbert-KAN, and GBF-KAN. It is closer to how KAN papers often compare architectures, but it gives KAN models more trainable edge-basis parameters than the MLP.

| Dataset | Model | Stage | Accuracy | Trainable parameters | Physical resources |
| --- | --- | --- | ---: | ---: | --- |
| MNIST | MLP | Software | `86.52%` | `25,450` | - |
| MNIST | Gilbert-KAN | Software | `71.88%` | `76,266` | - |
| MNIST | Gilbert-KAN | Physical mapped | `66.80%` | `152,490` | `152,448` memristors, `4,896` Gilbert multipliers |
| MNIST | GBF-KAN | Software | `9.57%` | `228,714` | - |
| MNIST | GBF-KAN | Physical mapped | `9.57%` | `457,386` | `457,344` memristors, `7,344` GBF cells/TIAs |
| Fashion-MNIST | MLP | Software | `77.93%` | `25,450` | - |
| Fashion-MNIST | Gilbert-KAN | Software | `71.29%` | `76,266` | - |
| Fashion-MNIST | Gilbert-KAN | Physical mapped | `72.07%` | `152,490` | `152,448` memristors, `4,896` Gilbert multipliers |
| Fashion-MNIST | GBF-KAN | Software | `55.66%` | `228,714` | - |
| Fashion-MNIST | GBF-KAN | Physical mapped | `55.27%` | `457,386` | `457,344` memristors, `7,344` GBF cells/TIAs |

## Unrestricted-Capacity Pilot Results

This run removes the node/parameter fairness constraint and asks which model works best under a reasonable CPU pilot budget. The tested settings were:

- MLP: `[784, 256, 10]`, 25 epochs, plus a larger `[784, 512, 10]`, 30 epoch check.
- Gilbert-KAN: `[784, 64, 10]`, 25 epochs, plus a larger `[784, 128, 10]`, 30 epoch check.
- GBF-KAN: `[784, 64, 10]`, 25 epochs, plus a larger `[784, 128, 10]`, 30 epoch check.

Best observed results:

| Dataset | Best model | Stage | Accuracy | Trainable parameters | Physical resources |
| --- | --- | --- | ---: | ---: | --- |
| MNIST | MLP `[784,512,10]` | Software | `94.38%` | `407,050` | - |
| MNIST | Gilbert-KAN `[784,128,10]` | Physical mapped | `87.79%` | `609,930` | `609,792` memristors, `5,472` Gilbert multipliers |
| MNIST | GBF-KAN `[784,128,10]` | Physical mapped | `76.27%` | `1,829,514` | `1,829,376` memristors, `8,208` GBF cells/TIAs |
| Fashion-MNIST | MLP `[784,256,10]` | Software | `84.42%` | `203,530` | - |
| Fashion-MNIST | Gilbert-KAN `[784,128,10]` | Physical mapped | `81.98%` | `609,930` | `609,792` memristors, `5,472` Gilbert multipliers |
| Fashion-MNIST | GBF-KAN `[784,64,10]` | Physical mapped | `79.44%` | `914,762` | `914,688` memristors, `7,632` GBF cells/TIAs |

The unrestricted result is much more favorable to KANs than the matched-parameter pilot. Gilbert-KAN becomes close to the MLP on Fashion-MNIST and reasonably strong on MNIST. GBF-KAN also improves strongly with capacity, especially on MNIST, but it still uses many more memristive devices and remains behind the MLP and Gilbert-KAN for raw flattened images.

## Interpretation

The MLP is the strongest raw-pixel baseline in these pilot runs. With the corrected same-node comparison, Gilbert-KAN becomes much stronger than in the matched-parameter run, reaching `66.80%` mapped accuracy on MNIST and `72.07%` on Fashion-MNIST. This confirms that hidden-node count matters for KAN-style comparisons. The physical mapping still loses some MNIST accuracy, consistent with earlier findings that the soft-clipped Gilbert multiplier changes the effective basis.

The fixed GBF-KAN performs very well on low-dimensional continuous inputs, but it is weaker on raw flattened images. Same-node comparison helps Fashion-MNIST but not MNIST; unrestricted capacity helps both, but at a high device-count cost. This is probably because fixed membership functions over individual pixel intensities do not capture spatial structure, and for MNIST many pixels are near the background value. Each pixel is treated independently on KAN edges, so the model has no convolutional locality or learned feature extractor.

This does not mean the GBF method is bad; it means the raw-pixel image setup is not the right showcase yet. Better next steps are:

- use PCA or patch features before the KAN layer;
- add a small convolutional or pooling front-end, then map the KAN classifier;
- sweep GBF centers/widths for pixel distributions while still keeping them fixed during training;
- train longer on larger subsets after the architecture is chosen.

For the current report, the honest conclusion is: same-node comparison is the fairer KAN-style comparison, and unrestricted capacity shows that both KAN variants can improve substantially. However, for raw MNIST/Fashion-MNIST the MLP remains the best observed model in this pilot. Gilbert-KAN is the stronger hardware KAN route for raw images, while fixed GBF-KAN likely needs a feature-extraction stage or basis redesign.

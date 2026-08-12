# MNIST and Fashion-MNIST Image Benchmark

This experiment adds MNIST and Fashion-MNIST as 10-class classification datasets and compares three routes:

- `mlp`: software dense MLP baseline.
- `software_kan`: original software cubic B-spline KAN baseline.
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
  --models mlp software_kan gilbert_kan gbf_kan `
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
  --models mlp software_kan gilbert_kan gbf_kan `
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
- Software B-spline KAN: `[784, 64, 10]`, 25 epochs, plus a larger `[784, 128, 10]`, 30 epoch check.
- Gilbert-KAN: `[784, 64, 10]`, 25 epochs, plus a larger `[784, 128, 10]`, 30 epoch check.
- GBF-KAN: `[784, 64, 10]`, 25 epochs, plus a larger `[784, 128, 10]`, 30 epoch check.

Best observed results:

| Dataset | Best model | Stage | Accuracy | Trainable parameters | Physical resources |
| --- | --- | --- | ---: | ---: | --- |
| MNIST | MLP `[784,512,10]` | Software | `94.38%` | `407,050` | - |
| MNIST | Software B-spline KAN `[784,64,10]` | Software | `90.09%` | `711,498` | - |
| MNIST | Gilbert-KAN `[784,128,10]` | Physical mapped | `87.79%` | `609,930` | `609,792` memristors, `5,472` Gilbert multipliers |
| MNIST | GBF-KAN `[784,128,10]` | Physical mapped | `76.27%` | `1,829,514` | `1,829,376` memristors, `8,208` GBF cells/TIAs |
| Fashion-MNIST | MLP `[784,256,10]` | Software | `84.42%` | `203,530` | - |
| Fashion-MNIST | Software B-spline KAN `[784,64,10]` | Software | `83.06%` | `711,498` | - |
| Fashion-MNIST | Gilbert-KAN `[784,128,10]` | Physical mapped | `81.98%` | `609,930` | `609,792` memristors, `5,472` Gilbert multipliers |
| Fashion-MNIST | GBF-KAN `[784,64,10]` | Physical mapped | `79.44%` | `914,762` | `914,688` memristors, `7,632` GBF cells/TIAs |

The unrestricted result is much more favorable to KANs than the matched-parameter pilot. The original software B-spline KAN is the strongest KAN-family model on MNIST in this run, but it does not beat the MLP and it has more trainable parameters. Gilbert-KAN becomes close to the MLP on Fashion-MNIST and reasonably strong on MNIST. GBF-KAN also improves strongly with capacity, especially on MNIST, but it still uses many more memristive devices and remains behind the MLP and Gilbert-KAN for raw flattened images.

## Full-Dataset Results

This run uses the full torchvision train/test splits for MNIST and Fashion-MNIST: `60,000` training images and `10,000` test images. To make the comparison closer to the FastKAN MNIST setup, all four models use hidden width `64`, `20` epochs, batch size `64`, and AdamW with learning rate `1e-3`.

Command pattern:

```powershell
python -m kan_memristor.experiments.image_benchmark `
  --datasets mnist fashion_mnist `
  --models mlp software_kan gilbert_kan gbf_kan `
  --n-train 60000 `
  --n-test 10000 `
  --hidden-width 64 `
  --mlp-hidden-width 64 `
  --spline-hidden-width 64 `
  --kan-hidden-width 64 `
  --gbf-hidden-width 64 `
  --epochs 20 `
  --batch-size 64 `
  --learning-rate 1e-3
```

The actual run was split by model to keep long CPU jobs easier to resume and inspect.

| Dataset | Model | Stage | Accuracy | Trainable parameters | Physical resources |
| --- | --- | --- | ---: | ---: | --- |
| MNIST | MLP `[784,64,10]` | Software | `97.51%` | `50,890` | - |
| MNIST | Software B-spline KAN `[784,64,10]` | Software | `91.53%` | `711,498` | - |
| MNIST | Gilbert-KAN `[784,64,10]` | Software pretrain | `94.64%` | `152,522` | - |
| MNIST | Gilbert-KAN `[784,64,10]` | Physical mapped | `93.84%` | `304,970` | `304,896` memristors, `5,088` Gilbert multipliers |
| MNIST | GBF-KAN `[784,64,10]` | Software pretrain | `89.65%` | `457,418` | - |
| MNIST | GBF-KAN `[784,64,10]` | Physical mapped | `89.46%` | `914,762` | `914,688` memristors, `7,632` GBF cells/TIAs |
| Fashion-MNIST | MLP `[784,64,10]` | Software | `87.98%` | `50,890` | - |
| Fashion-MNIST | Software B-spline KAN `[784,64,10]` | Software | `84.79%` | `711,498` | - |
| Fashion-MNIST | Gilbert-KAN `[784,64,10]` | Software pretrain | `85.53%` | `152,522` | - |
| Fashion-MNIST | Gilbert-KAN `[784,64,10]` | Physical mapped | `83.10%` | `304,970` | `304,896` memristors, `5,088` Gilbert multipliers |
| Fashion-MNIST | GBF-KAN `[784,64,10]` | Software pretrain | `82.65%` | `457,418` | - |
| Fashion-MNIST | GBF-KAN `[784,64,10]` | Physical mapped | `82.32%` | `914,762` | `914,688` memristors, `7,632` GBF cells/TIAs |

The full-data run explains why the FastKAN paper can report MNIST accuracy around `0.97`: even the small MLP baseline reaches `97.51%` when trained on the full dataset. The earlier pilot was mostly limited by the `4,096`-sample training subset.

Among the KAN-family models, Gilbert-KAN is strongest in this full-data image setting. Its physical mapping loses only `0.80` percentage points on MNIST, but loses `2.43` percentage points on Fashion-MNIST. GBF-KAN maps almost exactly from software to physical form, but the fixed basis is weaker on raw flattened pixels. The original software B-spline KAN is not competitive with the FastKAN-style result here, which likely reflects implementation and training differences: this repo's B-spline model does not use FastKAN's layer normalization, RBF approximation, scheduler, or SiLU base update.

## Interpretation

The MLP is the strongest raw-pixel baseline in these runs. On the full dataset, the MLP reaches `97.51%` on MNIST and `87.98%` on Fashion-MNIST with only `50,890` trainable parameters. The original software B-spline KAN is competitive in the small pilot, reaching `90.09%` on MNIST and `83.06%` on Fashion-MNIST at width 64, but it does not reproduce the FastKAN paper's MNIST result in this implementation. Increasing to width 128 lowered small-pilot test accuracy slightly while reducing training loss, suggesting overfitting in the small-data pilot. With the corrected same-node comparison, Gilbert-KAN becomes much stronger than in the matched-parameter run, reaching `66.80%` mapped accuracy on MNIST and `72.07%` on Fashion-MNIST. In the full-data run it reaches `93.84%` mapped MNIST accuracy and `83.10%` mapped Fashion-MNIST accuracy. This confirms that hidden-node count and training-set size both matter for KAN-style comparisons. The physical mapping still loses some accuracy, consistent with earlier findings that the soft-clipped Gilbert multiplier changes the effective basis.

The fixed GBF-KAN performs very well on low-dimensional continuous inputs, but it is weaker on raw flattened images. Same-node comparison helps Fashion-MNIST but not MNIST; unrestricted capacity helps both, but at a high device-count cost. This is probably because fixed membership functions over individual pixel intensities do not capture spatial structure, and for MNIST many pixels are near the background value. Each pixel is treated independently on KAN edges, so the model has no convolutional locality or learned feature extractor.

This does not mean the GBF method is bad; it means the raw-pixel image setup is not the right showcase yet. Better next steps are:

- use PCA or patch features before the KAN layer;
- add a small convolutional or pooling front-end, then map the KAN classifier;
- sweep GBF centers/widths for pixel distributions while still keeping them fixed during training;
- train longer on larger subsets after the architecture is chosen.

For the current report, the honest conclusion is: same-node comparison is the fairer KAN-style comparison, and unrestricted capacity shows that KAN variants can improve substantially. However, for raw MNIST/Fashion-MNIST the MLP remains the best observed model in this pilot. The original software B-spline KAN is a useful upper software reference, Gilbert-KAN is the stronger hardware KAN route for raw images, and fixed GBF-KAN likely needs a feature-extraction stage or basis redesign.

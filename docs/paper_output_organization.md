# Paper Output Organization

This note explains how the result artifacts are organized for a paper about implementing KANs on memristive devices with two hardware routes:

- Gilbert-multiplier polynomial KAN;
- generalized-bell-function (GBF) KAN.

The paper-facing output folder is:

```text
outputs/paper_ready/
```

Regenerate it with:

```powershell
python -m kan_memristor.experiments.prepare_paper_outputs
```

## Main Results To Use

The main paper should use the original two project datasets:

- `complicated_function`, reported by test MSE;
- `taglietti_yinyang`, reported by test accuracy.

The most useful result table is:

```text
outputs/paper_ready/tables/main_accuracy_energy_summary.csv
```

It keeps four routes:

| Route | Role in paper |
| --- | --- |
| MLP baseline | Conventional software baseline |
| Software B-spline KAN | Ideal KAN upper reference |
| Gilbert physical KAN | Polynomial hardware route using Gilbert-generated powers and RRAM weights |
| GBF physical KAN | Localized-basis hardware route using GBF cells, TIAs, and RRAM weights |

The corresponding paper-candidate figure is:

```text
outputs/paper_ready/figures/main_accuracy_energy_comparison.svg
```

## Result Selection

Use the normalized complete-physical Gilbert result as the main Gilbert result. It is more physically meaningful than the early mapped-only abstraction because it includes Gilbert-generated power rows, DynamicMemdiode-style conductance mapping, RRAM differential weights, and inter-layer normalization.

Use the fixed-basis mapped GBF result as the main GBF result. It is the cleanest result for the second hardware route because the GBF basis is already a physical frontend and the software-to-physical gap is small.

Use the software B-spline KAN only as an upper reference. It validates the KAN idea, but it is not the proposed hardware implementation.

## Supplementary Results

The polynomial basis expansion results are useful as supplementary evidence:

```text
outputs/paper_ready/tables/supplementary_polynomial_sweep.csv
```

They show that expanding the Gilbert route from odd powers to `1..5` or `1..7` improves ideal software regression, but does not automatically improve the mapped physical regression result. This supports the argument that hardware non-idealities and scaling matter, not only basis expressivity.

## Archived Outputs

The output folder now archives low-priority runs under:

```text
outputs/_archive/
```

This includes removed image benchmark outputs and smoke/sanity runs. They are preserved locally but should not be used as main paper evidence.

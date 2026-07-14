# Hardware Physical Models

This note records the behavioral hardware models used before building the full physical KAN network. The goal is not to replace SPICE. The goal is to create differentiable or vectorized Python models with parameters that can later be calibrated against SPICE or measurement data.

## Memristive Device Model

References:

- Aguirre, Sune, and Miranda (2022), "SPICE Implementation of the Dynamic Memdiode Model for Bipolar Resistive Switching Devices".
- Mehonic et al. (2019), "Simulation of Inference Accuracy Using Realistic RRAM Devices".

The model in `kan_memristor.hardware.memristor` has two levels.

First, `DynamicMemdiode` models the bipolar resistive-switching device itself. It uses a compact memdiode conduction law:

```text
I = I0(lambda) * sinh(alpha(lambda) * (V - R(lambda) * I))
I0(lambda) = Ioff + (Ion - Ioff) * lambda
```

`lambda = 0` corresponds to HRS and `lambda = 1` corresponds to LRS. The implicit current equation is solved with Newton iterations. The state update follows the SET/RESET balance structure:

```text
dlambda/dt = (1 - lambda) / tau_set(lambda, Vc) - lambda / tau_reset(lambda, Vc)
```

For practical simulation, SET and RESET are treated by voltage sign, matching the separated calibration approach described in Aguirre et al. The default parameters follow the paper's LTSpice example values where possible: `Ioff=1e-7 A`, `Ion=1e-2 A`, `eta_set=50`, `eta_reset=100`, `Vset=1.4 V`, `Vreset=-0.4 V`, `snapback_current=2e-4 A`, and an initial HRS/LRS state variable in `[0, 1]`.

Second, `RRAMWeightMapper` models how learned coefficients become physical weights. Following Mehonic et al., continuous weights are mapped proportionally onto finite conductance states and rounded to the nearest available state. Signed coefficients are represented with a differential pair: one device for the positive branch and one for the negative branch. The mapper can also inject simple non-idealities: finite HRS/LRS ratio, finite number of states, device-to-device variability, stuck-LRS devices, stuck-HRS devices, and approximate I/V nonlinearity.

## Gilbert Voltage Multiplier Model

Reference:

- Renduchintala, Hannah, and Kumar (2026), "Study of high-speed CMOS-based Gilbert voltage multiplier".

`kan_memristor.hardware.gilbert_multiplier` models the Gilbert cell as an analog voltage product block:

```text
Vout = k * Vx * Vy
```

The behavioral model includes finite input range, output clipping by the supply, a first-order low-pass response, static power, and operation-energy estimation. Defaults follow the cited 45 nm CMOS design where possible: `Vdd=1 V`, input transfer sweep about `+-400 mV`, `3 dB bandwidth=10 GHz`, `power=440 uW`, and reported voltage gain around `5.5 dB`. The product gain remains an explicit calibration parameter because the exact scale depends on circuit bias and load.

## Odd-Polynomial Edge Hardware Mapping

The physically simpler KAN edge function is:

```text
phi_oi(x_i) = c_oi,1*x_i + c_oi,3*x_i^3 + c_oi,5*x_i^5
```

The code path in `kan_memristor.hardware.odd_polynomial_edge` maps the three coefficients into RRAM conductance pairs and produces powers with cascaded Gilbert multipliers:

```text
x3 = multiply(multiply(x, x), x)
x5 = multiply(multiply(multiply(multiply(x, x), x), x), x)
```

This keeps the primitive odd-symmetric while making the physical cost explicit: each edge needs coefficient storage and repeated multiplication blocks for higher powers. The next step is to combine this hardware model with trained odd-polynomial KAN coefficients and sweep non-idealities to estimate accuracy degradation.

## Characterization CLI

Run:

```powershell
python -m kan_memristor.experiments.hardware_characterization --output-dir outputs/hardware_characterization
```

This writes plots and `metrics.json` for:

- memdiode I-V curves vs. memory state
- memdiode SET/RESET pulse response
- Gilbert multiplier product error
- RRAM conductance quantization error
- odd-polynomial hardware edge symmetry

## Initial Characterization Results

Run date: 2026-07-14. Command:

```powershell
python -m kan_memristor.experiments.hardware_characterization --output-dir outputs/hardware_characterization
```

| Block | Metric | Value |
| --- | --- | ---: |
| Memdiode | HRS current at 0.2 V | 4.11e-08 A |
| Memdiode | LRS current at 0.2 V | 1.82e-03 A |
| Memdiode | LRS/HRS conductance ratio at 0.2 V | 4.44e04 |
| Memdiode | peak lambda during SET/RESET schedule | 1.00 |
| Memdiode | final lambda after RESET schedule | 0.279 |
| Gilbert multiplier | product RMSE over +-0.4 V grid | 1.64e-02 V |
| Gilbert multiplier | max absolute product error | 6.72e-02 V |
| Gilbert multiplier | energy per multiply at 10 GHz | 4.40e-14 J |
| RRAM mapper | default HRS/LRS resistance ratio | 100 |
| RRAM mapper | finite conductance states | 16 |
| RRAM mapper | max coefficient quantization error | 0.0333 |
| Hardware edge | max odd-symmetry error | 0 |

These numbers are baseline behavioral outputs, not calibrated claims. They are useful for checking that the Python hardware model is internally consistent before connecting it to the trained odd-polynomial KAN network.

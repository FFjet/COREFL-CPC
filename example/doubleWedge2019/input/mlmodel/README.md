# Exported Header Format

This folder contains auto-generated `nnWeights`-style C++ headers exported from
trained surrogate models.

Current artifacts:

- `nnWeights_air5_transport_compact_mlp_sub2000.H`
- `nnWeights_air5_diffusion_compact_mlp_sub3000.H`

## What is inside

Each header defines one compact feed-forward `MLP` in the namespace:

```cpp
Foam::exportedNN
```

The file exposes:

- `inputCount`
- `hiddenCount`
- `hiddenLayers`
- `outputCount`
- `speciesCount`
- `inputScalerKind`
- `outputScalerKind`
- `inputLogEpsilon`
- `outputLogEpsilon`
- `supportsTvInput()`
- `weights()`
- `biases()`
- `inputMins()`
- `inputScales()`
- `outputMins()`
- `outputScales()`
- `speciesNames()`

The weights and biases are flattened in layer order for a dense MLP:

```text
input -> hidden_1 -> hidden_2 -> ... -> hidden_N -> output
```

The exporter assumes:

- only `MLP` headers are supported directly
- input columns must start with temperature and pressure
- optional `Tve`/`Tv` comes next
- species mass-fraction inputs follow after the thermo inputs

So the canonical input order for the Air-5 headers here is:

```text
T, p, Tve, N2, O2, NO, N, O
```

## Scaling

The runtime must apply the stored input scaling before the network and inverse
the stored output scaling after the network.

The scaler metadata is embedded in the header:

- `inputScalerKind`
- `outputScalerKind`
- `inputLogEpsilon`
- `outputLogEpsilon`

Current files in this folder use:

- transport header:
  - input scaler: `standard`
  - output scaler: `standard`
- diffusion header:
  - input scaler: `standard`
  - output scaler: `log_standard`

That means the diffusion outputs are trained/exported in log-scaled target
space and must be inverse-transformed accordingly at runtime.

## Output order

Transport header output order:

```text
mu, l_tr, l_int
```

Diffusion header output order:

```text
D_N2, D_O2, D_NO, D_N, D_O
```

## Source models

The paired model/scaler artifacts live in:

```text
data/models/
```

Matching exported artifacts:

- `air5_transport_compact_mlp_sub2000_export_*`
- `air5_diffusion_compact_mlp_sub3000_export_*`

Use those files if you need Python-side validation against the generated header.

# Two-Temperature Inviscid Flux Jacobian, WENO Characteristic Projection, and Code Audit

This note audits the two-temperature extension in `src` against the finite-difference inviscid-flux derivation in the supplied PDF `化学反应流动无粘通量雅可比矩阵及其特征向量.pdf`, the WENO flux-discretization workflow in `无粘通量离散.pdf`, and the [SU2 v7 thermochemical nonequilibrium form](https://su2code.github.io/docs_v7/Theory/).  The second PDF has a damaged cross-reference table in the repository copy; Ghostscript recovered only its first page.  That recovered page is enough to identify the intended workflow: reconstruct half-point inviscid fluxes with local Lax-Friedrichs splitting, use a Roe-averaged state at `i+1/2`, project the stencil with the left eigenvectors, reconstruct by WENO-Z or linear upwind depending on the shock sensor, and project the reconstructed characteristic flux back with the right eigenvectors.

## Governing Variables

SU2 writes the nonequilibrium conservative vector as species densities plus momentum, total energy, and vibrational-electronic energy.  COREFL stores the same information as one mixture density plus conservative scalar equations:

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E,
     rho*Y_1, ..., rho*Y_Ns, rho*Eve, ...]^T .
```

Here `Eve` is the mixture vibrational-electronic energy per unit mass.  In the current code it is appended as scalar index

```text
i_eve    = n_spec
i_eve_cv = 5 + n_spec
```

when `COREFL_ENABLE_TWO_TEMPERATURE=ON`, `species=1`, and the finite-rate mixture path is used.

The total specific energy closure is

```text
E = 0.5*(u^2 + v^2 + w^2) + e_tr(T,Y) + Eve
```

with

```text
h_tr,s(T) = h_eq,s(T) - E_ve,s(T)
e_tr,s(T) = h_tr,s(T) - R_s*T
e_tr(T,Y) = sum_s Y_s e_tr,s(T).
```

The code implements this split in `compute_total_energy` and primitive recovery: it subtracts the equilibrium `E_ve(T)` contribution from the NASA enthalpy at the translational temperature and then adds the transported `Eve`.

## Inviscid Flux in an Arbitrary Curvilinear Direction

Let

```text
k = (k1, k2, k3)^T,       |k| = sqrt(k1^2+k2^2+k3^2),
n = k/|k|,                U_k = k1*u + k2*v + k3*w,
U_n = n dot u.
```

The inviscid flux in direction `k` is

```text
F_k(Q) =
[ rho*U_k,
  rho*u*U_k + k1*p,
  rho*v*U_k + k2*p,
  rho*w*U_k + k3*p,
  (rho*E + p)*U_k,
  rho*Y_1*U_k,
  ...,
  rho*Y_Ns*U_k,
  rho*Eve*U_k,
  ... ]^T .
```

The additional two-temperature convective flux is therefore exactly `rho*Eve*U_k`, matching the SU2 form.

## Frozen Two-Temperature Pressure Differential

For the inviscid characteristic decomposition, chemistry and vibrational relaxation are frozen.  Define

```text
Gamma = gamma_tr - 1
c^2   = gamma_tr * R_mix * T
K     = 0.5*(u^2+v^2+w^2)
```

where `gamma_tr = cp_tr/cv_tr`.  Starting from

```text
rho*e = rho*E - (rho*u)^2/(2*rho) - (rho*v)^2/(2*rho) - (rho*w)^2/(2*rho)
      = rho*e_tr(T,Y) + rho*Eve
p     = T * sum_s (rho*Y_s) R_s,
```

the differential of pressure with respect to the conservative variables is

```text
dp = Gamma*(d(rho*E) - u*d(rho*u) - v*d(rho*v) - w*d(rho*w)
            + K*d(rho) - d(rho*Eve))
     + sum_s [R_s*T - Gamma*e_tr,s(T)] d(rho*Y_s).
```

Using `h_tr,s = e_tr,s + R_s*T`, the species coefficient becomes

```text
R_s*T - Gamma*e_tr,s = gamma_tr*R_s*T - Gamma*h_tr,s.
```

It is useful to divide the scalar pressure coefficients by `c^2`:

```text
alpha_s   = (gamma_tr*R_s*T - Gamma*h_tr,s) / c^2
alpha_eve = -Gamma / c^2
alpha_ps  = 0       for passive/turbulence scalars not entering p.
```

COREFL computes these in `compute_mixture_characteristic_thermo`:

```text
h_tr,s      = h_eq,s(T) - E_ve,s(T)
alpha_s     = (gamma*R_s*T - (gamma-1)*h_tr,s)/c^2
alpha_eve   = -(gamma-1)/c^2
energy_s    = -alpha_s*c^2/(gamma-1)
energy_eve  = 1
```

The `energy_*` coefficients are the total-energy entries of the scalar right eigenvectors, chosen so that scalar waves have `dp=0`.

## Eigenvalues

The inviscid Jacobian in direction `k` has the eigenvalues

```text
lambda_- = U_k - c*|k|
lambda_0 = U_k                 multiplicity: n_scalar_transported + 3
lambda_+ = U_k + c*|k|.
```

The three nonscalar `lambda_0` waves are the two shear waves and the entropy/contact wave.  The remaining `lambda_0` waves are transported scalar waves, including `rho*Eve`.

## Right Eigenvectors

Use an orthonormal frame `(n, sigma, tau)` with `sigma dot n = tau dot n = sigma dot tau = 0`.  Define

```text
U_sigma = sigma dot u
U_tau   = tau dot u
H       = E + p/rho
phi_a   = scalar_a/rho
chi_a   = -alpha_a*c^2/Gamma.
```

For species scalars, `chi_s = -alpha_s*c^2/Gamma`; for `Eve`, `chi_eve = 1`; for passive/turbulence scalars that do not enter pressure, `chi_a = 0`.

In conservative variables ordered as `[rho, rho*u, rho*v, rho*w, rho*E, rho*phi_a]`, a right eigenbasis is

```text
r_- =
[ 1,
  u - c*n_x,
  v - c*n_y,
  w - c*n_z,
  H - c*U_n,
  phi_1, ..., phi_M ]^T

r_sigma =
[ 0, sigma_x, sigma_y, sigma_z, U_sigma, 0, ..., 0 ]^T

r_tau =
[ 0, tau_x, tau_y, tau_z, U_tau, 0, ..., 0 ]^T

r_c =
[ 1,
  u,
  v,
  w,
  H - c^2/Gamma,
  phi_1, ..., phi_M ]^T

r_+ =
[ 1,
  u + c*n_x,
  v + c*n_y,
  w + c*n_z,
  H + c*U_n,
  phi_1, ..., phi_M ]^T

r_phi_a =
[ 0, 0, 0, 0, chi_a, 0, ..., 1 at scalar a, ..., 0 ]^T .
```

The scalar eigenvector for `rho*Eve` therefore has a unit entry in both `rho*E` and `rho*Eve`; this represents a frozen exchange of total energy and vibrational-electronic energy that leaves the translational pressure unchanged.

## Left Characteristic Projection Used by WENO

For a conservative perturbation or flux vector `q`, define

```text
q_rho = q[0]
q_m   = (q[1], q[2], q[3])
q_E   = q[4]
beta(q) = sum_a alpha_a*q_phi_a .
```

The left projections are

```text
L_- q =
0.5 * [ ((Gamma*K + c*U_n)/c^2)*q_rho
        - ((Gamma*u + c*n) dot q_m)/c^2
        + (Gamma/c^2)*q_E
        + beta(q) ]

L_sigma q = -U_sigma*q_rho + sigma dot q_m

L_tau q = -U_tau*q_rho + tau dot q_m

L_c q =
(1 - Gamma*K/c^2)*q_rho
  + (Gamma/c^2)*(u dot q_m)
  - (Gamma/c^2)*q_E
  - beta(q)

L_+ q =
0.5 * [ ((Gamma*K - c*U_n)/c^2)*q_rho
        - ((Gamma*u - c*n) dot q_m)/c^2
        + (Gamma/c^2)*q_E
        + beta(q) ]

L_phi_a q = q_phi_a - phi_a*q_rho .
```

This is exactly what `WENO.cu` evaluates:

```text
sumBetaQ = sum_a alpha_a * Q_phi_a
sumBetaF = sum_a alpha_a * F_phi_a

case 0: L_-      with +0.5*sumBeta
case 1: shear sigma
case 2: shear tau
case 3: contact  with -sumBeta
case 4: L_+      with +0.5*sumBeta
scalar: Q_phi_a - phi_a*Q_rho
```

The projected characteristic flux is split by local Lax-Friedrichs:

```text
f_char^+ = 0.5 * J * (L F + lambda_max L Q)
f_char^- = 0.5 * J * (L F - lambda_max L Q)
lambda_max = max_m(|U_k,m| + c_m*|k_m|)
```

over the WENO stencil.  `WENO5` or `WENO7` then reconstructs the half-point characteristic flux.

## Back Projection to Physical Flux

Let the reconstructed characteristic fluxes be

```text
a_- = fChar[0]
a_s = fChar[1]
a_t = fChar[2]
a_c = fChar[3]
a_+ = fChar[4]
a_phi = fChar[5 + scalar_index].
```

COREFL writes the right multiplication explicitly:

```text
S = a_- + a_c + a_+
D = a_- - a_+

F_rho = S

F_m = u*S - c*n*D + sigma*a_s + tau*a_t

F_E = H*S - c*U_n*D + U_sigma*a_s + U_tau*a_t
      - c^2/Gamma*a_c
      + sum_a chi_a*a_phi

F_phi_a = phi_a*S + a_phi .
```

The two-temperature term is the `chi_eve*a_eve = 1*a_eve` contribution to total energy plus `phi_eve*S + a_eve` in the transported `rho*Eve` flux.  This is consistent with the frozen two-temperature pressure differential above.

## Equation-to-Code Audit

| Equation term | Code location | Audit result |
| --- | --- | --- |
| Compile-time 2T switch | `CMakeLists.txt`, `src/Define.h` | `COREFL_ENABLE_TWO_TEMPERATURE` sets `constexpr bool kTwoTemperature`; CUDA kernels use `if constexpr` where possible. |
| Add conservative scalar `rho*Eve` | `src/Parameter.cpp` | `Eve` is appended after species, `Tve` is added to output-only variables. Correct for the COREFL variable layout. |
| Total energy split `E = K + e_tr(T,Y) + Eve` | `src/FieldOperation.cuh`, `src/FieldOperation.cu`, `src/Thermo.cu` | Correct: equilibrium NASA enthalpy at `T` is reduced by `E_ve(T)` and the transported `Eve` is added. Primitive recovery solves `e_tr(T)=E-K-Eve` and inverts `Tve` from `Eve`. |
| Convective flux `rho*Eve*U_k` | `src/InviscidScheme.cu`, `src/WENO.cu` | Correct: scalar convective flux loops include `i_eve`, and the WENO characteristic scalar wave includes the `Eve` total-energy coupling. |
| WENO 2T characteristic coefficients | `src/Thermo.cuh`, `src/WENO.cu` | Correct: `alpha_s`, `alpha_eve`, `energy_coeff_s`, and `energy_coeff_eve` match the pressure differential and right eigenvectors above. |
| Chemical source `sum_s omega_s E_ve,s(Tve)` | `src/FiniteRateChem.cu` | Correct for explicit RK3 units: `omega_s` is kg/(m3 s), `E_ve,s` is J/kg, so the source is J/(m3 s). Current 2T mode is restricted to transient RK3, so this source is explicit; point-implicit `rho*Eve` chemistry is not implemented. |
| Landau-Teller source `theta_tr:ve` | `src/Thermo.cu`, `src/FieldOperation.cuh`, `src/RK.cuh` | Correct for vibrational relaxation: the RK stages apply an exact exponential update, which is the intended unconditionally stable explicit Landau-Teller update. Electronic relaxation is not modeled except through the transported/inverted `Eve`. |
| Viscous total-energy diffusion | `src/ViscousScheme.cu`, `src/Thermo.cuh` | Fixed in this audit. The species diffusion enthalpy is now `h_eq(T)-E_ve(T)+E_ve(Tve) = h_tr(T)+E_ve(Tve)`, matching the nonequilibrium total energy. |
| Viscous `rho*Eve` flux | `src/ViscousScheme.cu` | Fixed in this audit. The flux is now `kappa_ve grad(Tve) + sum_s (-J_s) E_ve,s(Tve)`, consistent with the code convention that stored species viscous flux is `-J_s`. The 8th-order collocated path now writes the `rho*Eve` flux instead of leaving it unset. |
| Wall heat-flux split | `src/PostProcess.cu` | Fixed in this audit. The postprocessed `q_ve` and `q_total` now use the same species diffusion sign and nonequilibrium enthalpy as the solver viscous flux. |
| Diffusive time-step contribution | `src/TimeAdvanceFunc.cu`, `src/FieldOperation.cuh` | Present: `thermal_conductivity_ve/(rho*cv_ve)` is included in the diffusive spectral radius when 2T is enabled. |

## Current Scope Limitations

The code deliberately blocks two-temperature runs outside transient RK3:

```text
steady = 0
temporal_scheme = 3
species = 1
reaction != 2
```

As a consequence, the `rho*Eve` chemical source is explicit in the currently enabled 2T solver path.  The existing point-implicit chemical Jacobian machinery only covers species equations and is not wired for the coupled `rho*Eve` source or Landau-Teller update in steady, dual-time, or Wu-splitting paths.

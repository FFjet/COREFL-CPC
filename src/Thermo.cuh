#pragma once
#include "Define.h"
#include "DParameter.cuh"

namespace cfd{
struct DZone;
struct Species;

__device__ void compute_enthalpy(real t, real *enthalpy, const DParameter* param);
// __device__ void compute_enthalpy_1(real t, real *enthalpy, const DParameter* param);

__device__ void compute_cp(real t, real *cp, const DParameter *param);
// __device__ void compute_cp_1(real t, real *cp, DParameter* param);

__device__ void compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param);
// __device__ void compute_enthalpy_and_cp_1(real t, real *enthalpy, real *cp, const DParameter *param);

__device__ void compute_gibbs_div_rt(real t, const DParameter* param, real* gibbs_rt);
// __device__ void compute_gibbs_div_rt_1(real t, real* gibbs_rt);

__device__ real compute_ve_energy(int i_spec, real t, const DParameter *param);

__device__ real compute_ve_cv(int i_spec, real t, const DParameter *param);

__device__ real compute_vib_energy(int i_spec, real t, const DParameter *param);

__device__ real compute_vib_cv(int i_spec, real t, const DParameter *param);

__device__ real compute_mixture_ve_energy(real t, const real *y, const DParameter *param, real *cv_ve = nullptr);

__device__ real compute_mixture_vib_energy(real t, const real *y, const DParameter *param, real *cv_v = nullptr);

__device__ real compute_mixture_tr_energy(real t, const real *y, const DParameter *param, real *cv_tr = nullptr,
  real *cp_tr = nullptr, real *gas_const_mix = nullptr);

__device__ real invert_tve_from_ev(real ev_target, const real *y, real t_init, const DParameter *param);

__device__ real invert_tve_from_eve(real eve_target, const real *y, real t_init, const DParameter *param);

__device__ real compute_vt_relaxation_source(real density, real t, real tve, const real *y, const DParameter *param,
  real *eve_eq = nullptr, real *tau_eff = nullptr);

__device__ __forceinline__ real compute_nonequilibrium_diffusion_enthalpy(
  real h_eq, int i_spec, real t, real tve, const DParameter *param) {
  if (!param->two_temperature || param->i_eve < 0) return h_eq;
  return h_eq - compute_ve_energy(i_spec, t, param) + compute_ve_energy(i_spec, tve, param);
}

__device__ __forceinline__ void compute_mixture_characteristic_thermo(
  real t, const real *y, const DParameter *param, real *h_tr, real *scalar_alpha = nullptr,
  real *energy_coeff = nullptr, real *e_tr_mix = nullptr, real *r_mix = nullptr, real *cp_tr_mix = nullptr,
  real *cv_tr_mix = nullptr, real *gamma_mix = nullptr, real *sound_speed = nullptr) {
  const real t_eval = max(t, static_cast<real>(1.0));
  real h_eq[MAX_SPEC_NUMBER], cp_eq[MAX_SPEC_NUMBER];
  compute_enthalpy_and_cp(t_eval, h_eq, cp_eq, param);

  real e_tr_local = 0.0;
  real r_local = 0.0;
  real cp_local = 0.0;
  real cv_local = 0.0;

  for (int l = 0; l < param->n_spec; ++l) {
    const real yi = y[l];
    const real eve_eq = compute_ve_energy(l, t_eval, param);
    const real cv_ve_eq = compute_ve_cv(l, t_eval, param);
    const real h_tr_i = h_eq[l] - eve_eq;
    const real cp_tr_i = cp_eq[l] - cv_ve_eq;
    if (h_tr != nullptr) h_tr[l] = h_tr_i;
    e_tr_local += yi * (h_tr_i - param->gas_const[l] * t_eval);
    r_local += yi * param->gas_const[l];
    cp_local += yi * cp_tr_i;
    cv_local += yi * (cp_tr_i - param->gas_const[l]);
  }

  const real gamma_local = cp_local / max(cv_local, static_cast<real>(1e-8));
  const real c_local = sqrt(max(gamma_local * r_local * t_eval, static_cast<real>(1e-12)));

  if (e_tr_mix != nullptr) *e_tr_mix = e_tr_local;
  if (r_mix != nullptr) *r_mix = r_local;
  if (cp_tr_mix != nullptr) *cp_tr_mix = cp_local;
  if (cv_tr_mix != nullptr) *cv_tr_mix = cv_local;
  if (gamma_mix != nullptr) *gamma_mix = gamma_local;
  if (sound_speed != nullptr) *sound_speed = c_local;

  if (scalar_alpha == nullptr && energy_coeff == nullptr) return;

  const int n_scalar = param->n_scalar_transported;
  if (scalar_alpha != nullptr) {
    for (int l = 0; l < n_scalar; ++l) scalar_alpha[l] = 0.0;
  }
  if (energy_coeff != nullptr) {
    for (int l = 0; l < n_scalar; ++l) energy_coeff[l] = 0.0;
  }

  const real gm1 = gamma_local - 1.0;
  const real c2 = max(c_local * c_local, static_cast<real>(1e-30));
  const real inv_gm1 = 1.0 / max(gm1, static_cast<real>(1e-12));

  for (int l = 0; l < param->n_spec; ++l) {
    const real h_tr_i = h_tr != nullptr ? h_tr[l] : h_eq[l] - compute_ve_energy(l, t_eval, param);
    const real alpha = (gamma_local * param->gas_const[l] * t_eval - gm1 * h_tr_i) / c2;
    if (scalar_alpha != nullptr) scalar_alpha[l] = alpha;
    if (energy_coeff != nullptr) energy_coeff[l] = -alpha * c2 * inv_gm1;
  }

  if (param->i_eve >= 0) {
    if (scalar_alpha != nullptr) scalar_alpha[param->i_eve] = -gm1 / c2;
    if (energy_coeff != nullptr) energy_coeff[param->i_eve] = 1.0;
  }
}

template<typename ScalarField>
__device__ __forceinline__ void gather_species_mass_fractions(const ScalarField &sv, int i, int j, int k,
  const DParameter *param, real *y) {
  for (int l = 0; l < param->n_spec; ++l) {
    y[l] = sv(i, j, k, l);
  }
}
}

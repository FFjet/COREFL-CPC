#include "Thermo.cuh"
#include "DParameter.cuh"
#include "Constants.h"
#include "ChemData.h"

#ifdef HighTempMultiPart
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        enthalpy[i] =
            coeff(0, 0, i) * tt + 0.5 * coeff(1, 0, i) * tt2 + coeff(2, 0, i) * tt3 / 3 + 0.25 * coeff(3, 0, i) * tt4 +
            0.2 * coeff(4, 0, i) * tt5 + coeff(5, 0, i);
        const real cp =
            coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] =
            coeff(0, j, i) * tt + 0.5 * coeff(1, j, i) * tt2 + coeff(2, j, i) * tt3 / 3 + 0.25 * coeff(3, j, i) * tt4 +
            0.2 * coeff(4, j, i) * tt5 + coeff(5, j, i);
        const real cp =
            coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real hMin = coeff(0, j - 1, i) * Tmin + 0.5 * coeff(1, j - 1, i) * Tmin * Tmin
                            + coeff(2, j - 1, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j - 1, i) * Tmin * Tmin *
                            Tmin * Tmin
                            + 0.2 * coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j - 1, i);
                real hMax = coeff(0, j, i) * Tmax + 0.5 * coeff(1, j, i) * Tmax * Tmax
                            + coeff(2, j, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j, i) * Tmax * Tmax * Tmax *
                            Tmax
                            + 0.2 * coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real hMin = coeff(0, j, i) * Tmin + 0.5 * coeff(1, j, i) * Tmin * Tmin
                            + coeff(2, j, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j, i) * Tmin * Tmin * Tmin *
                            Tmin
                            + 0.2 * coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j, i);
                real hMax = coeff(0, j + 1, i) * Tmax + 0.5 * coeff(1, j + 1, i) * Tmax * Tmax
                            + coeff(2, j + 1, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j + 1, i) * Tmax * Tmax *
                            Tmax * Tmax
                            + 0.2 * coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j + 1, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              enthalpy[i] =
                  coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4
                  +
                  0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
            }
            break;
          }
        }
      }
      enthalpy[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, lnT{log(t)};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        const real cp = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                        + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        const real cp = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                        + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            break;
          }
        }
      }
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        enthalpy[i] =
            coeff(0, 0, i) * tt + 0.5 * coeff(1, 0, i) * tt2 + coeff(2, 0, i) * tt3 / 3 + 0.25 * coeff(3, 0, i) * tt4 +
            0.2 * coeff(4, 0, i) * tt5 + coeff(5, 0, i);
        cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) *
                tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] =
            coeff(0, j, i) * tt + 0.5 * coeff(1, j, i) * tt2 + coeff(2, j, i) * tt3 / 3 + 0.25 * coeff(3, j, i) * tt4 +
            0.2 * coeff(4, j, i) * tt5 + coeff(5, j, i);
        cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) *
                tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real cpMin = coeff(0, j - 1, i) + coeff(1, j - 1, i) * Tmin + coeff(2, j - 1, i) * Tmin * Tmin
                             + coeff(3, j - 1, i) * Tmin * Tmin * Tmin + coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j, i) + coeff(1, j, i) * Tmax + coeff(2, j, i) * Tmax * Tmax
                             + coeff(3, j, i) * Tmax * Tmax * Tmax + coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);

                real hMin = coeff(0, j - 1, i) * Tmin + 0.5 * coeff(1, j - 1, i) * Tmin * Tmin
                            + coeff(2, j - 1, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j - 1, i) * Tmin * Tmin *
                            Tmin
                            * Tmin
                            + 0.2 * coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j - 1, i);
                real hMax = coeff(0, j, i) * Tmax + 0.5 * coeff(1, j, i) * Tmax * Tmax
                            + coeff(2, j, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j, i) * Tmax * Tmax * Tmax *
                            Tmax
                            + 0.2 * coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real cpMin = coeff(0, j, i) + coeff(1, j, i) * Tmin + coeff(2, j, i) * Tmin * Tmin
                             + coeff(3, j, i) * Tmin * Tmin * Tmin + coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j + 1, i) + coeff(1, j + 1, i) * Tmax + coeff(2, j + 1, i) * Tmax * Tmax
                             + coeff(3, j + 1, i) * Tmax * Tmax * Tmax + coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);

                real hMin = coeff(0, j, i) * Tmin + 0.5 * coeff(1, j, i) * Tmin * Tmin
                            + coeff(2, j, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j, i) * Tmin * Tmin * Tmin *
                            Tmin
                            + 0.2 * coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j, i);
                real hMax = coeff(0, j + 1, i) * Tmax + 0.5 * coeff(1, j + 1, i) * Tmax * Tmax
                            + coeff(2, j + 1, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j + 1, i) * Tmax * Tmax *
                            Tmax
                            * Tmax
                            + 0.2 * coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j + 1, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              enthalpy[i] =
                  coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4
                  +
                  0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
              cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) *
                      t4;
            }
            break;
          }
        }
      }
      cp[i] *= R_u * param->imw[i];
      enthalpy[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, lnT{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_cp(real t, real *cp, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) *
                tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) *
                tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real cpMin = coeff(0, j - 1, i) + coeff(1, j - 1, i) * Tmin + coeff(2, j - 1, i) * Tmin * Tmin
                             + coeff(3, j - 1, i) * Tmin * Tmin * Tmin + coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j, i) + coeff(1, j, i) * Tmax + coeff(2, j, i) * Tmax * Tmax
                             + coeff(3, j, i) * Tmax * Tmax * Tmax + coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real cpMin = coeff(0, j, i) + coeff(1, j, i) * Tmin + coeff(2, j, i) * Tmin * Tmin
                             + coeff(3, j, i) * Tmin * Tmin * Tmin + coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j + 1, i) + coeff(1, j + 1, i) * Tmax + coeff(2, j + 1, i) * Tmax * Tmax
                             + coeff(3, j + 1, i) * Tmax * Tmax * Tmax + coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) *
                      t4;
            }
            break;
          }
        }
      }
      cp[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{log(t)};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        gibbs_rt[i] = coeff(0, 0, i) * (1.0 - log_tt) - 0.5 * coeff(1, 0, i) * tt - coeff(2, 0, i) * tt2 / 6.0 -
                      coeff(3, 0, i) * tt3 / 12.0 - coeff(4, 0, i) * tt4 * 0.05 + coeff(5, 0, i) * tt_inv -
                      coeff(6, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = coeff(0, j, i) * (1.0 - log_tt) - 0.5 * coeff(1, j, i) * tt - coeff(2, j, i) * tt2 / 6.0 -
                      coeff(3, j, i) * tt3 / 12.0 - coeff(4, j, i) * tt4 * 0.05 + coeff(5, j, i) * tt_inv -
                      coeff(6, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = coeff(0, j, i) * (1.0 - log_t) - 0.5 * coeff(1, j, i) * t - coeff(2, j, i) * t2 / 6.0 -
                          coeff(3, j, i) * t3 / 12.0 - coeff(4, j, i) * t4 * 0.05 + coeff(5, j, i) * t_inv -
                          coeff(6, j, i);
            break;
          }
        }
      }
    }
  } else {
    const real it{1.0 / t}, lnt{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        gibbs_rt[i] = -0.5 * coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt * (lntt + 1) +
                      coeff(2, 0, i) * (1.0 - lntt) - 0.5 * coeff(3, 0, i) * tt - coeff(4, 0, i) * tt2 / 6.0 -
                      coeff(5, 0, i) * tt3 / 12.0 - coeff(6, 0, i) * tt4 * 0.05 + coeff(7, 0, i) * itt -
                      coeff(8, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = -0.5 * coeff(0, j, i) * itt2 + coeff(1, j, i) * itt * (lntt + 1) +
                      coeff(2, j, i) * (1.0 - lntt) - 0.5 * coeff(3, j, i) * tt - coeff(4, j, i) * tt2 / 6.0 -
                      coeff(5, j, i) * tt3 / 12.0 - coeff(6, j, i) * tt4 * 0.05 + coeff(7, j, i) * itt -
                      coeff(8, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = -0.5 * coeff(0, j, i) * it2 + coeff(1, j, i) * it * (lnt + 1) +
                          coeff(2, j, i) * (1.0 - lnt) - 0.5 * coeff(3, j, i) * t - coeff(4, j, i) * t2 / 6.0 -
                          coeff(5, j, i) * t3 / 12.0 - coeff(6, j, i) * t4 * 0.05 + coeff(7, j, i) * it -
                          coeff(8, j, i);
            break;
          }
        }
      }
    }
  }
}
#else
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->low_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                      0.2 * coeff(i, 4) * t5 + coeff(i, 5);
      }
      enthalpy[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnT{log(t)};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        const real cp = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                        + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        const real cp = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                        + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            break;
          }
        }
      }
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const double tt = param->t_low[i];
        const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->low_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->t_high[i]) {
        const double tt = param->t_high[i];
        const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                      0.2 * coeff(i, 4) * t5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
      }
      enthalpy[i] *= param->gas_const[i];
      cp[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnT{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_cp(real t, real *cp, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  if (param->nasa_7_or_9 == 7) {
    for (auto i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        auto &coeff = param->low_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        auto &coeff = param->high_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
      }
      cp[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto &coeff = param->low_temp_coeff;
        gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                      coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto &coeff = param->high_temp_coeff;
        gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                      coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
      } else {
        const auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        gibbs_rt[i] =
            coeff(i, 0) * (1.0 - log_t) - 0.5 * coeff(i, 1) * t - coeff(i, 2) * t2 / 6.0 - coeff(i, 3) * t3 / 12.0 -
            coeff(i, 4) * t4 * 0.05 + coeff(i, 5) * t_inv - coeff(i, 6);
      }
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnt{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        gibbs_rt[i] = -0.5 * coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt * (lntt + 1) +
                      coeff(2, 0, i) * (1.0 - lntt) - 0.5 * coeff(3, 0, i) * tt - coeff(4, 0, i) * tt2 / 6.0 -
                      coeff(5, 0, i) * tt3 / 12.0 - coeff(6, 0, i) * tt4 * 0.05 + coeff(7, 0, i) * itt -
                      coeff(8, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = -0.5 * coeff(0, j, i) * itt2 + coeff(1, j, i) * itt * (lntt + 1) +
                      coeff(2, j, i) * (1.0 - lntt) - 0.5 * coeff(3, j, i) * tt - coeff(4, j, i) * tt2 / 6.0 -
                      coeff(5, j, i) * tt3 / 12.0 - coeff(6, j, i) * tt4 * 0.05 + coeff(7, j, i) * itt -
                      coeff(8, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = -0.5 * coeff(0, j, i) * it2 + coeff(1, j, i) * it * (lnt + 1) +
                          coeff(2, j, i) * (1.0 - lnt) - 0.5 * coeff(3, j, i) * t - coeff(4, j, i) * t2 / 6.0 -
                          coeff(5, j, i) * t3 / 12.0 - coeff(6, j, i) * t4 * 0.05 + coeff(7, j, i) * it -
                          coeff(8, j, i);
            break;
          }
        }
      }
    }
  }
}

#endif

__device__ real cfd::compute_ve_energy(int i_spec, real t, const DParameter *param) {
  if (!param->two_temperature || i_spec < 0 || i_spec >= param->n_spec) return 0.0;
  const real temp = max(t, static_cast<real>(1.0));
  const real r_spec = param->gas_const[i_spec];
  real e_ve = 0.0;
  e_ve += compute_vib_energy(i_spec, temp, param);
  const int n_level = param->n_electronic_level[i_spec];
  if (n_level > 0) {
    real z = 0.0, weighted = 0.0;
    for (int j = 0; j < n_level; ++j) {
      const real theta = param->electronic_theta(i_spec, j);
      const real g = param->electronic_g(i_spec, j);
      const real boltz = g * exp(-theta / temp);
      z += boltz;
      weighted += boltz * theta;
    }
    if (z > 0) e_ve += r_spec * weighted / z;
  }
  return e_ve;
}

__device__ real cfd::compute_vib_energy(int i_spec, real t, const DParameter *param) {
  if (!param->two_temperature || i_spec < 0 || i_spec >= param->n_spec) return 0.0;
  const real temp = max(t, static_cast<real>(1.0));
  const real r_spec = param->gas_const[i_spec];
  real e_v = 0.0;
  if (param->theta_v[i_spec] > 0) {
    const real x = param->theta_v[i_spec] / temp;
    if (x < 200) {
      e_v += r_spec * param->theta_v[i_spec] / expm1(x);
    }
  }
  return e_v;
}

__device__ real cfd::compute_ve_cv(int i_spec, real t, const DParameter *param) {
  if (!param->two_temperature || i_spec < 0 || i_spec >= param->n_spec) return 0.0;
  const real temp = max(t, static_cast<real>(1.0));
  const real r_spec = param->gas_const[i_spec];
  real cv_ve = 0.0;
  cv_ve += compute_vib_cv(i_spec, temp, param);
  const int n_level = param->n_electronic_level[i_spec];
  if (n_level > 0) {
    real z = 0.0, b = 0.0, c = 0.0;
    for (int j = 0; j < n_level; ++j) {
      const real theta = param->electronic_theta(i_spec, j);
      const real g = param->electronic_g(i_spec, j);
      const real boltz = g * exp(-theta / temp);
      z += boltz;
      b += boltz * theta;
      c += boltz * theta * theta;
    }
    if (z > 0) {
      const real mean = b / z;
      cv_ve += r_spec * (c / z - mean * mean) / (temp * temp);
    }
  }
  return cv_ve;
}

__device__ real cfd::compute_vib_cv(int i_spec, real t, const DParameter *param) {
  if (!param->two_temperature || i_spec < 0 || i_spec >= param->n_spec) return 0.0;
  const real temp = max(t, static_cast<real>(1.0));
  const real r_spec = param->gas_const[i_spec];
  real cv_v = 0.0;
  if (param->theta_v[i_spec] > 0) {
    const real x = param->theta_v[i_spec] / temp;
    if (x < 200) {
      const real ex = exp(x);
      const real denom = ex - 1.0;
      cv_v += r_spec * x * x * ex / (denom * denom);
    }
  }
  return cv_v;
}

__device__ real cfd::compute_mixture_ve_energy(real t, const real *y, const DParameter *param, real *cv_ve) {
  real eve = 0.0;
  real cve = 0.0;
  if (param->two_temperature) {
    for (int i = 0; i < param->n_spec; ++i) {
      const real yi = y[i];
      eve += yi * compute_ve_energy(i, t, param);
      cve += yi * compute_ve_cv(i, t, param);
    }
  }
  if (cv_ve != nullptr) *cv_ve = cve;
  return eve;
}

__device__ real cfd::compute_mixture_vib_energy(real t, const real *y, const DParameter *param, real *cv_v) {
  real ev = 0.0;
  real cvv = 0.0;
  if (param->two_temperature) {
    for (int i = 0; i < param->n_spec; ++i) {
      const real yi = y[i];
      ev += yi * compute_vib_energy(i, t, param);
      cvv += yi * compute_vib_cv(i, t, param);
    }
  }
  if (cv_v != nullptr) *cv_v = cvv;
  return ev;
}

__device__ real cfd::compute_mixture_tr_energy(real t, const real *y, const DParameter *param, real *cv_tr,
  real *cp_tr, real *gas_const_mix) {
  real h_i[MAX_SPEC_NUMBER];
  real cp_i[MAX_SPEC_NUMBER];
  compute_enthalpy_and_cp(t, h_i, cp_i, param);

  real e_mix = 0.0;
  real cp_mix = 0.0;
  real r_mix = 0.0;
  for (int i = 0; i < param->n_spec; ++i) {
    const real yi = y[i];
    cp_mix += yi * cp_i[i];
    r_mix += yi * param->gas_const[i];
    e_mix += yi * (h_i[i] - param->gas_const[i] * t);
  }
  real cv_ve_eq{};
  const real eve_eq = compute_mixture_ve_energy(t, y, param, &cv_ve_eq);
  const real cv_total = cp_mix - r_mix;
  if (cv_tr != nullptr) *cv_tr = cv_total - cv_ve_eq;
  if (cp_tr != nullptr) *cp_tr = cp_mix - cv_ve_eq;
  if (gas_const_mix != nullptr) *gas_const_mix = r_mix;
  return e_mix - eve_eq;
}

__device__ real cfd::invert_tve_from_eve(real eve_target, const real *y, real t_init, const DParameter *param) {
  real tve = max(t_init, static_cast<real>(1.0));
  for (int iter = 0; iter < 80; ++iter) {
    real cv_ve{};
    const real eve = compute_mixture_ve_energy(tve, y, param, &cv_ve);
    const real residual = eve - eve_target;
    if (abs(residual) < 1e-8 * max(static_cast<real>(1.0), abs(eve_target))) break;
    const real next_tve = max(static_cast<real>(1.0), tve - residual / max(cv_ve, static_cast<real>(1e-8)));
    if (abs(next_tve - tve) < 1e-8 * max(static_cast<real>(1.0), tve)) {
      tve = next_tve;
      break;
    }
    tve = next_tve;
  }
  return tve;
}

__device__ real cfd::invert_tve_from_ev(real ev_target, const real *y, real t_init, const DParameter *param) {
  real tve = max(t_init, static_cast<real>(1.0));
  for (int iter = 0; iter < 80; ++iter) {
    real cv_v{};
    const real ev = compute_mixture_vib_energy(tve, y, param, &cv_v);
    const real residual = ev - ev_target;
    if (abs(residual) < 1e-8 * max(static_cast<real>(1.0), abs(ev_target))) break;
    const real next_tve = max(static_cast<real>(1.0), tve - residual / max(cv_v, static_cast<real>(1e-8)));
    if (abs(next_tve - tve) < 1e-8 * max(static_cast<real>(1.0), tve)) {
      tve = next_tve;
      break;
    }
    tve = next_tve;
  }
  return tve;
}

namespace {
__device__ __forceinline__ real mw_from_imw(real imw) {
  return 1.0 / max(imw, static_cast<real>(1e-20));
}

__device__ __forceinline__ real default_lt_a(real reduced_mass, real theta_v) {
  return 1.16e-3 * sqrt(max(reduced_mass, static_cast<real>(1e-20))) *
         pow(max(theta_v, static_cast<real>(0.0)), static_cast<real>(4.0 / 3.0));
}

__device__ __forceinline__ real default_lt_b(real reduced_mass) {
  return 0.015 * pow(max(reduced_mass, static_cast<real>(1e-20)), static_cast<real>(0.25));
}

__device__ __forceinline__ real compute_pair_mw_relaxation_time(int i_vib, int i_partner, real t, real p,
  const cfd::DParameter *param) {
  if (p <= 1e-16 || t <= 1.0) return 1e30;

  const real mw_v = mw_from_imw(param->imw[i_vib]);
  const real mw_p = mw_from_imw(param->imw[i_partner]);
  const real reduced_mass = mw_v * mw_p / max(mw_v + mw_p, static_cast<real>(1e-20));

  real a_ij = param->lt_a(i_vib, i_partner);
  real b_ij = param->lt_b(i_vib, i_partner);
  if (a_ij <= 0.0 || b_ij <= 0.0) {
    a_ij = default_lt_a(reduced_mass, param->theta_v[i_vib]);
    b_ij = default_lt_b(reduced_mass);
  }

  const real mw_exponent = a_ij * (pow(t, static_cast<real>(-1.0 / 3.0)) - b_ij) - 18.421;
  return max(exp(mw_exponent) * cfd::p_atm / p, static_cast<real>(1e-30));
}

__device__ __forceinline__ real compute_species_vt_relaxation_time(int i_vib, real density, real t, real p,
  const real *y, const cfd::DParameter *param) {
  real conc = 0.0;
  real denom = 0.0;
  for (int j = 0; j < param->n_spec; ++j) {
    const real yj = max(y[j], static_cast<real>(0.0));
    if (yj <= 0.0) continue;
    const real mw_j = mw_from_imw(param->imw[j]);
    const real mole_like = yj / mw_j;
    const real tau_j = compute_pair_mw_relaxation_time(i_vib, j, t, p, param);
    conc += mole_like;
    denom += mole_like / tau_j;
  }
  if (conc <= 0.0 || denom <= 0.0) return 1e30;

  const real tau_mw = conc / denom;
  const real sigma = max(param->park_sigma[i_vib], static_cast<real>(1e-30)) *
                     (static_cast<real>(2.5e9) / max(t * t, static_cast<real>(1.0)));
  const real n_total = density * conc * cfd::avogadro;
  const real mw_v = mw_from_imw(param->imw[i_vib]);
  const real c_s = sqrt(max(static_cast<real>(8.0) * cfd::R_u * t / (cfd::pi * mw_v), static_cast<real>(1e-30)));
  const real tau_park = 1.0 / max(sigma * c_s * n_total, static_cast<real>(1e-30));
  return max(tau_mw + tau_park, static_cast<real>(1e-30));
}
}

__device__ real cfd::compute_vt_relaxation_source(real density, real t, real tve, const real *y,
  const DParameter *param, real *eve_eq, real *tau_eff) {
  if (!param->two_temperature || density <= 0.0) {
    if (eve_eq != nullptr) *eve_eq = 0.0;
    if (tau_eff != nullptr) *tau_eff = 1e30;
    return 0.0;
  }

  real r_mix = 0.0;
  for (int i = 0; i < param->n_spec; ++i) {
    r_mix += max(y[i], static_cast<real>(0.0)) * param->gas_const[i];
  }
  const real p = density * r_mix * max(t, static_cast<real>(1.0));

  real eve_eq_mix = 0.0;
  real eve_mix = 0.0;
  real src = 0.0;
  for (int i = 0; i < param->n_spec; ++i) {
    if (param->theta_v[i] <= 0.0) continue;
    const real yi = max(y[i], static_cast<real>(0.0));
    if (yi <= 0.0) continue;

    const real e_eq_i = compute_vib_energy(i, t, param);
    const real e_i = compute_vib_energy(i, tve, param);
    const real diff_i = e_eq_i - e_i;
    eve_eq_mix += yi * e_eq_i;
    eve_mix += yi * e_i;
    if (abs(diff_i) <= 1e-16) continue;

    const real tau_i = compute_species_vt_relaxation_time(i, density, t, p, y, param);
    if (tau_i < 1e29) {
      src += density * yi * diff_i / tau_i;
    }
  }

  if (eve_eq != nullptr) *eve_eq = eve_eq_mix;
  if (tau_eff != nullptr) {
    const real diff_mix = eve_eq_mix - eve_mix;
    if (abs(diff_mix) > 1e-16 && abs(src) > 1e-16) {
      *tau_eff = max(density * diff_mix / src, static_cast<real>(1e-30));
    } else {
      *tau_eff = 1e30;
    }
  }
  return src;
}

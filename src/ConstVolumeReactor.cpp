#include "ConstVolumeReactor.h"
#include "Constants.h"
#include <chrono>
#include <cstdlib>
#include <cmath>
#include "gxl_lib/Math.hpp"

namespace cfd {
#ifdef Combustion2Part
namespace {
real compute_mixture_vib_energy_host(real t, const std::vector<real> &y, const cfd::Species &species,
                                     real *cv_v);
}
#endif

void const_volume_reactor(Parameter &parameter) {
#ifdef Combustion2Part
  const Species species(parameter);
  const Reaction reaction(parameter, species);
  const int ns = species.n_spec;

  // *************************Initialize the thermodynamic field and species mass fractions*************************
  const auto condition = parameter.get_struct("constVolumeReactor");
  real T = std::get<real>(condition.at("initial_temperature"));
  real p = std::get<real>(condition.at("initial_pressure"));

  // Initialize the species mass fractions
  std::vector<real> rhoY(ns, 0), yk(ns, 0);
  real imw = 0.0;
  const int mole_mass = std::get<int>(condition.at("mole_mass"));
  if (mole_mass == 0) {
    for (const auto &[name, idx]: species.spec_list) {
      if (condition.find(name) != condition.cend()) {
        yk[idx] = std::get<real>(condition.at(name));
        imw += yk[idx] / species.mw[idx];
      }
    }
    real y_n2 = 1;
    for (int l = 0; l < ns - 1; ++l) {
      y_n2 -= yk[l];
    }
    yk[ns - 1] = y_n2;
  } else {
    // The yk is used as mole fraction first
    for (const auto &[name, idx]: species.spec_list) {
      if (condition.find(name) != condition.cend()) {
        yk[idx] = std::get<real>(condition.at(name));
        imw += yk[idx] * species.mw[idx];
      }
    }
    real x_n2 = 1;
    for (int l = 0; l < ns - 1; ++l) {
      x_n2 -= yk[l];
    }
    yk[ns - 1] = x_n2;
    // convert to mass fraction
    imw = 1.0 / imw;
    for (int i = 0; i < ns; ++i) {
      yk[i] = yk[i] * species.mw[i] * imw;
    }
  }

  const real rho = p / (T * R_u * imw);
  for (int i = 0; i < ns; ++i) {
    rhoY[i] = yk[i] * rho;
  }

  const bool use_two_temperature =
      kTwoTemperature && species.has_two_temperature_data &&
      (!condition.contains("two_temperature") || std::get<int>(condition.at("two_temperature")) != 0);

  if (use_two_temperature) {
    const real end_time = std::get<real>(condition.at("end_time"));
    const real dt = std::get<real>(condition.at("dt"));
    const int mechanism = std::get<int>(condition.at("mechanism"));
    const int conserve_mass_corefl = std::get<int>(condition.at("conserve_mass_corefl"));
    const int file_frequency = std::get<int>(condition.at("file_frequency"));
    const int screen_frequency = std::get<int>(condition.at("screen_frequency"));
    const int effective_advance_scheme =
        (std::get<int>(condition.at("advance_scheme")) == 0 || std::get<int>(condition.at("advance_scheme")) == 3)
            ? std::get<int>(condition.at("advance_scheme"))
            : 3;
    const int enable_chemical_source =
        condition.contains("enable_chemical_source") ? std::get<int>(condition.at("enable_chemical_source"))
                                                     : parameter.get_int("reaction");
    const int enable_vt_relaxation =
        condition.contains("enable_vt_relaxation") ? std::get<int>(condition.at("enable_vt_relaxation")) : 1;
    real Tve = condition.contains("initial_temperature_ve") ? std::get<real>(condition.at("initial_temperature_ve")) : T;

    std::vector<real> hk_2t(ns, 0), cpk_2t(ns, 0);
    species.compute_enthalpy_and_cp(T, hk_2t.data(), cpk_2t.data());
    const real eve_eq = species.compute_mixture_ve_energy(T, yk);
    real eve = species.compute_mixture_ve_energy(Tve, yk);
    real E = 0.0;
    for (int i = 0; i < ns; ++i) {
      E += yk[i] * (hk_2t[i] - R_u / species.mw[i] * T);
    }
    E = E - eve_eq + eve;
    real rho_eve = rho * eve;

    const int nr = reaction.n_reac;
    auto make_sources = [&]() {
      ReactorSources src;
      src.q1.assign(nr, 0.0);
      src.q2.assign(nr, 0.0);
      src.omega_d.assign(ns, 0.0);
      src.omega.assign(ns, 0.0);
      return src;
    };

    auto normalize_state = [&](std::vector<real> &rhoY_state, std::vector<real> &y_state, real &imw_state) {
      real sum_rho = 0.0;
      for (int l = 0; l < ns; ++l) {
        if (rhoY_state[l] < 0.0) rhoY_state[l] = 0.0;
        sum_rho += rhoY_state[l];
      }
      if (sum_rho <= 1e-30) {
        rhoY_state = rhoY;
        sum_rho = rho;
      }
      imw_state = 0.0;
      for (int l = 0; l < ns; ++l) {
        if (conserve_mass_corefl) {
          y_state[l] = rhoY_state[l] / sum_rho;
          rhoY_state[l] = y_state[l] * rho;
        } else {
          y_state[l] = rhoY_state[l] / rho;
        }
        imw_state += y_state[l] / species.mw[l];
      }
    };

    auto recover_temperatures = [&](const std::vector<real> &y_state, real imw_state, real &t_state, real &tve_state,
                                    real &p_state, real &eve_state, real rho_eve_state, real t_guess,
                                    real tve_guess) {
      eve_state = std::max(rho_eve_state / rho, static_cast<real>(0.0));
      tve_state = species.invert_tve_from_eve(eve_state, y_state, tve_guess);
      t_state = update_t_two_temperature(t_guess, species, y_state, E, eve_state);
      p_state = rho * R_u * imw_state * t_state;
    };

    auto evaluate_sources = [&](real t_state, real tve_state, const std::vector<real> &rhoY_state,
                                const std::vector<real> &y_state, ReactorSources &src) {
      std::fill(src.q1.begin(), src.q1.end(), 0.0);
      std::fill(src.q2.begin(), src.q2.end(), 0.0);
      std::fill(src.omega_d.begin(), src.omega_d.end(), 0.0);
      std::fill(src.omega.begin(), src.omega.end(), 0.0);
      src.chemical_eve_source = 0.0;
      src.vt_eve_source = 0.0;
      src.eve_eq = species.compute_mixture_ve_energy(t_state, y_state);
      src.tau_vt = 1e30;

      if (enable_chemical_source && parameter.get_int("reaction") == 1) {
        compute_src(mechanism, t_state, tve_state, species, reaction, rhoY_state, src.q1, src.q2, src.omega_d, src.omega);
        src.chemical_eve_source = compute_two_temperature_chemical_source(tve_state, species, src.omega);
      }
      if (enable_vt_relaxation) {
        src.vt_eve_source =
            compute_vt_relaxation_source(rho, t_state, tve_state, y_state, species, &src.eve_eq, &src.tau_vt);
      }
    };

    auto apply_lt_stage = [&](real dt_stage, std::vector<real> &rhoY_state, std::vector<real> &y_state,
                              real &imw_state, real &t_state, real &tve_state, real &p_state, real &eve_state,
                              real &rho_eve_state, ReactorSources &src) {
      if (!enable_vt_relaxation || dt_stage <= 0.0) return;
      evaluate_sources(t_state, tve_state, rhoY_state, y_state, src);
      if (src.tau_vt >= 1e29 || !std::isfinite(src.tau_vt)) {
        src.vt_eve_source = 0.0;
        return;
      }

      const real ev_old = compute_mixture_vib_energy_host(tve_state, y_state, species, nullptr);
      const real alpha = std::exp(-dt_stage / std::max(src.tau_vt, static_cast<real>(1e-30)));
      const real ev_new = ev_old * alpha + src.eve_eq * (1.0 - alpha);
      rho_eve_state = std::max(rho_eve_state + rho * (ev_new - ev_old), static_cast<real>(0.0));
      recover_temperatures(y_state, imw_state, t_state, tve_state, p_state, eve_state, rho_eve_state, t_state,
                           tve_state);
      src.vt_eve_source = rho * (ev_new - ev_old) / std::max(dt_stage, static_cast<real>(1e-30));
    };

    auto write_state = [&](FILE *fp_state, real time_state, real t_state, real tve_state, real p_state, real eve_state,
                           real rho_eve_state, const ReactorSources &src, int step_state, real imw_state,
                           const std::vector<real> &y_state) {
      fprintf(fp_state, "%.13e,%.13e,%.13e,%.13e,%.13e,%.13e,%.13e,%.13e,%.13e",
              time_state, t_state, tve_state, p_state, eve_state, rho_eve_state, src.chemical_eve_source,
              src.vt_eve_source, src.tau_vt);
      if (mole_mass == 1) {
        for (int l = 0; l < ns; ++l) {
          const real xl = y_state[l] / (species.mw[l] * imw_state);
          fprintf(fp_state, ",%.13e", xl);
        }
      } else {
        for (int l = 0; l < ns; ++l) {
          fprintf(fp_state, ",%.13e", y_state[l]);
        }
      }
      fprintf(fp_state, ",%d\n", step_state);
    };

    auto check_finite_state = [&](const char *tag, int step_state, real time_state, const std::vector<real> &rhoY_state,
                                  const std::vector<real> &y_state, real t_state, real tve_state, real p_state,
                                  real eve_state, real rho_eve_state) {
      bool ok = std::isfinite(t_state) && std::isfinite(tve_state) && std::isfinite(p_state) &&
                std::isfinite(eve_state) && std::isfinite(rho_eve_state);
      for (int l = 0; ok && l < ns; ++l) {
        ok = std::isfinite(rhoY_state[l]) && std::isfinite(y_state[l]);
      }
      if (ok) return;

      fprintf(stderr, "[2T-CVR] Non-finite state detected at %s, step=%d, time=%.13e\n", tag, step_state, time_state);
      fprintf(stderr, "  T=%.13e, Tve=%.13e, p=%.13e, Eve=%.13e, rhoEve=%.13e, E=%.13e\n",
              t_state, tve_state, p_state, eve_state, rho_eve_state, E);
      for (int l = 0; l < ns; ++l) {
        fprintf(stderr, "  spec[%d]=%s rhoY=%.13e Y=%.13e\n",
                l, species.spec_name[l].c_str(), rhoY_state[l], y_state[l]);
      }
      fflush(stderr);
      std::abort();
    };

    FILE *fp = fopen("const_volume_reactor_output.dat", "w");
    fprintf(fp,
            "variables=time(s),temperature(K),temperature_ve(K),pressure(Pa),Eve(J/kg),rhoEve(J/m3),src_eve_chem(J/m3/s),src_eve_vt(J/m3/s),tau_vt(s)");
    if (mole_mass == 1) {
      for (int l = 0; l < ns; ++l) {
        fprintf(fp, ",X<sub>%s</sub>", species.spec_name[l].c_str());
      }
    } else {
      for (int l = 0; l < ns; ++l) {
        fprintf(fp, ",Y<sub>%s</sub>", species.spec_name[l].c_str());
      }
    }
    fprintf(fp, ",step\n");

    auto start = std::chrono::high_resolution_clock::now();
    ReactorSources current_src = make_sources();
    evaluate_sources(T, Tve, rhoY, yk, current_src);
    write_state(fp, 0.0, T, Tve, p, eve, rho_eve, current_src, 0, imw, yk);

    constexpr real rk_stage_dt_scale[3]{1.0, 0.25, 2.0 / 3.0};
    real time = 0.0;
    int step = 0;
    while (time < end_time - 1e-30) {
      const real dt_step = std::min(dt, end_time - time);

      if (effective_advance_scheme == 0) {
        evaluate_sources(T, Tve, rhoY, yk, current_src);
        for (int l = 0; l < ns; ++l) {
          rhoY[l] += current_src.omega[l] * dt_step;
        }
        rho_eve += current_src.chemical_eve_source * dt_step;
        normalize_state(rhoY, yk, imw);
        recover_temperatures(yk, imw, T, Tve, p, eve, rho_eve, T, Tve);
        apply_lt_stage(dt_step, rhoY, yk, imw, T, Tve, p, eve, rho_eve, current_src);
      } else {
        const std::vector<real> rhoY0 = rhoY;
        const real rho_eve0 = rho_eve;
        const real T0 = T;
        const real Tve0 = Tve;

        ReactorSources src0 = make_sources();
        ReactorSources src1 = make_sources();
        ReactorSources src2 = make_sources();

        evaluate_sources(T0, Tve0, rhoY0, yk, src0);

        std::vector<real> rhoY1 = rhoY0;
        for (int l = 0; l < ns; ++l) {
          rhoY1[l] += dt_step * src0.omega[l];
        }
        real rho_eve1 = rho_eve0 + dt_step * src0.chemical_eve_source;
        std::vector<real> y1 = yk;
        real imw1 = imw;
        real T1 = T0, Tve1 = Tve0, p1 = p, eve1 = eve;
        normalize_state(rhoY1, y1, imw1);
        recover_temperatures(y1, imw1, T1, Tve1, p1, eve1, rho_eve1, T0, Tve0);
        check_finite_state("rk3-stage1-post-recover", step, time, rhoY1, y1, T1, Tve1, p1, eve1, rho_eve1);
        apply_lt_stage(rk_stage_dt_scale[0] * dt_step, rhoY1, y1, imw1, T1, Tve1, p1, eve1, rho_eve1, src1);
        check_finite_state("rk3-stage1-post-lt", step, time, rhoY1, y1, T1, Tve1, p1, eve1, rho_eve1);

        evaluate_sources(T1, Tve1, rhoY1, y1, src1);
        std::vector<real> rhoY2 = rhoY0;
        for (int l = 0; l < ns; ++l) {
          rhoY2[l] = 0.75 * rhoY0[l] + 0.25 * (rhoY1[l] + dt_step * src1.omega[l]);
        }
        real rho_eve2 = 0.75 * rho_eve0 + 0.25 * (rho_eve1 + dt_step * src1.chemical_eve_source);
        std::vector<real> y2 = yk;
        real imw2 = imw;
        real T2 = T1, Tve2 = Tve1, p2 = p1, eve2 = eve1;
        normalize_state(rhoY2, y2, imw2);
        recover_temperatures(y2, imw2, T2, Tve2, p2, eve2, rho_eve2, T1, Tve1);
        check_finite_state("rk3-stage2-post-recover", step, time, rhoY2, y2, T2, Tve2, p2, eve2, rho_eve2);
        apply_lt_stage(rk_stage_dt_scale[1] * dt_step, rhoY2, y2, imw2, T2, Tve2, p2, eve2, rho_eve2, src2);
        check_finite_state("rk3-stage2-post-lt", step, time, rhoY2, y2, T2, Tve2, p2, eve2, rho_eve2);

        evaluate_sources(T2, Tve2, rhoY2, y2, src2);
        for (int l = 0; l < ns; ++l) {
          rhoY[l] = (rhoY0[l] + 2.0 * (rhoY2[l] + dt_step * src2.omega[l])) / 3.0;
        }
        rho_eve = (rho_eve0 + 2.0 * (rho_eve2 + dt_step * src2.chemical_eve_source)) / 3.0;
        normalize_state(rhoY, yk, imw);
        recover_temperatures(yk, imw, T, Tve, p, eve, rho_eve, T2, Tve2);
        check_finite_state("rk3-final-post-recover", step, time, rhoY, yk, T, Tve, p, eve, rho_eve);
        apply_lt_stage(rk_stage_dt_scale[2] * dt_step, rhoY, yk, imw, T, Tve, p, eve, rho_eve, current_src);
        check_finite_state("rk3-final-post-lt", step, time, rhoY, yk, T, Tve, p, eve, rho_eve);
      }

      time += dt_step;
      ++step;
      evaluate_sources(T, Tve, rhoY, yk, current_src);

      if (screen_frequency > 0 && step % screen_frequency == 0) {
        printf("Step: %d, Time: %.6e s, T: %.6f K, Tve: %.6f K, P: %.6f Pa, Eve: %.6e, chemEve: %.6e, vtEve: %.6e\n",
               step, time, T, Tve, p, eve, current_src.chemical_eve_source, current_src.vt_eve_source);
      }
      if ((file_frequency > 0 && step % file_frequency == 0) || end_time - time <= 1e-30) {
        write_state(fp, time, T, Tve, p, eve, rho_eve, current_src, step, imw, yk);
      }
    }

    fclose(fp);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Total simulation time is %.6es.\n", duration.count());
    return;
  }
  // compute the total energy
  real E = 0;
  // real enthalpy = 0;
  std::vector<real> hk(ns, 0);
  species.compute_enthalpy(T, hk.data());
  for (int i = 0; i < ns; ++i) {
    E += yk[i] * hk[i];
  }
  // enthalpy = E;
  E -= p / rho;
  // *************************End of initialization*************************

  real time = 0.0;
  const real end_time = std::get<real>(condition.at("end_time"));
  const real dt = std::get<real>(condition.at("dt"));

  const int mechanism = std::get<int>(condition.at("mechanism"));
  const int conserve_mass_corefl = std::get<int>(condition.at("conserve_mass_corefl"));
  const int file_frequency = std::get<int>(condition.at("file_frequency"));
  const int screen_frequency = std::get<int>(condition.at("screen_frequency"));
  const int implicit_method = std::get<int>(condition.at("implicit_method"));
  const int advance_scheme = std::get<int>(condition.at("advance_scheme"));

  std::vector<real> omega_d(ns, 0);
  std::vector<real> omega(ns, 0);

  // *************************Prepare for output*************************
  FILE *fp = fopen("const_volume_reactor_output.dat", "w");
  fprintf(fp, "variables=time(s),temperature(K),pressure(Pa)");
  if (mole_mass == 1) {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",X<sub>%s</sub>", species.spec_name[l].c_str());
    }
  } else {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",Y<sub>%s</sub>", species.spec_name[l].c_str());
    }
  }
  fprintf(fp, ",step\n");
  for (int l = 0; l < ns; ++l) {
    yk[l] = rhoY[l] / rho;
  }
  fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
  if (mole_mass == 1) {
    for (int l = 0; l < ns; ++l) {
      const real xl = yk[l] / (species.mw[l] * imw);
      fprintf(fp, ",%.13e", xl);
    }
  } else {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",%.13e", yk[l]);
    }
  }
  fprintf(fp, ",%d\n", 0);
  // *************************End of output file opening*************************

  auto start = std::chrono::high_resolution_clock::now();
  real h = dt, err_last = 0;
  real RTol = 1e-6, ATol = 1e-8;
  int adaptive_time_step = 0, build_arnoldi = 0;
  if (advance_scheme == 1) {
    RTol = std::get<real>(condition.at("RTol"));
    ATol = std::get<real>(condition.at("ATol"));
    adaptive_time_step = std::get<int>(condition.at("adaptive_time_step"));
    build_arnoldi = std::get<int>(condition.at("build_arnoldi"));
  }

  real T_last = T;
  int step = 0;
  const int nr = reaction.n_reac;
  // ***************************Time marching loop*******************************
  while (time < end_time) {
    // The loop to advance the solution
    // Compute the concentration
    std::vector<real> q1(nr, 0), q2(nr, 0);
    compute_src(mechanism, T, species, reaction, rhoY, q1, q2, omega_d, omega);

    bool do_update{true};
    if (advance_scheme == 0) {
      if (implicit_method == 1) {
        // EPI method
        auto jac = compute_chem_src_jacobian(rhoY, species, reaction, q1, q2);
        EPI(jac, species, dt, omega);
      } else if (implicit_method == 2) {
        // DA method
        auto jac = compute_chem_src_jacobian_diagonal(rhoY, species, reaction, q1, q2);
        DA(jac, species, dt, omega);
      }

      // Update species mass fractions
      if (conserve_mass_corefl) {
        real sumRho = 0;
        for (int i = 0; i < ns; ++i) {
          rhoY[i] += omega[i] * dt;
          if (rhoY[i] < 0) {
            rhoY[i] = 0;
          }
          sumRho += rhoY[i];
        }
        for (int i = 0; i < ns; ++i) {
          yk[i] = rhoY[i] / sumRho;
          rhoY[i] = yk[i] * rho;
        }
      } else {
        for (int i = 0; i < ns; ++i) {
          rhoY[i] += omega[i] * dt;
          if (rhoY[i] < 0) {
            rhoY[i] = 0;
          }
        }
        for (int i = 0; i < ns; ++i) {
          yk[i] = rhoY[i] / rho;
        }
      }
    } else if (advance_scheme == 1) {
      // ROK4E
      const real h_step = std::min(h, end_time - time);
      std::vector<real> k1(ns, 0), k2(ns, 0), k3(ns, 0), k4(ns, 0);
      if (build_arnoldi) {
        constexpr int KrylovMaxDim = MAX_SPEC_NUMBER, KrylovMinDim = 4;
        real Q[KrylovMaxDim * MAX_SPEC_NUMBER]{};
        real H[KrylovMaxDim * KrylovMaxDim]{};
        const int krylovDim = buildArnoldi(species, rhoY, omega.data(), Q, H, KrylovMaxDim, mechanism, T, reaction, rho,
                                           E, KrylovMinDim);
        real g[KrylovMaxDim]{};

        // Stage 1
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], omega.data(), ns);
          // g[r] = h_step * gxl::vec_dot(&Q[r * ns], omega.data(), ns);
        }
        real LHS[KrylovMaxDim * KrylovMaxDim]{};
        // LHS = I - h * gamma * H
        real b[KrylovMaxDim]{};
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
          for (int j = 0; j < krylovDim; ++j) {
            if (i == j) {
              LHS[i * krylovDim + j] = 1.0 - h_step * rok4e::gamma * H[i * KrylovMaxDim + j];
            } else {
              LHS[i * krylovDim + j] = -h_step * rok4e::gamma * H[i * KrylovMaxDim + j];
            }
          }
        }
        auto ipiv = lu_decomp(LHS, krylovDim);
        real lhs[KrylovMaxDim * KrylovMaxDim]{};
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, b, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k1[l] = omega[l] - s;
        }

        // Stage 2
        std::vector<real> y_stage(ns, 0);
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * rok4e::alpha21 * k1[l];
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        real T_stage = update_t(T, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int l = 0; l < ns; ++l) {
          omega[l] += rok4e::gamma21Gamma * k1[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], omega.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, b, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k2[l] = omega[l] - s - rok4e::gamma21Gamma * k1[l];
        }
        // for (int l = 0; l < ns; ++l) {
          // k2[l] -= rok4e::gamma21Gamma * k1[l];
        // }

        // Stage 3
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * (rok4e::alpha31 * k1[l] + rok4e::alpha32 * k2[l]);
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        T_stage = update_t(T_stage, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        std::vector<real> f(ns, 0);
        for (int l = 0; l < ns; ++l) {
          f[l] = omega[l] + rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], f.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, g, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k3[l] = f[l] - s - rok4e::gamma31Gamma * k1[l] - rok4e::gamma32Gamma * k2[l];
        }

        // Stage 4
        for (int l = 0; l < ns; ++l) {
          f[l] = omega[l] + rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], f.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, g, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k4[l] = f[l] - s - rok4e::gamma41Gamma * k1[l] - rok4e::gamma42Gamma * k2[l] - rok4e::gamma43Gamma * k3[l];
        }
      } else { // First, compute the chemical jacobian matrix
        auto jac = compute_chem_src_jacobian(rhoY, species, reaction, q1, q2);
        // Assemble the LHS matrix and perform the LU decomposition
        // LHS = I - h * gamma * J
        std::vector<real> LHS(ns * ns, 0);
        for (int m = 0; m < ns; ++m) {
          for (int n = 0; n < ns; ++n) {
            if (m == n) {
              LHS[m * ns + n] = 1.0 - h_step * rok4e::gamma * jac[m * ns + n];
            } else {
              LHS[m * ns + n] = -h_step * rok4e::gamma * jac[m * ns + n];
            }
          }
        }
        auto ipiv = lu_decomp(LHS.data(), ns);

        // stage 1
        auto lhs1 = LHS;
        for (int i = 0; i < ns; ++i) {
          k1[i] = omega[i];
        }
        lu_to_solution(lhs1.data(), k1.data(), ns, ipiv);

        // stage 2
        lhs1 = LHS;
        std::vector<real> y_stage(ns, 0);
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * rok4e::alpha21 * k1[l];
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        // solve T with the new y_stage
        real T_stage = update_t(T, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int i = 0; i < ns; ++i) {
          k2[i] = omega[i] + rok4e::gamma21Gamma * k1[i];
        }
        lu_to_solution(lhs1.data(), k2.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k2[l] -= rok4e::gamma21Gamma * k1[l];
        }

        // stage 3
        lhs1 = LHS;
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * (rok4e::alpha31 * k1[l] + rok4e::alpha32 * k2[l]);
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        T_stage = update_t(T_stage, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int l = 0; l < ns; ++l) {
          k3[l] = omega[l] + rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }
        lu_to_solution(lhs1.data(), k3.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k3[l] -= rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }

        // stage 4
        for (int l = 0; l < ns; ++l) {
          k4[l] = omega[l] + rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
        lhs1 = LHS;
        lu_to_solution(lhs1.data(), k4.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k4[l] -= rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
      }

      if (adaptive_time_step) {
        // Update rhoY with error control
        std::vector<real> rhoY_new(ns, 0);
        real err{0};
        for (int l = 0; l < ns; ++l) {
          rhoY_new[l] = rhoY[l] + h_step * (rok4e::b1 * k1[l] + rok4e::b2 * k2[l] + rok4e::b4 * k4[l]);
          real dy = h_step * (rok4e::e1 * k1[l] + rok4e::e2 * k2[l] + rok4e::e3 * k3[l] + rok4e::e4 * k4[l]);
          dy /= RTol * rhoY_new[l] + ATol;
          err += dy * dy;
        }
        err = sqrt(err / ns);
        real hStar = h_step;
        if (step > 0) {
          hStar *= std::min(5.0, std::max(0.2, 0.8 * pow(err_last, 0.1) * pow(err + 1e-20, -0.175)));
        } else {
          // step == 0
          if (err > 1) {
            hStar *= 0.2;
          }
        }
        if (err <= 1) {
          // Accept the step
          for (int l = 0; l < ns; ++l) {
            rhoY[l] = rhoY_new[l];
            yk[l] = rhoY_new[l] / rho;
          }
          time += h_step;
          err_last = err;
          h = hStar;
          // if (abs(hStar - h_step) / h_step > 0.01)
            // printf("Step = %d, dt changes from %.3e to %.3e, err = %.3e\n", step, h_step, hStar, err);
        } else {
          // Reject the step
          do_update = false;
          h = hStar;
          // printf("Reject step = %d, dt changes from %.3e to %.3e, err = %.3e\n", step, h_step, hStar, err);
        }
      } else {
        for (int l = 0; l < ns; ++l) {
          rhoY[l] += h_step * (rok4e::b1 * k1[l] + rok4e::b2 * k2[l] + rok4e::b4 * k4[l]);
          yk[l] = rhoY[l] / rho;
        }
        time += h_step;
      }
    } else if (advance_scheme == 3) {
      // RK-3 explicit
      std::vector<real> rhoYN(ns, 0);
      // stage 1
      imw = 0.0;
      for (int l = 0; l < ns; ++l) {
        rhoYN[l] = rhoY[l] + omega[l] * dt;
        // if (rhoYN[l] < 0) {
        //   rhoYN[l] = 0;
        // }
        yk[l] = rhoYN[l] / rho;
        imw += yk[l] / species.mw[l];
      }
      // stage 2
      real T_stage = update_t(T, imw, species, yk, E);
      compute_src(mechanism, T_stage, species, reaction, rhoYN, q1, q2, omega_d, omega);
      imw = 0;
      for (int l = 0; l < ns; ++l) {
        rhoYN[l] = 0.75 * rhoY[l] + 0.25 * (rhoYN[l] + omega[l] * dt);
        // if (rhoYN[l] < 0) {
        //   rhoYN[l] = 0;
        // }
        yk[l] = rhoYN[l] / rho;
        imw += yk[l] / species.mw[l];
      }
      // stage 3
      T_stage = update_t(T_stage, imw, species, yk, E);
      compute_src(mechanism, T_stage, species, reaction, rhoYN, q1, q2, omega_d, omega);
      for (int l = 0; l < ns; ++l) {
        rhoY[l] = (rhoY[l] + 2.0 * (rhoYN[l] + omega[l] * dt)) / 3.0;
        if (rhoY[l] < 0) {
          rhoY[l] = 0;
        }
      }
      for (int i = 0; i < ns; ++i) {
        yk[i] = rhoY[i] / rho;
      }
    }

    // Update imw and T
    if (!do_update) {
      continue;
    }
    imw = 0.0;
    for (int i = 0; i < ns; ++i) {
      imw += yk[i] / species.mw[i];
    }
    T_last = T;
    // T = update_t_with_h(T_last, imw, species, yk, h);
    T = update_t(T_last, imw, species, yk, E);
    p = rho * R_u * imw * T;
    if (advance_scheme == 0 || advance_scheme == 3) {
      time += dt;
    }
    ++step;

    // Output results
    if (mole_mass == 1) {
      // use mole fraction for output
      if (step % screen_frequency == 0) {
        const real xH2 = yk[0] / (species.mw[0] * imw) * 100;
        const real xO2 = yk[2] / (species.mw[2] * imw) * 100;
        const real xH2O = yk[7] / (species.mw[7] * imw) * 100;
        printf("Step: %d, Time: %.3e s, T: %.5f K, P: %.2f Pa, H2%%: %.4e, O2%%: %.4e, H2O%%: %.4e\n", step, time, T, p,
               xH2, xO2, xH2O);
      }
      if (step % file_frequency == 0 || end_time - time < dt) {
        fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
        for (int l = 0; l < ns; ++l) {
          const real xl = yk[l] / (species.mw[l] * imw);
          fprintf(fp, ",%.13e", xl);
        }
        fprintf(fp, ",%d\n", step);
      }
    } else {
      if (step % screen_frequency == 0) {
        printf("Step: %d, Time: %.6e s, T: %.6f K, P: %.2f Pa, H2%%: %.4e, O2%%: %.4e, H2O%%: %.4e\n", step, time, T, p,
               yk[0], yk[2], yk[7]);
      }
      if (step % file_frequency == 0) {
        fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
        for (int l = 0; l < ns; ++l) {
          fprintf(fp, ",%.13e", yk[l]);
        }
        fprintf(fp, ",%d\n", step);
      }
    }
  }
  // ***************************End of time marching loop*******************************
  fclose(fp);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("Total simulation time is %.6es.\n", duration.count());
#endif
}

#ifdef Combustion2Part
namespace {
real smooth_reaction_temperature_host(real trx) {
  constexpr real T_min = 800.0;
  constexpr real epsilon = 80.0;
  return 0.5 * (trx + T_min + std::sqrt((trx - T_min) * (trx - T_min) + epsilon * epsilon));
}

real reaction_temperature_host(real t, real tve, real a, real b, const cfd::Reaction &reaction) {
  if (!reaction.two_temperature_reaction_temperature) return t;
  const real t_safe = std::max<real>(t, 1.0);
  const real tve_safe = std::max<real>(tve, 1.0);
  return smooth_reaction_temperature_host(std::pow(t_safe, a) * std::pow(tve_safe, b));
}

real compute_vib_energy_host(int i_spec, real t, const cfd::Species &species) {
  if (!species.has_two_temperature_data || i_spec < 0 || i_spec >= species.n_spec) return 0.0;
  const real temp = std::max<real>(t, 1.0);
  const real theta = species.theta_v[i_spec];
  if (theta <= 0.0) return 0.0;
  const real x = theta / temp;
  if (x >= 200.0) return 0.0;
  const real r_spec = cfd::R_u / species.mw[i_spec];
  return r_spec * theta / std::expm1(x);
}

real compute_vib_cv_host(int i_spec, real t, const cfd::Species &species) {
  if (!species.has_two_temperature_data || i_spec < 0 || i_spec >= species.n_spec) return 0.0;
  const real temp = std::max<real>(t, 1.0);
  const real theta = species.theta_v[i_spec];
  if (theta <= 0.0) return 0.0;
  const real x = theta / temp;
  if (x >= 200.0) return 0.0;
  const real ex = std::exp(x);
  const real denom = ex - 1.0;
  const real r_spec = cfd::R_u / species.mw[i_spec];
  return r_spec * x * x * ex / (denom * denom);
}

real compute_mixture_vib_energy_host(real t, const std::vector<real> &y, const cfd::Species &species,
                                     real *cv_v = nullptr) {
  real ev = 0.0;
  real cv = 0.0;
  for (int i = 0; i < species.n_spec; ++i) {
    const real yi = i < static_cast<int>(y.size()) ? y[i] : 0.0;
    ev += yi * compute_vib_energy_host(i, t, species);
    cv += yi * compute_vib_cv_host(i, t, species);
  }
  if (cv_v != nullptr) *cv_v = cv;
  return ev;
}

real compute_ve_energy_host(int i_spec, real t, const cfd::Species &species) {
  return species.compute_ve_energy(i_spec, t);
}

real compute_ve_cv_host(int i_spec, real t, const cfd::Species &species) {
  return species.compute_ve_cv(i_spec, t);
}

real compute_mixture_ve_energy_host(real t, const std::vector<real> &y, const cfd::Species &species,
                                    real *cv_ve = nullptr) {
  return species.compute_mixture_ve_energy(t, y, cv_ve);
}

real mw_from_imw_host(real imw) {
  return 1.0 / std::max(imw, static_cast<real>(1e-20));
}

real default_lt_a_host(real reduced_mass, real theta_v) {
  return 1.16e-3 * std::sqrt(std::max(reduced_mass, static_cast<real>(1e-20))) *
         std::pow(std::max(theta_v, static_cast<real>(0.0)), static_cast<real>(4.0 / 3.0));
}

real default_lt_b_host(real reduced_mass) {
  return 0.015 * std::pow(std::max(reduced_mass, static_cast<real>(1e-20)), static_cast<real>(0.25));
}

real compute_pair_mw_relaxation_time_host(int i_vib, int i_partner, real t, real p, const cfd::Species &species) {
  if (p <= 1e-16 || t <= 1.0) return 1e30;

  const real mw_v = species.mw[i_vib];
  const real mw_p = species.mw[i_partner];
  const real reduced_mass = mw_v * mw_p / std::max(mw_v + mw_p, static_cast<real>(1e-20));

  real a_ij = species.lt_a(i_vib, i_partner);
  real b_ij = species.lt_b(i_vib, i_partner);
  if (a_ij <= 0.0 || b_ij <= 0.0) {
    a_ij = default_lt_a_host(reduced_mass, species.theta_v[i_vib]);
    b_ij = default_lt_b_host(reduced_mass);
  }

  const real mw_exponent = a_ij * (std::pow(t, static_cast<real>(-1.0 / 3.0)) - b_ij) - 18.421;
  return std::max(std::exp(mw_exponent) * cfd::p_atm / p, static_cast<real>(1e-30));
}

real compute_species_vt_relaxation_time_host(int i_vib, real density, real t, real p, const std::vector<real> &y,
                                             const cfd::Species &species) {
  real conc = 0.0;
  real denom = 0.0;
  for (int j = 0; j < species.n_spec; ++j) {
    const real yj = std::max(y[j], static_cast<real>(0.0));
    if (yj <= 0.0) continue;
    const real mole_like = yj / species.mw[j];
    const real tau_j = compute_pair_mw_relaxation_time_host(i_vib, j, t, p, species);
    conc += mole_like;
    denom += mole_like / tau_j;
  }
  if (conc <= 0.0 || denom <= 0.0) return 1e30;

  const real tau_mw = conc / denom;
  const real sigma = std::max(species.park_sigma[i_vib], static_cast<real>(1e-30)) *
                     (static_cast<real>(2.5e9) / std::max(t * t, static_cast<real>(1.0)));
  const real n_total = density * conc * cfd::avogadro;
  const real c_s = std::sqrt(std::max(static_cast<real>(8.0) * cfd::R_u * t / (cfd::pi * species.mw[i_vib]),
                                      static_cast<real>(1e-30)));
  const real tau_park = 1.0 / std::max(sigma * c_s * n_total, static_cast<real>(1e-30));
  return std::max(tau_mw + tau_park, static_cast<real>(1e-30));
}
}

void compute_src(int mechanism, real t, const Species &species, const Reaction &reaction, const std::vector<real> &rhoY,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega) {
  compute_src(mechanism, t, t, species, reaction, rhoY, q1, q2, omega_d, omega);
}

void compute_src(int mechanism, real t, real tve, const Species &species, const Reaction &reaction,
  const std::vector<real> &rhoY, std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d,
  std::vector<real> &omega) {
  const int ns = species.n_spec;
  std::vector<real> c(ns, 0);
  for (int l = 0; l < ns; ++l) {
    c[l] = rhoY[l] / species.mw[l] * 1e-3; // Convert to mol/cm3
  }

  const int nr = reaction.n_reac;
  if (mechanism == 1) {
    // Hard-coded mechanism of Li's 9s19r H2
    chemical_source_hardCoded1(t, species, c, q1, q2, omega_d, omega);
  } else {
    // Any mechanism provided by the user
    auto kf = forward_reaction_rate(t, tve, species, reaction, c);
    auto kb = backward_reaction_rate(t, tve, species, reaction, kf, c);
    std::vector<real> q(nr, 0);
    rate_of_progress(kf, kb, c, q, q1, q2, species, reaction);
    chemical_source(q1, q2, omega_d, omega, species, reaction);
  }
}

real compute_vt_relaxation_source(real density, real t, real tve, const std::vector<real> &y, const Species &species,
  real *eve_eq, real *tau_eff) {
  if (!species.has_two_temperature_data || density <= 0.0) {
    if (eve_eq != nullptr) *eve_eq = 0.0;
    if (tau_eff != nullptr) *tau_eff = 1e30;
    return 0.0;
  }

  real r_mix = 0.0;
  for (int i = 0; i < species.n_spec; ++i) {
    r_mix += std::max(y[i], static_cast<real>(0.0)) * cfd::R_u / species.mw[i];
  }
  const real p = density * r_mix * std::max(t, static_cast<real>(1.0));

  real eve_eq_mix = 0.0;
  real eve_mix = 0.0;
  real src = 0.0;
  for (int i = 0; i < species.n_spec; ++i) {
    if (species.theta_v[i] <= 0.0) continue;
    const real yi = std::max(y[i], static_cast<real>(0.0));
    if (yi <= 0.0) continue;

    const real e_eq_i = compute_vib_energy_host(i, t, species);
    const real e_i = compute_vib_energy_host(i, tve, species);
    const real diff_i = e_eq_i - e_i;
    eve_eq_mix += yi * e_eq_i;
    eve_mix += yi * e_i;
    if (std::abs(diff_i) <= 1e-16) continue;

    const real tau_i = compute_species_vt_relaxation_time_host(i, density, t, p, y, species);
    if (tau_i < 1e29) {
      src += density * yi * diff_i / tau_i;
    }
  }

  if (eve_eq != nullptr) *eve_eq = eve_eq_mix;
  if (tau_eff != nullptr) {
    const real diff_mix = eve_eq_mix - eve_mix;
    if (std::abs(diff_mix) > 1e-16 && std::abs(src) > 1e-16) {
      *tau_eff = std::max(density * diff_mix / src, static_cast<real>(1e-30));
    } else {
      *tau_eff = 1e30;
    }
  }
  return src;
}

real compute_two_temperature_chemical_source(real tve, const Species &species, const std::vector<real> &omega) {
  real source = 0.0;
  for (int l = 0; l < species.n_spec; ++l) {
    source += omega[l] * species.compute_ve_energy(l, tve);
  }
  return source;
}

real log10_real(real x) {
  // log10(x) = ln(x) / ln(10)
  constexpr real INV_LN10 = 4.342944819032518e-01;
  return std::log(x) * INV_LN10;
}

real pow10_real(real p) {
  // 10^p = exp(ln(10) * p)
  constexpr real kLn10 = 2.302585092994046; // ln(10) 2.302585092994046
  return std::exp(kLn10 * p);
}

void chemical_source_hardCoded1(real t, const Species &species, const std::vector<real> &c,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega) {
  // Hard-coded chemical source term for Li's 9 species 19 reactions H2 mechanism
  // According to chemistry/2004-Li-IntJ.Chem.Kinet.inp
  // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2
  constexpr int ns = 9;

  // G/RT for Kc
  std::vector<real> gibbs_rt(ns, 0);
  compute_gibbs_div_rt(t, species, gibbs_rt);

  // thermodynamic scaling for kc and kb
  const real temp_t = p_atm / R_u * 1e-3 / t; // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  // Arrhenius parameters
  const real iT = 1.0 / t;
  const real iRcT = 1.0 / R_c * iT;
  const real logT = std::log(t);

  // Third body concentrations used here
  const real cc = c[0] * 2.5 + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] * 12 + c[8];

  // Production / destruction in mol/(cm3*s)
  real prod0{0}, prod1{0}, prod2{0}, prod3{0}, prod4{0}, prod5{0}, prod6{0}, prod7{0}, prod8{0};
  real dest0{0}, dest1{0}, dest2{0}, dest3{0}, dest4{0}, dest5{0}, dest6{0}, dest7{0}, dest8{0};

  // ----Reaction 0: H + O2 = O + OH
  {
    real kfb = 3.55e+15 * std::exp(-0.41 * logT - 1.66e+4 * iRcT);
    real qfb = kfb * c[1] * c[2];
    q1[0] = qfb;
    dest1 += qfb;
    dest2 += qfb;
    prod3 += qfb;
    prod4 += qfb;
    kfb = kfb * std::exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
    qfb = kfb * c[3] * c[4];
    q2[0] = qfb;
    dest3 += qfb;
    dest4 += qfb;
    prod1 += qfb;
    prod2 += qfb;
  }
  // ----Reaction 1: H2 + O = H + OH
  {
    real kfb = 5.08e+4 * std::exp(2.67 * logT - 6290 * iRcT);
    real qfb = kfb * c[0] * c[3];
    q1[1] = qfb;
    dest0 += qfb;
    dest3 += qfb;
    prod1 += qfb;
    prod4 += qfb;
    kfb *= std::exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
    qfb = kfb * c[1] * c[4];
    q2[1] = qfb;
    dest1 += qfb;
    dest4 += qfb;
    prod0 += qfb;
    prod3 += qfb;
  }
  // ----Reaction 2: H2 + OH = H2O + H
  {
    real kfb = 2.16e+8 * std::exp(1.51 * logT - 3430 * iRcT);
    real qfb = kfb * c[0] * c[4];
    q1[2] = qfb;
    dest0 += qfb;
    dest4 += qfb;
    prod7 += qfb;
    prod1 += qfb;
    kfb *= std::exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
    qfb = kfb * c[7] * c[1];
    q2[2] = qfb;
    dest7 += qfb;
    dest1 += qfb;
    prod0 += qfb;
    prod4 += qfb;
  }
  // ----Reaction 3: O + H2O = OH + OH
  {
    real kf = 2.97e+6 * std::exp(2.02 * logT - 13400 * iRcT);
    real qf = kf * c[3] * c[7];
    q1[3] = qf;
    dest3 += qf;
    dest7 += qf;
    prod4 += 2 * qf;
    kf *= std::exp(-gibbs_rt[3] - gibbs_rt[7] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[3] = qf;
    dest4 += 2 * qf;
    prod3 += qf;
    prod7 += qf;
  }
  // ----Reaction 4: H2 + M = H + H + M
  {
    real kf = 4.58e+19 * std::exp(-1.40 * logT - 1.0438e+5 * iRcT);
    kf *= cc;
    real qf = kf * c[0];
    q1[4] = qf;
    dest0 += qf;
    prod1 += 2 * qf;
    kf *= iTemp_t * std::exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
    qf = kf * c[1] * c[1];
    q2[4] = qf;
    dest1 += 2 * qf;
    prod0 += qf;
  }
  // ----Reaction 5: O + O + M = O2 + M
  {
    real kf = 6.16e+15 * sqrt(iT) * cc;
    real qf = kf * c[3] * c[3];
    q1[5] = qf;
    dest3 += 2 * qf;
    prod2 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
    qf = kf * c[2];
    q2[5] = qf;
    dest2 += qf;
    prod3 += 2 * qf;
  }
  // ----Reaction 6: O + H + M = OH + M
  {
    real kf = 4.71e+18 * iT * cc;
    real qf = kf * c[1] * c[3];
    q1[6] = qf;
    dest3 += qf;
    dest1 += qf;
    prod4 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[4];
    q2[6] = qf;
    dest4 += qf;
    prod3 += qf;
    prod1 += qf;
  }
  // ----Reaction 7: H + OH + M = H2O + M
  {
    real kf = 3.8e+22 * iT * iT * cc;
    real qf = kf * c[1] * c[4];
    q1[7] = qf;
    dest1 += qf;
    dest4 += qf;
    prod7 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[1] - gibbs_rt[4] + gibbs_rt[7]);
    qf = kf * c[7];
    q2[7] = qf;
    dest7 += qf;
    prod1 += qf;
    prod4 += qf;
  }
  // ----Reaction 8: H + O2 (+M) = HO2 (+M)
  {
    const real kf_high = 1.48e+12 * std::exp(0.6 * logT);
    const real kf_low = 6.37e+20 * std::exp(-1.72 * logT - 5.2e+2 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real cc2 = c[0] * 2 + c[1] + c[2] * 0.78 + c[3] + c[4] + c[5] + c[6] + c[7] * 11 + c[8];
      const real reduced_pressure = kf_low * cc2 / kf_high;
      constexpr real logFc = -0.09691001300805639; // log10(0.8) = -9.691001300805639e-02
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[1] * c[2];
    q1[8] = qf;
    dest1 += qf;
    dest2 += qf;
    prod5 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
    qf = kf * c[5];
    q2[8] = qf;
    dest5 += qf;
    prod1 += qf;
    prod2 += qf;
  }
  // ----Reaction 9: HO2 + H = H2 + O2
  {
    real kf = 1.66e+13 * std::exp(-820 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[9] = qf;
    dest5 += qf;
    dest1 += qf;
    prod0 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
    qf = kf * c[0] * c[2];
    q2[9] = qf;
    dest0 += qf;
    dest2 += qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 10: HO2 + H = OH + OH
  {
    real kf = 7.08e+13 * std::exp(-300 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[10] = qf;
    dest5 += qf;
    dest1 += qf;
    prod4 += 2 * qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[10] = qf;
    dest4 += 2 * qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 11: HO2 + O = O2 + OH
  {
    real qf = 3.25e+13 * c[5] * c[3];
    q1[11] = qf;
    dest5 += qf;
    dest3 += qf;
    prod2 += qf;
    prod4 += qf;
    const real kb = 3.25e+13 * std::exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
    qf = kb * c[2] * c[4];
    q2[11] = qf;
    dest2 += qf;
    dest4 += qf;
    prod5 += qf;
    prod3 += qf;
  }
  // ----Reaction 12: HO2 + OH = H2O + O2
  {
    real kf = 2.89e+13 * std::exp(500 * iRcT);
    real qf = kf * c[5] * c[4];
    q1[12] = qf;
    dest5 += qf;
    dest4 += qf;
    prod7 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
    qf = kf * c[7] * c[2];
    q2[12] = qf;
    dest7 += qf;
    dest2 += qf;
    prod5 += qf;
    prod4 += qf;
  }
  // ----Reaction 13: HO2 + HO2 = H2O2 + O2
  {
    real kf = 4.20e+14 * std::exp(-11980 * iRcT) + 1.30e+11 * std::exp(1630 * iRcT);
    real qf = kf * c[5] * c[5];
    q1[13] = qf;
    dest5 += 2 * qf;
    prod6 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
    qf = kf * c[6] * c[2];
    q2[13] = qf;
    dest6 += qf;
    dest2 += qf;
    prod5 += 2 * qf;
  }
  // ----Reaction 14: H2O2 (+ M) = OH + OH (+ M)
  {
    const real kf_high = 2.95e+14 * std::exp(-48400 * iRcT);
    const real kf_low = 1.20e+17 * std::exp(-45500 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real reduced_pressure = kf_low * cc / kf_high;
      constexpr real logFc = -3.010299956639812e-01; // log10(0.5) = -3.010299956639812e-01
      const real cT = -0.4 - 0.67 * logFc;
      const real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[6];
    q1[14] = qf;
    dest6 += qf;
    prod4 += 2 * qf;
    kf *= iTemp_t * std::exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[14] = qf;
    dest4 += 2 * qf;
    prod6 += qf;
  }
  // ----Reaction 15: H2O2 + H = H2O + OH
  {
    real kf = 2.41e+13 * std::exp(-3970 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[15] = qf;
    dest6 += qf;
    dest1 += qf;
    prod7 += qf;
    prod4 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
    qf = kf * c[7] * c[4];
    q2[15] = qf;
    dest7 += qf;
    dest4 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ---Reaction 16: H2O2 + H = HO2 + H2
  {
    real kf = 4.82e+13 * std::exp(-7950 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[16] = qf;
    dest6 += qf;
    dest1 += qf;
    prod5 += qf;
    prod0 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
    qf = kf * c[5] * c[0];
    q2[16] = qf;
    dest5 += qf;
    dest0 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ----Reaction 17: H2O2 + O = OH + HO2
  {
    real kf = 9.55e+6 * t * t * std::exp(-3970 * iRcT);
    real qf = kf * c[6] * c[3];
    q1[17] = qf;
    dest6 += qf;
    dest3 += qf;
    prod4 += qf;
    prod5 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
    qf = kf * c[4] * c[5];
    q2[17] = qf;
    dest4 += qf;
    dest5 += qf;
    prod6 += qf;
    prod3 += qf;
  }
  // ----Reaction 18: H2O2 + OH = HO2 + H2O
  {
    real kf = 1e+12 + 5.8e+14 * std::exp(-9560 * iRcT);
    real qf = kf * c[6] * c[4];
    q1[18] = qf;
    dest6 += qf;
    dest4 += qf;
    prod5 += qf;
    prod7 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
    qf = kf * c[5] * c[7];
    q2[18] = qf;
    dest5 += qf;
    dest7 += qf;
    prod6 += qf;
    prod4 += qf;
  }

  // Compute net production rates
  // Convert from mol/(cm3*s) to kg/(m3*s)
  omega_d[0] = dest0 * 2016;
  omega_d[1] = dest1 * 1008;
  omega_d[2] = dest2 * 31998;
  omega_d[3] = dest3 * 15999;
  omega_d[4] = dest4 * 17007;
  omega_d[5] = dest5 * 33006;
  omega_d[6] = dest6 * 34014;
  omega_d[7] = dest7 * 18015;
  omega_d[8] = dest8 * 28014;

  omega[0] = (prod0 - dest0) * 2016;
  omega[1] = (prod1 - dest1) * 1008;
  omega[2] = (prod2 - dest2) * 31998;
  omega[3] = (prod3 - dest3) * 15999;
  omega[4] = (prod4 - dest4) * 17007;
  omega[5] = (prod5 - dest5) * 33006;
  omega[6] = (prod6 - dest6) * 34014;
  omega[7] = (prod7 - dest7) * 18015;
  omega[8] = (prod8 - dest8) * 28014;
}

void compute_gibbs_div_rt(real t, const Species &species, std::vector<real> &gibbs_rt) {
  const int ns = species.n_spec;
  if (species.nasa_7_or_9 == 7) {
    const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
    for (int i = 0; i < ns; ++i) {
      if (t < species.t_low[i]) {
        const real tt = species.t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto &coeff = species.low_temp_coeff;
        gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                      coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
      } else {
        const auto &coeff = t < species.t_mid[i] ? species.low_temp_coeff : species.high_temp_coeff;
        gibbs_rt[i] =
            coeff(i, 0) * (1.0 - log_t) - 0.5 * coeff(i, 1) * t - coeff(i, 2) * t2 / 6.0 - coeff(i, 3) * t3 / 12.0 -
            coeff(i, 4) * t4 * 0.05 + coeff(i, 5) * t_inv - coeff(i, 6);
      }
    }
    return;
  }

  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, it{1.0 / t}, lnt{std::log(t)}, it2{it * it};
  const auto &coeff = species.therm_poly_coeff;
  for (int i = 0; i < ns; ++i) {
    if (t < species.temperature_range(i, 0)) {
      const real tt = species.temperature_range(i, 0);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      const real itt = 1.0 / tt, itt2 = itt * itt, lntt = std::log(tt);
      gibbs_rt[i] = -0.5 * coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt * (lntt + 1) +
                    coeff(2, 0, i) * (1.0 - lntt) - 0.5 * coeff(3, 0, i) * tt - coeff(4, 0, i) * tt2 / 6.0 -
                    coeff(5, 0, i) * tt3 / 12.0 - coeff(6, 0, i) * tt4 * 0.05 + coeff(7, 0, i) * itt -
                    coeff(8, 0, i);
    } else if (t > species.temperature_range(i, species.n_temperature_range[i])) {
      const real tt = species.temperature_range(i, species.n_temperature_range[i]);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      const real itt = 1.0 / tt, itt2 = itt * itt, lntt = std::log(tt);
      const int j = species.n_temperature_range[i] - 1;
      gibbs_rt[i] = -0.5 * coeff(0, j, i) * itt2 + coeff(1, j, i) * itt * (lntt + 1) +
                    coeff(2, j, i) * (1.0 - lntt) - 0.5 * coeff(3, j, i) * tt - coeff(4, j, i) * tt2 / 6.0 -
                    coeff(5, j, i) * tt3 / 12.0 - coeff(6, j, i) * tt4 * 0.05 + coeff(7, j, i) * itt -
                    coeff(8, j, i);
    } else {
      for (int j = 0; j < species.n_temperature_range[i]; ++j) {
        if (species.temperature_range(i, j) <= t && t <= species.temperature_range(i, j + 1)) {
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

real update_t(real T0, real imw, const Species &species, const std::vector<real> &yk, real E) {
  // Use Newton-Raphson method to update temperature
  real t = T0;
  const int ns = species.n_spec;
  constexpr int max_iter = 100;
  constexpr real eps = 1e-8;

  const real R = R_u * imw;
  real err = 1e+6;
  int iter = 0;
  while (err > eps && iter++ < max_iter) {
    std::vector<real> hk(ns, 0), cpk(ns, 0);
    species.compute_enthalpy_and_cp(t, hk.data(), cpk.data());
    real cp_tot{0}, h{0};
    for (int l = 0; l < ns; ++l) {
      cp_tot += cpk[l] * yk[l];
      h += hk[l] * yk[l];
    }
    const real et = h - R * t;
    const real cv = cp_tot - R;
    const real t1 = t - (et - E) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  return t;
}

real update_t_two_temperature(real T0, const Species &species, const std::vector<real> &yk, real E, real eve) {
  const int ns = species.n_spec;
  constexpr int max_iter = 80;
  constexpr real eps = 1e-8;

  auto evaluate_state = [&](real t_eval, real &residual, real &cv_tr) {
    std::vector<real> hk(ns, 0), cpk(ns, 0);
    species.compute_enthalpy_and_cp(t_eval, hk.data(), cpk.data());
    real e_mix = 0.0;
    real cv_mix = 0.0;
    for (int l = 0; l < ns; ++l) {
      const real r_spec = R_u / species.mw[l];
      e_mix += yk[l] * (hk[l] - r_spec * t_eval);
      cv_mix += yk[l] * (cpk[l] - r_spec);
    }
    real cv_v_eq = 0.0;
    const real e_ve_eq = compute_mixture_ve_energy_host(t_eval, yk, species, &cv_v_eq);
    residual = e_mix - e_ve_eq + eve - E;
    cv_tr = cv_mix - cv_v_eq;
    if (!std::isfinite(e_mix) || !std::isfinite(cv_mix) || !std::isfinite(e_ve_eq) || !std::isfinite(cv_v_eq) ||
        !std::isfinite(residual) || !std::isfinite(cv_tr) || cv_tr <= 0.0) {
      fprintf(stderr,
              "[2T-CVR] update_t_two_temperature evaluation failure: T=%.13e E=%.13e Eve=%.13e "
              "e_mix=%.13e e_ve_eq=%.13e cv_mix=%.13e cv_ve_eq=%.13e residual=%.13e cv_tr=%.13e\n",
              t_eval, E, eve, e_mix, e_ve_eq, cv_mix, cv_v_eq, residual, cv_tr);
      for (int l = 0; l < ns; ++l) {
        fprintf(stderr, "  y[%d]=%.13e hk=%.13e cp=%.13e\n", l, yk[l], hk[l], cpk[l]);
      }
      fflush(stderr);
      return false;
    }
    return true;
  };

  real t = std::max<real>(T0, 1.0);
  real residual = 0.0, cv_tr = 0.0;
  if (!evaluate_state(t, residual, cv_tr)) return std::max<real>(T0, 1.0);

  for (int iter = 0; iter < max_iter; ++iter) {
    const real scale = std::max<real>(1.0, std::abs(E) + std::abs(eve));
    if (std::abs(residual) < eps * scale) break;

    real dt_newton = -residual / cv_tr;
    const real limited_step = std::clamp(dt_newton, -0.2 * t, 0.2 * t);
    real candidate = std::max<real>(1.0, t + limited_step);
    real residual_candidate = 0.0, cv_tr_candidate = 0.0;
    bool accepted = false;

    for (int backtrack = 0; backtrack < 20; ++backtrack) {
      if (evaluate_state(candidate, residual_candidate, cv_tr_candidate) &&
          std::abs(residual_candidate) < std::abs(residual)) {
        accepted = true;
        break;
      }
      candidate = std::max<real>(1.0, 0.5 * (candidate + t));
    }

    if (!accepted) {
      break;
    }

    const real err = std::abs(candidate - t) / std::max<real>(1.0, t);
    t = candidate;
    residual = residual_candidate;
    cv_tr = cv_tr_candidate;
    if (err < eps) break;
  }
  return t;
}

real update_t_with_h(real T0, real imw, const Species &species, const std::vector<real> &yk, real h) {
  real t = T0;
  const int ns = species.n_spec;
  constexpr int max_iter = 100;
  constexpr real eps = 1e-8;

  const real R = R_u * imw;
  real err = 1e+6;
  int iter = 0;
  while (err > eps && iter++ < max_iter) {
    std::vector<real> hk(ns, 0), cpk(ns, 0);
    species.compute_enthalpy_and_cp(t, hk.data(), cpk.data());
    real cp_tot{0}, h_iter = 0;
    for (int l = 0; l < ns; ++l) {
      cp_tot += cpk[l] * yk[l];
      h_iter += hk[l] * yk[l];
    }
    const real t1 = t - (h_iter - h) / cp_tot;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  return t;
}

std::vector<real> forward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &c) {
  return forward_reaction_rate(t, t, species, reaction, c);
}

std::vector<real> forward_reaction_rate(real t, real tve, const Species &species, const Reaction &reaction,
  const std::vector<real> &c) {
  const int ns = species.n_spec, nr = reaction.n_reac;
  std::vector<real> kf(nr, 0);
  const auto &A = reaction.A, &b = reaction.b, &Ea = reaction.Ea;
  const auto &type = reaction.label;
  const auto &A2 = reaction.A2, &b2 = reaction.b2, &Ea2 = reaction.Ea2;
  const auto &third_body_coeff = reaction.third_body_coeff;
  const auto &alpha = reaction.troe_alpha, &t3 = reaction.troe_t3, &t1 = reaction.troe_t1, &t2 = reaction.troe_t2;
  const auto &tcf_a = reaction.tcf_a, &tcf_b = reaction.tcf_b;
  for (int i = 0; i < nr; ++i) {
    const real thf = reaction_temperature_host(t, tve, tcf_a[i], tcf_b[i], reaction);
    kf[i] = arrhenius(thf, A[i], b[i], Ea[i]);
    if (type[i] == 3) {
      // Duplicate reaction
      kf[i] += arrhenius(thf, A2[i], b2[i], Ea2[i]);
    } else if (type[i] > 3) {
      real cc{0};
      for (int l = 0; l < ns; ++l) {
        cc += c[l] * third_body_coeff(i, l);
      }
      if (type[i] == 4) {
        // Third body reaction
        kf[i] *= cc;
      } else {
        const real kf_low = arrhenius(thf, A2[i], b2[i], Ea2[i]);
        const real kf_high = kf[i];
        if (kf_high < 1e-25 && kf_low < 1e-25) {
          // If both kf_high and kf_low are too small, set kf to zero
          kf[i] = 0;
          continue;
        }
        const real reduced_pressure = kf_low * cc / kf_high;
        real F = 1.0;
        if (type[i] > 5) {
          // Troe form
          real f_cent = (1 - alpha[i]) * std::exp(-thf / t3[i]) + alpha[i] * std::exp(-thf / t1[i]);
          if (type[i] == 7) {
            f_cent += std::exp(-t2[i] / thf);
          }
          const real logFc = std::log10(f_cent);
          const real cn = -0.4 - 0.67 * logFc;
          const real n = 0.75 - 1.27 * logFc;
          const real logPr = std::log10(reduced_pressure);
          const real tempo = (logPr + cn) / (n - 0.14 * (logPr + cn));
          const real p = logFc / (1.0 + tempo * tempo);
          F = std::pow(10, p);
        }
        kf[i] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * F;
      }
    }
  }
  return std::move(kf);
}

std::vector<real> backward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &kf, const std::vector<real> &c) {
  return backward_reaction_rate(t, t, species, reaction, kf, c);
}

std::vector<real> backward_reaction_rate(real t, real tve, const Species &species, const Reaction &reaction,
  const std::vector<real> &kf, const std::vector<real> &c) {
  int n_gibbs{reaction.n_reac};
  const int nr = reaction.n_reac;
  const auto &type = reaction.label;
  const auto &tcb_a = reaction.tcb_a, &tcb_b = reaction.tcb_b;
  std::vector<real> kb(nr, 0);
  for (int i = 0; i < nr; ++i) {
    if (type[i] == 0) {
      // Irreversible reaction
      kb[i] = 0;
      --n_gibbs;
    } else if (reaction.rev_type[i] == 1) {
      // REV reaction
      const real thb = reaction_temperature_host(t, tve, tcb_a[i], tcb_b[i], reaction);
      kb[i] = arrhenius(thb, reaction.A2[i], reaction.b2[i], reaction.Ea2[i]);
      if (type[i] == 4) {
        // Third body required
        real cc{0};
        for (int l = 0; l < species.n_spec; ++l) {
          cc += c[l] * reaction.third_body_coeff(i, l);
        }
        kb[i] *= cc;
      }
      --n_gibbs;
    }
  }
  if (n_gibbs < 1)
    return std::move(kb);

  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  const auto order = reaction.order;
  for (int i = 0; i < nr; ++i) {
    if (type[i] != 2 && type[i] != 0) {
      const real thb = reaction_temperature_host(t, tve, tcb_a[i], tcb_b[i], reaction);
      std::vector<real> gibbs_rt(species.n_spec, 0);
      compute_gibbs_div_rt(thb, species, gibbs_rt);
      constexpr real temp_p = p_atm / R_u * 1e-3; // Convert the unit to mol*K/cm3
      const real temp_t = temp_p / thb;           // Unit is mol/cm3
      real d_gibbs{0};
      for (int l = 0; l < species.n_spec; ++l) {
        d_gibbs += gibbs_rt[l] * (stoi_b(i, l) - stoi_f(i, l));
      }
      const real kc{std::pow(temp_t, order[i]) * std::exp(-d_gibbs)};
      kb[i] = kf[i] / kc;
    }
  }
  return std::move(kb);
}

void rate_of_progress(const std::vector<real> &kf, const std::vector<real> &kb, const std::vector<real> &c,
  std::vector<real> &q, std::vector<real> &q1, std::vector<real> &q2, const Species &species,
  const Reaction &reaction) {
  const int ns{species.n_spec};
  const auto &stoi_f{reaction.stoi_f}, &stoi_b{reaction.stoi_b};
  for (int i = 0; i < reaction.n_reac; ++i) {
    if (reaction.label[i] != 0) {
      q1[i] = 1.0;
      q2[i] = 1.0;
      for (int j = 0; j < ns; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
        q2[i] *= std::pow(c[j], stoi_b(i, j));
      }
      q1[i] *= kf[i];
      q2[i] *= kb[i];
      q[i] = q1[i] - q2[i];
    } else {
      q1[i] = 1.0;
      q2[i] = 0.0;
      for (int j = 0; j < ns; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
      }
      q1[i] *= kf[i];
      q[i] = q1[i];
    }
  }
}

void chemical_source(const std::vector<real> &q1, const std::vector<real> &q2, std::vector<real> &omega_d,
  std::vector<real> &omega, const Species &species, const Reaction &reaction) {
  const int ns{species.n_spec};
  const int nr{reaction.n_reac};
  const auto &stoi_f = reaction.stoi_f, &stoi_b{reaction.stoi_b};
  const auto mw = species.mw;
  for (int i = 0; i < ns; ++i) {
    real creation = 0;
    omega_d[i] = 0;
    for (int j = 0; j < nr; ++j) {
      creation += q2[j] * stoi_f(j, i) + q1[j] * stoi_b(j, i);
      omega_d[i] += q1[j] * stoi_f(j, i) + q2[j] * stoi_b(j, i);
    }
    creation *= 1e+3 * mw[i];         // Unit is kg/(m3*s)
    omega_d[i] *= 1e+3 * mw[i];       // Unit is kg/(m3*s)
    omega[i] = creation - omega_d[i]; // Unit is kg/(m3*s)
  }
}

std::vector<real> compute_chem_src_jacobian(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2) {
  const int ns{species.n_spec}, nr{reaction.n_reac};
  std::vector<real> jac(ns * ns, 0);
  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  for (int m = 0; m < ns; ++m) {
    for (int n = 0; n < ns; ++n) {
      real zz{0};
      if (rhoY[n] > 1e-30) {
        for (int r = 0; r < nr; ++r) {
          // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
          zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, n) * q1[r] - stoi_b(r, n) * q2[r]);
        }
        zz /= rhoY[n];
      }
      jac[m * ns + n] = zz * 1e+3 * species.mw[m]; // //1e+3=1e-3(MW)*1e+6(cm->m)
    }
  }
  return std::move(jac);
}

std::vector<real> compute_chem_src_jacobian_diagonal(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2) {
  const int ns{species.n_spec}, nr{reaction.n_reac};
  std::vector<real> jac_diag(ns, 0);
  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  for (int m = 0; m < ns; ++m) {
    real zz{0};
    if (rhoY[m] > 1e-30) {
      for (int r = 0; r < nr; ++r) {
        // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
        zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, m) * q1[r] - stoi_b(r, m) * q2[r]);
      }
      zz /= rhoY[m];
    }
    jac_diag[m] = zz * 1e+3 * species.mw[m]; // //1e+3=1e-3(MW)*1e+6(cm->m)
  }
  return std::move(jac_diag);
}

void EPI(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega) {
  const int ns = species.n_spec;
  std::vector<real> lhs(ns * ns, 0);
  for (int m = 0; m < ns; ++m) {
    for (int n = 0; n < ns; ++n) {
      if (m == n) {
        lhs[m * ns + n] = 1.0 - dt * jac[m * ns + n];
      } else {
        lhs[m * ns + n] = -dt * jac[m * ns + n];
      }
    }
  }
  auto ipiv = lu_decomp(lhs.data(), ns);
  lu_to_solution(lhs.data(), omega.data(), ns, ipiv);
}

void DA(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega) {
  const int ns = species.n_spec;
  for (int l = 0; l < ns; ++l) {
    omega[l] /= 1 - dt * jac[l];
  }
}

std::vector<int> lu_decomp(real *lhs, int dim) {
  std::vector iPiv(dim, 0);
  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    iPiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }
  return std::move(iPiv);
}

void lu_to_solution(real *lhs, real *rhs, int dim, const std::vector<int> &ipiv) {
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = rhs[t];
      rhs[t] = rhs[m];
      rhs[m] = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
  }
  rhs[dim - 1] /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
    rhs[m] /= lhs[m * dim + m];
  }
}

int buildArnoldi(const Species &species, const std::vector<real> &rhoY, const real *f0, real *Q, real *H,
  int krylovMaxDim, int mechanism, real T, const Reaction &reaction, real rho, real E, int krylovMinDim) {
  const int ns = species.n_spec;
  const real norm_y = gxl::vec_norm(rhoY.data(), ns);
  const real eps = 1e-6 * (1 + norm_y);

  // v0 = f0 / ||f0||
  if (real norm_f0 = gxl::vec_norm<real>(f0, ns); norm_f0 < 1e-30) {
    // use unit vector to avoid zero basis
    for (int l = 0; l < ns; ++l) {
      Q[l] = 0;
    }
    Q[0] = 1.0;
    norm_f0 = 1.0;
  } else {
    for (int l = 0; l < ns; ++l) {
      Q[l] = f0[l] / norm_f0;
    }
  }
  int krylov_dim = 1;
  std::vector<real> q1(reaction.n_reac, 0), q2(reaction.n_reac, 0), omega_d(ns, 0), f_eps(ns, 0);
  for (int col = 0; col < krylovMaxDim; ++col) {
    // w = J * v_col
    // real y_eps[MAX_SPEC_NUMBER];
    std::vector<real> y_eps(ns, 0);
    std::vector<real> yk(ns, 0);
    real imw = 0;
    for (int l = 0; l < ns; ++l) {
      y_eps[l] = rhoY[l] + eps * Q[col * ns + l];
      yk[l] = y_eps[l] / rho;
      imw += yk[l] / species.mw[l];
    }
    real T_stage = update_t(T, imw, species, yk, E);
    // real f_eps[MAX_SPEC_NUMBER];
    compute_src(mechanism, T_stage, species, reaction, y_eps, q1, q2, omega_d, f_eps);
    real w[MAX_SPEC_NUMBER];
    for (int l = 0; l < ns; ++l) {
      w[l] = (f_eps[l] - f0[l]) / eps;
    }

    // Modified Gram-Schmidt
    for (int row = 0; row <= col; ++row) {
      const real hij = gxl::vec_dot(&Q[row * ns], w, ns);
      H[row * krylovMaxDim + col] = hij;
      for (int l = 0; l < ns; ++l) {
        w[l] -= hij * Q[row * ns + l];
      }
    }
    real h_next = gxl::vec_norm(w, ns);
    if (col + 1 < krylovMaxDim) {
      H[(col + 1) * krylovMaxDim + col] = h_next;
    }
    if (col + 1 < krylovMinDim) {
      if (h_next < 1e-30) h_next = 1e-30;
      for (int l = 0; l < ns; ++l) {
        Q[(col + 1) * ns + l] = w[l] / h_next;
      }
      krylov_dim = col + 2;
      continue;
    }
    if (h_next > 1e-30 && col + 1 < krylovMaxDim) {
      for (int l = 0; l < ns; ++l) {
        Q[(col + 1) * ns + l] = w[l] / h_next;
      }
      krylov_dim = col + 2;
    } else {
      krylov_dim = col + 1;
      break;
    }
  }
  return krylov_dim;
}
#endif
}

#include "PostProcess.h"
#include "Field.h"
#include "Constants.h"
#include "Thermo.cuh"
#include <filesystem>
#include <fstream>

namespace {
constexpr int kWallOutputStride = 6;

__device__ __forceinline__ real safe_face_area(real x, real y, real z) {
  return max(norm3d(x, y, z), static_cast<real>(1e-30));
}
}

void cfd::post_process(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, DParameter *param) {
  static const std::vector<int> processes{parameter.get_int_array("post_process")};
  if (processes.empty()) return;

  for (const auto process: processes) {
    switch (process) {
      case 0:
        wall_friction_heatflux_2d(mesh, field, parameter, param);
        break;
      case 1:
        wall_friction_heatFlux_3d(mesh, field, parameter, param);
        break;
      default:
        break;
    }
  }
}

void
cfd::wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter,
                               const DParameter *param) {
  const std::filesystem::path out_dir("output/wall");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  const auto path_name = out_dir.string();
  const int myid = parameter.get_int("myid");
  const bool is_parallel_output = parameter.get_int("n_proc") > 1;

  int size{mesh[0].mx};
  for (int blk = 1; blk < mesh.n_block; ++blk) {
    if (mesh[blk].mx > size) {
      size = mesh[blk].mx;
    }
  }
  std::vector<double> wall_data(size * kWallOutputStride, 0.0);
  real *wall_data_device = nullptr;
  cudaMalloc(&wall_data_device, size * kWallOutputStride * sizeof(real));

  const double rho_inf = parameter.get_real("rho_inf");
  const double v_inf = parameter.get_real("v_inf");
  const double dyn_pressure = 0.5 * rho_inf * v_inf * v_inf;
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    auto &block = mesh[blk];
    const int mx{block.mx};

    dim3 bpg((mx - 1) / 128 + 1, 1, 1);
    wall_friction_heatFlux_2d<<<bpg, 128>>>(field[blk].d_ptr, wall_data_device, param, dyn_pressure);
    cudaMemcpy(wall_data.data(), wall_data_device, size * kWallOutputStride * sizeof(real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::string file_name = "/friction_heatflux-block-" + std::to_string(blk) + ".dat";
    if (is_parallel_output) {
      file_name = "/friction_heatflux-proc-" + std::to_string(myid) + "-block-" + std::to_string(blk) + ".dat";
    }
    std::ofstream f(path_name + file_name);
    f << "variables = \"x\", \"tau_w\", \"cf\", \"q_tr\", \"q_ve\", \"q_total\", \"y_plus\"\n";
    for (int i = 0; i < mx; ++i) {
      const auto offset = i * kWallOutputStride;
      f << block.x(i, 0, 0) << '\t'
        << wall_data[offset] << '\t'
        << wall_data[offset + 1] << '\t'
        << wall_data[offset + 2] << '\t'
        << wall_data[offset + 3] << '\t'
        << wall_data[offset + 4] << '\t'
        << wall_data[offset + 5] << '\n';
    }
    f.close();
  }
  cudaFree(wall_data_device);
}

__global__ void cfd::wall_friction_heatFlux_2d(DZone *zone, real *wall_data, const DParameter *param, real dyn_pressure) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= zone->mx) return;

  constexpr int j = 0;
  constexpr int k = 0;
  auto &pv = zone->bv;
  real tx_raw{}, ty_raw{}, tz_raw{};
  if (i == 0) {
    tx_raw = zone->x(i + 1, j, k) - zone->x(i, j, k);
    ty_raw = zone->y(i + 1, j, k) - zone->y(i, j, k);
    tz_raw = zone->z(i + 1, j, k) - zone->z(i, j, k);
  } else if (i == zone->mx - 1) {
    tx_raw = zone->x(i, j, k) - zone->x(i - 1, j, k);
    ty_raw = zone->y(i, j, k) - zone->y(i - 1, j, k);
    tz_raw = zone->z(i, j, k) - zone->z(i - 1, j, k);
  } else {
    tx_raw = zone->x(i + 1, j, k) - zone->x(i - 1, j, k);
    ty_raw = zone->y(i + 1, j, k) - zone->y(i - 1, j, k);
    tz_raw = zone->z(i + 1, j, k) - zone->z(i - 1, j, k);
  }
  const real xi_mag = safe_face_area(tx_raw, ty_raw, tz_raw);
  const real tx = tx_raw / xi_mag, ty = ty_raw / xi_mag, tz = tz_raw / xi_mag;
  const auto &metric = zone->metric;
  const real xi_x = 0.5 * (metric(i, j, k, 0) + metric(i, j + 1, k, 0));
  const real xi_y = 0.5 * (metric(i, j, k, 1) + metric(i, j + 1, k, 1));
  const real xi_z = 0.5 * (metric(i, j, k, 2) + metric(i, j + 1, k, 2));
  const real eta_x = 0.5 * (metric(i, j, k, 3) + metric(i, j + 1, k, 3));
  const real eta_y = 0.5 * (metric(i, j, k, 4) + metric(i, j + 1, k, 4));
  const real eta_z = 0.5 * (metric(i, j, k, 5) + metric(i, j + 1, k, 5));

  const real dx = zone->x(i, j + 1, k) - zone->x(i, j, k);
  const real dy = zone->y(i, j + 1, k) - zone->y(i, j, k);
  const real dz = zone->z(i, j + 1, k) - zone->z(i, j, k);
  const real dn = safe_face_area(dx, dy, dz);

  // Reuse the eta-face discretization used by the 2nd-order viscous flux.
  const real u_xi =
      0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i - 1, j + 1, k, 1));
  const real u_eta = pv(i, j + 1, k, 1) - pv(i, j, k, 1);
  const real v_xi =
      0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i - 1, j + 1, k, 2));
  const real v_eta = pv(i, j + 1, k, 2) - pv(i, j, k, 2);
  const real w_xi =
      0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i - 1, j + 1, k, 3));
  const real w_eta = pv(i, j + 1, k, 3) - pv(i, j, k, 3);
  const real t_xi =
      0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i - 1, j + 1, k, 5));
  const real t_eta = pv(i, j + 1, k, 5) - pv(i, j, k, 5);

  real tve_xi{0.0}, tve_eta{0.0};
  if constexpr (kTwoTemperature) {
    if (param->i_eve >= 0) {
      const auto &tve_field = zone->temperature_ve;
      tve_xi = 0.25 * (tve_field(i + 1, j, k) - tve_field(i - 1, j, k) + tve_field(i + 1, j + 1, k) -
                       tve_field(i - 1, j + 1, k));
      tve_eta = tve_field(i, j + 1, k) - tve_field(i, j, k);
    }
  }

  const real u_x = u_xi * xi_x + u_eta * eta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i, j + 1, k));
  const real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  const real eta_x_div_jac =
      0.5 * (metric(i, j, k, 3) * zone->jac(i, j, k) + metric(i, j + 1, k, 3) * zone->jac(i, j + 1, k));
  const real eta_y_div_jac =
      0.5 * (metric(i, j, k, 4) * zone->jac(i, j, k) + metric(i, j + 1, k, 4) * zone->jac(i, j + 1, k));
  const real eta_z_div_jac =
      0.5 * (metric(i, j, k, 5) * zone->jac(i, j, k) + metric(i, j + 1, k, 5) * zone->jac(i, j + 1, k));
  const real face_area = safe_face_area(eta_x_div_jac, eta_y_div_jac, eta_z_div_jac);
  const real nx = eta_x_div_jac / face_area;
  const real ny = eta_y_div_jac / face_area;
  const real nz = eta_z_div_jac / face_area;

  const real traction_x = tau_xx * nx + tau_xy * ny + tau_xz * nz;
  const real traction_y = tau_xy * nx + tau_yy * ny + tau_yz * nz;
  const real traction_z = tau_xz * nx + tau_yz * ny + tau_zz * nz;
  const real tau_w = abs(traction_x * tx + traction_y * ty + traction_z * tz);
  const real cf = tau_w / max(dyn_pressure, static_cast<real>(1e-30));

  const real t_x = t_xi * xi_x + t_eta * eta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z;
  real conductivity{};
  if (param->n_spec > 0) {
    conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i, j + 1, k));
  } else {
    constexpr real cp_air{gamma_air * R_u / mw_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp_air;
  }

  real q_total_face = conductivity * (eta_x_div_jac * t_x + eta_y_div_jac * t_y + eta_z_div_jac * t_z);
  real q_ve_face = 0.0;

  if (param->n_spec > 0) {
    const auto &y = zone->sv;
    real diffusivity[MAX_SPEC_NUMBER];
    real yk[MAX_SPEC_NUMBER];
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    real sum_grad_eta_dot_grad_y_over_wl{0.0};
    real sum_rhoDkYk{0.0};
    real correction_velocity_term{0.0};
    real mw_tot{0.0};

    for (int l = 0; l < param->n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i, j + 1, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j + 1, k, l));

      const real y_xi =
          0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j + 1, k, l) - y(i - 1, j + 1, k, l));
      const real y_eta = y(i, j + 1, k, l) - y(i, j, k, l);

      const real y_x = y_xi * xi_x + y_eta * eta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z;
      const real grad_eta_dot_grad_y = eta_x_div_jac * y_x + eta_y_div_jac * y_y + eta_z_div_jac * y_z;
      diffusion_driven_force[l] = grad_eta_dot_grad_y;
      correction_velocity_term += diffusivity[l] * grad_eta_dot_grad_y;

      sum_grad_eta_dot_grad_y_over_wl += grad_eta_dot_grad_y * param->imw[l];
      mw_tot += yk[l] * param->imw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }

    mw_tot = 1.0 / max(mw_tot, static_cast<real>(1e-30));
    correction_velocity_term -= mw_tot * sum_rhoDkYk * sum_grad_eta_dot_grad_y_over_wl;

    if (param->gradPInDiffusionFlux) {
      const real p_xi =
          0.25 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4) + pv(i + 1, j + 1, k, 4) - pv(i - 1, j + 1, k, 4));
      const real p_eta = pv(i, j + 1, k, 4) - pv(i, j, k, 4);

      const real p_x = p_xi * xi_x + p_eta * eta_x;
      const real p_y = p_xi * xi_y + p_eta * eta_y;
      const real p_z = p_xi * xi_z + p_eta * eta_z;
      const real grad_eta_dot_grad_p_over_p =
          (eta_x_div_jac * p_x + eta_y_div_jac * p_y + eta_z_div_jac * p_z) /
          max(0.5 * (pv(i, j, k, 4) + pv(i, j + 1, k, 4)), static_cast<real>(1e-30));

      for (int l = 0; l < param->n_spec; ++l) {
        const real coefficient = (mw_tot * param->imw[l] - 1) * yk[l] * grad_eta_dot_grad_p_over_p;
        diffusion_driven_force[l] += coefficient;
        correction_velocity_term += coefficient * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j + 1, k, 5));
    compute_enthalpy(tm, h, param);

    real tve_m{tm};
    if constexpr (kTwoTemperature) {
      if (param->i_eve >= 0) {
        tve_m = 0.5 * (zone->temperature_ve(i, j, k) + zone->temperature_ve(i, j + 1, k));
        const real tve_x = tve_xi * xi_x + tve_eta * eta_x;
        const real tve_y = tve_xi * xi_y + tve_eta * eta_y;
        const real tve_z = tve_xi * xi_z + tve_eta * eta_z;
        const real conductivity_ve =
            0.5 * (zone->thermal_conductivity_ve(i, j, k) + zone->thermal_conductivity_ve(i, j + 1, k));
        q_ve_face = conductivity_ve * (eta_x_div_jac * tve_x + eta_y_div_jac * tve_y + eta_z_div_jac * tve_z);
      }
    }

    for (int l = 0; l < param->n_spec; ++l) {
      const real diffusion_flux = diffusivity[l] *
                                      (diffusion_driven_force[l] - mw_tot * yk[l] * sum_grad_eta_dot_grad_y_over_wl) -
                                  yk[l] * correction_velocity_term;
      q_total_face += h[l] * diffusion_flux;
      if constexpr (kTwoTemperature) {
        if (param->i_eve >= 0) {
          q_ve_face -= compute_ve_energy(l, tve_m, param) * diffusion_flux;
        }
      }
    }
  } else if constexpr (kTwoTemperature) {
    if (param->i_eve >= 0) {
      const real tve_x = tve_xi * xi_x + tve_eta * eta_x;
      const real tve_y = tve_xi * xi_y + tve_eta * eta_y;
      const real tve_z = tve_xi * xi_z + tve_eta * eta_z;
      const real conductivity_ve =
          0.5 * (zone->thermal_conductivity_ve(i, j, k) + zone->thermal_conductivity_ve(i, j + 1, k));
      q_ve_face = conductivity_ve * (eta_x_div_jac * tve_x + eta_y_div_jac * tve_y + eta_z_div_jac * tve_z);
    }
  }

  const real q_total = (q_total_face + q_ve_face) / face_area;
  const real q_ve = q_ve_face / face_area;
  const real q_tr = q_total - q_ve;

  const real rho_w = max(pv(i, j, k, 0), static_cast<real>(1e-30));
  const real u_tau = sqrt(max(tau_w / rho_w, static_cast<real>(0.0)));
  const real y_plus = rho_w * u_tau * dn / max(zone->mul(i, j, k), static_cast<real>(1e-30));

  const auto offset = i * kWallOutputStride;
  wall_data[offset] = tau_w;
  wall_data[offset + 1] = cf;
  wall_data[offset + 2] = q_tr;
  wall_data[offset + 3] = q_ve;
  wall_data[offset + 4] = q_total;
  wall_data[offset + 5] = y_plus;

}

void cfd::wall_friction_heatFlux_3d(const Mesh &mesh, const std::vector<Field> &field,
                                    const Parameter &parameter, DParameter *param) {
  const std::filesystem::path out_dir("output");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  const auto path_name = out_dir.string();
  const int myid = parameter.get_int("myid");
  const bool is_parallel_output = parameter.get_int("n_proc") > 1;

  bool stat_on{parameter.get_bool("if_collect_statistics")};
  bool spanwise_ave{parameter.get_bool("perform_spanwise_average")};
  for (int b = 0; b < mesh.n_block; ++b) {
    int mx{mesh[b].mx}, mz{mesh[b].mz};
    if (spanwise_ave) {
      mz = 1;
    }

    ggxl::VectorField2DHost<real> cfQw_host;
    printf("mx=%d,mz=%d\n", mx, mz);
    cfQw_host.allocate_memory(mx, mz, 2, 0);
    ggxl::VectorField2D<real> cfQw_device_hPtr;
    ggxl::VectorField2D<real> *cfQw_device = nullptr;
    cfQw_device_hPtr.allocate_memory(mx, mz, 2, 0);
    cudaMalloc(&cfQw_device, sizeof(ggxl::VectorField2D<real>));
    cudaMemcpy(cfQw_device, &cfQw_device_hPtr, sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);

    dim3 tpb(32, 1, 32);
    if (spanwise_ave) {
      tpb = dim3{128, 1, 1};
    }
    dim3 bpg((mx - 1) / tpb.x + 1, 1, (mz - 1) / tpb.z + 1);


    wall_friction_heatFlux_3d<<<bpg, tpb>>>(field[b].d_ptr, cfQw_device, param, stat_on, spanwise_ave);
    cudaMemcpy(cfQw_host.data(), cfQw_device_hPtr.data(), mx * mz * 2 * sizeof(real), cudaMemcpyDeviceToHost);
    if (!spanwise_ave) {
      std::string file_name = "/friction_heatFlux-block-" + std::to_string(b) + ".dat";
      if (is_parallel_output) {
        file_name = "/friction_heatFlux-proc-" + std::to_string(myid) + "-block-" + std::to_string(b) + ".dat";
      }
      std::ofstream f(path_name + file_name);
      f << "variables = \"x\", \"z\", \"cf\", \"y_plus\"\n";
      f << "zone,i=" << mx << ",j=" << mz << ",f=point\n";
      for (int kk = 0; kk < mz; ++kk) {
        for (int ii = 0; ii < mx; ++ii) {
          f << mesh[b].x(ii, 0, kk) << '\t' << mesh[b].z(ii, 0, kk) << '\t' << cfQw_host(ii, kk, 0) << '\t'
            << cfQw_host(ii, kk, 1) << '\n';
        }
      }
      f.close();
    } else {
      std::string file_name = "/spanaveraged_friction_heatFlux-block-" + std::to_string(b) + ".dat";
      if (is_parallel_output) {
        file_name =
            "/spanaveraged_friction_heatFlux-proc-" + std::to_string(myid) + "-block-" + std::to_string(b) + ".dat";
      }
      std::ofstream f(path_name + file_name);
      f << "variables = \"x\", \"cf\", \"y_plus\"\n";
      for (int ii = 0; ii < mx; ++ii) {
        f << mesh[b].x(ii, 0, 0) << '\t' << cfQw_host(ii, 0, 0) << '\t' << cfQw_host(ii, 0, 1) << '\n';
      }
      f.close();
    }


    cfQw_host.deallocate_memory();
    cfQw_device_hPtr.deallocate_memory();
  }
}

__global__ void
cfd::wall_friction_heatFlux_3d(DZone *zone, ggxl::VectorField2D<real> *cfQw, const DParameter *param, bool stat_on,
                               bool spanwise_ave) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= zone->mx || k >= zone->mz) return;

  constexpr int j = 1;
  auto &metric = zone->metric;
  const real d_wini = rnorm3d(metric(i, 0, k, 3), metric(i, 0, k, 4), metric(i, 0, k, 5));

  real u, v, w;
  real rho_w;
  const real dy = zone->y(i, j, k) - zone->y(i, j - 1, k);
  if (!stat_on) {
    auto &pv = zone->bv;
    u = pv(i, j, k, 1), v = pv(i, j, k, 2), w = pv(i, j, k, 3);
    rho_w = pv(i, 0, k, 0);
  } else {
//    auto &pv = zone->mean_value;
    auto &pv = zone->stat_favre_1st;
    u = pv(i, j, k, 0), v = pv(i, j, k, 1), w = pv(i, j, k, 2);
    rho_w = zone->stat_reynolds_1st(i, 0, k, 0);
  }
  const real rho_ref = param->rho_ref, v_ref = param->v_ref;
  gxl::Matrix<real, 3, 3, 1> bdjin;
  real d1 = metric(i, 0, k, 3);
  real d2 = metric(i, 0, k, 4);
  real d3 = metric(i, 0, k, 5);
  real kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(1, 1) = d1 / kk;
  bdjin(1, 2) = d2 / kk;
  bdjin(1, 3) = d3 / kk;

  d1 = bdjin(1, 2) - bdjin(1, 3);
  d2 = bdjin(1, 3) - bdjin(1, 1);
  d3 = bdjin(1, 1) - bdjin(1, 2);
  kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(2, 1) = d1 / kk;
  bdjin(2, 2) = d2 / kk;
  bdjin(2, 3) = d3 / kk;

  d1 = bdjin(1, 2) * bdjin(2, 3) - bdjin(1, 3) * bdjin(2, 2);
  d2 = bdjin(1, 3) * bdjin(2, 1) - bdjin(1, 1) * bdjin(2, 3);
  d3 = bdjin(1, 1) * bdjin(2, 2) - bdjin(1, 2) * bdjin(2, 1);
  kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(3, 1) = d1 / kk;
  bdjin(3, 2) = d2 / kk;
  bdjin(3, 3) = d3 / kk;

  const real vt = bdjin(2, 1) * u + bdjin(2, 2) * v + bdjin(2, 3) * w;
  const real vs = bdjin(3, 1) * u + bdjin(3, 2) * v + bdjin(3, 3) * w;
  const real velocity_tau = sqrt(vt * vt + vs * vs);

  const real tau = velocity_tau / d_wini * zone->mul(i, 0, k);
  const real cf = tau / (0.5 * (rho_ref * v_ref * v_ref));
  const real u_tau = sqrt(tau / rho_w);
  const real y_plus = rho_w * u_tau * dy / zone->mul(i, 0, k);

  (*cfQw)(i, k, 0) = cf;
  (*cfQw)(i, k, 1) = y_plus;
}

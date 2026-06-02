#include "PostProcess.h"
#include "Field.h"
#include "Constants.h"
#include "Thermo.cuh"
#include <filesystem>
#include <fstream>
#include <variant>

namespace {
constexpr int kWallOutputStride = 30;

__device__ __forceinline__ real safe_face_area(real x, real y, real z) {
  return max(norm3d(x, y, z), static_cast<real>(1e-30));
}

__device__ __forceinline__ real one_sided_derivative_nonuniform(real f0, real f1, real f2, real h1, real h2) {
  h1 = max(h1, static_cast<real>(1e-30));
  h2 = max(h2, h1 + static_cast<real>(1e-30));
  return -(h1 + h2) / (h1 * h2) * f0 + h2 / (h1 * (h2 - h1)) * f1 -
         h1 / (h2 * (h2 - h1)) * f2;
}

std::vector<int> wall_labels(const cfd::Parameter &parameter) {
  std::vector<int> labels;
  for (const auto &name: parameter.get_string_array("boundary_conditions")) {
    const auto &bc = parameter.get_struct(name);
    const auto type_iter = bc.find("type");
    const auto label_iter = bc.find("label");
    if (type_iter == bc.end() || label_iter == bc.end()) continue;
    if (std::get<std::string>(type_iter->second) == "wall") {
      labels.push_back(std::get<int>(label_iter->second));
    }
  }
  return labels;
}

bool contains_label(const std::vector<int> &labels, int label) {
  for (const auto item: labels) {
    if (item == label) return true;
  }
  return false;
}

__device__ __forceinline__ real derivative_i_bv(cfd::DZone *zone, int i, int j, int k, int v) {
  auto &f = zone->bv;
  const int mx = zone->mx;
  if (mx < 5) {
    if (i == 0) return f(i + 1, j, k, v) - f(i, j, k, v);
    if (i == mx - 1) return f(i, j, k, v) - f(i - 1, j, k, v);
    return 0.5 * (f(i + 1, j, k, v) - f(i - 1, j, k, v));
  }
  if (i == 0) {
    return -25.0 / 12 * f(i, j, k, v) + 4.0 * f(i + 1, j, k, v) - 3.0 * f(i + 2, j, k, v) +
           4.0 / 3 * f(i + 3, j, k, v) - 0.25 * f(i + 4, j, k, v);
  }
  if (i == 1) {
    return -0.25 * f(i - 1, j, k, v) - 5.0 / 6 * f(i, j, k, v) + 1.5 * f(i + 1, j, k, v) -
           0.5 * f(i + 2, j, k, v) + 1.0 / 12 * f(i + 3, j, k, v);
  }
  if (i == mx - 1) {
    return 25.0 / 12 * f(i, j, k, v) - 4.0 * f(i - 1, j, k, v) + 3.0 * f(i - 2, j, k, v) -
           4.0 / 3 * f(i - 3, j, k, v) + 0.25 * f(i - 4, j, k, v);
  }
  if (i == mx - 2) {
    return 0.25 * f(i + 1, j, k, v) + 5.0 / 6 * f(i, j, k, v) - 1.5 * f(i - 1, j, k, v) +
           0.5 * f(i - 2, j, k, v) - 1.0 / 12 * f(i - 3, j, k, v);
  }
  return -1.0 / 12 * (f(i + 2, j, k, v) - f(i - 2, j, k, v)) +
         2.0 / 3 * (f(i + 1, j, k, v) - f(i - 1, j, k, v));
}

__device__ __forceinline__ real derivative_j_lower_bv(cfd::DZone *zone, int i, int j, int k, int v) {
  auto &f = zone->bv;
  const int my = zone->my;
  if (my < 5) {
    if (j == 0) return f(i, j + 1, k, v) - f(i, j, k, v);
    if (j == my - 1) return f(i, j, k, v) - f(i, j - 1, k, v);
    return 0.5 * (f(i, j + 1, k, v) - f(i, j - 1, k, v));
  }
  if (j == 0) {
    return -25.0 / 12 * f(i, j, k, v) + 4.0 * f(i, j + 1, k, v) - 3.0 * f(i, j + 2, k, v) +
           4.0 / 3 * f(i, j + 3, k, v) - 0.25 * f(i, j + 4, k, v);
  }
  if (j == 1) {
    return -0.25 * f(i, j - 1, k, v) - 5.0 / 6 * f(i, j, k, v) + 1.5 * f(i, j + 1, k, v) -
           0.5 * f(i, j + 2, k, v) + 1.0 / 12 * f(i, j + 3, k, v);
  }
  if (j == my - 1) {
    return 25.0 / 12 * f(i, j, k, v) - 4.0 * f(i, j - 1, k, v) + 3.0 * f(i, j - 2, k, v) -
           4.0 / 3 * f(i, j - 3, k, v) + 0.25 * f(i, j - 4, k, v);
  }
  if (j == my - 2) {
    return 0.25 * f(i, j + 1, k, v) + 5.0 / 6 * f(i, j, k, v) - 1.5 * f(i, j - 1, k, v) +
           0.5 * f(i, j - 2, k, v) - 1.0 / 12 * f(i, j - 3, k, v);
  }
  return -1.0 / 12 * (f(i, j + 2, k, v) - f(i, j - 2, k, v)) +
         2.0 / 3 * (f(i, j + 1, k, v) - f(i, j - 1, k, v));
}

__device__ __forceinline__ real derivative_i_tve(cfd::DZone *zone, int i, int j, int k) {
  auto &f = zone->temperature_ve;
  const int mx = zone->mx;
  if (mx < 5) {
    if (i == 0) return f(i + 1, j, k) - f(i, j, k);
    if (i == mx - 1) return f(i, j, k) - f(i - 1, j, k);
    return 0.5 * (f(i + 1, j, k) - f(i - 1, j, k));
  }
  if (i == 0) {
    return -25.0 / 12 * f(i, j, k) + 4.0 * f(i + 1, j, k) - 3.0 * f(i + 2, j, k) +
           4.0 / 3 * f(i + 3, j, k) - 0.25 * f(i + 4, j, k);
  }
  if (i == 1) {
    return -0.25 * f(i - 1, j, k) - 5.0 / 6 * f(i, j, k) + 1.5 * f(i + 1, j, k) -
           0.5 * f(i + 2, j, k) + 1.0 / 12 * f(i + 3, j, k);
  }
  if (i == mx - 1) {
    return 25.0 / 12 * f(i, j, k) - 4.0 * f(i - 1, j, k) + 3.0 * f(i - 2, j, k) -
           4.0 / 3 * f(i - 3, j, k) + 0.25 * f(i - 4, j, k);
  }
  if (i == mx - 2) {
    return 0.25 * f(i + 1, j, k) + 5.0 / 6 * f(i, j, k) - 1.5 * f(i - 1, j, k) +
           0.5 * f(i - 2, j, k) - 1.0 / 12 * f(i - 3, j, k);
  }
  return -1.0 / 12 * (f(i + 2, j, k) - f(i - 2, j, k)) +
         2.0 / 3 * (f(i + 1, j, k) - f(i - 1, j, k));
}

__device__ __forceinline__ real derivative_j_lower_tve(cfd::DZone *zone, int i, int j, int k) {
  auto &f = zone->temperature_ve;
  const int my = zone->my;
  if (my < 5) {
    if (j == 0) return f(i, j + 1, k) - f(i, j, k);
    if (j == my - 1) return f(i, j, k) - f(i, j - 1, k);
    return 0.5 * (f(i, j + 1, k) - f(i, j - 1, k));
  }
  if (j == 0) {
    return -25.0 / 12 * f(i, j, k) + 4.0 * f(i, j + 1, k) - 3.0 * f(i, j + 2, k) +
           4.0 / 3 * f(i, j + 3, k) - 0.25 * f(i, j + 4, k);
  }
  if (j == 1) {
    return -0.25 * f(i, j - 1, k) - 5.0 / 6 * f(i, j, k) + 1.5 * f(i, j + 1, k) -
           0.5 * f(i, j + 2, k) + 1.0 / 12 * f(i, j + 3, k);
  }
  if (j == my - 1) {
    return 25.0 / 12 * f(i, j, k) - 4.0 * f(i, j - 1, k) + 3.0 * f(i, j - 2, k) -
           4.0 / 3 * f(i, j - 3, k) + 0.25 * f(i, j - 4, k);
  }
  if (j == my - 2) {
    return 0.25 * f(i, j + 1, k) + 5.0 / 6 * f(i, j, k) - 1.5 * f(i, j - 1, k) +
           0.5 * f(i, j - 2, k) - 1.0 / 12 * f(i, j - 3, k);
  }
  return -1.0 / 12 * (f(i, j + 2, k) - f(i, j - 2, k)) +
         2.0 / 3 * (f(i, j + 1, k) - f(i, j - 1, k));
}

__device__ __forceinline__ real derivative_i_sv(cfd::DZone *zone, int i, int j, int k, int l) {
  auto &f = zone->sv;
  const int mx = zone->mx;
  if (mx < 5) {
    if (i == 0) return f(i + 1, j, k, l) - f(i, j, k, l);
    if (i == mx - 1) return f(i, j, k, l) - f(i - 1, j, k, l);
    return 0.5 * (f(i + 1, j, k, l) - f(i - 1, j, k, l));
  }
  if (i == 0) {
    return -25.0 / 12 * f(i, j, k, l) + 4.0 * f(i + 1, j, k, l) - 3.0 * f(i + 2, j, k, l) +
           4.0 / 3 * f(i + 3, j, k, l) - 0.25 * f(i + 4, j, k, l);
  }
  if (i == 1) {
    return -0.25 * f(i - 1, j, k, l) - 5.0 / 6 * f(i, j, k, l) + 1.5 * f(i + 1, j, k, l) -
           0.5 * f(i + 2, j, k, l) + 1.0 / 12 * f(i + 3, j, k, l);
  }
  if (i == mx - 1) {
    return 25.0 / 12 * f(i, j, k, l) - 4.0 * f(i - 1, j, k, l) + 3.0 * f(i - 2, j, k, l) -
           4.0 / 3 * f(i - 3, j, k, l) + 0.25 * f(i - 4, j, k, l);
  }
  if (i == mx - 2) {
    return 0.25 * f(i + 1, j, k, l) + 5.0 / 6 * f(i, j, k, l) - 1.5 * f(i - 1, j, k, l) +
           0.5 * f(i - 2, j, k, l) - 1.0 / 12 * f(i - 3, j, k, l);
  }
  return -1.0 / 12 * (f(i + 2, j, k, l) - f(i - 2, j, k, l)) +
         2.0 / 3 * (f(i + 1, j, k, l) - f(i - 1, j, k, l));
}

__device__ __forceinline__ real derivative_j_lower_sv(cfd::DZone *zone, int i, int j, int k, int l) {
  auto &f = zone->sv;
  const int my = zone->my;
  if (my < 5) {
    if (j == 0) return f(i, j + 1, k, l) - f(i, j, k, l);
    if (j == my - 1) return f(i, j, k, l) - f(i, j - 1, k, l);
    return 0.5 * (f(i, j + 1, k, l) - f(i, j - 1, k, l));
  }
  if (j == 0) {
    return -25.0 / 12 * f(i, j, k, l) + 4.0 * f(i, j + 1, k, l) - 3.0 * f(i, j + 2, k, l) +
           4.0 / 3 * f(i, j + 3, k, l) - 0.25 * f(i, j + 4, k, l);
  }
  if (j == 1) {
    return -0.25 * f(i, j - 1, k, l) - 5.0 / 6 * f(i, j, k, l) + 1.5 * f(i, j + 1, k, l) -
           0.5 * f(i, j + 2, k, l) + 1.0 / 12 * f(i, j + 3, k, l);
  }
  if (j == my - 1) {
    return 25.0 / 12 * f(i, j, k, l) - 4.0 * f(i, j - 1, k, l) + 3.0 * f(i, j - 2, k, l) -
           4.0 / 3 * f(i, j - 3, k, l) + 0.25 * f(i, j - 4, k, l);
  }
  if (j == my - 2) {
    return 0.25 * f(i, j + 1, k, l) + 5.0 / 6 * f(i, j, k, l) - 1.5 * f(i, j - 1, k, l) +
           0.5 * f(i, j - 2, k, l) - 1.0 / 12 * f(i, j - 3, k, l);
  }
  return -1.0 / 12 * (f(i, j + 2, k, l) - f(i, j - 2, k, l)) +
         2.0 / 3 * (f(i, j + 1, k, l) - f(i, j - 1, k, l));
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

  int size{std::max(mesh[0].mx, mesh[0].my)};
  for (int blk = 1; blk < mesh.n_block; ++blk) {
    size = std::max(size, std::max(mesh[blk].mx, mesh[blk].my));
  }
  std::vector<double> wall_data(size * kWallOutputStride, 0.0);
  real *wall_data_device = nullptr;
  cudaMalloc(&wall_data_device, size * kWallOutputStride * sizeof(real));

  const double rho_inf = parameter.get_real("rho_inf");
  const double v_inf = parameter.get_real("v_inf");
  const double dyn_pressure = 0.5 * rho_inf * v_inf * v_inf;
  const auto wall_bc_labels = wall_labels(parameter);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    auto &block = mesh[blk];
    int wall_face_count = 0;
    for (int boundary_id = 0; boundary_id < static_cast<int>(block.boundary.size()); ++boundary_id) {
      const auto &boundary = block.boundary[boundary_id];
      if (!contains_label(wall_bc_labels, boundary.type_label)) continue;
      if (block.mz == 1) {
        if (boundary.range_start[2] > 0 || boundary.range_end[2] < 0) continue;
      } else if (boundary.range_start[2] != boundary.range_end[2]) {
        continue;
      }

      const int tangential_axis = boundary.face == 0 ? 1 : 0;
      const int axis_max = tangential_axis == 0 ? block.mx - 1 : block.my - 1;
      const int start_index = std::max(boundary.range_start[tangential_axis], 0);
      const int end_index = std::min(boundary.range_end[tangential_axis], axis_max);
      const int n_point = end_index - start_index + 1;
      if (n_point <= 0) continue;

      cudaMemset(wall_data_device, 0, size * kWallOutputStride * sizeof(real));
      dim3 bpg((n_point - 1) / 128 + 1, 1, 1);
      wall_friction_heatFlux_2d<<<bpg, 128>>>(field[blk].d_ptr, wall_data_device, param, dyn_pressure, boundary_id,
                                              start_index, n_point);
      cudaMemcpy(wall_data.data(), wall_data_device, size * kWallOutputStride * sizeof(real), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      std::string file_name = "/friction_heatflux-block-" + std::to_string(blk) + "-wall-" +
                              std::to_string(wall_face_count) + ".dat";
      if (is_parallel_output) {
        file_name = "/friction_heatflux-proc-" + std::to_string(myid) + "-block-" + std::to_string(blk) +
                    "-wall-" + std::to_string(wall_face_count) + ".dat";
      }
      ++wall_face_count;

      std::ofstream f(path_name + file_name);
      f << "variables = \"x\", \"y\", \"tau_w\", \"cf\", \"q_tr\", \"q_ve\", \"q_total\", \"y_plus\", "
           "\"q_tr_eta_face\", \"q_ve_eta_face\", \"q_eta_minus_wall\", \"q_tr_normal_1st\", "
           "\"q_ve_normal_1st\", \"q_normal_1st\", \"T_wall\", \"T_1\", \"Tve_wall\", \"Tve_1\", \"dn\", "
           "\"q_species_wall\", \"q_species_eta_face\", \"q_conduction_wall\", \"q_total_eta_face\", "
           "\"q_total_with_species\", \"T_2\", \"Tve_2\", \"dn_2\", \"q_tr_normal_2nd\", "
           "\"q_ve_normal_2nd\", \"q_normal_2nd\", \"q_total_2nd_with_species\", \"q_2nd_minus_1st\"\n";
      for (int local_i = 0; local_i < n_point; ++local_i) {
        const int idx = start_index + local_i;
        const int i = boundary.face == 0 ? boundary.range_start[0] : idx;
        const int j = boundary.face == 1 ? boundary.range_start[1] : idx;
        const auto offset = local_i * kWallOutputStride;
        f << block.x(i, j, 0) << '\t'
          << block.y(i, j, 0) << '\t'
          << wall_data[offset] << '\t'
          << wall_data[offset + 1] << '\t'
          << wall_data[offset + 2] << '\t'
          << wall_data[offset + 3] << '\t'
          << wall_data[offset + 4] << '\t'
          << wall_data[offset + 5] << '\t'
          << wall_data[offset + 6] << '\t'
          << wall_data[offset + 7] << '\t'
          << wall_data[offset + 8] << '\t'
          << wall_data[offset + 9] << '\t'
          << wall_data[offset + 10] << '\t'
          << wall_data[offset + 11] << '\t'
          << wall_data[offset + 12] << '\t'
          << wall_data[offset + 13] << '\t'
          << wall_data[offset + 14] << '\t'
          << wall_data[offset + 15] << '\t'
          << wall_data[offset + 16] << '\t'
          << wall_data[offset + 17] << '\t'
          << wall_data[offset + 18] << '\t'
          << wall_data[offset + 19] << '\t'
          << wall_data[offset + 20] << '\t'
          << wall_data[offset + 21] << '\t'
          << wall_data[offset + 22] << '\t'
          << wall_data[offset + 23] << '\t'
          << wall_data[offset + 24] << '\t'
          << wall_data[offset + 25] << '\t'
          << wall_data[offset + 26] << '\t'
          << wall_data[offset + 27] << '\t'
          << wall_data[offset + 28] << '\t'
          << wall_data[offset + 29] << '\n';
      }
      f.close();
    }
  }
  cudaFree(wall_data_device);
}

__global__ void cfd::wall_friction_heatFlux_2d(DZone *zone, real *wall_data, const DParameter *param,
                                               real dyn_pressure, int boundary_id, int start_index, int n_point) {
  if (boundary_id < 0 || boundary_id >= zone->n_boundary) return;
  const auto &boundary = zone->boundary[boundary_id];
  if (boundary.face < 0 || boundary.face > 1) return;

  const int local_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (local_i >= n_point) return;

  int idx[3]{boundary.range_start[0], boundary.range_start[1], boundary.range_start[2]};
  const int tangential_axis = boundary.face == 0 ? 1 : 0;
  idx[tangential_axis] = start_index + local_i;
  const int i = idx[0];
  const int j = idx[1];
  constexpr int k = 0;
  const int ni = i - (boundary.face == 0 ? boundary.direction : 0);
  const int nj = j - (boundary.face == 1 ? boundary.direction : 0);
  if (ni < 0 || ni >= zone->mx || nj < 0 || nj >= zone->my) return;
  const int n2i = ni - (boundary.face == 0 ? boundary.direction : 0);
  const int n2j = nj - (boundary.face == 1 ? boundary.direction : 0);
  const bool has_second_normal_point = n2i >= 0 && n2i < zone->mx && n2j >= 0 && n2j < zone->my;
  auto &pv = zone->bv;
  real tx_raw{}, ty_raw{}, tz_raw{};
  if (tangential_axis == 0) {
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
  } else {
    if (j == 0) {
      tx_raw = zone->x(i, j + 1, k) - zone->x(i, j, k);
      ty_raw = zone->y(i, j + 1, k) - zone->y(i, j, k);
      tz_raw = zone->z(i, j + 1, k) - zone->z(i, j, k);
    } else if (j == zone->my - 1) {
      tx_raw = zone->x(i, j, k) - zone->x(i, j - 1, k);
      ty_raw = zone->y(i, j, k) - zone->y(i, j - 1, k);
      tz_raw = zone->z(i, j, k) - zone->z(i, j - 1, k);
    } else {
      tx_raw = zone->x(i, j + 1, k) - zone->x(i, j - 1, k);
      ty_raw = zone->y(i, j + 1, k) - zone->y(i, j - 1, k);
      tz_raw = zone->z(i, j + 1, k) - zone->z(i, j - 1, k);
    }
  }
  const real xi_mag = safe_face_area(tx_raw, ty_raw, tz_raw);
  const real tx = tx_raw / xi_mag, ty = ty_raw / xi_mag, tz = tz_raw / xi_mag;
  const auto &metric = zone->metric;
  const real xi_x = 0.5 * (metric(i, j, k, 0) + metric(ni, nj, k, 0));
  const real xi_y = 0.5 * (metric(i, j, k, 1) + metric(ni, nj, k, 1));
  const real xi_z = 0.5 * (metric(i, j, k, 2) + metric(ni, nj, k, 2));
  const real eta_x = 0.5 * (metric(i, j, k, 3) + metric(ni, nj, k, 3));
  const real eta_y = 0.5 * (metric(i, j, k, 4) + metric(ni, nj, k, 4));
  const real eta_z = 0.5 * (metric(i, j, k, 5) + metric(ni, nj, k, 5));

  const real dx = zone->x(ni, nj, k) - zone->x(i, j, k);
  const real dy = zone->y(ni, nj, k) - zone->y(i, j, k);
  const real dz = zone->z(ni, nj, k) - zone->z(i, j, k);
  const real dn = safe_face_area(dx, dy, dz);
  real dn2 = dn;
  if (has_second_normal_point) {
    dn2 = safe_face_area(zone->x(n2i, n2j, k) - zone->x(i, j, k),
                         zone->y(n2i, n2j, k) - zone->y(i, j, k),
                         zone->z(n2i, n2j, k) - zone->z(i, j, k));
  }

  // Reuse the eta-face discretization used by the 2nd-order viscous flux.
  const real u_xi = derivative_i_bv(zone, i, j, k, 1);
  const real u_eta = derivative_j_lower_bv(zone, i, j, k, 1);
  const real v_xi = derivative_i_bv(zone, i, j, k, 2);
  const real v_eta = derivative_j_lower_bv(zone, i, j, k, 2);
  const real w_xi = derivative_i_bv(zone, i, j, k, 3);
  const real w_eta = derivative_j_lower_bv(zone, i, j, k, 3);
  const real t_xi = derivative_i_bv(zone, i, j, k, 5);
  const real t_eta = derivative_j_lower_bv(zone, i, j, k, 5);
  const real t_xi_wall = t_xi;
  const real t_eta_wall = t_eta;

  real tve_xi{0.0}, tve_eta{0.0}, tve_xi_wall{0.0}, tve_eta_wall{0.0};
  if constexpr (kTwoTemperature) {
    if (param->i_eve >= 0) {
      tve_xi = derivative_i_tve(zone, i, j, k);
      tve_eta = derivative_j_lower_tve(zone, i, j, k);
      tve_xi_wall = derivative_i_tve(zone, i, j, k);
      tve_eta_wall = derivative_j_lower_tve(zone, i, j, k);
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

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(ni, nj, k));
  const real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  const real eta_x_div_jac =
      0.5 * (metric(i, j, k, 3) * zone->jac(i, j, k) + metric(ni, nj, k, 3) * zone->jac(ni, nj, k));
  const real eta_y_div_jac =
      0.5 * (metric(i, j, k, 4) * zone->jac(i, j, k) + metric(ni, nj, k, 4) * zone->jac(ni, nj, k));
  const real eta_z_div_jac =
      0.5 * (metric(i, j, k, 5) * zone->jac(i, j, k) + metric(ni, nj, k, 5) * zone->jac(ni, nj, k));
  const real face_area = safe_face_area(eta_x_div_jac, eta_y_div_jac, eta_z_div_jac);
  const real nx_wall = dx / dn;
  const real ny_wall = dy / dn;
  const real nz_wall = dz / dn;

  const real traction_x = tau_xx * nx_wall + tau_xy * ny_wall + tau_xz * nz_wall;
  const real traction_y = tau_xy * nx_wall + tau_yy * ny_wall + tau_yz * nz_wall;
  const real traction_z = tau_xz * nx_wall + tau_yz * ny_wall + tau_zz * nz_wall;
  const real tau_w = abs(traction_x * tx + traction_y * ty + traction_z * tz);
  const real cf = tau_w / max(dyn_pressure, static_cast<real>(1e-30));

  const real t_x = t_xi * xi_x + t_eta * eta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z;
  const real t_x_wall = t_xi_wall * metric(i, j, k, 0) + t_eta_wall * metric(i, j, k, 3);
  const real t_y_wall = t_xi_wall * metric(i, j, k, 1) + t_eta_wall * metric(i, j, k, 4);
  const real t_z_wall = t_xi_wall * metric(i, j, k, 2) + t_eta_wall * metric(i, j, k, 5);
  real conductivity{};
  if (param->n_spec > 0) {
    conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(ni, nj, k));
  } else {
    constexpr real cp_air{gamma_air * R_u / mw_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp_air;
  }

  const real q_tr_conduction_face = conductivity * (eta_x_div_jac * t_x + eta_y_div_jac * t_y + eta_z_div_jac * t_z);
  real q_tr_face = q_tr_conduction_face;
  real q_ve_face = 0.0;
  real q_ve_conduction_face = 0.0;
  real q_tr_normal_first = 0.0;
  real q_ve_normal_first = 0.0;
  real q_species_wall = 0.0;
  real q_species_face = 0.0;
  const real t_wall = pv(i, j, k, 5);
  const real t_1 = pv(ni, nj, k, 5);
  const real t_2 = has_second_normal_point ? pv(n2i, n2j, k, 5) : t_1;
  real tve_wall = t_wall;
  real tve_1 = t_1;
  real tve_2 = t_2;
  real conductivity_wall{};
  real q_tr_normal_second = 0.0;
  if (param->n_spec > 0) {
    conductivity_wall = zone->thermal_conductivity(i, j, k);
    q_tr_normal_first = conductivity_wall * (t_1 - t_wall) / dn;
    q_tr_normal_second =
        conductivity_wall * one_sided_derivative_nonuniform(t_wall, t_1, t_2, dn, dn2);
  } else {
    constexpr real cp_air{gamma_air * R_u / mw_air / (gamma_air - 1)};
    conductivity_wall = zone->mul(i, j, k) / param->Pr * cp_air;
    q_tr_normal_first = conductivity_wall * (t_1 - t_wall) / dn;
    q_tr_normal_second =
        conductivity_wall * one_sided_derivative_nonuniform(t_wall, t_1, t_2, dn, dn2);
  }
  real q_tr_wall = conductivity_wall * (t_x_wall * nx_wall + t_y_wall * ny_wall + t_z_wall * nz_wall);

  if (param->n_spec > 0) {
    const auto &y = zone->sv;
    real diffusivity[MAX_SPEC_NUMBER];
    real yk[MAX_SPEC_NUMBER];
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    real sum_grad_eta_dot_grad_y_over_wl{0.0};
    real sum_rhoDkYk{0.0};
    real correction_velocity_term{0.0};
    real mw_tot{0.0};

    real diffusivity_wall[MAX_SPEC_NUMBER];
    real y_wall[MAX_SPEC_NUMBER];
    real diffusion_driven_force_wall[MAX_SPEC_NUMBER];
    real sum_grad_normal_dot_grad_y_over_wl{0.0};
    real sum_rhoDkYk_wall{0.0};
    real correction_velocity_term_wall{0.0};
    real mw_wall{0.0};

    for (int l = 0; l < param->n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(ni, nj, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(ni, nj, k, l));

      const real y_xi =
          0.5 * (derivative_i_sv(zone, i, j, k, l) + derivative_i_sv(zone, ni, nj, k, l));
      const real y_eta = 0.5 * (derivative_j_lower_sv(zone, i, j, k, l) + derivative_j_lower_sv(zone, ni, nj, k, l));

      const real y_x = y_xi * xi_x + y_eta * eta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z;
      const real grad_eta_dot_grad_y = eta_x_div_jac * y_x + eta_y_div_jac * y_y + eta_z_div_jac * y_z;
      diffusion_driven_force[l] = grad_eta_dot_grad_y;
      correction_velocity_term += diffusivity[l] * grad_eta_dot_grad_y;

      sum_grad_eta_dot_grad_y_over_wl += grad_eta_dot_grad_y * param->imw[l];
      mw_tot += yk[l] * param->imw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];

      y_wall[l] = y(i, j, k, l);
      diffusivity_wall[l] = zone->rho_D(i, j, k, l);
      const real grad_normal_dot_grad_y = (y(ni, nj, k, l) - y(i, j, k, l)) / dn;
      diffusion_driven_force_wall[l] = grad_normal_dot_grad_y;
      correction_velocity_term_wall += diffusivity_wall[l] * grad_normal_dot_grad_y;
      sum_grad_normal_dot_grad_y_over_wl += grad_normal_dot_grad_y * param->imw[l];
      mw_wall += y_wall[l] * param->imw[l];
      sum_rhoDkYk_wall += diffusivity_wall[l] * y_wall[l];
    }

    mw_tot = 1.0 / max(mw_tot, static_cast<real>(1e-30));
    correction_velocity_term -= mw_tot * sum_rhoDkYk * sum_grad_eta_dot_grad_y_over_wl;
    mw_wall = 1.0 / max(mw_wall, static_cast<real>(1e-30));
    correction_velocity_term_wall -= mw_wall * sum_rhoDkYk_wall * sum_grad_normal_dot_grad_y_over_wl;

    if (param->gradPInDiffusionFlux) {
      const real p_xi = 0.5 * (derivative_i_bv(zone, i, j, k, 4) + derivative_i_bv(zone, ni, nj, k, 4));
      const real p_eta = 0.5 * (derivative_j_lower_bv(zone, i, j, k, 4) + derivative_j_lower_bv(zone, ni, nj, k, 4));

      const real p_x = p_xi * xi_x + p_eta * eta_x;
      const real p_y = p_xi * xi_y + p_eta * eta_y;
      const real p_z = p_xi * xi_z + p_eta * eta_z;
      const real grad_eta_dot_grad_p_over_p =
          (eta_x_div_jac * p_x + eta_y_div_jac * p_y + eta_z_div_jac * p_z) /
          max(0.5 * (pv(i, j, k, 4) + pv(ni, nj, k, 4)), static_cast<real>(1e-30));

      for (int l = 0; l < param->n_spec; ++l) {
        const real coefficient = (mw_tot * param->imw[l] - 1) * yk[l] * grad_eta_dot_grad_p_over_p;
        diffusion_driven_force[l] += coefficient;
        correction_velocity_term += coefficient * diffusivity[l];
      }

      const real grad_normal_dot_grad_p_over_p =
          (pv(ni, nj, k, 4) - pv(i, j, k, 4)) /
          (dn * max(pv(i, j, k, 4), static_cast<real>(1e-30)));

      for (int l = 0; l < param->n_spec; ++l) {
        const real coefficient = (mw_wall * param->imw[l] - 1) * y_wall[l] * grad_normal_dot_grad_p_over_p;
        diffusion_driven_force_wall[l] += coefficient;
        correction_velocity_term_wall += coefficient * diffusivity_wall[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(ni, nj, k, 5));
    compute_enthalpy(tm, h, param);
    real h_wall[MAX_SPEC_NUMBER];
    compute_enthalpy(t_wall, h_wall, param);

    real tve_m{tm};
    if constexpr (kTwoTemperature) {
      if (param->i_eve >= 0) {
        tve_m = 0.5 * (zone->temperature_ve(i, j, k) + zone->temperature_ve(ni, nj, k));
        const real tve_x = tve_xi * xi_x + tve_eta * eta_x;
        const real tve_y = tve_xi * xi_y + tve_eta * eta_y;
        const real tve_z = tve_xi * xi_z + tve_eta * eta_z;
        const real conductivity_ve =
            0.5 * (zone->thermal_conductivity_ve(i, j, k) + zone->thermal_conductivity_ve(ni, nj, k));
        q_ve_conduction_face = conductivity_ve * (eta_x_div_jac * tve_x + eta_y_div_jac * tve_y + eta_z_div_jac * tve_z);
        q_ve_face = q_ve_conduction_face;
        tve_wall = zone->temperature_ve(i, j, k);
        tve_1 = zone->temperature_ve(ni, nj, k);
        tve_2 = has_second_normal_point ? zone->temperature_ve(n2i, n2j, k) : tve_1;
        q_ve_normal_first = zone->thermal_conductivity_ve(i, j, k) * (tve_1 - tve_wall) / dn;
      }
    }

    for (int l = 0; l < param->n_spec; ++l) {
      const real diffusion_flux = diffusivity[l] *
                                      (diffusion_driven_force[l] - mw_tot * yk[l] * sum_grad_eta_dot_grad_y_over_wl) -
                                  yk[l] * correction_velocity_term;
      real h_tr = h[l];
      real h_diff = h[l];
      if constexpr (kTwoTemperature) {
        if (param->i_eve >= 0) {
          const real eve_eq = compute_ve_energy(l, tm, param);
          const real eve = compute_ve_energy(l, tve_m, param);
          h_tr = h[l] - eve_eq;
          h_diff = h_tr + eve;
          q_ve_face += eve * diffusion_flux;
        }
      }
      q_tr_face += h_tr * diffusion_flux;

      const real diffusion_flux_wall = diffusivity_wall[l] *
                                           (diffusion_driven_force_wall[l] -
                                            mw_wall * y_wall[l] * sum_grad_normal_dot_grad_y_over_wl) -
                                       y_wall[l] * correction_velocity_term_wall;
      real h_wall_diff = h_wall[l];
      if constexpr (kTwoTemperature) {
        if (param->i_eve >= 0) {
          h_wall_diff = compute_nonequilibrium_diffusion_enthalpy(h_wall[l], l, t_wall, tve_wall, param);
        }
      }
      q_species_wall += h_wall_diff * diffusion_flux_wall;
      q_species_face += h_diff * diffusion_flux;
    }
  } else if constexpr (kTwoTemperature) {
    if (param->i_eve >= 0) {
      const real tve_x = tve_xi * xi_x + tve_eta * eta_x;
      const real tve_y = tve_xi * xi_y + tve_eta * eta_y;
      const real tve_z = tve_xi * xi_z + tve_eta * eta_z;
      const real conductivity_ve =
          0.5 * (zone->thermal_conductivity_ve(i, j, k) + zone->thermal_conductivity_ve(ni, nj, k));
      q_ve_conduction_face = conductivity_ve * (eta_x_div_jac * tve_x + eta_y_div_jac * tve_y + eta_z_div_jac * tve_z);
      q_ve_face = q_ve_conduction_face;
      tve_wall = zone->temperature_ve(i, j, k);
      tve_1 = zone->temperature_ve(ni, nj, k);
      tve_2 = has_second_normal_point ? zone->temperature_ve(n2i, n2j, k) : tve_1;
      q_ve_normal_first = zone->thermal_conductivity_ve(i, j, k) * (tve_1 - tve_wall) / dn;
    }
  }

  real q_ve_wall = 0.0;
  real q_ve_normal_second = 0.0;
  if constexpr (kTwoTemperature) {
    if (param->i_eve >= 0) {
      const real tve_x_wall = tve_xi_wall * metric(i, j, k, 0) + tve_eta_wall * metric(i, j, k, 3);
      const real tve_y_wall = tve_xi_wall * metric(i, j, k, 1) + tve_eta_wall * metric(i, j, k, 4);
      const real tve_z_wall = tve_xi_wall * metric(i, j, k, 2) + tve_eta_wall * metric(i, j, k, 5);
      q_ve_wall = zone->thermal_conductivity_ve(i, j, k) *
                  (tve_x_wall * nx_wall + tve_y_wall * ny_wall + tve_z_wall * nz_wall);
      q_ve_normal_second = zone->thermal_conductivity_ve(i, j, k) *
                           one_sided_derivative_nonuniform(tve_wall, tve_1, tve_2, dn, dn2);
    }
  }

  const real q_tr_eta = q_tr_face / face_area;
  const real q_ve_eta = q_ve_face / face_area;
  const real q_eta_total = q_tr_eta + q_ve_eta;
  const real q_conduction_wall = q_tr_wall + q_ve_wall;
  const real q_normal_second = q_tr_normal_second + q_ve_normal_second;
  const real q_total_first = q_tr_normal_first + q_ve_normal_first + q_species_wall;
  const real q_total_second = q_normal_second + q_species_wall;
  const real q_tr = q_tr_normal_second;
  const real q_ve = q_ve_normal_second;
  const real q_total = q_total_second;
  const real q_eta_minus_wall = q_eta_total - q_total;

  const real rho_w = max(pv(i, j, k, 0), static_cast<real>(1e-30));
  const real u_tau = sqrt(max(tau_w / rho_w, static_cast<real>(0.0)));
  const real y_plus = rho_w * u_tau * dn / max(zone->mul(i, j, k), static_cast<real>(1e-30));

  const auto offset = local_i * kWallOutputStride;
  wall_data[offset] = tau_w;
  wall_data[offset + 1] = cf;
  wall_data[offset + 2] = q_tr;
  wall_data[offset + 3] = q_ve;
  wall_data[offset + 4] = q_total;
  wall_data[offset + 5] = y_plus;
  wall_data[offset + 6] = q_tr_eta;
  wall_data[offset + 7] = q_ve_eta;
  wall_data[offset + 8] = q_eta_minus_wall;
  wall_data[offset + 9] = q_tr_normal_first;
  wall_data[offset + 10] = q_ve_normal_first;
  wall_data[offset + 11] = q_tr_normal_first + q_ve_normal_first;
  wall_data[offset + 12] = t_wall;
  wall_data[offset + 13] = t_1;
  wall_data[offset + 14] = tve_wall;
  wall_data[offset + 15] = tve_1;
  wall_data[offset + 16] = dn;
  wall_data[offset + 17] = q_species_wall;
  wall_data[offset + 18] = q_species_face / face_area;
  wall_data[offset + 19] = q_conduction_wall;
  wall_data[offset + 20] = q_eta_total;
  wall_data[offset + 21] = q_total;
  wall_data[offset + 22] = t_2;
  wall_data[offset + 23] = tve_2;
  wall_data[offset + 24] = dn2;
  wall_data[offset + 25] = q_tr_normal_second;
  wall_data[offset + 26] = q_ve_normal_second;
  wall_data[offset + 27] = q_normal_second;
  wall_data[offset + 28] = q_total_second;
  wall_data[offset + 29] = q_total_second - q_total_first;

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

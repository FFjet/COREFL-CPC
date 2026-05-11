#include "DigitalFilter.cuh"
#include "BoundCond.cuh"
#include <ctime>
#include <fstream>
#include <sstream>
#include <array>
#include <iomanip>
#include "mpi.h"
#include "Parallel.h"
#include "DParameter.cuh"

namespace cfd {
namespace {
// Spanwise periodic grid convention used here:
//   k=0 and k=mz-1 are the same physical point (duplicated endpoint).
// Therefore the number of unique periodic points is mz-1 and all periodic
// wrapping must use mz-1 rather than mz.
__host__ __device__ __forceinline__ int df_unique_span_count(int mz) {
  return mz > 1 ? mz - 1 : 1;
}

__device__ __forceinline__ int df_wrap_index(int k, int mz) {
  const int n_per = df_unique_span_count(mz);
  int r = k % n_per;
  if (r < 0) r += n_per;
  return r;
}

__device__ __forceinline__ real df_periodic_distance(real a, real b, real period) {
  real d = fabs(a - b);
  if (period > 0) {
    d = d - floor(d / period) * period;
    if (d > 0.5 * period) d = period - d;
  }
  return d;
}

__host__ __device__ __forceinline__ real df_max_real(real a, real b) { return a > b ? a : b; }
__host__ __device__ __forceinline__ real df_min_real(real a, real b) { return a < b ? a : b; }

constexpr int DF_RESTART_MAGIC = 20260501;
constexpr int DF_RESTART_VERSION = 2;
}

void DBoundCond::initialize_digital_filter(Parameter &parameter, Mesh &mesh) {
  parameter.update_parameter("use_df", false);

  int n_df = 0;
  std::vector<int> df_bc;
  std::vector<std::string> df_related_boundary{};
  auto &bcs = parameter.get_string_array("boundary_conditions");
  if (n_inflow > 0) {
    int i_inflow = 0;
    for (auto &bc_name: bcs) {
      auto &bc = parameter.get_struct(bc_name);
      auto &type = std::get<std::string>(bc.at("type"));
      if (type == "inflow") {
        int fluc = 0;
        if (const auto fluc_iter = bc.find("fluctuation_type"); fluc_iter != bc.end()) {
          fluc = std::get<int>(fluc_iter->second);
        }
        if (fluc == 11) {
          df_bc.push_back(std::get<int>(bc.at("label")));
          df_related_boundary.push_back(bc_name);
          df_label[i_inflow] = n_df;
          ++n_df;
        }
        ++i_inflow;
      }
    }
  }
  if (n_df == 0) return;

  df_mode = parameter.get_int("df_mode");
  df_velocity_scale = parameter.get_real("df_velocity_scale");
  if (df_velocity_scale <= 0) df_velocity_scale = parameter.get_real("v_inf");

  if (df_mode == 2) {
    // The turbulent-boundary-layer length scales are copied from URANOS, which may be modified later.
    // All length scales below are dimensional in the same units as the mesh coordinates.
    // y is normalized by df_y_ref, so df_y_ref should normally be the inlet boundary-layer thickness.
    const real delta_in = parameter.get_real("df_y_ref");
    const real re_tau = parameter.get_real("df_Re_tau");
    if (re_tau <= 0) {
      printf("Error: df_Re_tau must be positive when df_mode == 2.\n");
      MpiParallel::exit();
    }

    // Inner/outer transition in y/delta.
    df_y_zone_threshold = parameter.get_real("df_y_zone_threshold");
    df_y_zone_width = parameter.get_real("df_y_zone_width");

    // Streamwise integral length scales, in outer units.
    df_Lx[0] = parameter.get_real("df_LxU") * delta_in;
    df_Lx[1] = parameter.get_real("df_LxV") * delta_in;
    df_Lx[2] = parameter.get_real("df_LxW") * delta_in;

    // Spanwise length scales: inner values in wall units converted to delta units by 1/Re_tau;
    // outer values in delta units.  Values are clipped so that the inner scale does not exceed
    // the corresponding outer scale at low Re_tau.
    const real Lz_inner_delta[3] = {
      df_min_real(150.0 / re_tau, 0.40),
      df_min_real(75.0 / re_tau, 0.30),
      df_min_real(150.0 / re_tau, 0.40)
    };
    const real Lz_outer_delta[3] = {0.40, 0.30, 0.40};

    for (int c = 0; c < 3; ++c) {
      df_Lz_inner[c] = Lz_inner_delta[c] * delta_in;
      df_Lz_outer[c] = Lz_outer_delta[c] * delta_in;
      // Wall-normal length scales are proportional to the spanwise scales.
      df_Ly_inner[c] = 0.70 * df_Lz_inner[c];
      df_Ly_outer[c] = 0.70 * df_Lz_outer[c];
    }

    df_sra_coefficient = parameter.get_real("df_sra_coefficient");
    df_diagnostic_interval = parameter.get_int("df_diagnostic_interval");
  }

  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    printf("\t\tInitialize digital filter for turbulence inflow generation. df_mode = %d.\n", df_mode);
    printf(
      "\t\tThe z direction is assumed to be periodic, and the inflow is assumed to be in the yz(const x) plane!\n");
  }

  std::vector<int> N1, N2, iBlock;
  std::vector<std::vector<real>> scaled_y, y_ext, z_coord;
  std::vector<real> z_period;
  real y_ref{0}, dz{0};
  const int ngg{parameter.get_int("ngg")};
  if (df_mode == 1) y_ref = parameter.get_real("delta_omega");
  else y_ref = parameter.get_real("df_y_ref");

  for (int type = 0; type < n_df; ++type) {
    int label = df_bc[type];
    const auto &boundary_name = df_related_boundary[type];
    for (auto blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (auto &b: bs) {
        if (label == b.type_label) {
          ++n_df_face;
          df_related_block.push_back(blk);
          iBlock.push_back(blk);

          int n1{mesh[blk].my}, n2{mesh[blk].mz};
          if (b.face == 1) {
            printf("\tBoundary %s is on the xz plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
          } else if (b.face == 2) {
            printf("\tBoundary %s is on the xy plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
          }

          int i = b.direction == 1 ? mesh[blk].mx - 1 : 0;
          std::vector<real> scaled_y_temp(mesh[blk].my + 2 * ngg);
          for (int j = -ngg; j < mesh[blk].my + ngg; ++j) {
            scaled_y_temp[j + ngg] = mesh[blk].y(i, j, 0) / y_ref;
          }
          scaled_y.push_back(scaled_y_temp);

          std::vector<real> y_ext_temp(mesh[blk].my + 2 * ngg + 2 * DF_N);
          const int y_offset = ngg + DF_N;
          const real dy_low = mesh[blk].y(i, -ngg + 1, 0) - mesh[blk].y(i, -ngg, 0);
          const real dy_high = mesh[blk].y(i, mesh[blk].my + ngg - 1, 0) - mesh[blk].y(i, mesh[blk].my + ngg - 2, 0);
          for (int j = -ngg - DF_N; j < mesh[blk].my + ngg + DF_N; ++j) {
            if (j < -ngg) {
              y_ext_temp[j + y_offset] = mesh[blk].y(i, -ngg, 0) + (j + ngg) * dy_low;
            } else if (j >= mesh[blk].my + ngg) {
              y_ext_temp[j + y_offset] = mesh[blk].y(i, mesh[blk].my + ngg - 1, 0) + (j - (mesh[blk].my + ngg - 1)) *
                                         dy_high;
            } else {
              y_ext_temp[j + y_offset] = mesh[blk].y(i, j, 0);
            }
          }
          y_ext.push_back(y_ext_temp);

          std::vector<real> z_temp(n2);
          for (int k = 0; k < n2; ++k) z_temp[k] = mesh[blk].z(i, 0, k);
          z_coord.push_back(z_temp);
          // The spanwise mesh stores a duplicated periodic endpoint:
          // z(k=0) and z(k=mz-1) are the same physical point.  Therefore the period is
          // exactly z(mz-1)-z(0), with no half-cell correction.
          real zL = n2 > 1 ? fabs(z_temp[n2 - 1] - z_temp[0]) : 0;
          z_period.push_back(zL);

          real dz_this = 0;
          for (int k = 1; k < n2; ++k) dz_this += mesh[blk].z(i, 0, k) - mesh[blk].z(i, 0, k - 1);
          if (n2 > 1) dz_this /= n2 - 1;
          dz = dz_this;

          printf("\t\tInflow boundary %s with (%d, %d) grid points for digital filter on process %d, dz = %f.\n",
                 boundary_name.c_str(), n1, n2, myid, dz_this);

          N1.push_back(n1);
          N2.push_back(n2);
        }
      }
    }
  }

  bool use_df = n_df_face > 0;
  parameter.update_parameter("use_df", use_df);
  if (n_df_face == 0) return;

  initialize_df_memory(mesh, N1, N2);
  if (myid == 0) printf("\t\tThe memory for the digital filter is allocated.\n");

  get_digital_filter_lund_matrix(parameter, N1, scaled_y);
  if (myid == 0) printf("\t\tThe Lund matrix for the digital filter is computed.\n");

  get_digital_filter_convolution_kernel(parameter, scaled_y, y_ext, z_coord, N1, N2, z_period, dz);
  if (myid == 0) printf("\t\tThe convolution kernel for the digital filter is computed.\n");

  int init = parameter.get_int("initial");
  if (init == 1) {
    if (read_df(parameter, mesh, N1, N2)) {
      for (int i = 0; i < n_df_face; ++i) {
        dim3 TPB = {32, 32};
        dim3 BPG = {((N1[i] + 2 * ngg + TPB.x - 1) / TPB.x), ((N2[i] + 2 * ngg + TPB.y - 1) / TPB.y)};
        compute_fluctuations_first_step<<<BPG, TPB>>>(fluctuation_dPtr, df_lundMatrix_dPtr, df_velFluc_old_dPtr, i,
                                                      N1[i], N2[i], ngg);
      }
      if (myid == 0) {
        printf("\t\tDigital-filter RNG states and old filtered fields are restored from restart files.\n");
      }
      return;
    }
    if (myid == 0) {
      printf("\t\tDigital-filter restart failed or is incompatible. Reinitializing DF states.\n");
    }
  }

  for (int i = 0; i < n_df_face; ++i) {
    int sz = (N1[i] + 2 * DF_N + 2 * ngg) * (N2[i] + 2 * DF_N + 2 * ngg) * 3;
    dim3 TPB = {1024, 1, 1};
    dim3 BPG = {(sz - 1) / TPB.x + 1, 1, 1};
    time_t time_curr;
    initialize_rng<<<BPG, TPB>>>(rng_states_hPtr[i].data(), sz, time(&time_curr));

    generate_random_numbers(i, N1[i], N2[i], ngg);
    if (df_mode == 2) {
      apply_convolution_tbl(i, N1[i], N2[i], ngg);
    } else {
      apply_convolution(i, N1[i], N2[i], ngg);
    }
    sz = (N1[i] + 2 * ngg) * (N2[i] + 2 * ngg) * 3;
    cudaMemcpy(df_velFluc_old_hPtr[i].data(), df_velFluc_new_hPtr[i].data(), sz * sizeof(real),
               cudaMemcpyDeviceToDevice);
    TPB = {32, 32};
    BPG = {((N1[i] + 2 * ngg + TPB.x - 1) / TPB.x), ((N2[i] + 2 * ngg + TPB.y - 1) / TPB.y)};
    compute_fluctuations_first_step<<<BPG, TPB>>>(fluctuation_dPtr, df_lundMatrix_dPtr, df_velFluc_old_dPtr, i, N1[i],
                                                  N2[i], ngg);
  }
  if (myid == 0) printf("\t\tThe velocity fluctuations are computed.\n");
}

void DBoundCond::initialize_df_memory(const Mesh &mesh, const std::vector<int> &N1, const std::vector<int> &N2) {
  std::vector<ggxl::VectorField1D<real>> df_lundMatrix_hPtr(n_df_face);
  std::vector<ggxl::VectorField3D<real>> df_by_hPtr(n_df_face);
  std::vector<ggxl::VectorField3D<real>> df_bz_hPtr(n_df_face);
  rng_states_hPtr = new ggxl::VectorField2D<curandState>[n_df_face];
  random_values_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  std::vector<ggxl::VectorField2D<real>> df_fy_hPtr(n_df_face);
  df_velFluc_old_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  df_velFluc_new_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr(n_df_face);
  df_rng_state_cpu = new ggxl::VectorField2DHost<curandState>[n_df_face];
  df_velFluc_cpu = new ggxl::VectorField2DHost<real>[n_df_face];

  const int ngg = mesh.ngg;
  const int ng_filter = std::max(ngg, DF_N);
  for (int i = 0; i < n_df_face; ++i) {
    const int my = N1[i], mz = N2[i];
    df_lundMatrix_hPtr[i].allocate_memory(my, 6, ngg);
    df_by_hPtr[i].allocate_memory(my, mz, 1, 3, ng_filter);
    df_bz_hPtr[i].allocate_memory(my, mz, 1, 3, ng_filter);
    rng_states_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    random_values_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_fy_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_velFluc_old_hPtr[i].allocate_memory(my, mz, 3, ngg);
    df_velFluc_new_hPtr[i].allocate_memory(my, mz, 3, ngg);
    fluctuation_hPtr[i].allocate_memory(1, my, mz, 3, ngg);
    df_rng_state_cpu[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_velFluc_cpu[i].allocate_memory(my, mz, 3, ngg);
  }

  cudaMalloc(&df_lundMatrix_dPtr, n_df_face * sizeof(ggxl::VectorField1D<real>));
  cudaMemcpy(df_lundMatrix_dPtr, df_lundMatrix_hPtr.data(), n_df_face * sizeof(ggxl::VectorField1D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_by_dPtr, n_df_face * sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(df_by_dPtr, df_by_hPtr.data(), n_df_face * sizeof(ggxl::VectorField3D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&df_bz_dPtr, n_df_face * sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(df_bz_dPtr, df_bz_hPtr.data(), n_df_face * sizeof(ggxl::VectorField3D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&rng_states_dPtr, n_df_face * sizeof(ggxl::VectorField2D<curandState>));
  cudaMemcpy(rng_states_dPtr, rng_states_hPtr, n_df_face * sizeof(ggxl::VectorField2D<curandState>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&random_values_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(random_values_dPtr, random_values_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_fy_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_fy_dPtr, df_fy_hPtr.data(), n_df_face * sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&df_velFluc_old_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_velFluc_old_dPtr, df_velFluc_old_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_velFluc_new_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_velFluc_new_dPtr, df_velFluc_new_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&fluctuation_dPtr, n_df_face * sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(fluctuation_dPtr, fluctuation_hPtr.data(), n_df_face * sizeof(ggxl::VectorField3D<real>),
             cudaMemcpyHostToDevice);
}

void DBoundCond::get_digital_filter_lund_matrix(Parameter &parameter, const std::vector<int> &N1,
  const std::vector<std::vector<real>> &scaled_y) const {
  if (df_mode == 2) {
    read_touber_tbl_reynolds_stress(parameter, *this, N1, scaled_y);
    return;
  }
  const int method = parameter.get_int("reynolds_stress_supplier");
  if (method == 2) assume_gaussian_reynolds_stress(parameter, *this, N1, scaled_y);
  else {
    printf("The method %d is not supported for the Reynolds stress supplier.\n", method);
    MpiParallel::exit();
  }
}

void assume_gaussian_reynolds_stress(Parameter &parameter, const DBoundCond &dBoundCond, const std::vector<int> &N1,
  const std::vector<std::vector<real>> &y_scaled) {
  const auto Rij = parameter.get_real_array("df_reynolds_gaussian_peak");
  real *Rij_dPtr;
  cudaMalloc(&Rij_dPtr, 6 * sizeof(real));
  cudaMemcpy(Rij_dPtr, Rij.data(), 6 * sizeof(real), cudaMemcpyHostToDevice);
  const int ngg = parameter.get_int("ngg");
  for (int i = 0; i < dBoundCond.n_df_face; ++i) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[i].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[i].data(), y_scaled[i].size() * sizeof(real), cudaMemcpyHostToDevice);
    int TPB = 512;
    int BPG = (N1[i] + 2 * ngg + TPB - 1) / TPB;
    compute_lundMat_with_assumed_gaussian_reynolds_stress<<<BPG, TPB>>>(Rij_dPtr, dBoundCond.df_lundMatrix_dPtr, i,
                                                                        y_scaled_dPtr, N1[i], ngg);
    cudaFree(y_scaled_dPtr);
  }
  cudaFree(Rij_dPtr);
}

void read_touber_tbl_reynolds_stress(Parameter &parameter, const DBoundCond &dBoundCond, const std::vector<int> &N1,
  const std::vector<std::vector<real>> &y_scaled) {
  const auto filename = parameter.get_string("df_reynolds_stress_file");
  std::ifstream input(filename);
  if (!input) {
    printf("Cannot open df_reynolds_stress_file: %s\n", filename.c_str());
    MpiParallel::exit();
  }
  std::vector<real> y_file;
  const real y_ref = parameter.get_real("df_y_ref");
  std::vector<std::array<real, 6>> R_file;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::vector<real> vals;
    real v;
    while (iss >> v) vals.push_back(v);
    if (vals.size() == 5) {
      y_file.push_back(vals[0] / y_ref);
      R_file.push_back({vals[1], vals[4], 0.0, vals[2], 0.0, vals[3]}); // y R11 R22 R33 R12
    } else if (vals.size() >= 7) {
      y_file.push_back(vals[0] / y_ref);
      R_file.push_back({vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]}); // y R11 R12 R13 R22 R23 R33
    }
  }
  if (y_file.size() < 2) {
    printf("df_reynolds_stress_file should contain at least two valid rows.\n");
    MpiParallel::exit();
  }

  const int ngg = parameter.get_int("ngg");
  const int ny_file = static_cast<int>(y_file.size());
  for (int iFace = 0; iFace < dBoundCond.n_df_face; ++iFace) {
    std::vector<real> R_on_grid((N1[iFace] + 2 * ngg) * 6, 0.0);
    for (int j = -ngg; j < N1[iFace] + ngg; ++j) {
      const real y = y_scaled[iFace][j + ngg];
      int idx = 0;
      while (idx + 1 < ny_file && y_file[idx + 1] < y) ++idx;
      int idx2 = idx + 1 < ny_file ? idx + 1 : idx;
      real t = 0;
      if (idx2 != idx) t = (y - y_file[idx]) / (y_file[idx2] - y_file[idx]);
      if (t < 0) t = 0;
      if (t > 1) t = 1;
      for (int m = 0; m < 6; ++m) {
        R_on_grid[(j + ngg) * 6 + m] = (1 - t) * R_file[idx][m] + t * R_file[idx2][m];
      }
    }
    real *R_dPtr;
    cudaMalloc(&R_dPtr, R_on_grid.size() * sizeof(real));
    cudaMemcpy(R_dPtr, R_on_grid.data(), R_on_grid.size() * sizeof(real), cudaMemcpyHostToDevice);
    int TPB = 512;
    int BPG = (N1[iFace] + 2 * ngg + TPB - 1) / TPB;
    compute_lundMat_with_touber_reynolds_stress<<<BPG, TPB>>>(N1[iFace], ngg, R_dPtr, dBoundCond.df_lundMatrix_dPtr,
                                                              iFace);
    cudaFree(R_dPtr);
  }
}

void DBoundCond::get_digital_filter_convolution_kernel(Parameter &parameter,
  const std::vector<std::vector<real>> &y_scaled, const std::vector<std::vector<real>> &y_ext,
  const std::vector<std::vector<real>> &z_coord, const std::vector<int> &N1, const std::vector<int> &N2,
  const std::vector<real> &z_period, real dz) const {
  const int ngg = parameter.get_int("ngg");
  for (int iFace = 0; iFace < n_df_face; ++iFace) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[iFace].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[iFace].data(), y_scaled[iFace].size() * sizeof(real), cudaMemcpyHostToDevice);

    if (df_mode == 2) {
      real *y_ext_dPtr, *z_coord_dPtr;
      cudaMalloc(&y_ext_dPtr, y_ext[iFace].size() * sizeof(real));
      cudaMalloc(&z_coord_dPtr, z_coord[iFace].size() * sizeof(real));
      cudaMemcpy(y_ext_dPtr, y_ext[iFace].data(), y_ext[iFace].size() * sizeof(real), cudaMemcpyHostToDevice);
      cudaMemcpy(z_coord_dPtr, z_coord[iFace].data(), z_coord[iFace].size() * sizeof(real), cudaMemcpyHostToDevice);
      dim3 TPB(16, 16);
      dim3 BPG((N1[iFace] + 2 * ngg + TPB.x - 1) / TPB.x, (N2[iFace] + 2 * ngg + TPB.y - 1) / TPB.y);
      compute_convolution_kernel_tbl<<<BPG, TPB>>>(N1[iFace], N2[iFace], ngg, y_scaled_dPtr, df_y_zone_width,
                                                   df_y_zone_threshold, df_Ly_inner[0], df_Ly_inner[1], df_Ly_inner[2],
                                                   df_Ly_outer[0], df_Ly_outer[1], df_Ly_outer[2], df_Lz_inner[0],
                                                   df_Lz_inner[1], df_Lz_inner[2], df_Lz_outer[0], df_Lz_outer[1],
                                                   df_Lz_outer[2], df_by_dPtr, df_bz_dPtr, y_ext_dPtr, z_coord_dPtr,
                                                   z_period[iFace], iFace);
      cudaFree(y_ext_dPtr);
      cudaFree(z_coord_dPtr);
    } else {
      const real DF_IntegralLength = parameter.get_real("delta_omega");
      int TPB = 512;
      int BPG = (N1[iFace] + 2 * ngg + TPB - 1) / TPB;
      compute_convolution_kernel<<<BPG, TPB>>>(y_scaled_dPtr, df_by_dPtr, df_bz_dPtr, dz / DF_IntegralLength, iFace,
                                               N1[iFace], ngg);
    }
    cudaFree(y_scaled_dPtr);
  }
}

void DBoundCond::generate_random_numbers(int iFace, int my, int mz, int ngg) const {
  dim3 TPB(32, 32);
  dim3 BPG{((my + 2 * DF_N + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * DF_N + 2 * ngg + TPB.y - 1) / TPB.y)};
  generate_random_numbers_kernel<<<BPG, TPB>>>(rng_states_dPtr, random_values_dPtr, iFace, my, mz, ngg);
  TPB = 256;
  BPG = (my + 2 * DF_N + 2 * ngg - 1) / TPB.x + 1;
  remove_mean_spanwise<<<BPG, TPB>>>(random_values_dPtr, iFace, my, mz, ngg);
  TPB = {32, 32};
  BPG = {(my + 2 * DF_N + 2 * ngg - 1) / TPB.x + 1, ((DF_N + ngg + TPB.y - 1) / TPB.y)};
  apply_periodic_in_spanwise<<<BPG, TPB>>>(random_values_dPtr, iFace, my, mz, ngg);
}

void DBoundCond::apply_convolution(int iFace, int my, int mz, int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + 2 * DF_N + TPB.y - 1) / TPB.y)};
  perform_convolution_y<<<BPG, TPB>>>(random_values_dPtr, df_by_dPtr, df_fy_dPtr, iFace, my, mz, ngg);
  BPG = {((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  perform_convolution_z<<<BPG, TPB>>>(df_fy_dPtr, df_bz_dPtr, df_velFluc_new_dPtr, iFace, my, mz, ngg);
}

void DBoundCond::apply_convolution_tbl(int iFace, int my, int mz, int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + 2 * DF_N + TPB.y - 1) / TPB.y)};
  perform_convolution_y_tbl<<<BPG, TPB>>>(random_values_dPtr, df_by_dPtr, df_fy_dPtr, iFace, my, mz, ngg);
  BPG = {((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  perform_convolution_z_tbl<<<BPG, TPB>>>(df_fy_dPtr, df_bz_dPtr, df_velFluc_new_dPtr, iFace, my, mz, ngg);
}

void DBoundCond::compute_fluctuations(const DParameter *param, DZone *zone, const Inflow *inflowHere, int iFace, int my,
  int mz, int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  Castro_time_correlation_and_fluc_computation<<<BPG, TPB>>>(param, zone, inflowHere, df_velFluc_old_dPtr,
                                                             df_velFluc_new_dPtr, df_lundMatrix_dPtr, fluctuation_dPtr,
                                                             iFace, my, mz, ngg);
}

void DBoundCond::compute_fluctuations_tbl(const DParameter *param, DZone *zone, const Inflow *inflowHere,
  ggxl::VectorField3D<real> *profile_dPtr, int iFace, int my, int mz, int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  Touber_time_correlation_and_fluc_computation<<<BPG, TPB>>>(param, zone, inflowHere, profile_dPtr,
                                                             df_velFluc_old_dPtr, df_velFluc_new_dPtr,
                                                             df_lundMatrix_dPtr, fluctuation_dPtr,
                                                             df_Lx[0], df_Lx[1], df_Lx[2], iFace, my, mz, ngg);
}

__global__ void compute_lundMat_with_assumed_gaussian_reynolds_stress(const real *Rij,
  ggxl::VectorField1D<real> *df_lundMatrix_hPtr, int i_face, const real *y_scaled, int my, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) return;
  auto &mat = df_lundMatrix_hPtr[i_face];
  const real y_ = y_scaled[j + ngg];
  const real gaussian_y = exp(-y_ * y_ * 2.0);
  const real R11_y = Rij[0] * gaussian_y;
  const real R12_y = Rij[1] * gaussian_y;
  const real R22_y = Rij[2] * gaussian_y;
  const real R13_y = Rij[3] * gaussian_y;
  const real R23_y = Rij[4] * gaussian_y;
  const real R33_y = Rij[5] * gaussian_y;
  const real L11 = sqrt(df_max_real(R11_y, 0.0));
  const real L21 = L11 < 1e-40 ? 0 : R12_y / L11;
  const real L31 = L11 < 1e-40 ? 0 : R13_y / L11;
  const real L22 = sqrt(df_max_real(R22_y - L21 * L21, 0.0));
  const real L32 = L22 < 1e-40 ? 0 : (R23_y - L31 * L21) / L22;
  const real L33 = sqrt(df_max_real(R33_y - L31 * L31 - L32 * L32, 0.0));
  mat(j, 0) = L11;
  mat(j, 1) = L21;
  mat(j, 2) = L22;
  mat(j, 3) = L31;
  mat(j, 4) = L32;
  mat(j, 5) = L33;
}

__global__ void compute_lundMat_with_touber_reynolds_stress(int my, int ngg, const real *RStress,
  ggxl::VectorField1D<real> *df_lundMatrix_hPtr, int i_face) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) return;
  const int idx = (j + ngg) * 6;
  const real R11 = df_max_real(RStress[idx + 0], 0.0);
  const real R12 = RStress[idx + 1];
  const real R13 = RStress[idx + 2];
  const real R22 = df_max_real(RStress[idx + 3], 0.0);
  const real R23 = RStress[idx + 4];
  const real R33 = df_max_real(RStress[idx + 5], 0.0);
  auto &mat = df_lundMatrix_hPtr[i_face];
  const real L11 = sqrt(R11);
  const real L21 = L11 < 1e-40 ? 0.0 : R12 / L11;
  const real L31 = L11 < 1e-40 ? 0.0 : R13 / L11;
  const real L22 = sqrt(df_max_real(R22 - L21 * L21, 0.0));
  const real L32 = L22 < 1e-40 ? 0.0 : (R23 - L31 * L21) / L22;
  const real L33 = sqrt(df_max_real(R33 - L31 * L31 - L32 * L32, 0.0));
  mat(j, 0) = L11;
  mat(j, 1) = L21;
  mat(j, 2) = L22;
  mat(j, 3) = L31;
  mat(j, 4) = L32;
  mat(j, 5) = L33;
}

__global__ void compute_convolution_kernel(const real *y_scaled, ggxl::VectorField3D<real> *df_by,
  ggxl::VectorField3D<real> *df_bz, real dz_scaled, int iFace, int my, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) return;
  real dy = (j == -ngg) ? y_scaled[2 * ngg + 1] - y_scaled[2 * ngg] : y_scaled[j + ngg] - y_scaled[j + ngg - 1];
  auto &by = df_by[iFace];
  auto &bz = df_bz[iFace];
  real sum = 0;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real expValue = exp(-pi * fabs(static_cast<real>(jf)) * dy);
    by(j, 0, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real value = by(j, 0, jf, 0) / sum;
    by(j, 0, jf, 0) = value;
    by(j, 0, jf, 1) = value;
    by(j, 0, jf, 2) = value;
  }
  sum = 0;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real expValue = exp(-pi * dz_scaled * fabs(static_cast<real>(jf)));
    bz(j, 0, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real value = bz(j, 0, jf, 0) / sum;
    bz(j, 0, jf, 0) = value;
    bz(j, 0, jf, 1) = value;
    bz(j, 0, jf, 2) = value;
  }
}

__global__ void compute_convolution_kernel_tbl(int my, int mz, int ngg, const real *y_scaled, real y_zone_width,
  real y_zone_center, real LyUi, real LyVi, real LyWi, real LyUo, real LyVo, real LyWo, real LzUi, real LzVi, real LzWi,
  real LzUo, real LzVo, real LzWo, ggxl::VectorField3D<real> *df_by, ggxl::VectorField3D<real> *df_bz,
  const real *y_ext, const real *z_coord, real z_period, int iFace) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;

  const int y_offset = ngg + DBoundCond::DF_N;
  const real eta = y_scaled[j + ngg];
  const real width = y_zone_width > 1e-12 ? y_zone_width : 0.03;
  real blend = 0.5 * (1.0 + tanh((eta - y_zone_center) / width));
  if (blend < 0) blend = 0;
  if (blend > 1) blend = 1;
  const real Ly_i[3] = {LyUi, LyVi, LyWi};
  const real Ly_o[3] = {LyUo, LyVo, LyWo};
  const real Lz_i[3] = {LzUi, LzVi, LzWi};
  const real Lz_o[3] = {LzUo, LzVo, LzWo};

  auto &by = df_by[iFace];
  auto &bz = df_bz[iFace];
  const real y0 = y_ext[j + y_offset];
  const int km = df_wrap_index(k, mz);
  const real z0 = z_coord[km];

  for (int comp = 0; comp < 3; ++comp) {
    const real Ly = df_max_real((1.0 - blend) * Ly_i[comp] + blend * Ly_o[comp], 1e-30);
    const real Lz = df_max_real((1.0 - blend) * Lz_i[comp] + blend * Lz_o[comp], 1e-30);

    real sum_y = 0;
    for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
      const real dy = fabs(y_ext[j + jf + y_offset] - y0);
      const real v = exp(-pi * dy / Ly);
      by(j, k, jf, comp) = v;
      sum_y += v * v;
    }
    sum_y = sqrt(df_max_real(sum_y, 1e-300));
    for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) by(j, k, jf, comp) /= sum_y;

    real sum_z = 0;
    for (int kf = -DBoundCond::DF_N; kf <= DBoundCond::DF_N; ++kf) {
      const int kw = df_wrap_index(k + kf, mz);
      const real dz = df_periodic_distance(z_coord[kw], z0, z_period);
      const real v = exp(-pi * dz / Lz);
      bz(j, k, kf, comp) = v;
      sum_z += v * v;
    }
    sum_z = sqrt(df_max_real(sum_z, 1e-300));
    for (int kf = -DBoundCond::DF_N; kf <= DBoundCond::DF_N; ++kf) bz(j, k, kf, comp) /= sum_z;
  }
}

__global__ void generate_random_numbers_kernel(ggxl::VectorField2D<curandState> *rng_states,
  ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg || k >= mz + DBoundCond::DF_N + ngg) return;
  random_numbers[iFace](j, k, 0) = curand_normal_double(&rng_states[iFace](j, k, 0));
  random_numbers[iFace](j, k, 1) = curand_normal_double(&rng_states[iFace](j, k, 1));
  random_numbers[iFace](j, k, 2) = curand_normal_double(&rng_states[iFace](j, k, 2));
}

__global__ void remove_mean_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg) return;
  auto &r = random_numbers[iFace];
  const int mz_unique = df_unique_span_count(mz);
  real mean[3]{0, 0, 0}, rms[3]{0, 0, 0};
  // k=mz-1 is a duplicated endpoint and must not be counted twice.
  for (int k = 0; k < mz_unique; ++k) {
    for (int c = 0; c < 3; ++c) {
      mean[c] += r(j, k, c);
      rms[c] += r(j, k, c) * r(j, k, c);
    }
  }
  const real inv = 1.0 / static_cast<real>(mz_unique);
  for (int c = 0; c < 3; ++c) {
    mean[c] *= inv;
    rms[c] = sqrt(df_max_real(rms[c] * inv - mean[c] * mean[c], 1e-300));
  }
  for (int k = 0; k < mz_unique; ++k) {
    for (int c = 0; c < 3; ++c) r(j, k, c) = (r(j, k, c) - mean[c]) / rms[c];
  }
  // Make the stored duplicated endpoint exactly equal to k=0.
  if (mz > 1) {
    for (int c = 0; c < 3; ++c) r(j, mz - 1, c) = r(j, 0, c);
  }
}

__global__ void apply_periodic_in_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz,
  int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (j >= my + DBoundCond::DF_N + ngg || k > DBoundCond::DF_N + ngg) return;
  auto &r = random_numbers[iFace];
  const int src_plus = df_wrap_index(k, mz);
  const int src_minus = df_wrap_index(-k, mz);
  for (int c = 0; c < 3; ++c) {
    // Stored duplicated endpoint.
    if (mz > 1) r(j, mz - 1, c) = r(j, 0, c);
    // Periodic padding beyond the duplicated endpoint and before k=0.
    r(j, mz - 1 + k, c) = r(j, src_plus, c);
    r(j, -k, c) = r(j, src_minus, c);
  }
}

__global__ void perform_convolution_y(ggxl::VectorField2D<real> *random_numbers, ggxl::VectorField3D<real> *df_by,
  ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg - DBoundCond::DF_N;
  if (j >= my + ngg || k >= mz + DBoundCond::DF_N + ngg) return;
  auto &by = df_by[iFace];
  auto &r = random_numbers[iFace];
  auto &fy = df_fy[iFace];
  for (int c = 0; c < 3; ++c) {
    real val = 0;
    for (int jj = -DBoundCond::DF_N; jj <= DBoundCond::DF_N; ++jj) val += by(j, 0, jj, c) * r(j + jj, k, c);
    fy(j, k, c) = val;
  }
}

__global__ void perform_convolution_z(ggxl::VectorField2D<real> *df_fy, ggxl::VectorField3D<real> *df_bz,
  ggxl::VectorField2D<real> *velFluc, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;
  auto &fy = df_fy[iFace];
  auto &bz = df_bz[iFace];
  auto &vf = velFluc[iFace];
  for (int c = 0; c < 3; ++c) {
    real val = 0;
    for (int kk = -DBoundCond::DF_N; kk <= DBoundCond::DF_N; ++kk) val += bz(j, 0, kk, c) * fy(j, k + kk, c);
    vf(j, k, c) = val;
  }
}

__global__ void perform_convolution_y_tbl(ggxl::VectorField2D<real> *random_numbers,
  ggxl::VectorField3D<real> *df_by, ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg - DBoundCond::DF_N;
  if (j >= my + ngg || k >= mz + DBoundCond::DF_N + ngg) return;
  auto &by = df_by[iFace];
  auto &r = random_numbers[iFace];
  auto &fy = df_fy[iFace];
  const int kk = df_wrap_index(k, mz);
  for (int c = 0; c < 3; ++c) {
    real val = 0;
    for (int jj = -DBoundCond::DF_N; jj <= DBoundCond::DF_N; ++jj) val += by(j, kk, jj, c) * r(j + jj, k, c);
    fy(j, k, c) = val;
  }
}

__global__ void perform_convolution_z_tbl(ggxl::VectorField2D<real> *df_fy,
  ggxl::VectorField3D<real> *df_bz, ggxl::VectorField2D<real> *velFluc, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;
  auto &fy = df_fy[iFace];
  auto &bz = df_bz[iFace];
  auto &vf = velFluc[iFace];
  const int kk0 = df_wrap_index(k, mz);
  for (int c = 0; c < 3; ++c) {
    real val = 0;
    for (int kk = -DBoundCond::DF_N; kk <= DBoundCond::DF_N; ++kk) val += bz(j, kk0, kk, c) * fy(j, k + kk, c);
    vf(j, k, c) = val;
  }
}

__global__ void compute_fluctuations_first_step(ggxl::VectorField3D<real> *fluctuation_dPtr,
  ggxl::VectorField1D<real> *lundMatrix_dPtr, ggxl::VectorField2D<real> *df_velFluc_old_dPtr, int iFace, int my, int mz,
  int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;
  auto &fluc = fluctuation_dPtr[iFace];
  auto &lund = lundMatrix_dPtr[iFace];
  auto &velGen = df_velFluc_old_dPtr[iFace];
  fluc(0, j, k, 0) = lund(j, 0) * velGen(j, k, 0);
  fluc(0, j, k, 1) = lund(j, 1) * velGen(j, k, 0) + lund(j, 2) * velGen(j, k, 1);
  fluc(0, j, k, 2) = lund(j, 3) * velGen(j, k, 0) + lund(j, 4) * velGen(j, k, 1) + lund(j, 5) * velGen(j, k, 2);
}

__global__ void Castro_time_correlation_and_fluc_computation(const DParameter *param, DZone *zone, const Inflow *inflow,
  ggxl::VectorField2D<real> *velFluc_old, ggxl::VectorField2D<real> *velFluc_new,
  ggxl::VectorField1D<real> *lundMatrix_dPtr, ggxl::VectorField3D<real> *fluctuation_dPtr, int iFace, int my, int mz,
  int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;
  const real dt = 1.0 / 3.0 * param->dt;
  const real u_upper = inflow->u, u_lower = inflow->u_lower;
  const auto y = zone->y(0, j, k);
  const real y_ref = inflow->delta_omega;
  const real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / y_ref);
  const real tLen_x = y_ref / u;
  const real PiDtDivTInt = -pi * dt / tLen_x;
  const real arg1 = exp(PiDtDivTInt * 0.5);
  const real arg2 = sqrt(df_max_real(1 - exp(PiDtDivTInt), 0.0));
  auto &old = velFluc_old[iFace];
  auto &vf = velFluc_new[iFace];
  real val[3];
  for (int c = 0; c < 3; ++c) {
    val[c] = arg1 * old(j, k, c) + arg2 * vf(j, k, c);
    vf(j, k, c) = val[c];
    old(j, k, c) = val[c];
  }
  auto &lund = lundMatrix_dPtr[iFace];
  auto &fluc = fluctuation_dPtr[iFace];
  fluc(0, j, k, 0) = lund(j, 0) * val[0];
  fluc(0, j, k, 1) = lund(j, 1) * val[0] + lund(j, 2) * val[1];
  fluc(0, j, k, 2) = lund(j, 3) * val[0] + lund(j, 4) * val[1] + lund(j, 5) * val[2];
}

__global__ void Touber_time_correlation_and_fluc_computation(const DParameter *param, DZone *zone, const Inflow *inflow,
  ggxl::VectorField3D<real> *profile_dPtr, ggxl::VectorField2D<real> *velFluc_old,
  ggxl::VectorField2D<real> *velFluc_new, ggxl::VectorField1D<real> *lundMatrix_dPtr,
  ggxl::VectorField3D<real> *fluctuation_dPtr, real LxU, real LxV, real LxW, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) return;
  const real dt = 1.0 / 3.0 * param->dt;
  auto &prof = profile_dPtr[inflow->profile_idx];
  real u_mean = fabs(prof(0, j, k, 1));
  const real Ue = fabs(inflow->u) > 1e-30 ? fabs(inflow->u) : 1.0;
  u_mean = df_max_real(u_mean, 0.05 * Ue);
  const real Lx[3]{LxU, LxV, LxW};
  auto &old = velFluc_old[iFace];
  auto &vf = velFluc_new[iFace];
  real val[3];
  for (int c = 0; c < 3; ++c) {
    const real Tx = df_max_real(Lx[c] / u_mean, 1e-30);
    const real PiDtDivTInt = -pi * dt / Tx;
    const real arg1 = exp(PiDtDivTInt * 0.5);
    const real arg2 = sqrt(df_max_real(1.0 - exp(PiDtDivTInt), 0.0));
    val[c] = arg1 * old(j, k, c) + arg2 * vf(j, k, c);
    vf(j, k, c) = val[c];
    old(j, k, c) = val[c];
  }
  auto &lund = lundMatrix_dPtr[iFace];
  auto &fluc = fluctuation_dPtr[iFace];
  fluc(0, j, k, 0) = lund(j, 0) * val[0];
  fluc(0, j, k, 1) = lund(j, 1) * val[0] + lund(j, 2) * val[1];
  fluc(0, j, k, 2) = lund(j, 3) * val[0] + lund(j, 4) * val[1] + lund(j, 5) * val[2];
}

void DBoundCond::diagnose_digital_filter_tbl(const Block &block, int df_iFace, int step) const {
  if (df_mode != 2 || df_diagnostic_interval <= 0 || step < 0) return;
  if (step % df_diagnostic_interval != 0) return;

  const int my = block.my;
  const int mz = block.mz;
  const int ngg = block.ngg;

  ggxl::VectorField3D<real> fluc_meta;
  ggxl::VectorField1D<real> lund_meta;
  cudaMemcpy(&fluc_meta, fluctuation_dPtr + df_iFace, sizeof(ggxl::VectorField3D<real>), cudaMemcpyDeviceToHost);
  cudaMemcpy(&lund_meta, df_lundMatrix_dPtr + df_iFace, sizeof(ggxl::VectorField1D<real>), cudaMemcpyDeviceToHost);

  const int fluc_sz = fluc_meta.size();
  const int lund_sz = lund_meta.size();
  std::vector<real> fluc(fluc_sz * 3, 0.0);
  std::vector<real> lund(lund_sz * 6, 0.0);
  cudaMemcpy(fluc.data(), fluc_meta.data(), fluc.size() * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(lund.data(), lund_meta.data(), lund.size() * sizeof(real), cudaMemcpyDeviceToHost);

  const int disp2_fluc = 1 + 2 * ngg;
  const int disp1_fluc = (my + 2 * ngg) * disp2_fluc;
  const int dispt_fluc = (disp1_fluc + disp2_fluc + 1) * ngg;
  auto fluc_at = [&](int j, int k, int c) -> real {
    const int idx = k * disp1_fluc + j * disp2_fluc + dispt_fluc + c * fluc_sz;
    return fluc[idx] * df_velocity_scale;
  };
  auto lund_at = [&](int j, int c) -> real {
    return lund[j + ngg + c * lund_sz];
  };

  std::string filename = "./output/DF_TBL_statistics_b" + std::to_string(block.block_id) +
                         "_f" + std::to_string(df_iFace) + "_s" + std::to_string(step) + ".dat";
  std::ofstream out(filename);
  out.setf(std::ios::scientific);
  out << std::setprecision(16);
  out <<
      "variables= y mean_u mean_v mean_w uu vv ww uv uw vw target_uu target_vv target_ww target_uv target_uw target_vw\n";
  out << "zone i=" << my << ", f=point\n";
  for (int j = 0; j < my; ++j) {
    real mu = 0, mv = 0, mw = 0;
    real uu = 0, vv = 0, ww = 0, uv = 0, uw = 0, vw = 0;
    const int mz_unique = mz > 1 ? mz - 1 : 1;
    for (int k = 0; k < mz_unique; ++k) {
      const real u = fluc_at(j, k, 0);
      const real v = fluc_at(j, k, 1);
      const real w = fluc_at(j, k, 2);
      mu += u;
      mv += v;
      mw += w;
      uu += u * u;
      vv += v * v;
      ww += w * w;
      uv += u * v;
      uw += u * w;
      vw += v * w;
    }
    const real inv = 1.0 / static_cast<real>(mz_unique);
    mu *= inv;
    mv *= inv;
    mw *= inv;
    uu *= inv;
    vv *= inv;
    ww *= inv;
    uv *= inv;
    uw *= inv;
    vw *= inv;

    const real L11 = lund_at(j, 0), L21 = lund_at(j, 1), L22 = lund_at(j, 2);
    const real L31 = lund_at(j, 3), L32 = lund_at(j, 4), L33 = lund_at(j, 5);
    const real scale2 = df_velocity_scale * df_velocity_scale;
    const real R11 = L11 * L11 * scale2;
    const real R12 = L11 * L21 * scale2;
    const real R13 = L11 * L31 * scale2;
    const real R22 = (L21 * L21 + L22 * L22) * scale2;
    const real R23 = (L21 * L31 + L22 * L32) * scale2;
    const real R33 = (L31 * L31 + L32 * L32 + L33 * L33) * scale2;
    const real y = block.y(0, j, 0);
    out << y << ' ' << mu << ' ' << mv << ' ' << mw << ' '
        << uu << ' ' << vv << ' ' << ww << ' ' << uv << ' ' << uw << ' ' << vw << ' '
        << R11 << ' ' << R22 << ' ' << R33 << ' ' << R12 << ' ' << R13 << ' ' << R23 << '\n';
  }
}

bool DBoundCond::read_df(Parameter &parameter, const Mesh &mesh, const std::vector<int> &N1,
  const std::vector<int> &N2) const {
  const int myid = parameter.get_int("myid");
  const int ngg = mesh.ngg;

  for (int iFace = 0; iFace < n_df_face; ++iFace) {
    const int my = N1[iFace];
    const int mz = N2[iFace];
    const int sz_rng = (my + 2 * DF_N + 2 * ngg) * (mz + 2 * DF_N + 2 * ngg) * 3;
    const int sz_old = (my + 2 * ngg) * (mz + 2 * ngg) * 3;

    std::string filename = "./output/df-p" + std::to_string(myid) + "-f" + std::to_string(iFace) + ".bin";
    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
      printf("Warning: cannot open %s; restarting DF states.\n", filename.c_str());
      return false;
    }

    int first = 0;
    if (fread(&first, sizeof(int), 1, fp) != 1) {
      printf("Warning: cannot read DF restart header from %s.\n", filename.c_str());
      fclose(fp);
      return false;
    }

    int my_file = 0, mz_file = 0, ngg_file = 0, DF_N_file = 0, df_mode_file = -1;
    if (first == DF_RESTART_MAGIC) {
      int version = 0;
      if (fread(&version, sizeof(int), 1, fp) != 1 || version != DF_RESTART_VERSION) {
        printf("Warning: unsupported DF restart version in %s.\n", filename.c_str());
        fclose(fp);
        return false;
      }
      if (fread(&my_file, sizeof(int), 1, fp) != 1 ||
          fread(&mz_file, sizeof(int), 1, fp) != 1 ||
          fread(&ngg_file, sizeof(int), 1, fp) != 1 ||
          fread(&DF_N_file, sizeof(int), 1, fp) != 1 ||
          fread(&df_mode_file, sizeof(int), 1, fp) != 1) {
        printf("Warning: incomplete DF restart header in %s.\n", filename.c_str());
        fclose(fp);
        return false;
      }
      if (df_mode_file != df_mode) {
        printf("Warning: DF restart mode mismatch in %s: file=%d, current=%d.\n",
               filename.c_str(), df_mode_file, df_mode);
        fclose(fp);
        return false;
      }
    } else {
      // Legacy layout: my, mz, ngg, DF_N, rng_states, old_filtered_field.
      my_file = first;
      if (fread(&mz_file, sizeof(int), 1, fp) != 1 ||
          fread(&ngg_file, sizeof(int), 1, fp) != 1 ||
          fread(&DF_N_file, sizeof(int), 1, fp) != 1) {
        printf("Warning: incomplete legacy DF restart header in %s.\n", filename.c_str());
        fclose(fp);
        return false;
      }
    }

    if (my_file != my || mz_file != mz || ngg_file != ngg || DF_N_file != DF_N) {
      printf(
        "Warning: DF restart file %s is incompatible. File dimensions=(%d,%d), ngg=%d, DF_N=%d; current=(%d,%d), ngg=%d, DF_N=%d.\n",
        filename.c_str(), my_file, mz_file, ngg_file, DF_N_file, my, mz, ngg, DF_N);
      fclose(fp);
      return false;
    }

    if (fread(df_rng_state_cpu[iFace].data(), sizeof(curandState), sz_rng, fp) != static_cast<size_t>(sz_rng)) {
      printf("Warning: incomplete RNG state block in %s.\n", filename.c_str());
      fclose(fp);
      return false;
    }
    if (fread(df_velFluc_cpu[iFace].data(), sizeof(real), sz_old, fp) != static_cast<size_t>(sz_old)) {
      printf("Warning: incomplete old filtered field block in %s.\n", filename.c_str());
      fclose(fp);
      return false;
    }
    fclose(fp);

    cudaMemcpy(rng_states_hPtr[iFace].data(), df_rng_state_cpu[iFace].data(), sz_rng * sizeof(curandState),
               cudaMemcpyHostToDevice);
    cudaMemcpy(df_velFluc_old_hPtr[iFace].data(), df_velFluc_cpu[iFace].data(), sz_old * sizeof(real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(df_velFluc_new_hPtr[iFace].data(), df_velFluc_old_hPtr[iFace].data(), sz_old * sizeof(real),
               cudaMemcpyDeviceToDevice);
  }

  const auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Warning: CUDA error while restoring DF restart: %s.\n", cudaGetErrorString(err));
    return false;
  }
  return true;
}

void DBoundCond::write_df(Parameter &parameter, const Mesh &mesh) const {
  const int myid = parameter.get_int("myid");
  printf("Process %d is writing the digital filter to the file.\n", myid);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error before writing DF restart: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }
  for (int iFace = 0; iFace < n_df_face; ++iFace) {
    std::string filename = "./output/df-p" + std::to_string(myid) + "-f" + std::to_string(iFace) + ".bin";
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
      printf("Error: cannot open the file %s.\n", filename.c_str());
      MpiParallel::exit();
    }
    const int blk = df_related_block[iFace];
    int my = mesh[blk].my, mz = mesh[blk].mz, ngg = mesh.ngg;
    const int sz1 = (my + 2 * DF_N + 2 * ngg) * (mz + 2 * DF_N + 2 * ngg) * 3;
    const int sz2 = (my + 2 * ngg) * (mz + 2 * ngg) * 3;
    cudaMemcpy(df_rng_state_cpu[iFace].data(), rng_states_hPtr[iFace].data(), sz1 * sizeof(curandState),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(df_velFluc_cpu[iFace].data(), df_velFluc_old_hPtr[iFace].data(), sz2 * sizeof(real),
               cudaMemcpyDeviceToHost);

    const int magic = DF_RESTART_MAGIC;
    const int version = DF_RESTART_VERSION;
    const int mode = df_mode;
    fwrite(&magic, sizeof(int), 1, fp);
    fwrite(&version, sizeof(int), 1, fp);
    fwrite(&my, sizeof(int), 1, fp);
    fwrite(&mz, sizeof(int), 1, fp);
    fwrite(&ngg, sizeof(int), 1, fp);
    fwrite(&DF_N, sizeof(int), 1, fp);
    fwrite(&mode, sizeof(int), 1, fp);
    fwrite(df_rng_state_cpu[iFace].data(), sizeof(curandState), sz1, fp);
    fwrite(df_velFluc_cpu[iFace].data(), sizeof(real), sz2, fp);
    fclose(fp);
  }
}
} // namespace cfd

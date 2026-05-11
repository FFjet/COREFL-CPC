#include "Monitor.cuh"
#include "gxl_lib/MyString.h"
#include <filesystem>
#include "Parallel.h"
#include "Field.h"
#include "Mesh.h"
#include "DParameter.cuh"
#include "SinglePointStat.cuh"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <cmath>
#include "ChemData.h"

namespace cfd {
namespace {
struct MonitorBlockRecord {
  int pid{-1};
  int block_id{-1};
  int il{0}, ir{0}, jl{0}, jr{0}, kl{0}, kr{0};
  int frequency{0};
  int if_burst{0};
};

enum BurstMetricIndex : int {
  burst_sum_abs_flux  = 0,
  burst_max_abs_flux  = 1,
  burst_sum_abs_vflux = 2,
  burst_max_abs_vflux = 3,
  burst_max_abs_uY    = 4,
  burst_max_abs_vY    = 5,
  burst_metric_count  = 6
};

bool parse_monitor_block_record(const std::string &line, int default_frequency, MonitorBlockRecord &record) {
  if (line.empty()) {
    return false;
  }
  std::istringstream line_stream(line);
  if (!(line_stream >> record.pid >> record.block_id >> record.il >> record.ir >> record.jl >> record.jr >>
        record.kl >> record.kr)) {
    return false;
  }
  record.frequency = default_frequency;
  record.if_burst = 0;
  if (int frequency; line_stream >> frequency) {
    record.frequency = frequency;
    if (int if_burst; line_stream >> if_burst) {
      record.if_burst = if_burst != 0 ? 1 : 0;
    }
  }
  return true;
}

__device__ inline void atomic_max_positive_real(real *address, real value) {
  if (value <= 0) {
    return;
  }
  auto *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull;
  while (__longlong_as_double(old) < value) {
    const auto assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
    if (old == assumed) {
      break;
    }
  }
}

__device__ inline int classify_quadrant(real up, real vp, real yp) {
  if (up > 0 && vp > 0 && yp > 0) return 1;
  if (up < 0 && vp > 0 && yp > 0) return 2;
  if (up < 0 && vp > 0 && yp < 0) return 3;
  if (up > 0 && vp > 0 && yp < 0) return 4;
  if (up > 0 && vp < 0 && yp > 0) return 5;
  if (up < 0 && vp < 0 && yp > 0) return 6;
  if (up < 0 && vp < 0 && yp < 0) return 7;
  if (up > 0 && vp < 0 && yp < 0) return 8;
  return 0;
}

__global__ void accumulate_burst_trigger_for_block(
  DZone *zone, const DParameter *param, int il, int ir, int jl, int jr, int kl, int kr, int h2_species_index,
  real burst_H, int *quadrant_counts, unsigned long long *point_count, real *metrics) {
  const int nx = ir - il + 1;
  const int ny = jr - jl + 1;
  const int nz = kr - kl + 1;
  const unsigned long long plane_size = static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const unsigned long long total_points = plane_size * static_cast<unsigned long long>(nz);
  const unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * gridDim.x;

  int local_quadrant_counts[8]{};
  unsigned long long local_point_count = 0;
  real local_sum_abs_flux = 0;
  real local_max_abs_flux = 0;
  real local_sum_abs_vflux = 0;
  real local_max_abs_vflux = 0;
  real local_max_abs_uY = 0;
  real local_max_abs_vY = 0;

  for (unsigned long long linear = tid; linear < total_points; linear += stride) {
    const int k = kl + static_cast<int>(linear / plane_size);
    const auto rem = linear % plane_size;
    const int j = jl + static_cast<int>(rem / nx);
    const int i = il + static_cast<int>(rem % nx);

    const real u_mean = zone->stat_favre_1st(i, j, 0, 0);
    const real v_mean = zone->stat_favre_1st(i, j, 0, 1);
    const real y_mean = zone->stat_favre_1st(i, j, 0, 4 + h2_species_index);
    const real u_rms = sqrt(fmax(zone->stat_favre_2nd(i, j, 0, 0), 0.0));
    const real y_rms = sqrt(
      fmax(zone->stat_favre_2nd(i, j, 0, param->statFavre2ScalarVarOffset + h2_species_index), 0.0));
    const real abs_uY = fabs(zone->stat_favre_2nd(i, j, 0, param->statFavre2ScalarFluxUOffset + h2_species_index));
    const real abs_vY = fabs(zone->stat_favre_2nd(i, j, 0, param->statFavre2ScalarFluxVOffset + h2_species_index));
    local_max_abs_uY = fmax(local_max_abs_uY, abs_uY);
    local_max_abs_vY = fmax(local_max_abs_vY, abs_vY);

    const real up = zone->bv(i, j, k, 1) - u_mean;
    const real vp = zone->bv(i, j, k, 2) - v_mean;
    const real yp = zone->sv(i, j, k, h2_species_index) - y_mean;
    const real abs_flux = fabs(up * yp);
    const real abs_vflux = fabs(vp * yp);
    local_sum_abs_flux += abs_flux;
    local_sum_abs_vflux += abs_vflux;
    local_max_abs_flux = fmax(local_max_abs_flux, abs_flux);
    local_max_abs_vflux = fmax(local_max_abs_vflux, abs_vflux);
    ++local_point_count;

    if (abs_flux <= burst_H * u_rms * y_rms) {
      continue;
    }
    if (const int quadrant = classify_quadrant(up, vp, yp); quadrant > 0) {
      ++local_quadrant_counts[quadrant - 1];
    }
  }

  if (local_point_count > 0) {
    atomicAdd(point_count, local_point_count);
    atomicAdd(&metrics[burst_sum_abs_flux], local_sum_abs_flux);
    atomicAdd(&metrics[burst_sum_abs_vflux], local_sum_abs_vflux);
    atomic_max_positive_real(&metrics[burst_max_abs_flux], local_max_abs_flux);
    atomic_max_positive_real(&metrics[burst_max_abs_vflux], local_max_abs_vflux);
    atomic_max_positive_real(&metrics[burst_max_abs_uY], local_max_abs_uY);
    atomic_max_positive_real(&metrics[burst_max_abs_vY], local_max_abs_vY);
  }
  for (int q = 0; q < 8; ++q) {
    if (local_quadrant_counts[q] > 0) {
      atomicAdd(&quadrant_counts[q], local_quadrant_counts[q]);
    }
  }
}
} // namespace

Monitor::Monitor(Parameter &parameter, const Species &species, const Mesh &mesh_) :
  output_file{parameter.get_int("output_file")}, n_block{parameter.get_int("n_block")}, n_point(n_block, 0),
  mesh(mesh_) {
  block_monitor.initialize(parameter, mesh_);
  block_monitor.configure_burst(parameter, species);

  if (!parameter.get_int("if_monitor_points")) {
    return;
  }

  h_ptr = new DeviceMonitorData;

  // Set up the labels to monitor
  auto var_name_found{setup_labels_to_monitor(parameter, species)};
  const auto myid{parameter.get_int("myid")};
  if (myid == 0) {
    printf("The following variables will be monitored:\n");
    for (const auto &name: var_name_found) {
      printf("%s\t", name.c_str());
    }
    printf("\n");
  }

  // Read the points to be monitored
  auto monitor_file_name{parameter.get_string("monitor_file")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  int counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    int pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    int i, j, k, b;
    line_stream >> b >> i >> j >> k;
    is_h.push_back(i);
    js_h.push_back(j);
    ks_h.push_back(k);
    bs_h.push_back(b);
    printf("Process %d monitors block %d, point (%d, %d, %d).\n", myid, b, i, j, k);
    ++n_point[b];
    ++counter;
  }
  // copy the indices to GPU
  cudaMalloc(&h_ptr->bs_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->is_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->js_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->ks_d, sizeof(int) * counter);
  cudaMemcpy(h_ptr->bs_d, bs_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->is_d, is_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->js_d, js_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->ks_d, ks_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  n_point_total = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point_total);
  disp.resize(parameter.get_int("n_block"), 0);
  for (int b = 1; b < n_block; ++b) {
    disp[b] = disp[b - 1] + n_point[b - 1];
  }
  cudaMalloc(&h_ptr->disp, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->disp, disp.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);
  cudaMalloc(&h_ptr->n_point, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->n_point, n_point.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point_total, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point_total);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point_total, nullptr);
  for (int l = 0; l < n_point_total; ++l) {
    std::string file_name{
      "/monitor_" + std::to_string(myid) + '_' + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' +
      std::to_string(js_h[l]) + '_' + std::to_string(ks_h[l]) + ".dat"
    };
    std::filesystem::path whole_name_path{out_dir.string() + file_name};
    if (!exists(whole_name_path)) {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
      fprintf(files[l], "variables=step,");
      for (const auto &name: var_name_found) {
        fprintf(files[l], "%s,", name.c_str());
      }
      fprintf(files[l], "time\n");
    } else {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
    }
  }
}

Monitor::Monitor(const Parameter &parameter, const Mesh &mesh_) :
  output_file{parameter.get_int("output_file")}, n_block{parameter.get_int("n_block")}, n_point(n_block, 0),
  mesh(mesh_) {}

void Monitor::initialize(Parameter &parameter, const Species &species) {
  block_monitor.initialize(parameter, mesh);
  block_monitor.configure_burst(parameter, species);
  if (!parameter.get_int("if_monitor_points")) {
    return;
  }

  h_ptr = new DeviceMonitorData;

  // Set up the labels to monitor
  auto var_name_found{setup_labels_to_monitor(parameter, species)};
  const auto myid{parameter.get_int("myid")};
  if (myid == 0) {
    printf("The following variables will be monitored:\n");
    for (const auto &name: var_name_found) {
      printf("%s\t", name.c_str());
    }
    printf("\n");
  }

  // Read the points to be monitored
  auto monitor_file_name{parameter.get_string("monitor_file")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  int counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    int pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    int i, j, k, b;
    line_stream >> b >> i >> j >> k;
    is_h.push_back(i);
    js_h.push_back(j);
    ks_h.push_back(k);
    bs_h.push_back(b);
    printf("Process %d monitors block %d, point (%d, %d, %d).\n", myid, b, i, j, k);
    ++n_point[b];
    ++counter;
  }
  // copy the indices to GPU
  cudaMalloc(&h_ptr->bs_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->is_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->js_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->ks_d, sizeof(int) * counter);
  cudaMemcpy(h_ptr->bs_d, bs_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->is_d, is_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->js_d, js_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->ks_d, ks_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  n_point_total = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point_total);
  disp.resize(parameter.get_int("n_block"), 0);
  for (int b = 1; b < n_block; ++b) {
    disp[b] = disp[b - 1] + n_point[b - 1];
  }
  cudaMalloc(&h_ptr->disp, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->disp, disp.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);
  cudaMalloc(&h_ptr->n_point, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->n_point, n_point.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point_total, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point_total);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point_total, nullptr);
  for (int l = 0; l < n_point_total; ++l) {
    std::string file_name{
      "/monitor_" + std::to_string(myid) + '_' + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' +
      std::to_string(js_h[l]) + '_' + std::to_string(ks_h[l]) + ".dat"
    };
    std::filesystem::path whole_name_path{out_dir.string() + file_name};
    if (!exists(whole_name_path)) {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
      fprintf(files[l], "variables=step,");
      for (const auto &name: var_name_found) {
        fprintf(files[l], "%s,", name.c_str());
      }
      fprintf(files[l], "time\n");
    } else {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
    }
  }
}

std::vector<std::string> Monitor::setup_labels_to_monitor(const Parameter &parameter, const Species &species) {
  const auto n_spec{species.n_spec};
  auto &spec_list{species.spec_list};

  const auto var_name{parameter.get_string_array("monitor_var")};

  std::vector<int> bv_idx, sv_idx;
  auto n_found{0};
  std::vector<std::string> var_name_found;
  for (auto name: var_name) {
    name = gxl::to_upper(name);
    if (name == "DENSITY" || name == "RHO") {
      bv_idx.push_back(0);
      var_name_found.emplace_back("Density");
      ++n_found;
    } else if (name == "U") {
      bv_idx.push_back(1);
      var_name_found.emplace_back("U");
      ++n_found;
    } else if (name == "V") {
      bv_idx.push_back(2);
      var_name_found.emplace_back("V");
      ++n_found;
    } else if (name == "W") {
      bv_idx.push_back(3);
      var_name_found.emplace_back("W");
      ++n_found;
    } else if (name == "PRESSURE" || name == "P") {
      bv_idx.push_back(4);
      var_name_found.emplace_back("Pressure");
      ++n_found;
    } else if (name == "TEMPERATURE" || name == "T") {
      bv_idx.push_back(5);
      var_name_found.emplace_back("Temperature");
      ++n_found;
    } else if (name == "TKE") {
      sv_idx.push_back(n_spec);
      var_name_found.emplace_back("TKE");
      ++n_found;
    } else if (name == "OMEGA") {
      sv_idx.push_back(n_spec + 1);
      var_name_found.emplace_back("Omega");
      ++n_found;
    } else if (name == "MIXTUREFRACTION" || name == "Z") {
      sv_idx.push_back(n_spec + 2);
      var_name_found.emplace_back("MixtureFraction");
      ++n_found;
    } else if (name == "MIXTUREFRACTIONVARIANCE") {
      sv_idx.push_back(n_spec + 3);
      var_name_found.emplace_back("MixtureFractionVariance");
      ++n_found;
    } else if (n_spec > 0) {
      auto it = spec_list.find(name);
      if (it != spec_list.end()) {
        sv_idx.push_back(it->second);
        var_name_found.emplace_back(name);
        ++n_found;
      } else {
        if (parameter.get_int("myid") == 0) {
          printf("The variable %s is not found in the variable list.\n", name.c_str());
        }
      }
    } else if (parameter.get_int("n_ps") > 0) {
      const int n_ps{parameter.get_int("n_ps")};
      const int i_ps{parameter.get_int("i_ps")};
      if (name == "PS") {
        for (int i = 0; i < n_ps; ++i) {
          sv_idx.push_back(i_ps + i);
          var_name_found.emplace_back("PS" + std::to_string(i + 1));
          ++n_found;
        }
      } else {
        if (parameter.get_int("myid") == 0) {
          printf("The variable %s is not found in the variable list.\n", name.c_str());
        }
      }
    } else {
      if (parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  n_bv = static_cast<int>(bv_idx.size());
  n_sv = static_cast<int>(sv_idx.size());
  // The +1 is for physical time
  n_var = n_bv + n_sv + 1;
  h_ptr->n_bv = n_bv;
  h_ptr->n_sv = n_sv;
  h_ptr->n_var = n_var;
  cudaMalloc(&h_ptr->bv_label, sizeof(int) * n_bv);
  cudaMalloc(&h_ptr->sv_label, sizeof(int) * n_sv);
  cudaMemcpy(h_ptr->bv_label, bv_idx.data(), sizeof(int) * n_bv, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->sv_label, sv_idx.data(), sizeof(int) * n_sv, cudaMemcpyHostToDevice);

  return var_name_found;
}

Monitor::~Monitor() {
  for (const auto fp: files) {
    fclose(fp);
  }
}

void Monitor::monitor_point(int step, real physical_time, const std::vector<Field> &field) {
  if (counter_step == 0)
    step_start = step;

  for (int b = 0; b < n_block; ++b) {
    if (n_point[b] > 0) {
      constexpr auto tpb{128};
      const auto bpg{(n_point[b] - 1) / tpb + 1};
      record_monitor_data<<<bpg, tpb>>>(field[b].d_ptr, d_ptr, b, counter_step % output_file, physical_time);
    }
  }
  ++counter_step;
}

void Monitor::output_point_monitors() {
  cudaMemcpy(mon_var_h.data(), h_ptr->data.data(), sizeof(real) * n_var * output_file * n_point_total,
             cudaMemcpyDeviceToHost);

  for (int p = 0; p < n_point_total; ++p) {
    for (int l = 0; l < counter_step; ++l) {
      fprintf(files[p], "%d\t", step_start + l);
      for (int k = 0; k < n_var; ++k) {
        fprintf(files[p], "%e\t", mon_var_h(k, l, p));
      }
      fprintf(files[p], "\n");
    }
  }
  counter_step = 0;
}

bool Monitor::need_block_monitor_service(int step) const {
  return block_monitor.need_service(step);
}

bool Monitor::evaluate_block_burst(
  const Parameter &parameter, std::vector<Field> &field, real t, int step, int stat_count, DParameter *param) {
  return block_monitor.evaluate_burst(parameter, field, t, step, stat_count, param);
}

void Monitor::output_block_monitors(const Parameter &parameter, std::vector<Field> &field, real t, int step) {
  block_monitor.output_data(parameter, field, t, step);
}

void Monitor::stop_recording_blocks(const Parameter &parameter) const {
  block_monitor.stop_recording_blocks(parameter);
}

__global__ void record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos,
  real physical_time) {
  const auto idx = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  if (idx >= monitor_info->n_point[blk_id])
    return;
  const auto idx_tot = monitor_info->disp[blk_id] + idx;
  const auto i = monitor_info->is_d[idx_tot];
  const auto j = monitor_info->js_d[idx_tot];
  const auto k = monitor_info->ks_d[idx_tot];

  auto &data = monitor_info->data;
  const auto bv_label = monitor_info->bv_label;
  const auto sv_label = monitor_info->sv_label;
  const auto n_bv{monitor_info->n_bv};
  int var_counter{0};
  for (int l = 0; l < n_bv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->bv(i, j, k, bv_label[l]);
    ++var_counter;
  }
  for (int l = 0; l < monitor_info->n_sv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->sv(i, j, k, sv_label[l]);
    ++var_counter;
  }
  data(var_counter, counter_pos, idx_tot) = physical_time;
}

void BlockMonitor::initialize(Parameter &parameter, const Mesh &mesh_) {
  if (!parameter.get_int("if_monitor_blocks")) {
    return;
  }

  const int monitor_block_frequency = parameter.get_int("monitor_block_frequency");

  // First, get the variables to monitor.
  setup_labels_to_monitor(parameter);

  // for every group
  const auto monitor_file_name{parameter.get_string("monitor_block_file")};
  const auto myid{parameter.get_int("myid")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  int counter{0};
  while (gxl::getline(monitor_file, line)) {
    MonitorBlockRecord record{};
    if (!parse_monitor_block_record(line, monitor_block_frequency, record)) {
      continue;
    }
    if (myid != record.pid) {
      continue;
    }
    frequency.push_back(record.frequency);
    group_range.emplace_back(
      std::array<int, 7>{record.block_id, record.il, record.ir, record.jl, record.jr, record.kl, record.kr});
    burst_block_flag.push_back(record.if_burst);
    if (record.if_burst != 0) {
      burst_block_indices.push_back(n_block_mon);
      if (std::find(burst_unique_block_ids.begin(), burst_unique_block_ids.end(), record.block_id) ==
          burst_unique_block_ids.end()) {
        burst_unique_block_ids.push_back(record.block_id);
      }
    }
    ++n_block_mon;
    ++counter;
  }
  monitor_file.close();

  // Get the maximum public frequency
  if (n_block_mon > 0) {
    int nFreq = frequency[0];
    for (int i = 1; i < n_block_mon; ++i) {
      nFreq = std::gcd(nFreq, frequency[i]);
    }
    parameter.update_parameter("monitor_block_frequency", nFreq);
  }
  burst_state = {};
  burst_output_step = -1;
  burst_started_this_step = false;
  burst_output_quadrant_tag.clear();
  last_global_quadrant_counts.fill(0);

  const std::filesystem::path out_dir("output/monitor");
  if (exists(out_dir)) {
    // If we are starting a new simulation, we need to rename the old directory to avoid overwriting.
    if (parameter.get_int("initial") == 0) {
      if (parameter.get_int("myid") == 0) {
        std::filesystem::path old_dir = out_dir;
        old_dir += "_old";
        if (exists(old_dir)) {
          remove_all(old_dir);
        }
        std::filesystem::rename(out_dir, old_dir);
        printf("The output directory %s already exists. The old directory is renamed to %s.\n",
               out_dir.string().c_str(), old_dir.string().c_str());
      }
    }
  }
  if (!exists(out_dir) && parameter.get_int("myid") == 0) {
    create_directories(out_dir);
  }
  if (parameter.get_int("myid") == 0) {
    if (!std::filesystem::exists(out_dir.string() + "/info.txt")) {
      // open a file to write the variable names and other info
      std::ofstream info_file(out_dir.string() + "/info.txt");
      // First, print the global time step * monitor interval
      info_file << "Block monitor information:\n";
      real dt{parameter.get_real("dt")};
      if (!parameter.get_bool("fixed_time_step")) {
        printf("Warning: The time step is not fixed. The block monitor time interval may not be accurate.\n");
      }
      int monitor_interval{parameter.get_int("monitor_block_frequency")};
      info_file << "\tTime interval between two block monitor outputs: " << dt * monitor_interval << "\n";

      info_file << "The following variables are monitored in the block monitor:\n\t";
      auto var_names = parameter.get_string_array("monitor_block_var");
      // output the names, each line contains 5 names
      for (int i = 0; i < var_names.size(); ++i) {
        info_file << var_names[i] << "\t";
        if ((i + 1) % 5 == 0) {
          info_file << "\n\t";
        }
      }
      info_file << "\nTotal number of variables monitored: " << n_var << "\n";
      info_file << "Total number of block monitors: " << n_block_mon << "\n";
      info_file << "Total number of burst-enabled block monitors on this process: " << burst_block_indices.size() <<
          "\n";
      // Print the block monitor ranges
      info_file << "Block monitor ranges:\n";
      info_file << "\tpid\tBlockID\tIL\tIR\tJL\tJR\tKL\tKR\tFrequency\tIfBurst\n";
      monitor_file.open(monitor_file_name);
      // print all the lines except the comment line, including the ones for other processes
      gxl::getline(monitor_file, line); // The comment line
      while (gxl::getline(monitor_file, line)) {
        MonitorBlockRecord record{};
        if (!parse_monitor_block_record(line, monitor_block_frequency, record)) {
          continue;
        }
        info_file << "\t" << record.pid << '\t' << record.block_id << "\t" << record.il << "\t" << record.ir <<
            "\t" << record.jl << "\t" << record.jr << "\t" << record.kl << "\t" << record.kr << "\t" <<
            record.frequency << "\t" << record.if_burst << "\n";
      }
      // Open the file output/message/reference_state.txt, copy all the content to info_file
      std::ifstream ref_file("output/message/reference_state.txt");
      if (ref_file.is_open()) {
        info_file << "\nReference state information:\n\t";
        while (gxl::getline(ref_file, line)) {
          info_file << line << "\n\t";
        }
        ref_file.close();
      }
      info_file.close();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // write the mesh and jacobian
  ty = new MPI_Datatype[n_block_mon];
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];
    const auto &b = mesh_[bid];

    MPI_Datatype ty1;
    const int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const int il{group_range[blk][1]}, ir{group_range[blk][2]}, jl{group_range[blk][3]}, jr{group_range[blk][4]},
        kl{group_range[blk][5]}, kr{group_range[blk][6]};
    const auto nx{ir - il + 1}, ny{jr - jl + 1}, nz{kr - kl + 1};
    const int small_size[3]{nx, ny, nz};
    const int start_idx[3]{b.ngg + il, b.ngg + jl, b.ngg + kl};
    MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty1);
    MPI_Type_commit(&ty1);
    ty[blk] = ty1;

    MPI_File fp;
    char file_name[1024];
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-mesh.bin", out_dir.string().c_str(), myid, bid, il, ir, jl,
            jr, kl, kr);
    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    MPI_Status status;
    // write x, y, z
    MPI_File_write(fp, &nx, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &ny, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &nz, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &n_var, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, b.x.data(), 1, ty1, &status);
    MPI_File_write(fp, b.y.data(), 1, ty1, &status);
    MPI_File_write(fp, b.z.data(), 1, ty1, &status);
    MPI_File_write(fp, b.jacobian.data(), 1, ty1, &status);
    MPI_File_close(&fp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BlockMonitor::BlockMonitor(Parameter &parameter, const Mesh &mesh_) {
  if (!parameter.get_int("if_monitor_blocks")) {
    return;
  }

  const int monitor_block_frequency = parameter.get_int("monitor_block_frequency");

  // First, get the variables to monitor.
  setup_labels_to_monitor(parameter);

  // for every group
  const auto monitor_file_name{parameter.get_string("monitor_block_file")};
  const auto myid{parameter.get_int("myid")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  int counter{0};
  while (gxl::getline(monitor_file, line)) {
    MonitorBlockRecord record{};
    if (!parse_monitor_block_record(line, monitor_block_frequency, record)) {
      continue;
    }
    if (myid != record.pid) {
      continue;
    }
    frequency.push_back(record.frequency);
    group_range.emplace_back(
      std::array<int, 7>{record.block_id, record.il, record.ir, record.jl, record.jr, record.kl, record.kr});
    burst_block_flag.push_back(record.if_burst);
    if (record.if_burst != 0) {
      burst_block_indices.push_back(n_block_mon);
      if (std::find(burst_unique_block_ids.begin(), burst_unique_block_ids.end(), record.block_id) ==
          burst_unique_block_ids.end()) {
        burst_unique_block_ids.push_back(record.block_id);
      }
    }
    ++n_block_mon;
    ++counter;
  }
  monitor_file.close();

  // Get the maximum public frequency
  if (n_block_mon > 0) {
    int nFreq = frequency[0];
    for (int i = 1; i < n_block_mon; ++i) {
      nFreq = std::gcd(nFreq, frequency[i]);
    }
    parameter.update_parameter("monitor_block_frequency", nFreq);
  }
  burst_state = {};
  burst_output_step = -1;
  burst_started_this_step = false;
  burst_output_quadrant_tag.clear();
  last_global_quadrant_counts.fill(0);

  const std::filesystem::path out_dir("output/monitor");
  if (exists(out_dir)) {
    // If we are starting a new simulation, we need to rename the old directory to avoid overwriting.
    if (parameter.get_int("initial") == 0) {
      if (parameter.get_int("myid") == 0) {
        std::filesystem::path old_dir = out_dir;
        old_dir += "_old";
        if (exists(old_dir)) {
          remove_all(old_dir);
        }
        std::filesystem::rename(out_dir, old_dir);
        printf("The output directory %s already exists. The old directory is renamed to %s.\n",
               out_dir.string().c_str(), old_dir.string().c_str());
      }
    }
  }
  if (!exists(out_dir) && parameter.get_int("myid") == 0) {
    create_directories(out_dir);
  }
  if (parameter.get_int("myid") == 0) {
    if (!std::filesystem::exists(out_dir.string() + "/info.txt")) {
      // open a file to write the variable names and other info
      std::ofstream info_file(out_dir.string() + "/info.txt");
      // First, print the global time step * monitor interval
      info_file << "Block monitor information:\n";
      real dt{parameter.get_real("dt")};
      if (!parameter.get_bool("fixed_time_step")) {
        printf("Warning: The time step is not fixed. The block monitor time interval may not be accurate.\n");
      }
      int monitor_interval{parameter.get_int("monitor_block_frequency")};
      info_file << "\tTime interval between two block monitor outputs: " << dt * monitor_interval << "\n";

      info_file << "The following variables are monitored in the block monitor:\n\t";
      auto var_names = parameter.get_string_array("monitor_block_var");
      // output the names, each line contains 5 names
      for (int i = 0; i < var_names.size(); ++i) {
        info_file << var_names[i] << "\t";
        if ((i + 1) % 5 == 0) {
          info_file << "\n\t";
        }
      }
      info_file << "\nTotal number of variables monitored: " << n_var << "\n";
      info_file << "Total number of block monitors: " << n_block_mon << "\n";
      info_file << "Total number of burst-enabled block monitors on this process: " << burst_block_indices.size() <<
          "\n";
      // Print the block monitor ranges
      info_file << "Block monitor ranges:\n";
      info_file << "\tpid\tBlockID\tIL\tIR\tJL\tJR\tKL\tKR\tFrequency\tIfBurst\n";
      monitor_file.open(monitor_file_name);
      // print all the lines except the comment line, including the ones for other processes
      gxl::getline(monitor_file, line); // The comment line
      while (gxl::getline(monitor_file, line)) {
        MonitorBlockRecord record{};
        if (!parse_monitor_block_record(line, monitor_block_frequency, record)) {
          continue;
        }
        info_file << "\t" << record.pid << '\t' << record.block_id << "\t" << record.il << "\t" << record.ir <<
            "\t" << record.jl << "\t" << record.jr << "\t" << record.kl << "\t" << record.kr << "\t" <<
            record.frequency << "\t" << record.if_burst << "\n";
      }
      // Open the file output/message/reference_state.txt, copy all the content to info_file
      std::ifstream ref_file("output/message/reference_state.txt");
      if (ref_file.is_open()) {
        info_file << "\nReference state information:\n\t";
        while (gxl::getline(ref_file, line)) {
          info_file << line << "\n\t";
        }
        ref_file.close();
      }
      info_file.close();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // write the mesh and jacobian
  ty = new MPI_Datatype[n_block_mon];
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];
    const auto &b = mesh_[bid];

    MPI_Datatype ty1;
    const int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const int il{group_range[blk][1]}, ir{group_range[blk][2]}, jl{group_range[blk][3]}, jr{group_range[blk][4]},
        kl{group_range[blk][5]}, kr{group_range[blk][6]};
    const auto nx{ir - il + 1}, ny{jr - jl + 1}, nz{kr - kl + 1};
    const int small_size[3]{nx, ny, nz};
    const int start_idx[3]{b.ngg + il, b.ngg + jl, b.ngg + kl};
    MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty1);
    MPI_Type_commit(&ty1);
    ty[blk] = ty1;

    MPI_File fp;
    char file_name[1024];
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-mesh.bin", out_dir.string().c_str(), myid, bid, il, ir, jl,
            jr, kl, kr);
    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    MPI_Status status;
    // write x, y, z
    MPI_File_write(fp, &nx, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &ny, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &nz, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &n_var, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, b.x.data(), 1, ty1, &status);
    MPI_File_write(fp, b.y.data(), 1, ty1, &status);
    MPI_File_write(fp, b.z.data(), 1, ty1, &status);
    MPI_File_write(fp, b.jacobian.data(), 1, ty1, &status);
    MPI_File_close(&fp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void BlockMonitor::configure_burst(const Parameter &parameter, const Species &species) {
  burst_enabled = parameter.get_bool("if_monitor_block_burst");
  burst_quadrants = parameter.get_int_array("monitor_block_burst_quadrants");
  burst_frequency = parameter.get_int("monitor_block_burst_frequency");
  burst_check_frequency = parameter.get_int("monitor_block_burst_check_frequency");
  if (burst_check_frequency <= 0) {
    burst_check_frequency = parameter.get_int("monitor_block_frequency");
  }
  burst_duration = parameter.get_int("monitor_block_burst_duration");
  burst_cooldown = parameter.get_int("monitor_block_burst_cooldown");
  burst_min_count = parameter.get_int("monitor_block_burst_min_count");
  burst_H = parameter.get_real("monitor_block_burst_H");
  burst_use_abs_flux = parameter.get_bool("monitor_block_burst_use_abs_flux");
  fav2_scalar_var_offset = parameter.has_int("stat_favre2_scalar_var_offset")
                           ? parameter.get_int("stat_favre2_scalar_var_offset")
                           : 7;
  if (parameter.has_int("stat_favre2_scalar_flux_u_offset") && parameter.has_int("stat_favre2_scalar_flux_v_offset")) {
    fav2_scalar_flux_u_offset = parameter.get_int("stat_favre2_scalar_flux_u_offset");
    fav2_scalar_flux_v_offset = parameter.get_int("stat_favre2_scalar_flux_v_offset");
  } else {
    const int n_spec = parameter.get_int("n_spec");
    const int scalar_var_count = (parameter.get_bool("if_collect_spec_favreAvg") ? n_spec : 0) + parameter.
                                 get_int("n_ps");
    fav2_scalar_flux_u_offset = fav2_scalar_var_offset + scalar_var_count;
    fav2_scalar_flux_v_offset = fav2_scalar_flux_u_offset +
                                (parameter.get_bool("if_collect_scalar_flux") ? n_spec : 0);
  }

  if (!burst_enabled) {
    return;
  }
  if (!parameter.get_int("if_monitor_blocks")) {
    printf("Error: if_monitor_block_burst requires if_monitor_blocks = 1.\n");
    MpiParallel::exit();
  }
  if (!parameter.get_bool("if_collect_statistics")) {
    printf("Error: if_monitor_block_burst requires if_collect_statistics = 1.\n");
    MpiParallel::exit();
  }
  if (!parameter.get_bool("perform_spanwise_average")) {
    printf("Error: if_monitor_block_burst requires perform_spanwise_average = 1.\n");
    MpiParallel::exit();
  }
  if (!parameter.get_bool("if_collect_spec_favreAvg")) {
    printf("Error: if_monitor_block_burst requires if_collect_spec_favreAvg = 1.\n");
    MpiParallel::exit();
  }
  if (!parameter.get_bool("if_collect_2nd_moments")) {
    printf("Error: if_monitor_block_burst requires if_collect_2nd_moments = 1.\n");
    MpiParallel::exit();
  }
  if (!parameter.get_bool("if_collect_scalar_flux")) {
    printf("Error: if_monitor_block_burst requires if_collect_scalar_flux = 1.\n");
    MpiParallel::exit();
  }
  if (burst_frequency <= 0 || burst_check_frequency <= 0) {
    printf("Error: burst monitor frequencies must be positive.\n");
    MpiParallel::exit();
  }
  if (burst_duration < 0 || burst_cooldown < 0 || burst_min_count <= 0) {
    printf("Error: burst monitor duration/cooldown/min_count is invalid.\n");
    MpiParallel::exit();
  }
  if (burst_quadrants.empty()) {
    printf("Error: monitor_block_burst_quadrants cannot be empty.\n");
    MpiParallel::exit();
  }
  for (const int quadrant: burst_quadrants) {
    if (quadrant < 1 || quadrant > 8) {
      printf("Error: burst quadrant %d is invalid. Expect values in [1, 8].\n", quadrant);
      MpiParallel::exit();
    }
  }

  h2_species_index = -1;
  for (const auto &[name, index]: species.spec_list) {
    if (gxl::to_upper(name) == "H2") {
      h2_species_index = index;
      break;
    }
  }
  if (h2_species_index < 0) {
    printf("Error: if_monitor_block_burst requires H2 in the species list.\n");
    MpiParallel::exit();
  }

  const int local_burst_block_count = static_cast<int>(burst_block_indices.size());
  MPI_Allreduce(&local_burst_block_count, &global_burst_block_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (global_burst_block_count <= 0) {
    printf("Error: if_monitor_block_burst requires at least one monitor block with if_burst = 1 across all ranks.\n");
    MpiParallel::exit();
  }

  if (d_burst_quadrant_counts == nullptr) {
    cudaMalloc(&d_burst_quadrant_counts, 8 * sizeof(int));
  }
  if (d_burst_point_count == nullptr) {
    cudaMalloc(&d_burst_point_count, sizeof(unsigned long long));
  }
  if (d_burst_metrics == nullptr) {
    cudaMalloc(&d_burst_metrics, burst_metric_count * sizeof(real));
  }

  if (parameter.get_int("myid") == 0) {
    std::ofstream info_file("output/monitor/info.txt", std::ios::app);
    if (info_file.is_open()) {
      info_file << "\nBurst block monitor settings:\n";
      info_file << "\tEnabled\t1\n";
      info_file << "\tGlobalBurstBlockCount\t" << global_burst_block_count << "\n";
      info_file << "\tCheckFrequency\t" << burst_check_frequency << "\n";
      info_file << "\tBurstFrequency\t" << burst_frequency << "\n";
      info_file << "\tDuration\t" << burst_duration << "\n";
      info_file << "\tCooldown\t" << burst_cooldown << "\n";
      info_file << "\tH\t" << burst_H << "\n";
      info_file << "\tMinCount\t" << burst_min_count << "\n";
      info_file << "\tQuadrants\t";
      for (size_t i = 0; i < burst_quadrants.size(); ++i) {
        if (i > 0) info_file << ',';
        info_file << burst_quadrants[i];
      }
      info_file << "\n";
    }
  }
}

bool BlockMonitor::need_regular_output(int blk, int step) const {
  return step % frequency[blk] == 0;
}

bool BlockMonitor::need_trigger_check(int step) const {
  if (!burst_enabled) {
    return false;
  }
  if (burst_state.active || step < burst_state.cooldown_end_step) {
    return false;
  }
  return step % burst_check_frequency == 0;
}

bool BlockMonitor::need_service(int step) const {
  bool need_local_regular_output = false;
  for (int blk = 0; blk < n_block_mon; ++blk) {
    if (need_regular_output(blk, step)) {
      need_local_regular_output = true;
      break;
    }
  }
  if (!burst_enabled) {
    return need_local_regular_output;
  }
  if (burst_state.active) {
    const int next_state_step = burst_state.next_output_step >= 0
                                ? std::min(burst_state.next_output_step, burst_state.end_step)
                                : burst_state.end_step;
    return need_local_regular_output || step >= next_state_step;
  }
  return need_local_regular_output || need_trigger_check(step);
}

const char *BlockMonitor::quadrant_name(int quadrant) {
  switch (quadrant) {
    case 1: return "O1";
    case 2: return "O2";
    case 3: return "O3";
    case 4: return "O4";
    case 5: return "O5";
    case 6: return "O6";
    case 7: return "O7";
    case 8: return "O8";
    default: return "O0";
  }
}

bool BlockMonitor::is_target_quadrant(int quadrant, real up, real vp, real yp) {
  switch (quadrant) {
    case 1: return up > 0 && vp > 0 && yp > 0;
    case 2: return up < 0 && vp > 0 && yp > 0;
    case 3: return up < 0 && vp > 0 && yp < 0;
    case 4: return up > 0 && vp > 0 && yp < 0;
    case 5: return up > 0 && vp < 0 && yp > 0;
    case 6: return up < 0 && vp < 0 && yp > 0;
    case 7: return up < 0 && vp < 0 && yp < 0;
    case 8: return up > 0 && vp < 0 && yp < 0;
    default: return false;
  }
}

BlockBurstTriggerResult BlockMonitor::check_local_burst_trigger(const Parameter &parameter, std::vector<Field> &field,
  int stat_count, DParameter *param) const {
  BlockBurstTriggerResult result{};
  if (!burst_enabled || h2_species_index < 0 || param == nullptr || stat_count <= 0 || burst_block_indices.empty()) {
    return result;
  }

  dim3 stat_tpb{8, 8, 4};
  for (const int bid: burst_unique_block_ids) {
    const auto mx = field[bid].block.mx;
    const auto my = field[bid].block.my;
    if (field[bid].block.mz == 1) {
      stat_tpb = {16, 16, 1};
    } else {
      stat_tpb = {8, 8, 4};
    }
    const dim3 stat_bpg{((mx - 1) / stat_tpb.x + 1), ((my - 1) / stat_tpb.y + 1), 1};
    compute_statistical_data_spanwise_average<<<stat_bpg, stat_tpb>>>(field[bid].d_ptr, param, stat_count);
  }

  cudaMemset(d_burst_quadrant_counts, 0, 8 * sizeof(int));
  cudaMemset(d_burst_point_count, 0, sizeof(unsigned long long));
  cudaMemset(d_burst_metrics, 0, burst_metric_count * sizeof(real));

  constexpr int tpb = 256;
  for (const int blk: burst_block_indices) {
    const auto &range = group_range[blk];
    const int nx = range[2] - range[1] + 1;
    const int ny = range[4] - range[3] + 1;
    const int nz = range[6] - range[5] + 1;
    const auto n_point = static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny) *
                         static_cast<unsigned long long>(nz);
    if (n_point == 0) {
      continue;
    }
    const int bpg = static_cast<int>(std::min<unsigned long long>((n_point + tpb - 1) / tpb, 1024));
    accumulate_burst_trigger_for_block<<<bpg, tpb>>>(
      field[range[0]].d_ptr, param, range[1], range[2], range[3], range[4], range[5], range[6], h2_species_index,
      burst_H, d_burst_quadrant_counts, d_burst_point_count, d_burst_metrics);
  }

  std::array<real, burst_metric_count> host_metrics{};
  cudaMemcpy(result.quadrant_counts.data(), d_burst_quadrant_counts, 8 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&result.point_count, d_burst_point_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_metrics.data(), d_burst_metrics, burst_metric_count * sizeof(real), cudaMemcpyDeviceToHost);
  result.sum_abs_flux = host_metrics[burst_sum_abs_flux];
  result.max_abs_flux = host_metrics[burst_max_abs_flux];
  result.sum_abs_vflux = host_metrics[burst_sum_abs_vflux];
  result.max_abs_vflux = host_metrics[burst_max_abs_vflux];
  result.max_abs_uY = host_metrics[burst_max_abs_uY];
  result.max_abs_vY = host_metrics[burst_max_abs_vY];
  return result;
}

void BlockMonitor::write_block_snapshot(const Parameter &parameter, const Field &f, int blk, real t, int step,
  bool is_burst, int burst_id, const std::string &quadrant_tag) const {
  const std::filesystem::path out_dir("output/monitor");
  const int myid{parameter.get_int("myid")};
  const auto bid = group_range[blk][0];
  const auto &range = group_range[blk];

  char file_name[1024];
  if (is_burst) {
    sprintf(file_name,
            "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-burst%s-B%06d-step%010d-data.bin",
            out_dir.string().c_str(), myid, bid, range[1], range[2], range[3], range[4], range[5], range[6],
            quadrant_tag.empty() ? "O0" : quadrant_tag.c_str(), burst_id, step);
  } else {
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-data.bin", out_dir.string().c_str(), myid, bid, range[1],
            range[2], range[3], range[4], range[5], range[6]);
  }

  const auto now_tp = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now_tp);
  const std::tm *tm = std::localtime(&now_c);
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp.time_since_epoch()) % 1000;
  char msbuf[8];
  std::snprintf(msbuf, sizeof(msbuf), "_%03d", static_cast<int>(ms.count()));
  std::string new_file_name = std::string(file_name) + "_" + std::string(buffer) + std::string(msbuf);
  if (std::filesystem::exists(file_name)) {
    std::filesystem::rename(file_name, new_file_name);
  }

  MPI_File fp;
  MPI_File_open(MPI_COMM_SELF, new_file_name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND,
                MPI_INFO_NULL, &fp);
  MPI_Status status;
  MPI_File_write(fp, &t, 1, MPI_DOUBLE, &status);
  const auto &ty1 = ty[blk];
  for (int l = 0; l < n_bv; ++l) {
    MPI_File_write(fp, f.bv[bv_label[l]], 1, ty1, &status);
  }
  for (int l = 0; l < n_sv; ++l) {
    MPI_File_write(fp, f.sv[sv_label[l]], 1, ty1, &status);
  }
  for (int l = 0; l < n_ov; ++l) {
    MPI_File_write(fp, f.ov[ov_label[l]], 1, ty1, &status);
  }
  MPI_File_close(&fp);
}

bool BlockMonitor::evaluate_burst(const Parameter &parameter, std::vector<Field> &field, real t, int step,
  int stat_count, DParameter *param) {
  bool need_local_output = false;
  burst_output_step = -1;
  burst_started_this_step = false;
  burst_output_quadrant_tag.clear();
  last_global_quadrant_counts.fill(0);

  for (int blk = 0; blk < n_block_mon; ++blk) {
    if (need_regular_output(blk, step)) {
      need_local_output = true;
      break;
    }
  }
  if (!burst_enabled) {
    return need_local_output;
  }

  bool just_deactivated = false;
  if (burst_state.active && step >= burst_state.end_step) {
    burst_state.active = false;
    burst_state.cooldown_end_step = step + burst_cooldown;
    just_deactivated = true;
  }
  if (burst_state.active) {
    if (step >= burst_state.next_output_step && step < burst_state.end_step) {
      burst_output_step = step;
      burst_output_quadrant_tag = burst_state.last_trigger_quadrant_tag;
      burst_state.next_output_step += burst_frequency;
      if (!burst_block_indices.empty()) {
        need_local_output = true;
      }
    }
    return need_local_output;
  }
  if (just_deactivated || !need_trigger_check(step)) {
    return need_local_output;
  }

  const auto local_trigger = check_local_burst_trigger(parameter, field, stat_count, param);
  std::array<int, 8> global_quadrant_counts{};
  std::array<real, 2> local_sum_metrics{local_trigger.sum_abs_flux, local_trigger.sum_abs_vflux};
  std::array<real, 2> global_sum_metrics{};
  std::array<real, 4> local_max_metrics{
    local_trigger.max_abs_flux, local_trigger.max_abs_vflux, local_trigger.max_abs_uY, local_trigger.max_abs_vY
  };
  std::array<real, 4> global_max_metrics{};
  unsigned long long global_point_count = 0;

  MPI_Allreduce(local_trigger.quadrant_counts.data(), global_quadrant_counts.data(), 8, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_trigger.point_count, &global_point_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_sum_metrics.data(), global_sum_metrics.data(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_max_metrics.data(), global_max_metrics.data(), 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  last_global_quadrant_counts = global_quadrant_counts;
  burst_state.last_global_strong_count = 0;
  std::string quadrant_tag;
  for (const int quadrant: burst_quadrants) {
    if (global_quadrant_counts[quadrant - 1] >= burst_min_count) {
      if (!quadrant_tag.empty()) {
        quadrant_tag += '-';
      }
      quadrant_tag += quadrant_name(quadrant);
      burst_state.last_global_strong_count = std::max(
        burst_state.last_global_strong_count, global_quadrant_counts[quadrant - 1]);
    }
  }
  if (quadrant_tag.empty()) {
    return need_local_output;
  }

  burst_state.active = true;
  burst_state.end_step = step + burst_duration;
  burst_state.cooldown_end_step = -1;
  burst_state.next_output_step = step + burst_frequency;
  burst_state.burst_id += 1;
  burst_state.last_trigger_quadrant_tag = quadrant_tag;
  burst_state.last_max_abs_uY = global_max_metrics[2];
  burst_state.last_max_abs_vY = global_max_metrics[3];
  burst_output_step = step;
  burst_started_this_step = true;
  burst_output_quadrant_tag = quadrant_tag;
  if (!burst_block_indices.empty()) {
    need_local_output = true;
  }

  if (parameter.get_int("myid") == 0) {
    std::ostringstream msg;
    msg << "=== BURST START === step=" << step << " time=" << t << " burst_id=" << burst_state.burst_id
        << " quadrants=" << quadrant_tag;
    for (const int quadrant: burst_quadrants) {
      if (global_quadrant_counts[quadrant - 1] >= burst_min_count) {
        msg << " global_count(" << quadrant_name(quadrant) << ")=" << global_quadrant_counts[quadrant - 1];
      }
    }
    printf("%s\n", msg.str().c_str());
    printf("Global burst diagnostics: max|u''Y_H2''|=%e max|v''Y_H2''|=%e\n", burst_state.last_max_abs_uY,
           burst_state.last_max_abs_vY);
    if (burst_use_abs_flux) {
      const real avg_abs_flux =
          global_point_count > 0 ? global_sum_metrics[0] / static_cast<real>(global_point_count) : 0.0;
      const real avg_abs_vflux =
          global_point_count > 0 ? global_sum_metrics[1] / static_cast<real>(global_point_count) : 0.0;
      printf(
        "Global burst flux diagnostics: avg|u''Y_H2''|_inst=%e max|u''Y_H2''|_inst=%e avg|v''Y_H2''|_inst=%e max|v''Y_H2''|_inst=%e\n",
        avg_abs_flux, global_max_metrics[0], avg_abs_vflux, global_max_metrics[1]);
    }
  }
  return need_local_output;
}

void BlockMonitor::output_data(const Parameter &parameter, std::vector<Field> &field, real t, int step) {
  if (n_block_mon <= 0) {
    return;
  }

  const bool write_burst = burst_enabled && burst_output_step == step && !burst_output_quadrant_tag.empty();
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];
    const auto &f = field[bid];
    if (need_regular_output(blk, step)) {
      write_block_snapshot(parameter, f, blk, t, step, false, 0, "");
    }
    if (write_burst && burst_block_flag[blk] != 0) {
      write_block_snapshot(parameter, f, blk, t, step, true, burst_state.burst_id, burst_output_quadrant_tag);
    }
  }
}

void BlockMonitor::stop_recording_blocks(const Parameter &parameter) const {
  if (n_block_mon <= 0)
    return;

  const std::filesystem::path out_dir("output/monitor");
  const int myid{parameter.get_int("myid")};
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];

    const int il{group_range[blk][1]}, ir{group_range[blk][2]};

    char file_name[1024];
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-data.bin", out_dir.string().c_str(), myid, bid, il, ir,
            group_range[blk][3], group_range[blk][4], group_range[blk][5], group_range[blk][6]);
    // Rename the file with the current date and time
    if (std::filesystem::exists(file_name)) {
      // The date should be in format YYYYMMDD_HHMMSS, and it should be in front of '.bin'
      // Get the current time
      auto now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      const std::tm *tm = std::localtime(&now_c);
      char buffer[64];
      std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm);

      std::string new_file_name = file_name;
      new_file_name += "_" + std::string(buffer);
      std::filesystem::rename(file_name, new_file_name);
    }
  }
}

void BlockMonitor::setup_labels_to_monitor(const Parameter &parameter) {
  const auto &var_names = parameter.VNs;

  const auto mon_var = parameter.get_string_array("monitor_block_var");
  std::vector<int> var_found;
  const int n_ps{parameter.get_int("n_ps")};
  for (auto name: mon_var) {
    name = gxl::to_upper(name);
    bool found{false};
    for (auto &[N, l]: var_names) {
      if (N == name) {
        var_found.push_back(l);
        found = true;
        break;
      }
    }
    if (!found) {
      if (n_ps > 0 && name == "PS") {
        for (int i = 0; i < n_ps; ++i) {
          var_found.push_back(100 + i);
        }
        found = true;
      }
      if (!found && parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  const auto ov_labels = parameter.get_int_array("ov_labels");
  for (auto l: var_found) {
    if (l < 6) {
      ++n_bv;
      bv_label.push_back(l);
    } else if (l >= 1000) {
      ++n_sv;
      sv_label.push_back(l - 1000);
    } else if (l >= 100) {
      ++n_sv;
      sv_label.push_back(l - 100);
    } else {
      for (int ll = 0; ll < ov_labels.size(); ++ll) {
        if (l == ov_labels[ll]) {
          ov_label.push_back(ll);
          ++n_ov;
          break;
        }
      }
    }
  }
  n_var = n_bv + n_sv + n_ov;
}
} // cfd

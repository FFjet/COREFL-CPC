#pragma once

#include <array>
#include <mpi.h>
#include <string>
#include "Parameter.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct Field;
class Mesh;
struct Species;
struct DParameter;

struct DeviceMonitorData {
  int n_bv{0}, n_sv{0}, n_var{0};
  int *bv_label{nullptr};
  int *sv_label{nullptr};
  int *bs_d{nullptr}, *is_d{nullptr}, *js_d{nullptr}, *ks_d{nullptr};
  int *disp{nullptr}, *n_point{nullptr};
  ggxl::Array3D<real> data;
};

struct GlobalBurstState {
  bool active{false};
  int end_step{-1};
  int cooldown_end_step{-1};
  int next_output_step{-1};
  int burst_id{0};
  int last_global_strong_count{0};
  real last_max_abs_uY{0};
  real last_max_abs_vY{0};
  std::string last_trigger_quadrant_tag;
};

struct BlockBurstTriggerResult {
  std::array<int, 8> quadrant_counts{};
  unsigned long long point_count{0};
  real max_abs_uY{0};
  real max_abs_vY{0};
  real sum_abs_flux{0};
  real max_abs_flux{0};
  real sum_abs_vflux{0};
  real max_abs_vflux{0};
};

class BlockMonitor {
public:
  explicit BlockMonitor() = default;

  void initialize(Parameter &parameter, const Mesh &mesh_);

  explicit BlockMonitor(Parameter &parameter, const Mesh &mesh_);

  void configure_burst(const Parameter &parameter, const Species &species);

  [[nodiscard]] bool need_service(int step) const;

  [[nodiscard]] bool evaluate_burst(
    const Parameter &parameter, std::vector<Field> &field, real t, int step, int stat_count, DParameter *param);

  void output_data(const Parameter &parameter, std::vector<Field> &field, real t, int step);

  void stop_recording_blocks(const Parameter &parameter) const;

private:
  int n_bv{0}, n_sv{0}, n_ov{0}, n_var{0};
  std::vector<int> bv_label, sv_label, ov_label;

  int n_block_mon{0};
  int n_group{0};
  std::vector<std::string> group_name;
  std::vector<std::array<int, 7>> group_range;
  std::vector<int> frequency;
  std::vector<int> burst_block_flag;
  std::vector<int> burst_block_indices;
  std::vector<int> burst_unique_block_ids;
  bool burst_enabled{false};
  bool burst_use_abs_flux{false};
  int burst_frequency{1};
  int burst_check_frequency{1};
  int burst_duration{0};
  int burst_cooldown{0};
  int burst_min_count{1};
  int global_burst_block_count{0};
  int h2_species_index{-1};
  int fav2_scalar_var_offset{7};
  int fav2_scalar_flux_u_offset{7};
  int fav2_scalar_flux_v_offset{7};
  real burst_H{2};
  std::vector<int> burst_quadrants;
  GlobalBurstState burst_state;
  int burst_output_step{-1};
  bool burst_started_this_step{false};
  std::string burst_output_quadrant_tag;
  std::array<int, 8> last_global_quadrant_counts{};
  int *d_burst_quadrant_counts{nullptr};
  unsigned long long *d_burst_point_count{nullptr};
  real *d_burst_metrics{nullptr};

  MPI_Datatype *ty{nullptr};

  // Utility functions
  void setup_labels_to_monitor(const Parameter &parameter);
  [[nodiscard]] bool need_regular_output(int blk, int step) const;
  [[nodiscard]] bool need_trigger_check(int step) const;
  [[nodiscard]] BlockBurstTriggerResult check_local_burst_trigger(
    const Parameter &parameter, std::vector<Field> &field, int stat_count, DParameter *param) const;
  void write_block_snapshot(const Parameter &parameter, const Field &field, int blk, real t, int step, bool is_burst,
    int burst_id, const std::string &quadrant_tag) const;
  [[nodiscard]] static bool is_target_quadrant(int quadrant, real up, real vp, real yp);
  [[nodiscard]] static const char *quadrant_name(int quadrant);
};

class Monitor {
public:
  explicit Monitor(Parameter &parameter, const Species &species, const Mesh &mesh_);

  explicit Monitor(const Parameter &parameter, const Mesh &mesh_);

  void initialize(Parameter &parameter, const Species &species);

  void monitor_point(int step, real physical_time, const std::vector<Field> &field);

  void output_point_monitors();

  [[nodiscard]] bool need_block_monitor_service(int step) const;

  [[nodiscard]] bool evaluate_block_burst(
    const Parameter &parameter, std::vector<Field> &field, real t, int step, int stat_count, DParameter *param);

  void output_block_monitors(const Parameter &parameter, std::vector<Field> &field, real t, int step);

  void stop_recording_blocks(const Parameter &parameter) const;

  ~Monitor();

private:
  BlockMonitor block_monitor{};
  int output_file{0};
  int step_start{0};
  int counter_step{0};
  int n_block{0};
  int n_bv{0}, n_sv{0}, n_var{0};
  std::vector<int> bs_h, is_h, js_h, ks_h;
  int n_point_total{0};
  std::vector<int> n_point;
  std::vector<int> disp;
  ggxl::Array3DHost<real> mon_var_h;
  DeviceMonitorData *h_ptr{nullptr}, *d_ptr{nullptr};
  std::vector<FILE *> files;

  const Mesh &mesh;

private:
  // Utility functions
  std::vector<std::string> setup_labels_to_monitor(const Parameter &parameter, const Species &species);
};

struct DZone;
__global__ void record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos,
  real physical_time);
} // cfd

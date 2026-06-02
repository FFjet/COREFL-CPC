#pragma once

#include "Define.h"
#include <vector>

namespace cfd {
struct DParameter;
struct Parameter;
struct Species;

struct MlMlpHostModel {
  int input_count = 0;
  int hidden_count = 0;
  int hidden_layers = 0;
  int output_count = 0;
  float output_log_epsilon = 0.0f;
  bool output_log_standard = false;
  std::vector<float> weights;
  std::vector<float> biases;
  std::vector<float> input_mins;
  std::vector<float> input_scales;
  std::vector<float> output_mins;
  std::vector<float> output_scales;
};

struct MlTransportHostData {
  MlMlpHostModel transport;
  MlMlpHostModel diffusion;
};

MlTransportHostData load_ml_transport_models(const Parameter &parameter, const Species &species);

void upload_ml_transport_models(DParameter &d_param, const MlTransportHostData &models);
} // namespace cfd

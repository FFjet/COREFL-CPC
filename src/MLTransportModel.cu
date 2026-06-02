#include "MLTransportModel.h"
#include "DParameter.cuh"
#include "Parameter.h"
#include "ChemData.h"
#include "Parallel.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open ML model header: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

int parse_uint_constant(const std::string &text, const char *name) {
  const std::regex pattern(std::string(name) + R"(\s*=\s*([0-9]+))");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error(std::string("Missing ML model integer constant: ") + name);
  }
  return std::stoi(match[1].str());
}

double parse_double_constant(const std::string &text, const char *name) {
  const std::regex pattern(std::string(name) + R"(\s*=\s*([-+0-9.eE]+))");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error(std::string("Missing ML model real constant: ") + name);
  }
  return std::stod(match[1].str());
}

std::string parse_string_constant(const std::string &text, const char *name) {
  const std::regex pattern(std::string(name) + "\\s*=\\s*\"([^\"]+)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error(std::string("Missing ML model string constant: ") + name);
  }
  return match[1].str();
}

std::vector<float> parse_numeric_array(const std::string &text, const char *function_name) {
  const std::string needle = std::string(function_name) + "()";
  const auto function_pos = text.find(needle);
  if (function_pos == std::string::npos) {
    throw std::runtime_error(std::string("Missing ML model array function: ") + function_name);
  }
  const auto data_pos = text.find("data = {", function_pos);
  if (data_pos == std::string::npos) {
    throw std::runtime_error(std::string("Missing ML model array data block: ") + function_name);
  }
  const auto begin = data_pos + std::string("data = {").size();
  const auto end = text.find("};", begin);
  if (end == std::string::npos) {
    throw std::runtime_error(std::string("Unterminated ML model array data block: ") + function_name);
  }

  const std::string body = text.substr(begin, end - begin);
  const std::regex number_pattern(R"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)");
  std::vector<float> values;
  for (std::sregex_iterator it(body.begin(), body.end(), number_pattern), last; it != last; ++it) {
    values.push_back(static_cast<float>(std::stod((*it).str())));
  }
  return values;
}

std::vector<std::string> parse_species_names(const std::string &text) {
  const std::string needle = "speciesNames()";
  const auto function_pos = text.find(needle);
  if (function_pos == std::string::npos) {
    throw std::runtime_error("Missing ML model speciesNames() function");
  }
  const auto data_pos = text.find("data = {", function_pos);
  const auto begin = data_pos + std::string("data = {").size();
  const auto end = text.find("};", begin);
  const std::string body = text.substr(begin, end - begin);
  const std::regex name_pattern("\"([^\"]+)\"");
  std::vector<std::string> names;
  for (std::sregex_iterator it(body.begin(), body.end(), name_pattern), last; it != last; ++it) {
    std::string name = (*it)[1].str();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    names.push_back(name);
  }
  return names;
}

cfd::MlMlpHostModel parse_model_header(const std::filesystem::path &path, int expected_outputs,
  bool require_log_standard) {
  const std::string text = read_text_file(path);
  cfd::MlMlpHostModel model;
  model.input_count = parse_uint_constant(text, "inputCount");
  model.hidden_count = parse_uint_constant(text, "hiddenCount");
  model.hidden_layers = parse_uint_constant(text, "hiddenLayers");
  model.output_count = parse_uint_constant(text, "outputCount");
  model.output_log_epsilon = static_cast<float>(parse_double_constant(text, "outputLogEpsilon"));
  model.output_log_standard = parse_string_constant(text, "outputScalerKind") == "log_standard";
  model.weights = parse_numeric_array(text, "weights");
  model.biases = parse_numeric_array(text, "biases");
  model.input_mins = parse_numeric_array(text, "inputMins");
  model.input_scales = parse_numeric_array(text, "inputScales");
  model.output_mins = parse_numeric_array(text, "outputMins");
  model.output_scales = parse_numeric_array(text, "outputScales");

  const int expected_weight_count =
      model.input_count * model.hidden_count +
      std::max(0, model.hidden_layers - 1) * model.hidden_count * model.hidden_count +
      model.hidden_count * model.output_count;
  const int expected_bias_count = model.hidden_layers * model.hidden_count + model.output_count;
  if (model.output_count != expected_outputs || static_cast<int>(model.weights.size()) != expected_weight_count ||
      static_cast<int>(model.biases.size()) != expected_bias_count ||
      static_cast<int>(model.input_mins.size()) != model.input_count ||
      static_cast<int>(model.input_scales.size()) != model.input_count ||
      static_cast<int>(model.output_mins.size()) != model.output_count ||
      static_cast<int>(model.output_scales.size()) != model.output_count ||
      model.output_log_standard != require_log_standard) {
    throw std::runtime_error("ML model header has unexpected dimensions or scaler kind: " + path.string());
  }
  if (model.input_count > 8 || model.hidden_count > 32) {
    throw std::runtime_error("ML model is larger than the CUDA fixed work arrays support: " + path.string());
  }

  return model;
}

void validate_species_order(const std::vector<std::string> &model_species, const cfd::Species &species,
  const std::filesystem::path &path) {
  if (species.n_spec != 5 || static_cast<int>(model_species.size()) != species.n_spec) {
    throw std::runtime_error("Air-5 ML transport requires exactly 5 species: " + path.string());
  }
  for (int l = 0; l < species.n_spec; ++l) {
    std::string spec_name = species.spec_name[l];
    std::transform(spec_name.begin(), spec_name.end(), spec_name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (spec_name != model_species[l]) {
      throw std::runtime_error("ML model species order must match mechanism order: " + path.string());
    }
  }
}

void copy_float_vector_to_device(const std::vector<float> &host, const float **device) {
  float *ptr = nullptr;
  cudaMalloc(&ptr, host.size() * sizeof(float));
  cudaMemcpy(ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  *device = ptr;
}

void upload_model(cfd::MlMlpHostModel const &model, int &input_count, int &hidden_count, int &hidden_layers,
  int &output_count, float &output_log_epsilon, bool &output_log_standard, const float **weights, const float **biases,
  const float **input_mins, const float **input_scales, const float **output_mins, const float **output_scales) {
  input_count = model.input_count;
  hidden_count = model.hidden_count;
  hidden_layers = model.hidden_layers;
  output_count = model.output_count;
  output_log_epsilon = model.output_log_epsilon;
  output_log_standard = model.output_log_standard;
  copy_float_vector_to_device(model.weights, weights);
  copy_float_vector_to_device(model.biases, biases);
  copy_float_vector_to_device(model.input_mins, input_mins);
  copy_float_vector_to_device(model.input_scales, input_scales);
  copy_float_vector_to_device(model.output_mins, output_mins);
  copy_float_vector_to_device(model.output_scales, output_scales);
}
} // namespace

cfd::MlTransportHostData cfd::load_ml_transport_models(const Parameter &parameter, const Species &species) {
  MlTransportHostData data;
  if constexpr (!kUseMlTransport) {
    return data;
  } else {
    const std::filesystem::path model_dir{"input/mlmodel"};
    const auto transport_path = model_dir / "nnWeights_air5_transport_compact_mlp_sub2000.H";
    const auto diffusion_path = model_dir / "nnWeights_air5_diffusion_compact_mlp_sub3000.H";

    try {
      const std::string transport_text = read_text_file(transport_path);
      validate_species_order(parse_species_names(transport_text), species, transport_path);
      const std::string diffusion_text = read_text_file(diffusion_path);
      validate_species_order(parse_species_names(diffusion_text), species, diffusion_path);

      data.transport = parse_model_header(transport_path, 3, false);
      data.diffusion = parse_model_header(diffusion_path, species.n_spec, true);
      if (parameter.get_int("myid") == 0) {
        printf("\t->-> %-20s : ML transport model loaded\n", transport_path.string().c_str());
        printf("\t->-> %-20s : ML diffusion model loaded\n", diffusion_path.string().c_str());
      }
    } catch (const std::exception &ex) {
      if (parameter.get_int("myid") == 0) {
        printf("Failed to load ML transport models: %s\n", ex.what());
      }
      MpiParallel::exit();
    }
    return data;
  }
}

void cfd::upload_ml_transport_models(DParameter &d_param, const MlTransportHostData &models) {
  if constexpr (!kUseMlTransport) {
    (void)d_param;
    (void)models;
  } else {
    upload_model(models.transport, d_param.ml_transport_input_count, d_param.ml_transport_hidden_count,
                 d_param.ml_transport_hidden_layers, d_param.ml_transport_output_count,
                 d_param.ml_transport_output_log_epsilon, d_param.ml_transport_output_log_standard,
                 &d_param.ml_transport_weights, &d_param.ml_transport_biases, &d_param.ml_transport_input_mins,
                 &d_param.ml_transport_input_scales, &d_param.ml_transport_output_mins,
                 &d_param.ml_transport_output_scales);

    upload_model(models.diffusion, d_param.ml_diffusion_input_count, d_param.ml_diffusion_hidden_count,
                 d_param.ml_diffusion_hidden_layers, d_param.ml_diffusion_output_count,
                 d_param.ml_diffusion_output_log_epsilon, d_param.ml_diffusion_output_log_standard,
                 &d_param.ml_diffusion_weights, &d_param.ml_diffusion_biases, &d_param.ml_diffusion_input_mins,
                 &d_param.ml_diffusion_input_scales, &d_param.ml_diffusion_output_mins,
                 &d_param.ml_diffusion_output_scales);
  }
}

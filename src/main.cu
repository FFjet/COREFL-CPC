/**
 * @file main.cu
 * @brief Main function for the DNS program.
 * @details This program is designed to run a CFD simulation for various mixture types, including Air as calorically perfect gas, and thermally perfect gas mixture.
 */
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"
#include "Simulate.cuh"
#include "ConstVolumeReactor.h"
#include "PostProcess.h"
#include <filesystem>

namespace {
namespace fs = std::filesystem;

void clear_wall_output() {
  std::error_code ec;
  fs::remove_all("output/wall", ec);
}

void move_wall_output_to(const fs::path &target) {
  std::error_code ec;
  fs::remove_all(target, ec);
  fs::create_directories(target.parent_path(), ec);
  fs::rename("output/wall", target, ec);
}

template<MixtureModel mix_model>
void run_driver(cfd::Parameter &parameter, cfd::Mesh &mesh) {
  cfd::Driver<mix_model> driver(parameter, mesh);
  driver.initialize_computation();
  const auto &post_process_files = parameter.get_string_array("post_process_flowfield_files");
  if (!post_process_files.empty()) {
    const fs::path post_dir{parameter.get_string("post_process_output_dir")};
    for (int i = 0; i < static_cast<int>(post_process_files.size()); ++i) {
      if (driver.myid == 0) {
        clear_wall_output();
      }
      MPI_Barrier(MPI_COMM_WORLD);
      driver.reload_flowfield(post_process_files[i], false);
      cfd::post_process(driver.mesh, driver.field, driver.parameter, driver.param);
      MPI_Barrier(MPI_COMM_WORLD);
      if (driver.myid == 0) {
        move_wall_output_to(post_dir / ("snapshot_" + std::to_string(i)));
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (driver.myid == 0) {
        printf("\tPost-processed snapshot %d/%d: %s\n", i + 1, static_cast<int>(post_process_files.size()),
               post_process_files[i].c_str());
      }
    }
  } else if (parameter.get_bool("post_process_only")) {
    cfd::post_process(driver.mesh, driver.field, driver.parameter, driver.param);
  } else {
    simulate(driver);
  }
  driver.deallocate();
}
}

int main(int argc, char *argv[]) {
  cfd::Parameter parameter(&argc, &argv);

  if (parameter.get_string("canonical_problem") == "constVolumeReactor") {
    cfd::const_volume_reactor(parameter);
    return 0;
  }

  cfd::Mesh mesh(parameter);

  const int species = parameter.get_int("species");
  if (const bool turbulent_laminar = parameter.get_bool("turbulence"); !turbulent_laminar) {
    parameter.update_parameter("turbulence_method", 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (species == 1) {
    // Multiple species
    // Laminar & DNS
    run_driver<MixtureModel::Mixture>(parameter, mesh);
  } else {
    if constexpr (cfd::kTwoTemperature) {
      if (parameter.get_int("myid") == 0) {
        printf("Two-temperature mode is only available for species-based mixture simulations.\n");
      }
      MPI_Finalize();
      return 1;
    }
    // Air simulation
    // Laminar and air
    run_driver<MixtureModel::Air>(parameter, mesh);
  }

  if (parameter.get_int("myid") == 0) {
    printf("Yeah, baby, we are ok now\n");
    std::ofstream out("Man, we are Finished.txt");
    out << "Yeah, baby, we are ok now\n";
    out.close();
  }
  MPI_Finalize();
  return 0;
}

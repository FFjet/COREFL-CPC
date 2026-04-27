/**
 * @brief The file for declarations of post process procedures
 * @details Current available procedures:
 *    0 - compute wall friction and heat flux in 2D, which assumes a j=0 wall
 *    1 - compute wall friction and heat flux in 3D, which assumes a j=0 wall
 */
#pragma once

#include "Define.h"
#include "Field.h"
#include "Parameter.h"

#include <vector>

namespace cfd {
class Mesh;

struct DZone;
struct DParameter;

void post_process(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, DParameter *param);

// Compute the wall friction and heat flux in 2D. Assume the wall is the j=0 plane
// Procedure 0
void wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter,
                               const DParameter *param);

__global__ void wall_friction_heatFlux_2d(DZone *zone, real *wall_data, const DParameter *param, real dyn_pressure);

// Compute the wall friction and heat flux in 3D. Assume the wall is the j=0 plane
// Procedure 1
void wall_friction_heatFlux_3d(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, DParameter *param);

__global__ void wall_friction_heatFlux_3d(DZone *zone, ggxl::VectorField2D<real> *cfQw, const DParameter *param, bool stat_on,
                                          bool spanwise_ave);
}

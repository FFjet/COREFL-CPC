# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should not beyond 55, which is set by allowing the computation of CH4 in Grimech mechanism.
#     If larger mechanism is to be used, the variable should be modified to be larger or equal to the species number.
#     However, the correctness, if the resource is available for the corresponding kernel may need to be tested.

list(APPEND src_file
    BoundCond.h BoundCond.cpp BoundCond.cu BoundCond.cuh BoundaryIO.h
    ChemData.h ChemData.cpp Constants.h ConstVolumeReactor.cpp ConstVolumeReactor.h
    DataCommunication.cuh DataCommunication.cu Define.h DParameter.cu DParameter.cuh DPLUR.cuh DPLUR.cu Driver.cu Driver.cuh DualTimeStepping.cuh DualTimeStepping.cu
    Element.h Element.cpp
    Field.h Field.cu FieldOperation.cuh FieldOperation.cu FiniteRateChem.cu FiniteRateChem.cuh FieldIO.h Fluctuation.cu Fluctuation.cuh
    ImplicitTreatmentHPP.cuh Initialize.cuh Initialize.cu InviscidScheme.cuh InviscidScheme.cu IOManager.h
    kernels.cuh kernels.cu kernels.h
    Limiter.cuh Mesh.cu Mesh.cuh
    main.cu Mesh.h Mesh.cpp MixtureFraction.cpp MixtureFraction.h Monitor.cuh Monitor.cu MixingLayer.cuh MixingLayer.cu
    Parallel.h Parallel.cpp Parameter.h Parameter.cpp PostProcess.h PostProcess.cu
    Reconstruction.cu Reconstruction.cuh Residual.cu Residual.cuh RK.cuh RiemannSolver.cuh
    SteadySim.cuh Simulate.cuh
    SpongeLayer.cu SpongeLayer.cuh SinglePointStat.cu SinglePointStat.cuh StrangSplitting.cuh StrangSplitting.cu
    Transport.cuh Transport.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Thermo.cuh Thermo.cu
    DigitalFilter.cu DigitalFilter.cuh
    UDIO.h UDIO.cu
    ViscousScheme.cuh ViscousScheme.cu
    WallDistance.cuh WENO.cu

    stat_lib/TkeBudget.cu stat_lib/TkeBudget.cuh stat_lib/SpeciesStat.cu stat_lib/SpeciesStat.cuh

    gxl_lib/Array.hpp gxl_lib/Array.cuh
    gxl_lib/MyString.cpp gxl_lib/MyAlgorithm.cpp gxl_lib/MyAlgorithm.h gxl_lib/Math.cuh
    gxl_lib/Time.cpp gxl_lib/Time.h
    gxl_lib/MyAtomic.cuh
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_policy(SET CMP0074 NEW)
find_package(MPI QUIET)

add_executable(corefl ${src_file})

set(COREFL_MPI_INCLUDE_DIRS
    ${MPI_CXX_HEADER_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${MPI_CXX_COMPILER_INCLUDE_DIRS}
    ${MPI_CXX_ADDITIONAL_INCLUDE_DIRS}
)

list(REMOVE_DUPLICATES COREFL_MPI_INCLUDE_DIRS)

set(COREFL_MPI_LINK_LIBRARIES)
set(COREFL_MPI_LINK_OPTIONS)
set(COREFL_MPI_FORCE_INCLUDE_DIRS)

if(MPI_FOUND)
    list(APPEND COREFL_MPI_LINK_LIBRARIES MPI::MPI_CXX)
else()
    if(DEFINED ENV{OPAL_PREFIX} AND EXISTS "$ENV{OPAL_PREFIX}/include/mpi.h")
        set(COREFL_MPI_ROOT "$ENV{OPAL_PREFIX}")
    elseif(EXISTS "/opt/dmx-hpcx/hpcx/ompi/include/mpi.h")
        set(COREFL_MPI_ROOT "/opt/dmx-hpcx/hpcx/ompi")
    endif()

    if(COREFL_MPI_ROOT)
        list(APPEND COREFL_MPI_INCLUDE_DIRS
            "${COREFL_MPI_ROOT}/include"
            "${COREFL_MPI_ROOT}/include/openmpi"
            "${COREFL_MPI_ROOT}/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include"
            "${COREFL_MPI_ROOT}/include/openmpi/opal/mca/event/libevent2022/libevent"
            "${COREFL_MPI_ROOT}/include/openmpi/opal/mca/event/libevent2022/libevent/include"
        )
        list(APPEND COREFL_MPI_LINK_LIBRARIES "${COREFL_MPI_ROOT}/lib/libmpi.so")
        list(APPEND COREFL_MPI_LINK_OPTIONS
            "-Wl,-rpath,${COREFL_MPI_ROOT}/lib"
            "-Wl,--enable-new-dtags"
        )
        message(WARNING "MPI wrapper detection failed; falling back to manual MPI settings from ${COREFL_MPI_ROOT}.")
    else()
        message(FATAL_ERROR "MPI was not found and no manual fallback path was detected. Set OPAL_PREFIX or install a working MPI toolchain.")
    endif()
endif()

if(MPI_CXX_HEADER_DIR)
    list(APPEND COREFL_MPI_FORCE_INCLUDE_DIRS "${MPI_CXX_HEADER_DIR}")
endif()
if(COREFL_MPI_ROOT)
    list(APPEND COREFL_MPI_FORCE_INCLUDE_DIRS "${COREFL_MPI_ROOT}/include")
endif()

list(REMOVE_DUPLICATES COREFL_MPI_INCLUDE_DIRS)
list(REMOVE_DUPLICATES COREFL_MPI_FORCE_INCLUDE_DIRS)

if(COREFL_MPI_INCLUDE_DIRS)
    target_include_directories(corefl PRIVATE ${COREFL_MPI_INCLUDE_DIRS})
endif()

foreach(COREFL_MPI_FORCE_INCLUDE_DIR IN LISTS COREFL_MPI_FORCE_INCLUDE_DIRS)
    if(EXISTS "${COREFL_MPI_FORCE_INCLUDE_DIR}/mpi.h")
        target_compile_options(corefl PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-I${COREFL_MPI_FORCE_INCLUDE_DIR}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-I${COREFL_MPI_FORCE_INCLUDE_DIR}>"
        )
    endif()
endforeach()

if(COREFL_MPI_LINK_OPTIONS)
    target_link_options(corefl PRIVATE ${COREFL_MPI_LINK_OPTIONS})
endif()

set_target_properties(corefl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    #INTERPROCEDURAL_OPTIMIZATION TRUE
)

set_target_properties(corefl
    PROPERTIES
    CXX_STANDARD 20 # A newer version CMake 3.25.2 would support this
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    #  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
)

#target_include_directories(corefl
#    PUBLIC
#    ${CMAKE_SOURCE_DIR}/depends/include/
#)

#target_link_directories(corefl
#    PUBLIC#PRIVATE
#    "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
#    "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
#)

#target_compile_options(2_CLionGPU
#  PUBLIC#private
#  ${flags}
#  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
#  )

target_link_libraries(corefl
    PUBLIC#PRIVATE
#    "$<$<CONFIG:Debug>:fmtd>"
#    "$<$<CONFIG:Release>:fmt>"
    ${COREFL_MPI_LINK_LIBRARIES}
)

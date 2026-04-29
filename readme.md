# COREFL: A compressible reactive flow solver on generalized curvilinear coordinates

COREFL performs direct numerical simulations of compressible reactive flows on GPU based on finite difference method.

The docs are under construction at https://corefl.readthedocs.io/en/latest/, and the current readme is just a brief introduction to the code, including the environment requirements, compilation, and running. A simple example case is also included to illustrate the usage of the code.

## Environment requirements

- **CUDA compiler**: supporting C++17 (nvcc > 11.0)
- **C++ compiler**: supporting C++20
- **MPI library**: supporting CUDA-aware MPI (E.g., OpenMPI > 1.8)
- **CMake**: supporting CUDA language

comments on the environments:

- MPI: Only a few vendors' MPI support CUDA-aware MPI, and only on Linux systems. Therefore, only Linux system supports the parallel running of COREFL. However, any MPI version supports the compilation and running in serial modes.

COREFL has been compiled and tested on Nvidia A100 GPU.
The most frequently used configuration by us on A100 is given for reference:
**CUDA 11.8 / gcc 11.3 / openmpi 4.1.5 / cmake 3.26.3**

## Compilation

All compilations are performed on **Linux** system. For Windows system, we successfully built the code with Visual Studio 2022 and CUDA, Microsoft MPI. You can also build it with CLion, but the toolchain must be the msvc instead of mingw. I would not include that here because large scale computations are always performed on Linux clusters.

The structure of the current COREFL folder is:

- docs/*
- example/*
- src/
  - gxl_lib/*
  - stat_lib/*
  - *.cpp/.h/.cuh/.cu/.hpp
- tools/*
- CMakeLists.txt

The folder contains the COREFL codes, documentations, example cases, and some tools can be used. The concrete usage of them will be discussed throughout the documentation.

### The compilation of COREFL

The compilation consists of the following steps:

1. Navigate to the COREFL folder (the "`code`" folder here).
2. Modify the CMakeLists.txt:
   1. Modify the number in `set(CMAKE_CUDA_ARCHITECTURES 60)` according to the GPU compute capability. For example, this number is 60 for P100, 70 for V100, and 80 for A100.
   2. Modify `add_compile_definitions(MAX_SPEC_NUMBER=9)` according to the problem. The number should be larger than or equal to the species number to be used in computations. If no species is included, set it to 1.
   3. Modify `add_compile_definitions(MAX_REAC_NUMBER=19)` according to the problem. The number should be larger than or equal to the reaction number to be used in computations. If no reaction is included, set it to 1.
   4. Modify `add_compile_definitions(Combustion2Part)` according to the problem. If a general chemical reaction case is considered (combustion), set it to `Combustion2Part`. If a high-temerature air chemistry is to be simulated, set to `HighTempMultiPart`. Only the 5 species 6 reactions mechanism of air is supported currently when using `HighTempMultiPart`.
3. Load the compilation environment. For example, `module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1 cmake/3.26.3`
4. `cmake -Bbuild -DCMAKE_BUILD_TYPE=Release`
5. `cmake --build build --parallel 16`

The executable COREFL should appear in ths current folder.

### Two-temperature hypersonic mode

The two-temperature implementation is selected at compile time. Configure with `COREFL_ENABLE_TWO_TEMPERATURE=ON`:

```bash
cmake -S . -B build-2t -DCMAKE_BUILD_TYPE=Release -DCOREFL_ENABLE_TWO_TEMPERATURE=ON
cmake --build build-2t --parallel 16
cp corefl corefl-2t
```

The top-level `CMakeLists.txt` currently defaults to `MAX_SPEC_NUMBER=9`, `MAX_REAC_NUMBER=19`, `MAX_PASSIVE_SCALAR_NUMBER=1`, NASA9-capable thermodynamics, and `Combustion2Part`. These defaults match the provided AIR5 two-temperature example. Increase `MAX_SPEC_NUMBER` or `MAX_REAC_NUMBER` before configuring if the mechanism is larger.

Runtime constraints for the two-temperature path are enforced in `src/Parameter.cpp`:

```text
bool steady = 0
int temporal_scheme = 3
int species = 1
int reaction = 1
```

The flamelet path, steady update, dual-time stepping, and Wu splitting are intentionally blocked for now because `rhoEve` chemistry/VT relaxation is not wired into those update paths. In the enabled RK3 path, the chemical `rhoEve` source is explicit.

The added conservative variable is `rhoEve`, stored as scalar `Eve` after the species mass fractions. The output variable `Tve` is also available. The implemented frozen two-temperature energy split is:

```text
E = 0.5 * (u^2 + v^2 + w^2) + e_tr(T,Y) + Eve
h_tr,s(T) = h_eq,s(T) - E_ve,s(T)
F_Eve^inv = rho * Eve * U_k
S_Eve = theta_tr:ve + sum_s omega_s * E_ve,s(Tve)
```

The viscous sign convention follows the existing species equation implementation. The physical species diffusion flux is `J_s`, so the species equation contains `div(-J_s)`. COREFL stores the species viscous flux as:

```text
D_s^code = -J_s
```

This is visible in `src/ViscousScheme.cu`: the stored species flux is proportional to `+rhoD_s grad(Y_s)` in the simple Fick limit, and the viscous residual adds the divergence of the stored flux. With that convention, the two-temperature viscous fluxes implemented in the solver are:

```text
F_Eve^vis = kappa_ve * grad(Tve)
           + sum_s D_s^code * E_ve,s(Tve)

F_E^vis = u dot tau + kappa_tr * grad(T) + kappa_ve * grad(Tve)
          + sum_s D_s^code * (h_tr,s(T) + E_ve,s(Tve))
```

The nonequilibrium diffusion enthalpy helper in `src/Thermo.cuh` computes `h_eq,s(T) - E_ve,s(T) + E_ve,s(Tve)`, which is exactly `h_tr,s(T) + E_ve,s(Tve)`. The 2nd-order face kernels and the 8th-order collocated kernels in `src/ViscousScheme.cu` use this same expression for total energy. The 8th-order scalar kernel also explicitly writes the `rhoEve` viscous flux into `fFlux/gFlux/hFlux` before the 8th-order derivative is taken, so the `rhoEve` derivative does not read an old or unset flux value.

The wall heat-flux post-processing in `src/PostProcess.cu` uses the same split:

```text
q_tr = kappa_tr * n dot grad(T)
       + sum_s D_s^code * h_tr,s(T)

q_ve = kappa_ve * n dot grad(Tve)
       + sum_s D_s^code * E_ve,s(Tve)

q_total = q_tr + q_ve
```

A useful consistency check is not to add `q_ve` to a total-energy diffusion term that already used `h_tr,s(T) + E_ve,s(Tve)` for species diffusion; doing so would count `sum_s D_s^code * E_ve,s(Tve)` twice. `src/PostProcess.cu` keeps `q_tr` and `q_ve` separate first, then forms `q_total`.

The main implementation files are:

- `src/Define.h`, `CMakeLists.txt`: compile-time `kTwoTemperature` switch.
- `src/Parameter.cpp`, `src/DParameter.*`, `src/Field.*`: variable indexing, `Eve`, `Tve`, and storage.
- `src/ChemData.*`, `src/Thermo.*`: NASA thermodynamics split into translational-rotational and vibrational-electronic parts, `two_temperature.dat`, and Landau-Teller data.
- `src/FieldOperation.*`: conservative/primitive conversion, `Tve` inversion, total-energy closure, and exact exponential VT update.
- `src/FiniteRateChem.cu`: `sum_s omega_s E_ve,s(Tve)` source and two-temperature reaction-control temperatures.
- `src/InviscidScheme.cu`, `src/WENO.cu`: convective `rhoEve` flux and characteristic WENO projection.
- `src/ViscousScheme.cu`: `Tve` conduction, `rhoEve` viscous flux, and nonequilibrium enthalpy in total-energy diffusion.
- `src/PostProcess.cu`: wall `q_tr`, `q_ve`, and `q_total` split.
- `src/RK.cuh`: RK3-stage VT relaxation.

The detailed characteristic derivation and equation-to-code audit are in `docs/two_temperature_weno_characteristic_derivation.md`.

### The use of readGrid

COREFL reads the structured multiblock grid files in `Plot3D` format. We can not partition the blocks automatically. Therefore, if you want a parallel computation with multiple blocks, you need to partition the blocks manually.

The grid files read by COREFL is not the `Plot3D` file outputted directly from softwares, but the ones treated by our **readGrid** tool. 

The **readGrid** tool will read the "`gridFile.dat`" and "`gridFile.inp`", where "`gridFile`" is the filename, and output two folders `grid` and `boundary_condition`, which should be moved to the `input` folder and read by COREFL.

The advantage of using such a file is that the corresponding processes need only read their own blocks instead of waiting for asigning. But this may be improved or integrated in COREFL in the future.

This part is in the folder `readGrid`, and the compilation is straightforward and omitted here. This code can be compiled on Windows systems. A pre-compiled executable is also given in the folder if anyone uses this tool on Windows before uploading the grid to linux clusters (which is what we do).

## Running

### Grid generation

The grid for COREFL is structured grid in `Plot3D` format. The `Plot3D` files including `name.dat` and `name.inp` contain the grid coordinates and boundary condition information, respectively.

> You can generate the grid with a commercial software supporting `Plot3D` or by writing codes. For example, **Pointwise** and **Gridgen** both support this file format. In **Pointwise**, you choose the solver "Gridgen Generic", and by exporting CAE, you get those two files.

> Note that, we do not have automatic blocking in COREFL. You need to partition the blocks manually when generating grid in softwares.

With the two files, we need a tool called **readGrid** to convert the `Plot3D` files into two folders that COREFL reads. The **readGrid** tool is also included in this folder, and any C++ compiler supporting C++20 works.

The tool would output two folders called `grid` and `boundary_condition`, these should be put into the `input` folder in the work directory.

### Run the CFD code

A typical folder for running COREFL is as follows:
- input/
  - grid/*
  - boundary_condition/*
  - chemistry/*
  - setup.txt
- run.sh
- corefl(optional)

All settings are set in the `setup.txt` file in the following manner:

`type name = value`

There should be space between each two symbols.

As we always run COREFL on clusters, we need a script file to run it. The executable is optional because we can specify the path to the exectuble in the script.

In our environment, we have 4 Nvidia A100s on a node, and 4 nodes. We can write the script as follows:
```bash
#!/bin/bash
#SBATCH --job-name=case1    # name of the job
#SBATCH --nodes=2           # number of nodes to use
#SBATCH --ntasks-per-node=4 # number of tasks per nodes
#SBATCH --gres=gpu:4        # number of GPUs per nodes
#SBATCH --qos=gpugpu        # included when more than 1 node is used

module purge
module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

### Job ID
JOB_ID="${SLURM_JOB_ID}"
### hosfile
HOSTFILE="hostfile.${JOB_ID}"
GPUS=4

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

mpirun -n 8 \ # total number of processes to be started
  --mca btl tcp,self \
  --mca btl_tcp_if_include eth0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile ${HOSTFILE} \
  /path/to/corefl
```

With the above script, the corefl will be started. An `output` folder will be created which contains all output files.

In the output folder, a file named `flowfield.plt` will be created, which is the instantaneous flowfield file. A folder called `message` will also be created. The flowfield file, and the message folder, are necessary for starting a computation with existing results.

### Run the two-temperature AIR5 example

The example case is `example/squareCylinder2D_M10_air5_dns`. It contains `input/setup_2t.txt`, `input/chemistry/air5_park.inp`, and `input/chemistry/two_temperature.dat`.

After building a two-temperature executable as `corefl-2t`, run:

```bash
cd example/squareCylinder2D_M10_air5_dns
./run_2t.sh
```

If the executable has a different path, set `COREFL_2T_BIN`:

```bash
COREFL_2T_BIN=/path/to/corefl ./run_2t.sh
```

The script regenerates the mesh, copies `input/setup_2t.txt` to `input/setup.txt`, clears previous output, and starts the executable. Important setup entries are:

```text
int therm_nasa_7_9 = 9
string mechanism_file = chemistry/air5_park.inp
string two_temperature_file = chemistry/two_temperature.dat
bool enable_vt_relaxation = 1
real Tve = 300.0
```

Boundary/profile input may provide either `EVE` directly or `Tve`; if `Tve` is provided, COREFL computes the corresponding mixture `Eve` from the local species composition.

### Optional wall blowing for 1T hypersonic startup

For the Mach-10 AIR5 square-cylinder case, the one-temperature calculation can lock onto a collapsed near-wall branch during startup before a detached bow shock is established. A short wall-normal blowing stage can be enabled on the wall boundary to seed the detached shock without overriding the wall pressure.

The option is configured inside the wall boundary block in `setup.txt`:

```text
struct square_wall {
    string type = wall
    int label = 2
    string thermal_type = adiabatic
    int catalytic_type = 0
    int if_blow_shock_wave = 1
    real blow_shock_wave_coefficient = 1.0
    int blow_shock_wave_start_step = 3000
    int blow_shock_wave_end_step = 6000
}
```

The parameters are:

- `if_blow_shock_wave`: enables the temporary wall-normal blowing when set to `1`; the default is `0`.
- `blow_shock_wave_coefficient`: multiplier for the local incoming normal velocity. The tested value for this case is `1.0`; a smaller value such as `0.3` was not enough to keep the 1T solution away from the collapsed branch.
- `blow_shock_wave_start_step` and `blow_shock_wave_end_step`: inclusive RK-step window for the forcing. With `dt = 5.0e-10`, the `3000..6000` window corresponds to `1.5e-6..3.0e-6 s`.

For a fresh 1T AIR5 run to `2.0e-5 s`, the corresponding control settings used in the square-cylinder test were:

```text
int total_step = 40000
real total_simulation_time = 2.0e-5
int output_time_series = 1000
```

On a workstation with multiple GPUs, select an idle GPU with `CUDA_VISIBLE_DEVICES`:

```bash
cd example/squareCylinder2D_M10_air5_dns
CUDA_VISIBLE_DEVICES=1 /path/to/corefl
```

A useful sanity check for this setup is the centerline bow-shock standoff distance at `2.0e-5 s`. In the tested AIR5 square-cylinder case, the 1T blow-only run gave about `8.55e-4 m`, while the 2T reference was about `9.19e-4 m`; the collapsed 1T branch stayed near `2.33e-5 m`. No wall pressure override is needed for this configuration.

## Example setup

### Reactive shock tube

The reactive shock tube is presented as an example case because the grid file is small and easy to upload. The folder is as follows:

- case/1-reactiveShockTube/
  - 1-grid/
    - generateMeshShockTube.py
    - readGrid.exe
  - 2-compute/
    - input/
      - grid/grid   0.dat
      - boundary_condition/
        - boundary   0.txt
        - inner   0.txt
        - parallel   0.txt
      - chemistry/
        - therm.dat
        - tran.dat
        - H2_mech-PREMIX.inp
      - setup.txt
    - run.sh


#### Generating grids

The first step would be generating grids for the computation. Although this is a 1D case, we need to generate a 2D grid because COREFL does not support 1D. As we want to use a high-order scheme, for example, 7th-order WENO, we need to have at least 5 additional layers for the ghost layers info. Therefore, 7 layers are adopted for the y direction. And in x direction, we can set a equal dx. We offer a mesh generation file in the subfolder `1-grid`. You can run the python file and the files will be generated in a folder called `readGrid`. You move the exe and the `input.txt` to the folder, and modify the `input.txt` as instructed by the comments. Run the exe, you will get two folders called `grid` and `boundary_condition`. Upload them to the `input` folder of the computational folder, you can use that grid.

#### Set up and run the case

We have already supply a converted grid file in the `2-compute/input/` subfolder in case you just want to run the case.

Here, such a small case, we just need 1 GPU card. Modify the run file to the following form.

```bash
#!/bin/bash
#SBATCH --job-name=ReactiveShockTube    # name of the job
#SBATCH --nodes=1                       # number of nodes to use
#SBATCH --ntasks-per-node=1             # number of tasks per nodes
#SBATCH --gres=gpu:1                    # number of GPUs per nodes
#SBATCH --qos=gpugpu                    # included when more than 1 node is used

module purge
module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

### Job ID
JOB_ID="${SLURM_JOB_ID}"
### hosfile
HOSTFILE="hostfile.${JOB_ID}"
GPUS=4

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

mpirun -n 1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_include eth0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile ${HOSTFILE} \
  /home/bingxing2/home/scx6d0j/GuoXL/code/corefl-cpc/corefl # change the directory to your path to corefl.
```

With these files, and a compiled executable corefl, the test can be started.

### Settings

Maybe you care about the settings, and let me explain the ones related to the current simulation in brief.

First, about the controls:

```c++
int gridIsBinary = 1  // the grid file is in ASCII (0) or binary (1).
real gridScale = 1    // In which scale is the grid generated. If the grid is generated in millimeters, the value is 0.001. In this case, we generate it in meters (1).
int total_step = 100000   // Total steps to compute
int output_file = 10000   // Frequency of outputting flowfield files
int output_screen = 1000  // Frequency of outputting residual info on screen
int output_time_series = 10000 // Frequency of outputting a flowfield named by the physical time. Because we want to compare some transient info with exsiting profiles. If this value is 0, no time series will be outputted.
```

Next, about the temporal schemes:

```c++
bool steady = 0     // If the simulation is steady or not. Steady (1), transient (0)
real dt = 1e-9      // The physical time step in second.
real total_simulation_time = 2.4e-4 // We have data at 0.23ms, so we want the computation to stop after that
```

Third, about the spatial schemes:

```c++
int shock_sensor = 2   // Ducros sensor (0), modified Jameson sensor (1), sensor based on density and pressure jump (2)
real shockSensor_threshold = -0.2 // A negative value means all points are computed by WENO scheme.
int viscous_order = 2   // inviscid(0), 2nd order (2), 8th order (8)
array int viscous_flux_tpb_2d {
 16 16 1
}
array int viscous_derivative_tpb_2d {
 32 16 1
}
```

The viscous-kernel CUDA block sizes can be overridden from `setup.txt` with `viscous_flux_tpb_2d`,
`viscous_flux_tpb_3d`, `viscous_derivative_tpb_2d`, and `viscous_derivative_tpb_3d`. Each array must
contain three positive integers and no more than 1024 total threads. The default 2D flux block is
`16 16 1`, which is the safe setting for the register-heavy two-temperature 8th-order scalar viscous
flux kernel; the default 2D derivative block is `32 16 1`.

Fourth, about the chemistry

```c++
int species = 1     // Air (0), multi-component simulation (1)
string mechanism_file = chemistry/H2PREMIX.inp  // The path is relative to the "input/" folder.
int reaction = 1    // No reaction (0), Finite rate chemistry based on the mechanism (1)
```

Fifth, about the boundary conditions.

```c++
array string boundary_conditions {
  wall  outflow   // Write all boundary conditions' names here
}
struct wall {
  string type = wall  // Specify the type of this bc
  int label = 2       // This label must be consistent with the label of the bc when generating grid
  string  thermal_type    =   adiabatic  // Thermal wall type can be 1. "adiabatic" wall; 2. "isothermal" wall
  real    temperature     =   300        // If the wall is isothermal, the temperature should be given. As the wall in this case is adiabatic, this value will not be used.
}
struct outflow {
  string type = outflow
  int label   =   6
}

// other info about the flow in this case
string  reference_state =   left  // Specify the reference state for the simulation.
string default_init = left  // The default initialization info for the whole flowfield
struct  left {
  string  type            =   inflow
  int     label           =   5
  int     inflow_type     =   0   // 0 for constant inflow, 1 for profile inflow
  real    density         =   0.072
  real    velocity        =   0
  real    pressure        =   7173
  real    u               =   1
  real    v               =   0
  real    w               =   0
  real    H2              =   0.012772428
  real    O2              =   0.101362139
  real    AR              =   0.885865433
}
int groups_init = 2 // Because this case needs two parts with different conditions, we use this initialize in group function of COREFL. The restriction is that the names of groups other than the "default_init" must be named as "init_cond_l", where "l" is indexed from 0 to "group_init-1".
struct init_cond_0 {
  real x0 = 0.06
  real x1 = 0.15
  real y0 = -1
  real y1 = 1
  real z0 = -2
  real z1 = 2
  string name = right  // This tells the code to find a struct named as this value
}
struct  right {
    string  type            =   inflow
    int     label           =   5
    int     inflow_type     =   0   // 0 for constant inflow, 1 for profile inflow
    real    density     =   0.18075
    real    velocity            =   487.34
    real    pressure        =   35594
    real    u               =   -1
    real    v               =   0
    real    w               =   0
    real    H2              =   0.012772428
    real    O2              =   0.101362139
    real    AR              =   0.885865433
}
```

## Other comments

If anyone is interested in our code in more cases, you can email me at <guoxinliang@buaa.edu.cn> freely. I would be happy to cooperate with you for more usages.

Below are some nonsense of mine.

The exampled one is a very simple case, but illustrates many interesting functions such as initialization in group of COREFL. We did not include the readGrid codes in this repository currently because the folder of the above case does not use that.

I have to admit that, because I am a PhD student in the fifth year, a more detailed introduction may cost more energy. If you want to use them, let us talk in private. Maybe I can give you some existing cases to save your time.

Besides, the compilation and environment issues always occur. If there are troubles, please also contact me at the email. I really want my effort of 2-3 years to be used instead of protected in my own computer.

Anyway, we have already conducted many researches based on our GPU code, which is much faster than the CPU codes. It is really fascinating to run a DNS within several days instead of weaks or months.

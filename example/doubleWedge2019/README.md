# Double-Wedge 2019 Mesh

This case uses the double-wedge wall from Exposito and Rana, Aerospace Science
and Technology 92 (2019), 839-846, with a revised parabola-top computational
domain.  The upper slanted boundary is replaced by a right-opening parabola
with left vertex `V=(-0.01, 0.0)` and right point `P=(0.07, 0.07)`.
Points `1` and `2` are projected normally to the parabola, and those two
normal projection lines split the domain into three structured blocks.
The default mesh uses matched endpoint tangential spacing on both sides of each
block interface so the transition across `1-Q1` and `2-Q2` is smooth.  The
default left/base wall-parallel target spacing is `1.0e-5 m`.  The first
interface keeps that fine spacing with `--first-interface-tangent-factor 1.0`,
and the mesh then coarsens toward the two right blocks with the default
`--right-tangent-factor 2.0`.  The generator also relaxes the first
`--interface-relax-cells 96` columns on both sides of every block interface so
the adjacent i-direction spacing is continuous across the join.

Generate the Plot3D mesh, COREFL `grid/` files, boundary files, and preview
figures with:

```bash
python3 example/doubleWedge2019/generate_mesh.py --n-proc 1
```

Main outputs are written to `example/doubleWedge2019/input/`:

- `doubleWedge2019.xyz`: binary multiblock Plot3D grid.
- `doubleWedge2019.inp`: Gridgen-style boundary file for `read_grid`.
- `grid/` and `boundary_condition/`: COREFL-ready files.
- `geometry.png`, `block_layout.png`, `mesh_preview.png`, `mesh_closeup.png`.
- `mesh_summary.txt`: geometry, mesh quality, interface checks, and y+ estimates
  for the low- and high-enthalpy freestreams in the paper.

Boundary labels:

- `2`: wall, O-1-2-3.
- `5`: inflow/farfield, O-V and V-Q1-Q2-P.
- `6`: outflow, 3-P.
- `-1`: block interfaces, 1-Q1 and 2-Q2.

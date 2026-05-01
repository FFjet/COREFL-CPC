# Double-Wedge 2019 Mesh

This case uses the double-wedge wall from Exposito and Rana, Aerospace Science
and Technology 92 (2019), 839-846, with a simplified rectangular-top
computational domain.  The left side starts at `V=(-0.01, 0.0)`, the top-left
point is directly above `V`, and the top-right corner is placed 0.08 m above
point `3`, i.e. `TR=(0.0776, 0.1262)`.  The domain is split by vertical lines
through `O`, `1`, and `2`.

The wedge wall is written as a no-slip wall boundary.  The full top boundary is
symmetry and is intentionally left coarse in the normal direction.  The default
mesh uses the same `1.0e-5 m` X-direction spacing in all four blocks, a first
wall-normal spacing of `2.93e-6 m` at the wedge wall, and 640 normal cells.

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

- `2`: no-slip wall, bottom double-wedge.
- `3`: symmetry, V-O and full rectangular top.
- `5`: inflow, left side.
- `6`: outflow, 3-TR.
- `-1`: block interfaces through O, 1, and 2.

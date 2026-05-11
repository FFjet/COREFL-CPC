#!/usr/bin/env python3
"""Static guardrails for the two-temperature merge.

The checks are intentionally narrow: they catch the index and formula regressions
that are easy to reintroduce while merging the one-temperature upstream branch.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []

    parameter = read("src/Parameter.cpp")
    thermo = read("src/Thermo.cuh")
    viscous = read("src/ViscousScheme.cu")
    weno = read("src/WENO.cu")
    chem = read("src/FiniteRateChem.cu")
    field_op = read("src/FieldOperation.cuh")

    require('i_eve = get_int("n_spec")' in parameter,
            "Parameter.cpp must place Eve immediately after species scalars", failures)
    require('i_eve_cv = 5 + get_int("n_spec")' in parameter,
            "Parameter.cpp must place rhoEve at conservative index 5+n_spec", failures)
    require('++i_ps_cv' in parameter and '++i_ps' in parameter,
            "Parameter.cpp must shift passive scalar indices after inserting Eve", failures)

    require("scalar_alpha[param->i_eve] = -gm1 / c2" in thermo,
            "Thermo.cuh must project Eve with scalar_alpha=-Gamma/c^2", failures)
    require("energy_coeff[param->i_eve] = 1.0" in thermo,
            "Thermo.cuh must recover total energy with unit Eve coefficient", failures)
    require("compute_nonequilibrium_diffusion_enthalpy" in thermo and
            "h_eq - compute_ve_energy(i_spec, t, param) + compute_ve_energy(i_spec, tve, param)" in thermo,
            "Thermo.cuh must use h_tr(T)+Eve_s(Tve) for diffusion enthalpy", failures)

    require("zone->vis_flux" not in viscous,
            "ViscousScheme.cu must not use the removed shifted vis_flux buffer", failures)
    require("lc - 1" not in viscous,
            "Second-order passive scalar viscous flux must use conservative index lc", failures)
    require("4 + param->i_eve" not in viscous,
            "rhoEve viscous flux must use 5+i_eve, not 4+i_eve", failures)
    require("auto &fv = zone->fFlux" in viscous and "auto &gv = zone->gFlux" in viscous and
            "auto &hv = zone->hFlux" in viscous,
            "Second-order viscous kernels must write fFlux/gFlux/hFlux", failures)
    require("fv(i, j, k, 5 + param->i_eve)" in viscous and
            "gv(i, j, k, 5 + param->i_eve)" in viscous and
            "hv(i, j, k, 5 + param->i_eve)" in viscous,
            "rhoEve viscous flux must be written in all three directions", failures)
    require("fv(i, j, k, 5 + l) = diffusion_flux" in viscous and
            "gv(i, j, k, 5 + l) = diffusion_flux" in viscous and
            "hv(i, j, k, 5 + l) = diffusion_flux" in viscous,
            "Species diffusion fluxes must be written at conservative indices 5+l", failures)
    require("fv(i, j, k, 4) += eve_conduction" in viscous and
            "gv(i, j, k, 4) += eve_conduction" in viscous and
            "hv(i, j, k, 4) += eve_conduction" in viscous,
            "Total energy viscous flux must include VE conduction in all directions", failures)

    require(weno.count("compute_mixture_characteristic_thermo") >= 3,
            "WENO characteristic kernels must use two-temperature mixture thermo", failures)
    require("energy_coeff[l] * fChar[l + 5]" in weno,
            "WENO recovery must include scalar energy compensation", failures)
    require("param->x_sponge_start" in weno and "param->y_sponge_start" in weno and "in_sponge" in weno,
            "WENO must retain upstream sponge downgrade logic", failures)

    require("compute_two_temperature_chemical_source" in chem and
            "compute_ve_energy(l, tve, param)" in chem,
            "rhoEve chemical source must use species Eve at Tve", failures)
    require("zone->dq(i, j, k, 5 + param->i_eve)" in chem,
            "rhoEve chemical source must be added at conservative index 5+i_eve", failures)

    require("cv(i, j, k, 5 + param->i_eve)" in field_op and
            "exp(-dt_stage / max(tau_eff" in field_op,
            "VT relaxation must update rhoEve with the split exponential step", failures)
    require(re.search(r"total_energy \+= zone->sv\(i, j, k, param->i_eve\)", field_op) is not None,
            "Total energy closure must add independent Eve scalar", failures)

    if failures:
        print("Two-temperature merge checks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Two-temperature merge checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

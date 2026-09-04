# Setup-1 grids (PGLIB-OPF derivatives)

These `*_corrected.m` files are the **in-memory protocol** (paper setup 1) base cases
read by `scripts/runtime/pure_julia/run_matrix.jl`.

They are derived from [PGLIB-OPF](https://github.com/power-grid-lib/pglib-opf)
IEEE / GOC cases, with local corrections used by this repository.

Only the seven networks in the paper runtime matrix are tracked here:

- `pglib_opf_case14_ieee_corrected.m`
- `pglib_opf_case30_ieee_corrected.m`
- `pglib_opf_case57_ieee_corrected.m`
- `pglib_opf_case118_ieee_corrected.m`
- `pglib_opf_case500_goc_corrected.m`
- `pglib_opf_case2000_goc_corrected.m`
- `pglib_opf_case10000_goc_corrected.m`

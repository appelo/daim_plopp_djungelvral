# DN-tolerance robustness study  (fig:tolstudy, Sec. "Truncation tolerance, rank,
# and robustness" of notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Converged error vs the DN tolerance tol_DD for the three solvers on a coarse grid
# (uses tolerance_study from the solver file); writes notes/figs/tolerance_study.png.
#     julia --project=. examples/experiments/tolerance_study.jl
include(joinpath(@__DIR__, "common.jl"))

function fig_tolerance_study()
    println("\n=== solver / tolerance study (M=4, nx=25, ny=101) ===")
    fig = tolerance_study(; M = 4, nx = 25, ny = 101)   # defined in the solver file
    dst = joinpath(FIGDIR, "tolerance_study.png")
    save(dst, fig)
    println("  saved ", dst)
end

Random.seed!(SEED)
fig_tolerance_study()

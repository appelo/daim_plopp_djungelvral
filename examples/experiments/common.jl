# Shared setup for the per-experiment reproducibility scripts in this directory.
#
# Each `examples/experiments/<name>.jl` script reproduces exactly one figure/table
# of `notes/sbp_poisson_2d_lowrank_solvers.tex` and can be run on its own:
#
#     julia --project=. examples/experiments/<name>.jl
#
# It `include`s this file, which loads the solver once and defines the shared
# constants and rank helpers.  The guard makes repeated inclusion (e.g. by the
# `examples/experiments_2d_lowrank.jl` run-all driver) a no-op.

if !@isdefined(_LRDD_EXPERIMENTS_COMMON)
    _LRDD_EXPERIMENTS_COMMON = true

    using LinearAlgebra, Printf, Random, CairoMakie

    # The 2D low-rank Poisson DD solver (run_poisson_2d_lr, eig2d, cg2d, eig2d_lr,
    # tolerance_study, …).
    include(joinpath(@__DIR__, "..", "sbp_poisson_bvp_2d_lowrank.jl"))

    # The low-rank forcing uses a random crossDEIM start; fix the seed so every
    # script is deterministic.  (crossDEIM converges to the same forcing regardless,
    # so results are essentially seed-independent, but this pins them exactly.)
    SEED = 20240531

    # Figures referenced by the note are written here.
    FIGDIR = abspath(joinpath(@__DIR__, "..", "..", "notes", "figs"))
    isdir(FIGDIR) || mkpath(FIGDIR)

    # Where run_poisson_2d_lr(doplot=true) writes its 3-panel figure.
    DRIVER_PNG = abspath(joinpath(@__DIR__, "..", "sbp_poisson_bvp_2d_lowrank.png"))

    # Rank summaries used across experiments.
    maxrank(r)  = maximum(r.rankhist[end])                       # final max subdomain rank
    meanrank(r) = sum(r.rankhist[end]) / length(r.rankhist[end]) # final mean subdomain rank
    peakrank(r) = maximum(maximum(rk) for rk in r.rankhist)      # max rank over all sweeps
    avgrank(r)  = [sum(rk) / length(rk) for rk in r.rankhist]    # mean rank per sweep
end

# Run-all driver for the reproducibility scripts of
# `notes/sbp_poisson_2d_lowrank_solvers.tex`.
#
# Each experiment lives in its own self-contained, individually runnable file under
# `examples/experiments/` (and reproduces exactly one figure/table of the note).
# This script just `include`s them in the order they appear in the note; the shared
# solver is loaded once (guarded in `experiments/common.jl`).
#
#     julia --project=. examples/experiments_2d_lowrank.jl          # run everything
#     julia --project=. examples/experiments/dn_convergence.jl      # or one at a time
#
# Mapping (file  ->  note result):
#   dn_convergence.jl     -> fig:dn        (Convergence of the DN iteration)
#   order_of_accuracy.jl  -> tab:refine    (Order of accuracy)
#   solver_comparison.jl  -> tab:solvers   (Comparison of the three solvers)
#   rank_vs_tolerance.jl  -> tab:rank, fig:rank        (Truncation tolerance & rank)
#   tolerance_study.jl    -> fig:tolstudy  (DN-tolerance robustness)
#   schedule_theta.jl     -> tab:theta, fig:sched      (Schedule: choosing theta)
#   schedule_residual.jl  -> tab:sched-src, fig:sched-src  (Operator-residual schedule)
#   timing_vs_gridsize.jl -> tab:timing, fig:timing    (Timing vs grid size)
#   subdomains.jl         -> tab:sub, fig:sub          (Iterations vs #subdomains)

const _EXPDIR = joinpath(@__DIR__, "experiments")
for f in ("dn_convergence.jl", "order_of_accuracy.jl", "solver_comparison.jl",
          "rank_vs_tolerance.jl", "tolerance_study.jl", "schedule_theta.jl",
          "schedule_residual.jl", "timing_vs_gridsize.jl", "subdomains.jl")
    include(joinpath(_EXPDIR, f))
end

println("\nAll experiments done.  Figures in ",
        abspath(joinpath(@__DIR__, "..", "notes", "figs")))

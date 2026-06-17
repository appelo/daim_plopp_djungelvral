# Comparison of the three subdomain solvers  (tab:solvers, Sec. "Comparison of the
# three subdomain solvers" of notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Fixed problem (M=4, nx=ny=80, p=6, tol_DD=1e-7) with the final truncation
# tolerance matched across solvers; prints the table.
#     julia --project=. examples/experiments/solver_comparison.jl
include(joinpath(@__DIR__, "common.jl"))

function table_solver_comparison()
    println("\n=== solver comparison (M=4, nx=ny=80, p=6, tol_DD=1e-7) ===")
    tt = 1e-10
    cases = [
        ("eig2d (direct)",  (; solver = eig2d,    tol_domain_solver = 1e-11, tol_trunc = tt, tol_DD = 1e-7)),
        ("cg2d (iterative)",(; solver = cg2d,     tol_domain_solver = 1e-11, tol_trunc = tt, tol_DD = 1e-7)),
        ("eig2d_lr",        (; solver = eig2d_lr, tol_domain_solver = tt,     tol_trunc = tt, tol_DD = 1e-7)),
    ]
    rows = NamedTuple[]
    for (label, opts) in cases
        r = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, θ = 0.5,
                              maxiter = 600, doplot = false, opts = opts)
        push!(rows, (; label, sweeps = r.sweeps, err = r.errhist[end],
                       maxrank = maxrank(r), meanrank = meanrank(r)))
        @printf("  %-18s sweeps=%3d  err=%.3e  max rank=%2d  mean rank=%.1f\n",
                label, r.sweeps, r.errhist[end], maxrank(r), meanrank(r))
    end
    return rows
end

Random.seed!(SEED)
table_solver_comparison()

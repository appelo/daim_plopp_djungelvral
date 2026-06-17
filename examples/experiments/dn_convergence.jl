# Convergence of the Dirichlet--Neumann iteration  (fig:dn, Sec. "Convergence of
# the Dirichlet--Neumann iteration" of notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Representative eig2d_lr run (M=4, nx=ny=80, p=6); writes notes/figs/dn_convergence.png.
#     julia --project=. examples/experiments/dn_convergence.jl
include(joinpath(@__DIR__, "common.jl"))

function fig_dn_convergence()
    println("\n=== DN convergence (eig2d_lr, M=4, nx=ny=80, p=6) ===")
    r = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, θ = 0.5,
                          maxiter = 400, plotkind = :surface, doplot = true,
                          opts = (; solver = eig2d_lr, tol_domain_solver = 1e-10,
                                    tol_trunc = 1e-10, tol_DD = 1e-7))
    dst = joinpath(FIGDIR, "dn_convergence.png")
    cp(DRIVER_PNG, dst; force = true)
    @printf("  sweeps=%d  final err=%.3e  max rank=%d\n", r.sweeps, r.errhist[end], maxrank(r))
    println("  saved ", dst)
    return r
end

Random.seed!(SEED)
fig_dn_convergence()

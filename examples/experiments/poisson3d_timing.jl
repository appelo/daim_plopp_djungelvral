# Timing comparison eig3d vs eig3d_lr for the 3D Tucker solver, used in the
# "Three-dimensional Tucker analogue" section of
# notes/sbp_poisson_2d_lowrank_solvers.tex.
#
# Times the full Dirichlet--Neumann solution (M=2, p=4, tol_DD=1e-7) at grid sizes
# N = nx = ny = nz.  WARNING: large N is heavy — N=300 takes several minutes and a
# few GB of RAM per solver.
#
#   julia --project=. examples/experiments/poisson3d_timing.jl 80 160     # calibration
#   julia --project=. examples/experiments/poisson3d_timing.jl 300        # headline

using Printf
include(joinpath(@__DIR__, "..", "sbp_poisson_bvp_3d_lowrank.jl"))

fr(r) = maximum(maximum(c) for c in r.rankhist[end])   # final max Tucker core rank
optE   = (; solver = eig3d,    tol_trunc = 1e-10, tol_DD = 1e-7)                      # dense direct
optLRd = (; solver = eig3d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7)               # dense RHS
optLRc = (; solver = eig3d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7, rhs = :crosssum)  # matrix-free RHS
runN(N, o) = run_poisson_3d_lr(; M = 2, nx = N, ny = N, nz = N, accuracy = 4,
                               maxiter = 200, doplot = false, opts = o)

runN(12, optE); runN(12, optLRd); runN(12, optLRc)   # warmup (compile)

timed(N, o) = (GC.gc(); t = @elapsed r = runN(N, o); (; t, sweeps = r.sweeps, err = r.errhist[end], rank = fr(r)))

Ns = isempty(ARGS) ? [80, 160] : parse.(Int, ARGS)
for N in Ns
    e  = timed(N, optE)
    ld = timed(N, optLRd)
    lc = timed(N, optLRc)
    @printf("N=%3d  eig3d %8.2f s (rank %d)   eig3d_lr:dense %8.2f s (rank %d)   eig3d_lr:crosssum %8.2f s (rank %d)   err≈%.2e\n",
            N, e.t, e.rank, ld.t, ld.rank, lc.t, lc.rank, e.err)
    @printf("       speedup eig3d/lr:dense = %.2f ; eig3d/lr:crosssum = %.2f ; lr:dense/lr:crosssum = %.2f\n",
            e.t / ld.t, e.t / lc.t, ld.t / lc.t)
end
println("DONE")

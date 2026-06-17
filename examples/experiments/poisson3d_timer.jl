# Bottleneck timing of run_poisson_3d_lr via TimerOutputs.jl.  Runs eig3d and
# eig3d_lr (M=2) and prints a per-operation breakdown (`show_timer=true`).
#
#   julia --project=. examples/experiments/poisson3d_timer.jl        # N=120
#   julia --project=. examples/experiments/poisson3d_timer.jl 200    # other N

include(joinpath(@__DIR__, "..", "sbp_poisson_bvp_3d_lowrank.jl"))

N = isempty(ARGS) ? 120 : parse(Int, ARGS[1])

println("\n##### eig3d  (N=$N, M=2) #####")
run_poisson_3d_lr(; M = 2, nx = N, ny = N, nz = N, accuracy = 4, maxiter = 200,
                  doplot = false, show_timer = true,
                  opts = (; solver = eig3d, tol_trunc = 1e-10, tol_DD = 1e-7))

println("\n##### eig3d_lr  (N=$N, M=2) #####")
run_poisson_3d_lr(; M = 2, nx = N, ny = N, nz = N, accuracy = 4, maxiter = 200,
                  doplot = false, show_timer = true,
                  opts = (; solver = eig3d_lr, tol_domain_solver = 1e-10,
                            tol_trunc = 1e-10, tol_DD = 1e-7))

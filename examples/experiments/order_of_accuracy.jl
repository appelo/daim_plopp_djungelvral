# Order of accuracy  (tab:refine, Sec. "Order of accuracy" of
# notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Grid refinement with the exact solver eig2d so the measured error is the
# discretization error; prints the table.
#     julia --project=. examples/experiments/order_of_accuracy.jl
include(joinpath(@__DIR__, "common.jl"))

function table_refinement()
    println("\n=== grid refinement (eig2d, M=4) ===")
    Ns = [20, 40, 80, 160]
    rows = NamedTuple[]
    for p in (4, 6)
        prev_err = NaN; prev_h = NaN
        for N in Ns
            r = run_poisson_2d_lr(; M = 4, nx = N, ny = N, accuracy = p, θ = 0.5,
                                  maxiter = 2000, doplot = false,
                                  opts = (; solver = eig2d, tol_domain_solver = 1e-12,
                                            tol_trunc = 1e-12, tol_DD = 1e-11))
            h = 5.0 / (4 * (N - 1))                      # representative x mesh width
            err = r.errhist[end]
            rate = isnan(prev_err) ? NaN : log(prev_err / err) / log(prev_h / h)
            push!(rows, (; p, N, err, rate, sweeps = r.sweeps))
            @printf("  p=%d  N=%3d  err=%.3e  rate=%s  sweeps=%d\n",
                    p, N, err, isnan(rate) ? "  --" : @sprintf("%.2f", rate), r.sweeps)
            prev_err = err; prev_h = h
        end
    end
    return rows
end

Random.seed!(SEED)
table_refinement()

# Timing vs grid size  (tab:timing + fig:timing, Sec. "Timing versus grid size" of
# notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Wall-clock time to solve (M=4, p=6, tol_DD=1e-7) for the dense eig2d vs the
# scheduled eig2d_lr as N=nx=ny is refined; writes notes/figs/timing.png.
# NOTE: wall-clock timings are machine/load dependent; run on a quiet machine.
#     julia --project=. examples/experiments/timing_vs_gridsize.jl
include(joinpath(@__DIR__, "common.jl"))

function table_and_fig_timing(; Ns = [40, 80, 160, 320, 640])
    println("\n=== timing vs grid size (M=4, p=6, tol_DD=1e-7) ===")
    optE  = (; solver = eig2d,    tol_domain_solver = 1e-11, tol_trunc = 1e-10, tol_DD = 1e-7)
    optLR = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7,
               tol_schedule = res -> clamp(0.02 * res, 1e-10, 1.0))
    run1(N, o) = (Random.seed!(7);
                  @elapsed run_poisson_2d_lr(; M = 4, nx = N, ny = N, accuracy = 6,
                                             maxiter = 400, doplot = false, opts = o))
    Random.seed!(1); run_poisson_2d_lr(; M = 4, nx = 24, ny = 24, accuracy = 6, maxiter = 200, doplot = false, opts = optE)
    Random.seed!(1); run_poisson_2d_lr(; M = 4, nx = 24, ny = 24, accuracy = 6, maxiter = 200, doplot = false, opts = optLR)
    rows = NamedTuple[]
    for N in Ns
        tE = min(run1(N, optE),  run1(N, optE))
        tL = min(run1(N, optLR), run1(N, optLR))
        push!(rows, (; N, tE, tL))
        @printf("  N=%4d: eig2d %.4fs   eig2d_lr(sched) %.4fs   speedup(eig2d/lr)=%.2f\n", N, tE, tL, tE / tL)
    end
    Nv = [r.N for r in rows]; tEv = [r.tE for r in rows]; tLv = [r.tL for r in rows]
    fig = Figure(size = (720, 520))
    ax = Axis(fig[1, 1]; xlabel = "N  (nx = ny)", ylabel = "wall time [s]",
              xscale = log10, yscale = log10, title = "solve time vs grid size (M=4, p=6)")
    scatterlines!(ax, Nv, tEv; label = "eig2d (dense)",            marker = :rect)
    scatterlines!(ax, Nv, tLv; label = "eig2d_lr (scheduled)",     marker = :circle)
    # reference slopes anchored at the largest N
    lines!(ax, Nv, tEv[end] .* (Nv ./ Nv[end]).^3; color = :gray, linestyle = :dash, label = "∝ N³")
    lines!(ax, Nv, tLv[end] .* (Nv ./ Nv[end]).^2; color = :black, linestyle = :dot,  label = "∝ N²")
    axislegend(ax; position = :lt)
    dst = joinpath(FIGDIR, "timing.png"); save(dst, fig)
    println("  saved ", dst)
    return rows
end

Random.seed!(SEED)
table_and_fig_timing()

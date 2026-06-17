# Truncation tolerance vs rank  (tab:rank + fig:rank, Sec. "Truncation tolerance,
# rank, and robustness" of notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# For eig2d_lr the truncation tolerance is the rank/accuracy knob: prints the table
# and writes notes/figs/rank_vs_tol.png.
#     julia --project=. examples/experiments/rank_vs_tolerance.jl
include(joinpath(@__DIR__, "common.jl"))

function table_and_fig_rank_knob()
    println("\n=== truncation tolerance vs rank (eig2d_lr, nx=ny=80, p=6) ===")
    tols = [1e-6, 1e-8, 1e-10, 1e-12]
    rows = NamedTuple[]
    runs = Dict{Float64,Any}()
    for tol in tols
        r = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, θ = 0.5,
                              maxiter = 400, doplot = false,
                              opts = (; solver = eig2d_lr, tol_domain_solver = tol,
                                        tol_trunc = tol, tol_DD = 1e-7))
        runs[tol] = r
        push!(rows, (; tol, sweeps = r.sweeps, err = r.errhist[end],
                       maxrank = maxrank(r), meanrank = meanrank(r)))
        @printf("  tol=%.0e  sweeps=%3d  err=%.3e  max rank=%2d  mean rank=%.1f\n",
                tol, r.sweeps, r.errhist[end], maxrank(r), meanrank(r))
    end
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1]; xlabel = "DN sweep", ylabel = "mean subdomain rank",
              title = "Truncation tolerance sets the rank (eig2d_lr)")
    for tol in (1e-6, 1e-8, 1e-12)
        r = runs[tol]
        scatterlines!(ax, 1:length(r.rankhist), avgrank(r);
                      label = "tol = " * @sprintf("%.0e", tol), marker = :circle)
    end
    axislegend(ax; position = :rt)
    dst = joinpath(FIGDIR, "rank_vs_tol.png")
    save(dst, fig)
    println("  saved ", dst)
    return rows
end

Random.seed!(SEED)
table_and_fig_rank_knob()

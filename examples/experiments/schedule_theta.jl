# Tolerance schedule: choosing theta and rank growth  (tab:theta + fig:sched,
# Sec. "The tolerance schedule: choosing theta and rank growth" of
# notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Schedule eps = clamp(theta*||dlambda||, floor, 1): (a) robustness vs theta,
# (b) floor sensitivity, (c) rank growth scheduled vs fixed.
# Writes notes/figs/scheduling.png.
#     julia --project=. examples/experiments/schedule_theta.jl
include(joinpath(@__DIR__, "common.jl"))

function table_and_fig_scheduling()
    println("\n=== tolerance schedule eps = theta*||dlambda|| ===")
    prob = (; M = 4, nx = 80, ny = 80, accuracy = 6, θ = 0.5, maxiter = 400)
    base = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7)
    sched(θ; flr = 1e-10) = res -> clamp(θ * res, flr, 1.0)

    println("  (a) robustness vs theta (floor = 1e-10):")
    θs = [0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
    rows_theta = NamedTuple[]
    for θ in θs
        r = run_poisson_2d_lr(; prob..., doplot = false,
                              opts = merge(base, (; tol_schedule = sched(θ))))
        conv = r.sweeps < prob.maxiter
        push!(rows_theta, (; θ, conv, sweeps = r.sweeps, err = r.errhist[end],
                             finalrank = maxrank(r), peakrank = peakrank(r)))
        @printf("    theta=%.3f  conv=%s  sweeps=%3d  err=%.3e  final rank=%2d  peak rank=%2d\n",
                θ, conv ? "Y" : "N", r.sweeps, r.errhist[end], maxrank(r), peakrank(r))
    end

    println("  (b) floor sensitivity at theta = 0.02:")
    rows_floor = NamedTuple[]
    for flr in (1e-7, 1e-8, 1e-10, 1e-12)
        r = run_poisson_2d_lr(; prob..., doplot = false,
                              opts = merge(base, (; tol_schedule = sched(0.02; flr = flr))))
        push!(rows_floor, (; flr, conv = r.sweeps < prob.maxiter, sweeps = r.sweeps,
                             err = r.errhist[end], finalrank = maxrank(r)))
        @printf("    floor=%.0e  conv=%s  sweeps=%3d  err=%.3e  final rank=%2d\n",
                flr, r.sweeps < prob.maxiter ? "Y" : "N", r.sweeps, r.errhist[end], maxrank(r))
    end

    # (c) figure: accuracy decay for several theta + rank growth (scheduled vs fixed).
    rstall = run_poisson_2d_lr(; prob..., doplot = false, opts = merge(base, (; tol_schedule = sched(0.10))))
    rmid   = run_poisson_2d_lr(; prob..., doplot = false, opts = merge(base, (; tol_schedule = sched(0.05))))
    rconv  = run_poisson_2d_lr(; prob..., doplot = false, opts = merge(base, (; tol_schedule = sched(0.02))))
    rfixed = run_poisson_2d_lr(; prob..., doplot = false, opts = base)

    fig = Figure(size = (1150, 460))
    axL = Axis(fig[1, 1]; xlabel = "DN sweep", ylabel = "max|w − w_exact|", yscale = log10,
               title = "accuracy vs sweep for several θ")
    for (r, lab) in ((rstall, "θ = 0.10 (stalls)"), (rmid, "θ = 0.05"), (rconv, "θ = 0.02 (converges)"))
        scatterlines!(axL, 1:length(r.errhist), max.(r.errhist, 1e-16); label = lab, marker = :circle)
    end
    hlines!(axL, [2.47e-5]; color = :black, linestyle = :dash, label = "discretization error")
    axislegend(axL; position = :rt)

    axR = Axis(fig[1, 2]; xlabel = "DN sweep", ylabel = "mean subdomain rank",
               title = "rank growth: scheduled vs fixed")
    scatterlines!(axR, 1:length(rconv.rankhist),  avgrank(rconv);  label = "scheduled θ = 0.02", marker = :circle)
    scatterlines!(axR, 1:length(rfixed.rankhist), avgrank(rfixed); label = "fixed tol = 1e-10",   marker = :rect)
    axislegend(axR; position = :rt)

    dst = joinpath(FIGDIR, "scheduling.png")
    save(dst, fig)
    println("  saved ", dst)
    @printf("  scheduled θ=0.02: peak rank=%d, final rank=%d ;  fixed 1e-10: peak rank=%d, final rank=%d\n",
            peakrank(rconv), maxrank(rconv), peakrank(rfixed), maxrank(rfixed))
    return (; rows_theta, rows_floor)
end

Random.seed!(SEED)
table_and_fig_scheduling()

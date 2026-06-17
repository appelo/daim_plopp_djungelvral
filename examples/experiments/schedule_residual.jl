# Alternative schedule source: operator residual vs ||dlambda||  (tab:sched-src +
# fig:sched-src, Sec. "The tolerance schedule …" of
# notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# Drive eps = clamp(theta*r, eps_min, 1) by the solution's operator residual r
# (sched_on=:residual) vs the interface increment ||dlambda|| (:lambda): (a) theta
# robustness at M=4, (b) largest converging theta vs M, (c) figure.
# Writes notes/figs/scheduling_residual.png.
#     julia --project=. examples/experiments/schedule_residual.jl
include(joinpath(@__DIR__, "common.jl"))

function table_and_fig_schedule_residual()
    println("\n=== schedule driven by operator residual vs ||dlambda|| ===")
    schd(θ) = res -> clamp(θ * res, 1e-10, 1.0)
    opt(θ, src) = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10,
                     tol_DD = 1e-7, tol_schedule = schd(θ), sched_on = src)
    yn(b) = b ? "Y" : "N"

    println("  (a) theta-robustness at M=4 (nx=ny=80, p=6): :lambda vs :residual")
    θs = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    rows = NamedTuple[]
    for θ in θs
        Random.seed!(7); rl = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, maxiter = 400, doplot = false, opts = opt(θ, :lambda))
        Random.seed!(7); rr = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, maxiter = 400, doplot = false, opts = opt(θ, :residual))
        push!(rows, (; θ, convL = rl.sweeps < 400, swL = rl.sweeps, errL = rl.errhist[end], rkL = maxrank(rl),
                       convR = rr.sweeps < 400, swR = rr.sweeps, errR = rr.errhist[end], rkR = maxrank(rr)))
        @printf("    theta=%.2f | lambda: %s sw=%3d err=%.2e rk=%2d | residual: %s sw=%3d err=%.2e rk=%2d\n",
                θ, yn(rl.sweeps<400), rl.sweeps, rl.errhist[end], maxrank(rl),
                   yn(rr.sweeps<400), rr.sweeps, rr.errhist[end], maxrank(rr))
    end

    println("  (b) largest converging theta vs M (nx=ny=48): :lambda vs :residual")
    for M in (4, 8, 16)
        msg = @sprintf("    M=%2d:", M)
        for (src, tag) in ((:lambda, "lam"), (:residual, "res"))
            best = 0.0
            for θ in (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
                Random.seed!(7)
                r = run_poisson_2d_lr(; M, nx = 48, ny = 48, accuracy = 6, maxiter = 800, doplot = false, opts = opt(θ, src))
                if r.sweeps < 800; best = θ; break; end   # θ list is descending → first converging is the largest
            end
            msg *= @sprintf("  %s θ*≈%.3f", tag, best)
        end
        println(msg)
    end

    # (c) figure: error vs sweep at theta=0.10 (lambda stalls, residual converges) + rank growth.
    Random.seed!(7); el = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, maxiter = 400, doplot = false, opts = opt(0.10, :lambda))
    Random.seed!(7); er = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, maxiter = 400, doplot = false, opts = opt(0.10, :residual))
    Random.seed!(7); ef = run_poisson_2d_lr(; M = 4, nx = 80, ny = 80, accuracy = 6, maxiter = 400, doplot = false,
                                            opts = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7))
    fig = Figure(size = (1150, 460))
    axL = Axis(fig[1, 1]; xlabel = "DN sweep", ylabel = "max|w − w_exact|", yscale = log10,
               title = "schedule source at θ = 0.10 (M=4)")
    scatterlines!(axL, 1:length(el.errhist), max.(el.errhist, 1e-16); label = "‖Δλ‖ (stalls)",     marker = :rect)
    scatterlines!(axL, 1:length(er.errhist), max.(er.errhist, 1e-16); label = "operator residual", marker = :circle)
    hlines!(axL, [2.47e-5]; color = :black, linestyle = :dash, label = "discretization error")
    axislegend(axL; position = :rt)
    axR = Axis(fig[1, 2]; xlabel = "DN sweep", ylabel = "mean subdomain rank",
               title = "rank growth (residual schedule vs fixed)")
    scatterlines!(axR, 1:length(er.rankhist), avgrank(er); label = "residual θ = 0.10", marker = :circle)
    scatterlines!(axR, 1:length(ef.rankhist), avgrank(ef); label = "fixed tol = 1e-10",  marker = :rect)
    axislegend(axR; position = :rt)
    dst = joinpath(FIGDIR, "scheduling_residual.png"); save(dst, fig)
    println("  saved ", dst)
    @printf("  residual θ=0.10: sweeps=%d peak rank=%d ;  fixed: peak rank=%d\n",
            er.sweeps, peakrank(er), peakrank(ef))
    return rows
end

Random.seed!(SEED)
table_and_fig_schedule_residual()

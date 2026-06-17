# Experiments for the 3D Tucker low-rank solver, used in
# `notes/sbp_poisson_2d_lowrank_solvers.tex` (Sec. "Three-dimensional Tucker
# analogue").  Generates the representative convergence figure and the scheduling
# figure, and prints the θ-robustness table (:lambda vs :residual).
#
#     julia --project=. examples/experiments/poisson3d_scheduling.jl

using Printf, CairoMakie
include(joinpath(@__DIR__, "..", "sbp_poisson_bvp_3d_lowrank.jl"))

const FIGDIR = abspath(joinpath(@__DIR__, "..", "..", "notes", "figs"))
isdir(FIGDIR) || mkpath(FIGDIR)

const P = (; M = 3, nx = 16, ny = 24, nz = 24, accuracy = 4, θ = 0.5, maxiter = 400)
yn(b) = b ? "Y" : "N"
mrank(r) = [maximum(maximum(c) for c in r.rankhist[k]) for k in 1:length(r.rankhist)]  # max core rank / sweep
peakrank(r) = maximum(mrank(r))
finalrank(r) = maximum(maximum(c) for c in r.rankhist[end])
schd(θ) = res -> clamp(θ * res, 1e-10, 1.0)
opt(θ, src) = (; solver = eig3d, tol_domain_solver = 1e-10, tol_trunc = 1e-10,
                 tol_DD = 1e-7, tol_schedule = schd(θ), sched_on = src)

# --- representative convergence + ranks figure (the driver's own doplot) -----
function fig_convergence()
    println("\n=== 3D representative run (eig3d, M=$(P.M), nx=$(P.nx), ny=nz=$(P.ny)) ===")
    r = run_poisson_3d_lr(; P..., doplot = true,
                          plotfile = joinpath(FIGDIR, "poisson3d_convergence.png"),
                          opts = (; solver = eig3d, tol_trunc = 1e-10, tol_DD = 1e-7))
    @printf("  sweeps=%d  err=%.3e  final rank=%d\n", r.sweeps, r.errhist[end], finalrank(r))
end

# --- scheduling study: θ-robustness table + figure --------------------------
function table_and_fig_scheduling()
    println("\n=== 3D scheduling: θ-robustness (eig3d), :lambda vs :residual ===")
    θs = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    for src in (:lambda, :residual)
        for θ in θs
            r = run_poisson_3d_lr(; P..., doplot = false, opts = opt(θ, src))
            @printf("  %-9s θ=%.2f  conv=%s  sweeps=%3d  err=%.3e  final rank=%2d  peak rank=%2d\n",
                    src, θ, yn(r.sweeps < P.maxiter), r.sweeps, r.errhist[end], finalrank(r), peakrank(r))
        end
    end

    # runs feeding the figure.
    rl_stall = run_poisson_3d_lr(; P..., doplot = false, opts = opt(0.50, :lambda))    # large θ stalls
    rl_conv  = run_poisson_3d_lr(; P..., doplot = false, opts = opt(0.02, :lambda))    # small θ converges
    rr_big   = run_poisson_3d_lr(; P..., doplot = false, opts = opt(0.50, :residual))  # residual rescues θ=0.50
    rr_conv  = run_poisson_3d_lr(; P..., doplot = false, opts = opt(0.10, :residual))
    rfixed   = run_poisson_3d_lr(; P..., doplot = false, opts = (; solver = eig3d, tol_trunc = 1e-10, tol_DD = 1e-7))

    fig = Figure(size = (1150, 460))
    axL = Axis(fig[1, 1]; xlabel = "DN sweep", ylabel = "max|w − w_exact|", yscale = log10,
               title = "3D scheduling: accuracy vs sweep (θ = 0.50)")
    scatterlines!(axL, 1:length(rl_stall.errhist), max.(rl_stall.errhist, 1e-16); label = "‖Δλ‖ source (stalls)",     marker = :rect)
    scatterlines!(axL, 1:length(rr_big.errhist),   max.(rr_big.errhist, 1e-16);   label = "residual source (converges)", marker = :diamond)
    hlines!(axL, [8.48e-5]; color = :black, linestyle = :dash, label = "discretization error")
    axislegend(axL; position = :rt)

    axR = Axis(fig[1, 2]; xlabel = "DN sweep", ylabel = "max Tucker core rank",
               title = "3D rank growth: scheduled vs fixed")
    scatterlines!(axR, 1:length(rr_conv.rankhist), mrank(rr_conv); label = "scheduled (residual, θ=0.10)", marker = :circle)
    scatterlines!(axR, 1:length(rl_conv.rankhist), mrank(rl_conv); label = "scheduled (‖Δλ‖, θ=0.02)",     marker = :diamond)
    scatterlines!(axR, 1:length(rfixed.rankhist),  mrank(rfixed);  label = "fixed tol = 1e-10",            marker = :rect)
    axislegend(axR; position = :rt)

    dst = joinpath(FIGDIR, "poisson3d_scheduling.png")
    save(dst, fig)
    println("  saved ", dst)
    @printf("  fixed: peak rank=%d, final rank=%d ; scheduled(res θ=0.1): peak=%d ; scheduled(λ θ=0.02): peak=%d\n",
            peakrank(rfixed), finalrank(rfixed), peakrank(rr_conv), peakrank(rl_conv))
end

fig_convergence()
table_and_fig_scheduling()
println("\nDONE")

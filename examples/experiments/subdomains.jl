# Iteration count vs number of subdomains  (tab:sub + fig:sub, Sec. "Iteration
# count versus the number of subdomains" of notes/sbp_poisson_2d_lowrank_solvers.tex).
#
# DN sweeps to converge vs M (nx=ny=48, p=6) for eig2d vs eig2d_lr (fixed tol), and
# the robust scheduling constant theta vs M; writes notes/figs/subdomains.png.
#     julia --project=. examples/experiments/subdomains.jl
include(joinpath(@__DIR__, "common.jl"))

function table_and_fig_subdomains(; Ms = [2, 4, 8, 16, 32])
    println("\n=== DN sweeps vs number of subdomains (nx=ny=48, p=6, tol_DD=1e-7) ===")
    optE   = (; solver = eig2d,    tol_domain_solver = 1e-11, tol_trunc = 1e-10, tol_DD = 1e-7)
    optLRf = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7)
    Random.seed!(1); run_poisson_2d_lr(; M = 2, nx = 24, ny = 24, accuracy = 6, maxiter = 200, doplot = false, opts = optLRf)
    rows = NamedTuple[]
    for M in Ms
        Random.seed!(7); rE  = run_poisson_2d_lr(; M, nx = 48, ny = 48, accuracy = 6, maxiter = 1500, doplot = false, opts = optE)
        Random.seed!(7); rLf = run_poisson_2d_lr(; M, nx = 48, ny = 48, accuracy = 6, maxiter = 1500, doplot = false, opts = optLRf)
        push!(rows, (; M, swE = rE.sweeps, rkE = maxrank(rE), swL = rLf.sweeps, rkL = maxrank(rLf)))
        @printf("  M=%2d: eig2d sweeps=%4d (rank %2d)   eig2d_lr sweeps=%4d (rank %2d)\n",
                M, rE.sweeps, maxrank(rE), rLf.sweeps, maxrank(rLf))
    end
    Mv = [r.M for r in rows]
    fig = Figure(size = (720, 520))
    ax = Axis(fig[1, 1]; xlabel = "number of subdomains M", ylabel = "DN sweeps to converge",
              xscale = log10, yscale = log10, title = "DN sweeps vs M (nx=ny=48, p=6)")
    scatterlines!(ax, Mv, [r.swE for r in rows]; label = "eig2d",    marker = :rect,   markersize = 16)
    scatterlines!(ax, Mv, [r.swL for r in rows]; label = "eig2d_lr", marker = :circle, markersize = 9)
    sw0 = rows[end].swE
    lines!(ax, Mv, sw0 .* (Mv ./ Mv[end]).^2; color = :gray, linestyle = :dash, label = "∝ M²")
    axislegend(ax; position = :lt)
    dst = joinpath(FIGDIR, "subdomains.png"); save(dst, fig)
    println("  saved ", dst)

    # Robust scheduling constant theta as a function of M (which tested theta converge),
    # for the scheduled low-rank solver.
    println("  -- robust schedule constant theta vs M (scheduled eig2d_lr) --")
    schd(θ) = res -> clamp(θ * res, 1e-10, 1.0)
    for M in (4, 8, 16)
        msg = ""
        for θ in (0.05, 0.02, 0.01, 0.005, 0.002)
            Random.seed!(7)
            r = run_poisson_2d_lr(; M, nx = 48, ny = 48, accuracy = 6, maxiter = 800, doplot = false,
                                  opts = (; solver = eig2d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10,
                                            tol_DD = 1e-7, tol_schedule = schd(θ)))
            msg *= @sprintf(" θ=%.3f:%s", θ, r.sweeps < 800 ? "conv" : "STALL")
        end
        println("    M=", M, msg)
    end
    return rows
end

Random.seed!(SEED)
table_and_fig_subdomains()

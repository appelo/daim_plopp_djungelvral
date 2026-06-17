# Per-phase timing breakdown of the 3D Tucker solver, dense eig3d vs low-rank
# eig3d_lr (new Cross-DEIM gfun caching ON), via the TO3D TimerOutput embedded in
# run_poisson_3d_lr (reset at the start of every run).
#
# Writes two files next to this script:
#   eig3d_timing_breakdown.txt  — the formatted TimerOutputs report per (N, solver)
#   eig3d_timing_breakdown.csv  — one row per section: N,solver,section,ncalls,time_s,alloc_GiB,pct
#
#   julia --project=. examples/experiments/eig3d_timing_breakdown.jl 80 160 240
using Printf, TimerOutputs, Dates
include(joinpath(@__DIR__, "..", "sbp_poisson_bvp_3d_lowrank.jl"))

optE   = (; solver = eig3d,    tol_trunc = 1e-10, tol_DD = 1e-7)
optLR  = (; solver = eig3d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7, cache = true)
optLRc = (; solver = eig3d_lr, tol_domain_solver = 1e-10, tol_trunc = 1e-10, tol_DD = 1e-7, cache = true, rhs = :crosssum)

runN(N, o) = run_poisson_3d_lr(; M = 2, nx = N, ny = N, nz = N, accuracy = 4,
                               maxiter = 200, doplot = false, opts = o)
runN(12, optE); runN(12, optLR); runN(12, optLRc)   # warmup / compile (resets TO3D internally)

Ns = isempty(ARGS) ? [80, 160, 240] : parse.(Int, ARGS)
cases = [("eig3d", optE), ("eig3d_lr", optLR), ("eig3d_lr_crosssum", optLRc)]

txt = joinpath(@__DIR__, "eig3d_timing_breakdown.txt")
csv = joinpath(@__DIR__, "eig3d_timing_breakdown.csv")

open(csv, "w") do fc
    println(fc, "N,solver,section,ncalls,time_s,alloc_GiB,pct_of_total")
    open(txt, "w") do ft
        println(ft, "3D Tucker solver per-phase timing  (M=2, p=4, tol_DD=1e-7, cache=on)")
        println(ft, "generated ", string(now()), "\n")
        for N in Ns, (name, o) in cases
            GC.gc()
            r = runN(N, o)                         # TO3D now holds this run's breakdown
            tot = TimerOutputs.tottime(TO3D)       # ns
            hdr = @sprintf("===== N=%d  solver=%s  (wall=%.2fs, sweeps=%d, err=%.2e) =====",
                           N, name, tot / 1e9, r.sweeps, r.errhist[end])
            println(ft, hdr)
            print_timer(ft, TO3D; sortby = :time)
            println(ft, "\n")
            for (sec, t) in TO3D.inner_timers
                println(fc, join((N, name, '"' * sec * '"',
                                  TimerOutputs.ncalls(t),
                                  round(TimerOutputs.time(t) / 1e9, digits = 4),
                                  round(TimerOutputs.allocated(t) / 2^30, digits = 4),
                                  round(100 * TimerOutputs.time(t) / tot, digits = 2)), ','))
            end
            flush(fc); flush(ft)
            @printf("done N=%d %s  wall=%.2fs\n", N, name, tot / 1e9); flush(stdout)
        end
    end
end
println("wrote ", txt, " and ", csv)
println("DONE")

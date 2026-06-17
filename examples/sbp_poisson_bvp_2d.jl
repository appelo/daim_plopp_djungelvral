# 2D Poisson solver with a 1D (in x) Dirichlet–Neumann domain decomposition.
#
# Solve  u_xx + u_yy = f  on [xL,xR]×[yB,yT] with
#   * Dirichlet data on the bottom (y = yB) and outer left/right (x = xL, xR),
#   * Neumann   data on the top    (y = yT, outward normal derivative u_n = +∂y u).
#
# The domain is split IN X into M strips Ω_i = [x_i, x_{i+1}]×[yB,yT]; the
# decomposition is one dimensional but every subdomain is two dimensional.  The
# subdomain unknown is a MATRIX U (nx×ny), U[a,b] indexing x by a and y by b, and
# the stack of them is  us = Vector{Array{Float64}}(undef, M).
#
# Discretization (same as the 1D examples, applied in each direction): SBP–SAT
# second derivatives with the energy-stable (symmetric) SAT.  Per direction the
# matrix form  u_dd ≈ A*U(+/⋅)  + boundary injection  comes from
# `sbp_sat_matrices`.  The 2D operator acting on U is the Sylvester form
#
#     L(U) = A1*U + U*A2ᵀ          (A1 acts on x-rows, A2 on y-columns).
#
# Symmetrization.  With diagonal SBP norms Hx, Hy and sx = √diag(Hx), sy = √diag(Hy),
# A1S = Hx^{1/2} A1 Hx^{-1/2},  A2S = Hy^{1/2} A2 Hy^{-1/2}  are symmetric (neg. def.).
# Writing Û = Hx^{1/2} U Hy^{1/2} turns  A1*U + U*A2ᵀ = RHS  into the symmetric
# Sylvester system
#
#     A1S*Û + Û*A2S = F_S ,        F_S = Hx^{1/2} RHS Hy^{1/2},
#
# solved by a conjugate gradient that works DIRECTLY on matrices: the MatVec is
# L_S(V) = A1S*V + V*A2S and the inner product is Frobenius ⟨U,V⟩ = sum(U.*V).
# The physical solution is recovered as U = Hx^{-1/2} Û Hy^{-1/2}.
#
# Dirichlet–Neumann sweep in x: Ω_1 is Dirichlet on the outer-left edge, every
# later Ω_i is Neumann (interface flux) on its left edge; all are Dirichlet on the
# right edge (interface value λ_i, or the outer-right edge for Ω_M).  The interface
# unknowns λ_i are now functions of y (length-ny vectors).
#
# Run from the package root:
#   julia --project=. examples/sbp_poisson_bvp_2d.jl
#
# Required packages:
#   ] add SummationByPartsOperators LinearAlgebra Plots

using LinearAlgebra
using SummationByPartsOperators
import Plots   # imported (not `using`) so it does not shadow `grid` etc.

include(joinpath(@__DIR__, "sbp_second_derivative_matrix.jl"))

# --- matrix-form conjugate gradient (Sylvester operator) -------------------

dot2(U, V) = sum(U .* V)                       # Frobenius inner product

# Solve  A1S*X + X*A2S = B  for the matrix X, with A1S, A2S symmetric and the
# operator L(X) = A1S*X + X*A2S negative definite, so M = -L is SPD.  CG works
# directly on matrices and may be warm-started with `X0`; returns (X, iters).
function cg2d(A1S, A2S, B; X0 = zero(B), tol = 1e-11, maxiter = 10 * length(B))
    Mop(X) = .-(A1S * X .+ X * A2S)       # SPD operator -L
    b  = .-B                              # solve  Mop(X) = b
    X  = copy(X0)
    R  = b .- Mop(X)                      # residual at the (possibly warm) start
    P  = copy(R)
    rs = dot2(R, R)
    bnorm = sqrt(dot2(b, b))
    for k in 1:maxiter
        MP = Mop(P)
        α  = rs / dot2(P, MP)
        @. X += α * P
        @. R -= α * MP
        rs_new = dot2(R, R)
        sqrt(rs_new) <= tol * bnorm && return X, k
        @. P = R + (rs_new / rs) * P
        rs = rs_new
    end
    return X, maxiter
end

# Solve one 2D subdomain  u_xx + u_yy = F  with x-boundary data gL, gR (length ny)
# and y-boundary data gB, gT (length nx), using the symmetrized matrix CG.
# `mx`, `my` are the x/y `sbp_sat_matrices`; `A1S, A2S, sx, sy` the symmetrized
# operators and norm square roots.  Returns the physical solution U (nx×ny).
function solve_subdomain_2d(mx, my, A1S, A2S, sx, sy, gL, gR, gB, gT, F; U0 = nothing)
    # A1*U + U*A2ᵀ = RHS,  RHS = F − x-injection − y-injection.
    xinj = mx.BL * gL' .+ mx.BR * gR'         # nx×ny  (outer products)
    yinj = gB * my.BL' .+ gT * my.BR'         # nx×ny
    RHS  = F .- xinj .- yinj
    F_S  = sx .* RHS .* sy'                    # Hx^{1/2} RHS Hy^{1/2}
    # Warm start: previous physical guess U0 maps to Û0 = Hx^{1/2} U0 Hy^{1/2}.
    X0 = U0 === nothing ? zero(F_S) : (sx .* U0 .* sy')
    Uhat, iters = cg2d(A1S, A2S, F_S; X0 = X0)
    U = (Uhat ./ sx) ./ sy'                    # Hx^{-1/2} Û Hy^{-1/2}
    return U, iters
end

# --- driver ----------------------------------------------------------------

function run_poisson_2d(; M = 4, nx = 25, ny = 101, accuracy = 6, θ = 0.5,
                        xL = -1.0, xR = 1.0, yB = 0.0, yT = 1.0,
                        tol = 1e-9, maxiter = 1000, plotevery = 1, pause = 0.05,
                        plotkind = :surface)        # :surface or :contour
    M >= 1 || throw(ArgumentError("need at least one subdomain"))

    # Manufactured solution for Poisson  u_xx + u_yy = f  (non-harmonic).
    u_exact(x, y)  = exp(x) * cos(y) + x^3
    uy_exact(x, y) = -exp(x) * sin(y)          # ∂y u_exact, for the top Neumann
    f_force(x, y)  = 6 * x                      # forcing = u_xx + u_yy of u_exact

    xnodes = collect(range(xL, xR; length = M + 1))   # subdomain edges in x

    # Shared y-operator: Dirichlet bottom (left), Neumann top (right) → bc [1,0,0,1].
    D1y = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = yB, xmax = yT, N = ny)
    y   = grid(D1y)
    my  = sbp_sat_matrices(D1y, [1.0, 0.0, 0.0, 1.0]; symmetric = true)
    sy  = sqrt.(diag(my.H))
    A2S = (sy .* my.A) ./ sy'

    # Per-subdomain x-operators (DN roles) and their symmetrizations.
    Dxs = Vector{Any}(undef, M)
    xs  = Vector{Vector{Float64}}(undef, M)
    mxs = Vector{Any}(undef, M)
    sxs = Vector{Vector{Float64}}(undef, M)
    A1Ss = Vector{Matrix{Float64}}(undef, M)
    for i in 1:M
        Dxs[i] = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                     accuracy_order = accuracy,
                                     xmin = xnodes[i], xmax = xnodes[i + 1], N = nx)
        xs[i]  = grid(Dxs[i])
        bc = i == 1 ? [1.0, 0.0, 1.0, 0.0] : [0.0, 1.0, 1.0, 0.0]
        mxs[i]  = sbp_sat_matrices(Dxs[i], bc; symmetric = true)
        sxs[i]  = sqrt.(diag(mxs[i].H))
        A1Ss[i] = (sxs[i] .* mxs[i].A) ./ sxs[i]'
    end

    Fs = [f_force.(xs[i], y') for i in 1:M]     # Poisson volume forcing per subdomain

    # y-boundary data per subdomain (functions of x on that subdomain's grid).
    gBs = [u_exact.(xs[i], yB)  for i in 1:M]   # bottom Dirichlet value
    gTs = [uy_exact.(xs[i], yT) for i in 1:M]   # top Neumann  (u_n = +∂y u)

    println("2D Poisson, Dirichlet–Neumann DD (1D in x), each subdomain 2D")
    println("  domain   : [$xL, $xR]×[$yB, $yT],  M = $M strips,  nx = $nx, ny = $ny")
    println("  y-BCs    : Dirichlet bottom, Neumann top ;  x-outer BCs: Dirichlet")
    println("  relax θ  : $θ ,  subdomain solves: symmetrized matrix CG (A1S*U + U*A2S)")
    println()
    println("  ", rpar("sweep", 7), rpar("‖Δλ‖ residual", 16), rpar("CG its / sweep", 16))

    us = Vector{Array{Float64}}(undef, M)       # physical subdomain solutions

    # One forward DN sweep (interface unknowns λ_i are length-ny vectors).
    function forward_sweep!(λ)
        newλ = [copy(λ[i]) for i in 1:M - 1]
        q = zeros(ny)                            # interface flux ∂x U from the left
        cgs = 0
        for i in 1:M
            gL = i == 1 ? u_exact.(xL, y) : -q   # outer Dirichlet(y) or Neumann flux
            gR = i == M ? u_exact.(xR, y) : λ[i] # outer Dirichlet(y) or interface λ_i
            U0 = isassigned(us, i) ? us[i] : nothing   # warm start from previous sweep
            us[i], its = solve_subdomain_2d(mxs[i], my, A1Ss[i], A2S, sxs[i], sy,
                                            gL, gR, gBs[i], gTs[i], Fs[i]; U0 = U0)
            cgs += its
            if i < M           # flux ∂x U at the right edge, per y-row → length ny
                q = [derivative_right(Dxs[i], view(us[i], :, b), Val(1)) for b in 1:ny]
            end
            i >= 2 && (newλ[i - 1] = us[i][1, :]) # left trace at the interface
        end
        return newλ, cgs
    end

    # Live per-sweep plot: assembled solution as a 3D surface (or contour).
    zlo, zhi = extrema(u_exact(x, yy) for x in range(xL, xR; length = 50)
                                      for yy in range(yB, yT; length = 50))
    function show_frame(k, res)
        fp = plotkind === :contour ?
            Plots.plot(xlabel = "x", ylabel = "y", title = "DN sweep $k   ‖Δλ‖ = $(fmt(res))") :
            Plots.plot(xlabel = "x", ylabel = "y", zlabel = "u",
                       zlims = (zlo, zhi), camera = (35, 30),
                       title = "DN sweep $k   ‖Δλ‖ = $(fmt(res))")
        for i in 1:M
            if plotkind === :contour
                Plots.contour!(fp, xs[i], y, us[i]'; colorbar = (i == M), levels = 12)
            else
                Plots.surface!(fp, xs[i], y, us[i]'; colorbar = false,
                               label = "", clims = (zlo, zhi))
            end
        end
        display(fp)
        pause > 0 && sleep(pause)
    end

    λ = [zeros(ny) for _ in 1:M - 1]            # interface traces (wrong guess)
    hist = Float64[]
    total_cg = 0
    for k in 1:maxiter
        newλ, sweep_cg = forward_sweep!(λ)
        res = M > 1 ? maximum(maximum(abs, newλ[i] .- λ[i]) for i in 1:M - 1) : 0.0
        converged = M == 1 || res < tol
        (k == 1 || k % plotevery == 0 || converged) && show_frame(k, res)
        for i in 1:M - 1
            @. λ[i] = θ * newλ[i] + (1 - θ) * λ[i]
        end
        total_cg += sweep_cg
        push!(hist, res)
        if k <= 5 || k % 10 == 0 || converged
            println("  ", rpar(string(k), 7), rpar(fmt(res), 16), rpar(string(sweep_cg), 16))
        end
        converged && break
    end

    forward_sweep!(λ)                            # final consistent sweep
    show_frame(length(hist), isempty(hist) ? 0.0 : hist[end])

    err = maximum(maximum(abs, us[i] .- u_exact.(xs[i], y')) for i in 1:M)
    println()
    println("  converged in $(length(hist)) sweeps, $total_cg total CG iterations")
    println("  max|U − u_exact| over all subdomains = ", fmt(err))

    # Final figure: solution surface + DN convergence history.
    p1 = plotkind === :contour ?
        Plots.plot(xlabel = "x", ylabel = "y", title = "2D Poisson DD (M = $M)") :
        Plots.plot(xlabel = "x", ylabel = "y", zlabel = "u", camera = (35, 30),
                   title = "2D Poisson DD (M = $M)")
    for i in 1:M
        if plotkind === :contour
            Plots.contour!(p1, xs[i], y, us[i]'; colorbar = (i == M), levels = 12)
        else
            Plots.surface!(p1, xs[i], y, us[i]'; colorbar = false)
        end
    end
    p2 = Plots.plot(1:length(hist), max.(hist, 1e-16); label = "‖Δλ‖", lw = 2,
                    marker = :circle, ms = 3, yscale = :log10, xlabel = "DN sweep",
                    ylabel = "interface residual", title = "convergence", color = :red)
    plt = Plots.plot(p1, p2; layout = (1, 2), size = (1100, 450))
    outfile = joinpath(@__DIR__, "sbp_poisson_bvp_2d.png")
    Plots.savefig(plt, outfile)
    println("  saved plot to ", outfile)
    display(plt)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_poisson_2d()
end

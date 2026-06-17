# 3D Poisson solver with a 1D (in x) Dirichlet–Neumann domain decomposition.
#
# Solve  u_xx + u_yy + u_zz = f  on [xL,xR]×[yB,yT]×[zB,zT] with
#   * Dirichlet on the outer x faces (and the x-interfaces of the decomposition),
#   * Dirichlet on the y-bottom, Neumann on the y-top (u_n = +∂y u),
#   * Dirichlet on both z faces.
#
# The domain is split IN X into M slabs Ω_i = [x_i,x_{i+1}]×[yB,yT]×[zB,zT]; the
# decomposition is one dimensional but each subdomain is three dimensional.  The
# subdomain unknown is a 3D ARRAY U (nx×ny×nz), U[a,b,c] indexing x,y,z, and the
# stack is  us = Vector{Array{Float64}}(undef, M).
#
# Discretization (same SBP–SAT, energy-stable / symmetric SAT, applied in each
# direction).  The 3D operator acting on U is the sum of three tensor
# mode-products of the symmetrized SBP operators:
#
#     L(U) = A1S·₁U + A2S·₂U + A3S·₃U          (A_kS contracted into dim k of U).
#
# Symmetrization.  With diagonal SBP norms Hx,Hy,Hz and sx=√diag(Hx), sy,sz, the
# operators A1S = Hx^{1/2}A1Hx^{-1/2}, A2S, A3S are symmetric (neg. def.).  Writing
# Û = S .* U with S[a,b,c] = sx[a]·sy[b]·sz[c] turns  A1·₁U + A2·₂U + A3·₃U = RHS
# into the symmetric system  A1S·₁Û + A2S·₂Û + A3S·₃Û = F_S,  F_S = S .* RHS,
# solved by a conjugate gradient that works DIRECTLY on 3D arrays (MatVec L,
# Frobenius inner product sum(U.*V)).  The physical solution is U = Û ./ S.
#
# Dirichlet–Neumann sweep in x (same as the 2D file): Ω_1 Dirichlet on the
# outer-left face, later Ω_i Neumann (interface flux) on the left face, all
# Dirichlet on the right face (interface value λ_i, or the outer-right face for
# Ω_M).  The interface unknowns λ_i are now y–z planes (ny×nz matrices).
#
# Run from the package root:
#   julia --project=. examples/sbp_poisson_bvp_3d.jl
#
# Required packages:
#   ] add SummationByPartsOperators LinearAlgebra Plots

using LinearAlgebra
using SummationByPartsOperators
import Plots   # imported (not `using`) so it does not shadow `grid` etc.

include(joinpath(@__DIR__, "sbp_second_derivative_matrix.jl"))

# --- tensor-mode operator and 3D conjugate gradient ------------------------

dot3(U, V) = sum(U .* V)                        # Frobenius inner product

# L(U) = A1S·₁U + A2S·₂U + A3S·₃U  (each A_kS symmetric, so each mode is a plain
# matrix multiply on the matricized array).
function applyL3(A1S, A2S, A3S, U)
    nx, ny, nz = size(U)
    V = reshape(A1S * reshape(U, nx, ny * nz), nx, ny, nz)        # mode 1
    @inbounds for c in 1:nz                                       # mode 2
        @views V[:, :, c] .+= U[:, :, c] * A2S
    end
    V .+= reshape(reshape(U, nx * ny, nz) * A3S, nx, ny, nz)      # mode 3
    return V
end

# Solve  L(X) = B  with L symmetric negative definite (M = -L is SPD).  Works
# directly on 3D arrays, warm-started with `X0`; returns (X, iters).
function cg3d(A1S, A2S, A3S, B; X0 = zero(B), tol = 1e-11, maxiter = 10 * length(B))
    Mop(X) = .-applyL3(A1S, A2S, A3S, X)        # SPD operator -L
    b  = .-B
    X  = copy(X0)
    R  = b .- Mop(X)
    P  = copy(R)
    rs = dot3(R, R)
    bnorm = sqrt(dot3(b, b))
    for k in 1:maxiter
        MP = Mop(P)
        α  = rs / dot3(P, MP)
        @. X += α * P
        @. R -= α * MP
        rs_new = dot3(R, R)
        sqrt(rs_new) <= tol * bnorm && return X, k
        @. P = R + (rs_new / rs) * P
        rs = rs_new
    end
    return X, maxiter
end

# Solve one 3D subdomain  u_xx+u_yy+u_zz = F.  Boundary data are 2D arrays over
# the other two dimensions: gL,gR (ny×nz), gyB,gyT (nx×nz), gzB,gzT (nx×ny).
# Uses the symmetrized tensor CG; returns the physical solution U (nx×ny×nz).
function solve_subdomain_3d(mx, my, mz, A1S, A2S, A3S, sx, sy, sz,
                            gL, gR, gyB, gyT, gzB, gzT, F; U0 = nothing)
    nx, ny, nz = length(sx), length(sy), length(sz)
    # boundary injection by tensor outer products (B-vector ⊗ boundary data)
    xinj = reshape(mx.BL, nx, 1, 1) .* reshape(gL, 1, ny, nz) .+
           reshape(mx.BR, nx, 1, 1) .* reshape(gR, 1, ny, nz)
    yinj = reshape(gyB, nx, 1, nz) .* reshape(my.BL, 1, ny, 1) .+
           reshape(gyT, nx, 1, nz) .* reshape(my.BR, 1, ny, 1)
    zinj = reshape(gzB, nx, ny, 1) .* reshape(mz.BL, 1, 1, nz) .+
           reshape(gzT, nx, ny, 1) .* reshape(mz.BR, 1, 1, nz)
    RHS = F .- xinj .- yinj .- zinj
    S   = sx .* reshape(sy, 1, ny) .* reshape(sz, 1, 1, nz)       # Hx^{1/2}⊗Hy^{1/2}⊗Hz^{1/2}
    F_S = S .* RHS
    X0  = U0 === nothing ? zero(F_S) : (S .* U0)                  # warm start in Û space
    Uhat, iters = cg3d(A1S, A2S, A3S, F_S; X0 = X0)
    return Uhat ./ S, iters                                       # physical U
end

# --- driver ----------------------------------------------------------------

function run_poisson_3d(; M = 3, nx = 20, ny = 100, nz = 100, accuracy = 4, θ = 0.5,
                        xL = -1.0, xR = 1.0, yB = 0.0, yT = 1.0, zB = 0.0, zT = 1.0,
                        tol = 1e-9, maxiter = 1000, plotevery = 1, pause = 0.05,
                        plotkind = :surface)        # :surface or :contour (mid-z slice)
    M >= 1 || throw(ArgumentError("need at least one subdomain"))

    # Manufactured solution for Poisson  Δu = f  (non-harmonic).
    u_exact(x, y, z)  = exp(sqrt(2) * x) * cos(y) * cos(z) + x^3
    uy_exact(x, y, z) = -exp(sqrt(2) * x) * sin(y) * cos(z)        # ∂y u, for y-top Neumann
    f_force(x, y, z)  = 6 * x                                      # forcing = Δu of u_exact

    xnodes = collect(range(xL, xR; length = M + 1))

    # Shared y-operator (Dirichlet bottom / Neumann top) and z-operator (Dirichlet both).
    D1y = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = yB, xmax = yT, N = ny)
    D1z = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = zB, xmax = zT, N = nz)
    y   = grid(D1y);  z = grid(D1z)
    my  = sbp_sat_matrices(D1y, [1.0, 0.0, 0.0, 1.0]; symmetric = true)
    mz  = sbp_sat_matrices(D1z, [1.0, 0.0, 1.0, 0.0]; symmetric = true)
    sy  = sqrt.(diag(my.H));  sz = sqrt.(diag(mz.H))
    A2S = (sy .* my.A) ./ sy';  A3S = (sz .* mz.A) ./ sz'

    # Per-subdomain x-operators (DN roles), symmetrizations, and right-flux functionals.
    Dxs = Vector{Any}(undef, M)
    xs  = Vector{Vector{Float64}}(undef, M)
    mxs = Vector{Any}(undef, M)
    sxs = Vector{Vector{Float64}}(undef, M)
    A1Ss = Vector{Matrix{Float64}}(undef, M)
    dRx = Vector{Vector{Float64}}(undef, M)        # right-derivative functional (∂x at x_right)
    for i in 1:M
        Dxs[i] = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                     accuracy_order = accuracy,
                                     xmin = xnodes[i], xmax = xnodes[i + 1], N = nx)
        xs[i]  = grid(Dxs[i])
        bc = i == 1 ? [1.0, 0.0, 1.0, 0.0] : [0.0, 1.0, 1.0, 0.0]
        mxs[i]  = sbp_sat_matrices(Dxs[i], bc; symmetric = true)
        sxs[i]  = sqrt.(diag(mxs[i].H))
        A1Ss[i] = (sxs[i] .* mxs[i].A) ./ sxs[i]'
        t = zeros(nx); mul_transpose_derivative_right!(t, Dxs[i], Val(1), 1.0, false)
        dRx[i] = t
    end

    # Poisson volume forcing per subdomain (nx×ny×nz; f depends only on x here).
    Fs = [f_force.(xs[i], reshape(y, 1, ny), reshape(z, 1, 1, nz)) for i in 1:M]

    # Transverse boundary data per subdomain, as 2D arrays over the other two dims:
    #   y faces → nx×nz (functions of x,z);  z faces → nx×ny (functions of x,y).
    gyBs = Vector{Matrix{Float64}}(undef, M)        # y-bottom Dirichlet value
    gyTs = Vector{Matrix{Float64}}(undef, M)        # y-top    Neumann (u_n = +∂y u)
    gzBs = Vector{Matrix{Float64}}(undef, M)        # z-bottom Dirichlet value
    gzTs = Vector{Matrix{Float64}}(undef, M)        # z-top    Dirichlet value
    for i in 1:M
        gyBs[i] = [u_exact(xs[i][a], yB, z[c])  for a in 1:nx, c in 1:nz]
        gyTs[i] = [uy_exact(xs[i][a], yT, z[c]) for a in 1:nx, c in 1:nz]
        gzBs[i] = [u_exact(xs[i][a], y[b], zB)  for a in 1:nx, b in 1:ny]
        gzTs[i] = [u_exact(xs[i][a], y[b], zT)  for a in 1:nx, b in 1:ny]
    end

    # x-face Dirichlet data (ny×nz) on the outer boundaries.
    gxL = [u_exact(xL, y[b], z[c]) for b in 1:ny, c in 1:nz]
    gxR = [u_exact(xR, y[b], z[c]) for b in 1:ny, c in 1:nz]

    println("3D Poisson, Dirichlet–Neumann DD (1D in x), each subdomain 3D")
    println("  domain   : [$xL,$xR]×[$yB,$yT]×[$zB,$zT],  M = $M slabs,  nx=$nx ny=$ny nz=$nz")
    println("  BCs      : x Dirichlet/interface ; y Dir-bottom/Neu-top ; z Dirichlet both")
    println("  relax θ  : $θ ,  subdomain solves: tensor CG (A1S·₁U + A2S·₂U + A3S·₃U)")
    println()
    println("  ", rpar("sweep", 7), rpar("‖Δλ‖ residual", 16), rpar("CG its / sweep", 16))

    us = Vector{Array{Float64}}(undef, M)           # physical 3D subdomain solutions

    function forward_sweep!(λ)
        newλ = [copy(λ[i]) for i in 1:M - 1]
        q = zeros(ny, nz)                            # interface flux plane ∂x U from the left
        cgs = 0
        for i in 1:M
            gL = i == 1 ? gxL : -q                   # outer Dirichlet plane or Neumann flux
            gR = i == M ? gxR : λ[i]                 # outer Dirichlet plane or interface λ_i
            U0 = isassigned(us, i) ? us[i] : nothing  # warm start from previous sweep
            us[i], its = solve_subdomain_3d(mxs[i], my, mz, A1Ss[i], A2S, A3S,
                                            sxs[i], sy, sz, gL, gR,
                                            gyBs[i], gyTs[i], gzBs[i], gzTs[i], Fs[i]; U0 = U0)
            cgs += its
            if i < M       # flux ∂x U at the right face: contract dim 1 with dRx → ny×nz
                q = reshape(dRx[i]' * reshape(us[i], nx, ny * nz), ny, nz)
            end
            i >= 2 && (newλ[i - 1] = us[i][1, :, :])  # left-face trace at the interface
        end
        return newλ, cgs
    end

    # Live per-sweep plot of the mid-z slice u(x, y, z_mid) as a 3D surface.
    c0 = (nz + 1) ÷ 2
    zlo, zhi = extrema(u_exact(x, yy, z[c0]) for x in range(xL, xR; length = 40)
                                             for yy in range(yB, yT; length = 40))
    function show_frame(k, res)
        fp = plotkind === :contour ?
            Plots.plot(xlabel = "x", ylabel = "y",
                       title = "DN sweep $k   ‖Δλ‖ = $(fmt(res))  (z = $(round(z[c0],digits=2)))") :
            Plots.plot(xlabel = "x", ylabel = "y", zlabel = "u", zlims = (zlo, zhi),
                       camera = (35, 30),
                       title = "DN sweep $k   ‖Δλ‖ = $(fmt(res))  (z = $(round(z[c0],digits=2)))")
        for i in 1:M
            sl = us[i][:, :, c0]'
            plotkind === :contour ?
                Plots.contour!(fp, xs[i], y, sl; colorbar = (i == M), levels = 12) :
                Plots.surface!(fp, xs[i], y, sl; colorbar = false)
        end
        display(fp)
        pause > 0 && sleep(pause)
    end

    λ = [zeros(ny, nz) for _ in 1:M - 1]            # interface planes (wrong guess)
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

    forward_sweep!(λ)                               # final consistent sweep
    show_frame(length(hist), isempty(hist) ? 0.0 : hist[end])

    err = maximum(maximum(abs, us[i] .- [u_exact(xs[i][a], y[b], z[c])
                  for a in 1:nx, b in 1:ny, c in 1:nz]) for i in 1:M)
    println()
    println("  converged in $(length(hist)) sweeps, $total_cg total CG iterations")
    println("  max|U − u_exact| over all subdomains = ", fmt(err))

    # Final figure: mid-z slice surface + DN convergence history.
    p1 = plotkind === :contour ?
        Plots.plot(xlabel = "x", ylabel = "y", title = "3D Poisson DD, mid-z slice (M = $M)") :
        Plots.plot(xlabel = "x", ylabel = "y", zlabel = "u", camera = (35, 30),
                   title = "3D Poisson DD, mid-z slice (M = $M)")
    for i in 1:M
        sl = us[i][:, :, c0]'
        plotkind === :contour ?
            Plots.contour!(p1, xs[i], y, sl; colorbar = (i == M), levels = 12) :
            Plots.surface!(p1, xs[i], y, sl; colorbar = false)
    end
    p2 = Plots.plot(1:length(hist), max.(hist, 1e-16); label = "‖Δλ‖", lw = 2,
                    marker = :circle, ms = 3, yscale = :log10, xlabel = "DN sweep",
                    ylabel = "interface residual", title = "convergence", color = :red)
    plt = Plots.plot(p1, p2; layout = (1, 2), size = (1100, 450))
    outfile = joinpath(@__DIR__, "sbp_poisson_bvp_3d.png")
    Plots.savefig(plt, outfile)
    println("  saved plot to ", outfile)
    display(plt)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_poisson_3d()
end

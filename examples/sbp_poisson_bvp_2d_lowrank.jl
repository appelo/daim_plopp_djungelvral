# Low-rank (LRSVD) 2D Poisson Dirichlet–Neumann domain-decomposition solver, with the
# live + final plotting from `sbp_poisson_bvp_2d.jl` (the dense reference version)
# re-added.  Per-sweep console tables are omitted; subdomain solutions are LRSVDs and
# are densified only for visualization.
#
# Solve  w_xx + w_yy = f  on [xL,xR]×[yB,yT], split IN X into M strips, with
#   * Dirichlet data on the bottom (y = yB) and outer left/right (x = xL, xR),
#   * Neumann   data on the top    (y = yT, w_n = +∂y w).
# Each subdomain unknown is a low-rank MATRIX W = U·S·Vᵀ (nx×ny).  U, V hold the
# left/right singular vectors, so the PDE solution is written w (not u) throughout.

using LinearAlgebra
using SummationByPartsOperators
using LRDD
using CairoMakie   # static plotting backend; `grid` is not exported, so no clash

include(joinpath(@__DIR__, "sbp_second_derivative_matrix.jl"))

# --- precomputed operators -------------------------------------------------
# All per-direction quantities are built once at setup (the symmetrized operators
# are constant across the DN iteration) and bundled here, including the symmetric
# eigendecompositions used by `eig2d`.

mutable struct XOperator          # per subdomain (x-direction)
    Dx
    x::Vector{Float64}
    mx
    sx::Vector{Float64}
    A1S::Matrix{Float64}
    dRx::Vector{Float64}          # right-derivative functional (∂x at x_right)
    eigA1                         # eigen(Symmetric(A1S))
end

mutable struct YOperator          # shared (y-direction)
    D1y
    y::Vector{Float64}
    my
    sy::Vector{Float64}
    A2S::Matrix{Float64}
    eigA2                         # eigen(Symmetric(A2S))
end

# --- matrix-form conjugate gradient (Sylvester operator) -------------------

dot2(U, V) = sum(U .* V)                       # Frobenius inner product

# Solve  A1S*X + X*A2S = B  for the matrix X, with A1S = xb.A1S, A2S = yb.A2S
# symmetric and L(X) = A1S*X + X*A2S negative definite, so M = -L is SPD.  CG works
# directly on matrices and may be warm-started with `X0`; returns (X, iters).
function cg2d(xb::XOperator, yb::YOperator, B; X0 = zero(B), tol = 1e-11, maxiter = 10 * length(B))
    A1S, A2S = xb.A1S, yb.A2S
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

# Dense Sylvester solver by symmetric eigen-diagonalization — a direct, drop-in
# alternative to `cg2d` with the same `(xb, yb, B; X0, tol)` signature.  Uses the
# precomputed eigendecompositions A1S = Q1 Λ1 Q1ᵀ (xb.eigA1) and A2S = Q2 Λ2 Q2ᵀ
# (yb.eigA2): A1S*X + X*A2S = B becomes (Λ1[i]+Λ2[j]) Y[i,j] = (Q1ᵀ B Q2)[i,j], so
# Y = (Q1ᵀ B Q2) ./ (λ1 .+ λ2ᵀ) and X = Q1 Y Q2ᵀ.  `X0`/`tol` are ignored (direct).
function eig2d(xb::XOperator, yb::YOperator, B; X0 = nothing, tol = 0)
    λ1, Q1 = xb.eigA1.values, xb.eigA1.vectors
    λ2, Q2 = yb.eigA2.values, yb.eigA2.vectors
    Y = (Q1' * B * Q2) ./ (λ1 .+ λ2')
    return Q1 * Y * Q2', 0
end

# Low-rank counterpart of `eig2d`: the right-hand side `B` is an LRSVD and the eigenbasis
# solve  Y[i,j] = (Q1ᵀ B Q2)[i,j] / (λ1[i] + λ2[j])  is built as a low-rank matrix with
# `crossDEIM` (matrix-free, sampling entries of Y).  The core G = Q1ᵀ B Q2 stays low-rank,
# G = (Q1ᵀB.U) diag(B.S) (Q2ᵀB.V)ᵀ, so its entries are cheap.  As the actual solve,
# `crossDEIM` uses the solver tolerance `opts.tol_domain_solver` (the final solution
# truncation by `tol_trunc` happens in the caller).  Returns the solution X = Q1 Y Q2ᵀ
# as an LRSVD (and the crossDEIM iteration count).
function eig2d_lr(xb::XOperator, yb::YOperator, B::LRSVD;
                  opts = (; tol_domain_solver = 1e-11, tol_trunc = 1e-11))
    λ1, Q1 = xb.eigA1.values, xb.eigA1.vectors
    λ2, Q2 = yb.eigA2.values, yb.eigA2.vectors
    # core G = Q1ᵀ B Q2 in factored form; G[i,j] = P1[i,:] · SP2[j,:]
    P1  = Q1' * B.U                 # nx × r
    SP2 = (Q2' * B.V) .* B.S'       # ny × r  (right factor scaled by the singular values)
    gfun = (i, j) -> dot(view(P1, i, :), view(SP2, j, :)) / (λ1[i] + λ2[j])
    # crossDEIM tolerance = opts.tol_domain_solver (the solve accuracy); seed with B's factors.
    rmax   = min(size(B.U, 1), size(B.V, 1))
    cdopts = (; tol = get(opts, :tol_domain_solver, get(opts, :tol_trunc, 1e-11)),
                r_max = rmax, r_in = rmax, max_iter = 20)
    Ylr, info = crossDEIM(gfun, LRSVD(B.U, B.S, B.V), cdopts)
    # X = Q1 Y Q2ᵀ, kept low-rank by rotating the factors back through Q1, Q2.
    return LRSVD(Q1 * Ylr.U, Ylr.S, Q2 * Ylr.V), info[1]
end

# Spectrally-truncated mirror of `eig2d_lr`: identical, except the eigenbasis is
# restricted to the `p` *most important* eigenvectors per direction.  Importance is
# ranked by |eigenvalue|, smaller-is-more-important: A1S, A2S are symmetric negative
# definite, so the smallest-|λ| modes are the smoothest, and they dominate the eigenbasis
# solve because the entry weight 1/(λ1[i]+λ2[j]) is largest there.  Keeping the leading
# p1 = min(p,nx), p2 = min(p,ny) eigenvectors gives a coarse spectral solve whose solution
# lives in the leading p1×p2 eigenspace (rank ≤ min(p1,p2)).  `p` is read from `opts.p`
# (default `typemax(Int)`, i.e. full basis ⇒ exactly `eig2d_lr`).  Same signature/return
# as `eig2d_lr` (the LRSVD solution and the crossDEIM iteration count).
function eig2d_lr_p(xb::XOperator, yb::YOperator, B::LRSVD;
                    opts = (; tol_domain_solver = 1e-11, tol_trunc = 1e-11, p = typemax(Int)))
    λ1f, Q1f = xb.eigA1.values, xb.eigA1.vectors
    λ2f, Q2f = yb.eigA2.values, yb.eigA2.vectors
    p = get(opts, :p, typemax(Int))
    # indices of the p smallest |eigenvalue| (most important = smoothest modes).
    i1 = sortperm(abs.(λ1f))[1:min(p, length(λ1f))]
    i2 = sortperm(abs.(λ2f))[1:min(p, length(λ2f))]
    λ1, Q1 = λ1f[i1], Q1f[:, i1]    # Q1: nx × p1
    λ2, Q2 = λ2f[i2], Q2f[:, i2]    # Q2: ny × p2
    # core G = Q1ᵀ B Q2 on the reduced eigenspace, factored: G[i,j] = P1[i,:] · SP2[j,:]
    P1   = Q1' * B.U                # p1 × r
    Q2tV = Q2' * B.V                # p2 × r
    SP2  = Q2tV .* B.S'             # p2 × r  (right factor scaled by the singular values)
    gfun = (i, j) -> dot(view(P1, i, :), view(SP2, j, :)) / (λ1[i] + λ2[j])
    rmax   = min(size(Q1, 2), size(Q2, 2))   # = min(p1, p2)
    cdopts = (; tol = get(opts, :tol_domain_solver, get(opts, :tol_trunc, 1e-11)),
                r_max = rmax, r_in = rmax, max_iter = 20)
    # seed crossDEIM with the projected RHS factors (the numerator G); recompress to
    # orthonormal factors of rank ≤ min(p1,p2) so the seed is never wider than the
    # reduced p1×p2 space (B's rank r may exceed p when p is small).
    seed = truncsum([LRSVD(P1, B.S, Q2tV)], 1e-14, rmax, LRSVD)
    Ylr, info = crossDEIM(gfun, seed, cdopts)
    # X = Q1 Y Q2ᵀ, rotated back through the reduced eigenvectors.
    return LRSVD(Q1 * Ylr.U, Ylr.S, Q2 * Ylr.V), info[1]
end

# Solve one 2D subdomain  w_xx + w_yy = F  with x-boundary data gL, gR (length ny)
# and y-boundary data gB, gT (length nx).  The forcing `F` and the warm start `W0`
# are low-rank (LRSVD); the boundary injections and the RHS are assembled in low-rank
# form (RHS via `truncsum`), and the symmetrizer Hx^{1/2}(·)Hy^{1/2} acts directly on
# the factors.  `opts` is a NamedTuple of solver options: `opts.solver` selects the
# dense Sylvester solver (`cg2d` or `eig2d`), `opts.tol_domain_solver` is the solver
# tolerance, and `opts.tol_trunc` is the tolerance for truncating the returned LRSVD
# (defaults to `opts.tol_domain_solver` if absent).  The symmetrized system is densified
# for the (still dense) solve — to be replaced by a low-rank solver later.
# Returns the truncated LRSVD solution and the solver iteration count.
function solve_subdomain_2d_lr(xb::XOperator, yb::YOperator, gL, gR, gB, gT, F;
                               W0 = nothing, opts = (; solver = cg2d, tol_domain_solver = 1e-11, tol_trunc = 1e-11))
    # RHS = F − x-injection − y-injection, all low-rank (LRSVD).
    xinj = LRSVD(hcat(xb.mx.BL, xb.mx.BR), [1.0, 1.0], hcat(gL, gR))   # rank-2: BL⊗gL + BR⊗gR
    yinj = LRSVD(hcat(gB, gT), [1.0, 1.0], hcat(yb.my.BL, yb.my.BR))   # rank-2: gB⊗BL + gT⊗BR
    rmax = min(size(F.U, 1), size(F.V, 1))
    # near-exact RHS assembly (1e-14), kept independent of opts.tol_trunc so a loose
    # solution-truncation tolerance never degrades the right-hand side.
    RHS  = truncsum([F, LRSVD(xinj.U, -xinj.S, xinj.V), LRSVD(yinj.U, -yinj.S, yinj.V)],
                    1e-14, rmax, LRSVD)
    # F_S = Hx^{1/2} RHS Hy^{1/2}, applied directly to the factors of RHS.
    F_S = LRSVD(xb.sx .* RHS.U, RHS.S, yb.sy .* RHS.V)

    # Warm start in Û space: X0 = Hx^{1/2} W0 Hy^{1/2} (also low-rank).
    X0 = W0 === nothing ? nothing : LRSVD(xb.sx .* W0.U, W0.S, yb.sy .* W0.V)

    # cg2d / eig2d use a dense bridge (densify → solve → re-factor); the low-rank
    # solvers eig2d_lr and eig2d_lr_p work directly on the LRSVD factors (no densification).
    if opts.solver === cg2d || opts.solver === eig2d
        F_S_dense = F_S.U * Diagonal(F_S.S) * F_S.V'
        X0_dense  = X0 === nothing ? zero(F_S_dense) : X0.U * Diagonal(X0.S) * X0.V'
        What, iters = opts.solver(xb, yb, F_S_dense; X0 = X0_dense, tol = get(opts, :tol_domain_solver, 0.0))
        W = svd((What ./ xb.sx) ./ yb.sy')         # Hx^{-1/2} Ŵ Hy^{-1/2}, factored
        Wout = trunclr(LRSVD(W.U, W.S, W.V), get(opts, :tol_trunc, get(opts, :tol_domain_solver, 0.0)))
    elseif opts.solver === eig2d_lr || opts.solver === eig2d_lr_p
        What, iters = opts.solver(xb, yb, F_S; opts = opts)          # Ŵ as an LRSVD
        # unwind the symmetrizer on the factors: W = Hx^{-1/2} Ŵ Hy^{-1/2}, then recompress.
        Wlr  = LRSVD(What.U ./ xb.sx, What.S, What.V ./ yb.sy)
        Wout = truncsum([Wlr], get(opts, :tol_trunc, get(opts, :tol_domain_solver, 0.0)), rmax, LRSVD)
    else
        error("solver $(opts.solver) is not supported; use cg2d, eig2d, eig2d_lr, or eig2d_lr_p")
    end

    return Wout, iters   # truncated LRSVD solution
end

# --- driver ----------------------------------------------------------------

# Keyword arguments
#   M, nx, ny  : number of x-subdomains, and grid points per subdomain (x, y).
#   accuracy   : SBP accuracy order.
#   θ          : Dirichlet–Neumann interface relaxation factor in (0, 1].
#   xL,xR,yB,yT: domain bounds.
#   maxiter    : maximum number of DN sweeps.
#   plotkind   : :surface (3D) or :contour, for the final solution panel (CairoMakie).
#   animate    : if true, record a movie of the assembled solution at every DN sweep
#                (reproduces the old per-iteration live view; CairoMakie is static).
#   animfile   : output path for the animation; format inferred from the extension
#                (.mp4 default, .gif also works).
#   framerate  : animation frame rate (frames per second).
#   opts       : NamedTuple of solver / domain-decomposition options:
#                  opts.solver            — the dense Sylvester solver for A1S*X + X*A2S = B:
#                                             cg2d  → matrix conjugate gradient (iterative, warm-started)
#                                             eig2d → symmetric eigen-diagonalization (direct, exact)
#                                             eig2d_lr   → low-rank eigen-diagonalization (Cross-DEIM)
#                                             eig2d_lr_p → eig2d_lr on the p smallest-|λ| eigenvectors
#                  opts.p                 — (eig2d_lr_p only) number of most-important eigenvectors
#                                            (smallest |eigenvalue|) kept per direction.
#                  opts.tol_domain_solver — subdomain solver tolerance (CG residual; ignored by eig2d).
#                  opts.tol_trunc         — tolerance for `trunclr`-truncating the returned LRSVD
#                                            (defaults to opts.tol_domain_solver if omitted).
#                  opts.tol_DD            — DN sweep convergence tolerance on the interface update ‖Δλ‖.
#                  opts.tol_schedule      — (eig2d_lr only, optional) function res -> tol mapping the
#                                            previous-sweep residual to the per-sweep solver AND
#                                            truncation tolerance (inexact/adaptive solves; loose
#                                            early, tight late).
#                  opts.sched_on          — (optional) which residual feeds tol_schedule:
#                                             :lambda   (default) — the interface increment ‖Δλ‖;
#                                             :residual          — the relative SBP-SAT operator
#                                                                   residual of the solution ws.
#
# Returns (; ws, xs, y, sweeps): the per-subdomain solutions `ws` (each a W),
# the subdomain x-grids `xs`, the shared y-grid `y`, and the number of DN sweeps.
#
# Example calls for the different subdomain solvers, and — for the low-rank solver —
# with / without tolerance "scaling" (the adaptive schedule eps = clamp(theta*rho,
# floor, 1) set via `tol_schedule`).  `tol_DD` is the DN convergence tolerance and
# `tol_trunc` truncates the returned low-rank solution:
#
#   # cg2d — matrix conjugate gradient (iterative, dense bridge):
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6,opts=(; solver=cg2d, tol_domain_solver=1e-11, tol_trunc=1e-10, tol_DD=1e-7))
#
#   # eig2d — direct symmetric eigen-diagonalization (exact, dense):
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6,opts=(; solver=eig2d, tol_trunc=1e-10, tol_DD=1e-7))
#
#   # eig2d_lr — low-rank (Cross-DEIM), NO scaling: fixed solver & truncation tol:
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6,opts=(; solver=eig2d_lr, tol_domain_solver=1e-10, tol_trunc=1e-10, tol_DD=1e-7))
#
#   # eig2d_lr — WITH scaling, schedule driven by the interface increment rho=‖Δλ‖
#   #            (sched_on=:lambda is the default); ‖Δλ‖ is a step size, so it needs a
#   #            small constant theta (~0.01–0.03) to stay robust:
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6,opts=(; solver=eig2d_lr, tol_DD=1e-7,tol_schedule = res -> clamp(0.02*res, 1e-10, 1.0)))
#
#   # eig2d_lr — WITH scaling, schedule driven by the operator residual rho=R
#   #            (sched_on=:residual); a true residual, far more robust, so a larger
#   #            theta (~0.1) is fine:
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6,opts=(; solver=eig2d_lr, tol_DD=1e-7, sched_on=:residual,tol_schedule = res -> clamp(0.1*res, 1e-10, 1.0)))
#
#   # eig2d_lr_p — low-rank on the p smallest-|λ| eigenvectors per direction.
#   #              Coarse spectral solve; too small a p destabilizes the DN coupling and
#   #              the call exits with a clear "DN iteration diverged" error (the case
#   #              below, p=70, does this).  Use p near nx for a converging run; p omitted
#   #              ⇒ full basis ⇒ exactly eig2d_lr:
#   run_poisson_2d_lr(; M=4, nx=80, ny=80, accuracy=6, opts=(; solver=eig2d_lr_p, p=70, tol_domain_solver=1e-10, tol_trunc=1e-10, tol_DD=1e-7,tol_schedule = res -> clamp(0.1*res, 1e-10, 1.0)))
#
#   # Add `animate=true` (optionally `animfile=...`, `framerate=...`) to any call to
#   # record a per-sweep movie; `doplot=false` suppresses the static PNG.
function run_poisson_2d_lr(; M = 4, nx = 100, ny = 100, accuracy = 6, θ = 0.5,
                           xL = 0.0, xR = 5.0, yB = 0.0, yT = 1.0,
                           maxiter = 1000,
                           plotkind = :surface,        # :surface or :contour
                           doplot = true,              # set false to suppress all plotting
                           animate = false,            # record a per-sweep solution movie
                           animfile = joinpath(@__DIR__, "sbp_poisson_bvp_2d_lowrank.mp4"),
                           framerate = 8,
                           opts = (; solver = cg2d, tol_domain_solver = 1e-11,
                                     tol_trunc = 1e-11, tol_DD = 1e-9))  # solver + DD options
    M >= 1 || throw(ArgumentError("need at least one subdomain"))

    # Manufactured solution for Poisson  w_xx + w_yy = f.
    w_exact(x, y)  = exp(sin(x*y + x))
    wy_exact(x, y) = x * cos(x*y + x) * exp(sin(x*y + x))     # ∂y w_exact, for the top Neumann
    # forcing = w_xx + w_yy of w_exact  (g = x*y + x):
    f_force(x, y)  = exp(sin(x*y + x)) * ((y + 1)^2 + x^2) * (cos(x*y + x)^2 - sin(x*y + x))

    xnodes = collect(range(xL, xR; length = M + 1))   # subdomain edges in x

    # Shared y-operator (Dirichlet bottom / Neumann top → bc [1,0,0,1]); diagonalized once.
    D1y = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = yB, xmax = yT, N = ny)
    my  = sbp_sat_matrices(D1y, [1.0, 0.0, 0.0, 1.0]; symmetric = true)
    sy  = sqrt.(diag(my.H))
    A2S = (sy .* my.A) ./ sy'
    yb  = YOperator(D1y, grid(D1y), my, sy, A2S, eigen(Symmetric(A2S)))
    y   = yb.y                                       # shared y-grid (used widely below)

    # Per-subdomain x-operators (DN roles), symmetrizations, and eigendecompositions.
    xblocks = Vector{XOperator}(undef, M)
    for i in 1:M
        Dx = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                 accuracy_order = accuracy,
                                 xmin = xnodes[i], xmax = xnodes[i + 1], N = nx)
        bc = i == 1 ? [1.0, 0.0, 1.0, 0.0] : [0.0, 1.0, 1.0, 0.0]
        mx = sbp_sat_matrices(Dx, bc; symmetric = true)
        sx = sqrt.(diag(mx.H))
        A1S = (sx .* mx.A) ./ sx'
        t = zeros(nx); mul_transpose_derivative_right!(t, Dx, Val(1), 1.0, false)
        xblocks[i] = XOperator(Dx, grid(Dx), mx, sx, A1S, t, eigen(Symmetric(A1S)))
    end

    # Low-rank (LRSVD) forcing per subdomain, built matrix-free with crossDEIM on
    # F_i[a,b] = f_force(xblocks[i].x[a], y[b]).
    Fs = Vector{LRSVD{Float64}}(undef, M)
    for i in 1:M
        gfun = (a, b) -> f_force(xblocks[i].x[a], y[b])
        U0 = reshape(normalize(randn(nx)), nx, 1)   # rank-1 starting guess
        V0 = reshape(normalize(randn(ny)), ny, 1)
        Fs[i], _ = crossDEIM(gfun, LRSVD(U0, [1.0], V0))
    end

    # y-boundary data per subdomain (functions of x on that subdomain's grid).
    gBs = [w_exact.(xblocks[i].x, yB)  for i in 1:M]   # bottom Dirichlet value
    gTs = [wy_exact.(xblocks[i].x, yT) for i in 1:M]   # top Neumann  (w_n = +∂y w)

    ws = Vector{LRSVD{Float64}}(undef, M)       # physical subdomain solutions (each an LRSVD W)

    # One forward DN sweep (interface unknowns λ_i are length-ny vectors).
    function forward_sweep!(λ, sopts)
        newλ = [copy(λ[i]) for i in 1:M - 1]
        q = zeros(ny)                            # interface flux ∂x W from the left
        for i in 1:M
            gL = i == 1 ? w_exact.(xL, y) : -q   # outer Dirichlet(y) or Neumann flux
            gR = i == M ? w_exact.(xR, y) : λ[i] # outer Dirichlet(y) or interface λ_i
            W0 = isassigned(ws, i) ? ws[i] : nothing   # warm start from previous sweep
            ws[i], _ = solve_subdomain_2d_lr(xblocks[i], yb, gL, gR, gBs[i], gTs[i], Fs[i];
                                             W0 = W0, opts = sopts)
            if i < M           # flux ∂x W at the right edge, per y-row → length ny
                # W = U·diag(S)·Vᵀ and ∂x|_right is linear in x, so differentiate every U
                # column at once: dUr = dRx·U, then recombine  q = V·(S .* dUr).
                dUr = vec(xblocks[i].dRx' * ws[i].U)   # length-R: ∂x|_right of each U column
                q   = ws[i].V * (ws[i].S .* dUr)
            end
            # interface (row-1) trace from the factors:  W[1,:] = V·(S .* U[1,:])
            wtrace = ws[i].V * (ws[i].S .* ws[i].U[1, :])
            i >= 2 && (newλ[i - 1] = wtrace)      # left trace at the interface
        end
        return newλ
    end

    # The subdomain solutions are LRSVDs, densified here only for visualization.
    Wfull(F) = F.U * Diagonal(F.S) * F.V'

    # Discrete SBP-SAT operator residual evaluated from the subdomain solutions only (no λ):
    # for each subdomain  r_i = A1·W_i + W_i·A2ᵀ + xinj + yinj − F_i,  with A1 = mx.A, A2 = my.A
    # (solution-side SAT included) and xinj/yinj the boundary-data SAT injections.  Outer faces
    # use the exact BCs; at interfaces the Dirichlet value comes from the neighbour's trace and
    # the Neumann flux from the neighbour's ∂x trace (coupling taken from the solutions, not λ).
    # `zerosol=true` evaluates it at W=0 (≈ the load) for use as a normalization.
    function operator_residual_abs(zerosol::Bool)
        r = 0.0
        for i in 1:M
            Wi = zerosol ? zeros(nx, ny) : Wfull(ws[i])
            gL = i == 1 ? w_exact.(xL, y) :
                 (zerosol ? zeros(ny) : -vec(xblocks[i-1].dRx' * Wfull(ws[i-1])))
            gR = i == M ? w_exact.(xR, y) :
                 (zerosol ? zeros(ny) : Wfull(ws[i+1])[1, :])
            xinj = xblocks[i].mx.BL * gL' .+ xblocks[i].mx.BR * gR'
            yinj = gBs[i] * yb.my.BL' .+ gTs[i] * yb.my.BR'
            ri = xblocks[i].mx.A * Wi .+ Wi * yb.my.A' .+ xinj .+ yinj .- Wfull(Fs[i])
            r = max(r, norm(ri))
        end
        return r
    end
    res0 = operator_residual_abs(true)            # zero-solution residual (the load), computed once
    res0 = res0 == 0 ? 1.0 : res0
    # relative SBP-SAT operator residual of the current solution.
    solution_residual() = operator_residual_abs(false) / res0

    zlo, zhi = extrema(w_exact(x, yy) for x in range(xL, xR; length = 50)
                                      for yy in range(yB, yT; length = 50))
    # Draw the assembled solution (all subdomains) into an existing CairoMakie axis.
    # Makie's surface!/contourf! take z[i,j] = w(x_i, y_j), i.e. the LRSVD matrix as-is.
    function plot_solution!(ax)
        local plt = nothing
        for i in 1:M
            Z = Wfull(ws[i])
            plt = plotkind === :contour ?
                contourf!(ax, xblocks[i].x, y, Z; levels = range(zlo, zhi; length = 13)) :
                surface!(ax, xblocks[i].x, y, Z; colorrange = (zlo, zhi))
        end
        return plt
    end

    # Optional tolerance schedule (eig2d_lr only): set the per-sweep solver AND truncation
    # tolerance from a residual of the previous sweep via opts.tol_schedule(res).  The
    # quantity used as `res` is selected by opts.sched_on:
    #   :lambda   (default) — the interface increment ‖Δλ‖ (a step size);
    #   :residual          — the relative SBP-SAT operator residual of the solution ws
    #                        (the true PDE residual, the analog of lrAA's ‖G(X)−X‖).
    tol_sched = get(opts, :tol_schedule, nothing)
    sched_on  = get(opts, :sched_on, :lambda)
    function opts_for(res)
        (opts.solver === eig2d_lr && tol_sched !== nothing) || return opts
        t = tol_sched(res)
        return merge(opts, (; tol_domain_solver = t, tol_trunc = t))
    end

    λ = [zeros(ny) for _ in 1:M - 1]            # interface traces (wrong guess)
    hist = Float64[]                             # interface residual ‖Δλ‖ per sweep
    errhist = Float64[]                          # max|W − w_exact| over all subdomains per sweep
    solreshist = Float64[]                       # relative SBP-SAT operator residual (vs zero solution)
    rankhist = Vector{Vector{Int}}()            # solution rank per subdomain per sweep
    snaps = Vector{Vector{LRSVD{Float64}}}()    # per-sweep solution snapshots (for `animate`)
    sweeps = 0
    sched_prev = Inf                             # drives the schedule (loosest on the first sweep)
    for k in 1:maxiter
        sweeps = k
        newλ = forward_sweep!(λ, opts_for(sched_prev))
        # snapshot this sweep's solution (ws[i] is reassigned, not mutated, each sweep,
        # so storing the references is safe and cheap — the factors are low-rank).
        animate && push!(snaps, [ws[i] for i in 1:M])
        res = M > 1 ? maximum(maximum(abs, newλ[i] .- λ[i]) for i in 1:M - 1) : 0.0
        err = maximum(maximum(abs, Wfull(ws[i]) .- w_exact.(xblocks[i].x, y')) for i in 1:M)
        # Divergence guard: bail out gracefully (before the next sweep overflows into a
        # cryptic LAPACK error) if the iteration blows up.  This happens, e.g., when
        # eig2d_lr_p uses too few eigenvectors: the truncated solve drops the interface-flux
        # content and destabilizes the Dirichlet–Neumann coupling.
        (isfinite(res) && isfinite(err) && err < 1e8) ||
            error("DN iteration diverged at sweep $k (max error = $err). The subdomain " *
                  "solver/options are unstable for this configuration — e.g. eig2d_lr_p with " *
                  "too small `p` loses the interface-flux content and destabilizes the DD " *
                  "coupling. Increase `p`, or use eig2d / cg2d / eig2d_lr.")
        sres = solution_residual()
        converged = M == 1 || res < opts.tol_DD
        for i in 1:M - 1
            @. λ[i] = θ * newλ[i] + (1 - θ) * λ[i]
        end
        push!(hist, res)
        push!(errhist, err)
        push!(solreshist, sres)
        push!(rankhist, [length(ws[i].S) for i in 1:M])
        # convergence is always tested on ‖Δλ‖; only the *schedule source* changes.
        sched_prev = sched_on === :residual ? sres : res
        converged && break
    end

    forward_sweep!(λ, opts_for(sched_prev))      # final consistent sweep

  if animate && !isempty(snaps)
    # Per-sweep movie of the assembled solution (reproduces the old live frame).
    # Observables let `record` update the surfaces/contours in place each frame.
    figA = Figure(size = (900, 600))
    axA = plotkind === :contour ?
        Axis(figA[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect()) :
        Axis3(figA[1, 1]; xlabel = "x", ylabel = "y", zlabel = "w",
              azimuth = 0.6π, elevation = 0.30π)
    Zobs = [Observable(Wfull(snaps[1][i])) for i in 1:M]   # one solution field per subdomain
    for i in 1:M
        plotkind === :contour ?
            contourf!(axA, xblocks[i].x, y, Zobs[i]; levels = range(zlo, zhi; length = 13)) :
            surface!(axA, xblocks[i].x, y, Zobs[i]; colorrange = (zlo, zhi))
    end
    plotkind === :surface && zlims!(axA, zlo, zhi)
    record(figA, animfile, 1:length(snaps); framerate = framerate) do k
        for i in 1:M
            Zobs[i][] = Wfull(snaps[k][i])
        end
        axA.title = "DN sweep $k   ‖Δλ‖ = $(round(hist[k]; sigdigits = 3))"
    end
    println("  saved animation to ", animfile)
  end

  if doplot
    # Final figure: solution surface + DN convergence history + subdomain ranks.
    fig = Figure(size = (1650, 450))
    ax1 = plotkind === :contour ?
        Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect(),
             title = "2D Poisson DD low-rank (M = $M)") :
        Axis3(fig[1, 1]; xlabel = "x", ylabel = "y", zlabel = "w", azimuth = 0.6π,
              elevation = 0.30π, title = "2D Poisson DD low-rank (M = $M)")
    pl = plot_solution!(ax1)
    plotkind === :surface ? zlims!(ax1, zlo, zhi) : Colorbar(fig[1, 1, Right()], pl)

    sw = 1:length(hist)
    ax2 = Axis(fig[1, 2]; xlabel = "DN sweep", ylabel = "residual / error",
               yscale = log10, title = "convergence")
    scatterlines!(ax2, sw, max.(hist, 1e-16);       label = "‖Δλ‖",                  marker = :circle,  color = :red)
    scatterlines!(ax2, sw, max.(errhist, 1e-16);    label = "max|w − w_exact|",      marker = :rect,    color = :blue)
    scatterlines!(ax2, sw, max.(solreshist, 1e-16); label = "rel. op. residual (SAT)", marker = :diamond, color = :green)
    axislegend(ax2; position = :rt)

    ax3 = Axis(fig[1, 3]; xlabel = "DN sweep", ylabel = "rank", title = "subdomain ranks")
    for i in 1:M
        scatterlines!(ax3, sw, [rankhist[k][i] for k in sw]; label = "Ω$i", marker = :circle)
    end
    axislegend(ax3; position = :rb)

    outfile = joinpath(@__DIR__, "sbp_poisson_bvp_2d_lowrank.png")
    save(outfile, fig)
    println("  saved plot to ", outfile)
  end  # if doplot

    return (; ws, xs = [xb.x for xb in xblocks], y, sweeps, rankhist, hist, errhist, solreshist)
end

# --- tolerance study --------------------------------------------------------

# Run `run_poisson_2d_lr` across the three subdomain solvers and a sweep of tolerances,
# plotting the achieved error max|w − w_exact| vs the DN tolerance tol_DD.  cg2d varies its
# solver tolerance; eig2d_lr varies solver + truncation tolerance together; eig2d is exact.
function tolerance_study(; M = 4, nx = 25, ny = 101)
    tol_DDs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # (label, solver, tol_domain_solver, tol_trunc)
    combos = [
        ("eig2d (exact)",      eig2d,    1e-11, 1e-11),
        ("cg2d tol_s=1e-7",    cg2d,     1e-7,  1e-11),
        ("cg2d tol_s=1e-9",    cg2d,     1e-9,  1e-11),
        ("cg2d tol_s=1e-11",   cg2d,     1e-11, 1e-11),
        ("eig2d_lr tol=1e-7",  eig2d_lr, 1e-7,  1e-7),    # solver & truncation tied
        ("eig2d_lr tol=1e-9",  eig2d_lr, 1e-9,  1e-9),    # tied
        ("eig2d_lr tol=1e-11", eig2d_lr, 1e-11, 1e-11),   # tied
    ]
    fig = Figure(size = (800, 520))
    ax = Axis(fig[1, 1]; xlabel = "tol_DD", ylabel = "max|w − w_exact|",
              xscale = log10, yscale = log10, xreversed = true,
              limits = (nothing, nothing, 1e-3, 1e-1),
              title = "solver / tolerance study  (M=$M, nx=$nx, ny=$ny)")
    for (label, solver, ts, tt) in combos
        errs = Float64[]
        for tDD in tol_DDs
            r = run_poisson_2d_lr(; M, nx, ny, doplot = false,
                    opts = (; solver, tol_domain_solver = ts, tol_trunc = tt, tol_DD = tDD))
            push!(errs, r.errhist[end])
        end
        println(rpad(label, 20), " errors = ", errs)
        scatterlines!(ax, tol_DDs, max.(errs, 1e-16); label = label, marker = :circle)
    end
    axislegend(ax; position = :lt)
    out = joinpath(@__DIR__, "sbp_poisson_bvp_2d_lowrank_tolstudy.png")
    save(out, fig)
    println("saved study plot to ", out)
    return fig
end

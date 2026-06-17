# Low-rank (Tucker3) 3D Poisson Dirichlet–Neumann domain-decomposition solver — the
# 3D analogue of `sbp_poisson_bvp_2d_lowrank.jl`, with the matrix unknown (LRSVD)
# replaced by a 3-tensor unknown in Tucker3 format.  Graphical output is omitted.
#
# Solve  w_xx + w_yy + w_zz = f  on [xL,xR]×[yB,yT]×[zB,zT], split IN X into M slabs, with
#   * Dirichlet on the outer x faces and the x-interfaces of the decomposition,
#   * Dirichlet on the y-bottom, Neumann on the y-top (w_n = +∂y w),
#   * Dirichlet on both z faces.
# Each subdomain unknown is a low-rank 3-TENSOR W (nx×ny×nz) stored as a Tucker3,
# W = G ×₁U[1] ×₂U[2] ×₃U[3].  The interface unknowns λ_i are y–z planes (ny×nz).
# A 2-panel figure (DN convergence history + per-subdomain Tucker ranks) is saved
# when `doplot = true`.
#
# Solver status (cf. the 2D file): cg3d and eig3d are implemented (dense bridge);
# the genuine low-rank solvers eig3d_lr / eig3d_lr_p are placeholders for now.

using LinearAlgebra
using SummationByPartsOperators
using LRDD
using CairoMakie   # static plotting backend; `grid` is not exported, so no clash
using TimerOutputs  # bottleneck timing (run_poisson_3d_lr; pass show_timer=true)

# Shared timer; reset at the start of each run_poisson_3d_lr call.  Hot blocks are
# wrapped with `@timeit TO3D "label" …` so a per-operation breakdown can be printed.
const TO3D = TimerOutput()

include(joinpath(@__DIR__, "sbp_second_derivative_matrix.jl"))

# --- precomputed operators -------------------------------------------------
# Per-direction quantities built once at setup, including the symmetric
# eigendecompositions used by `eig3d`.  Struct names carry a `3` suffix so this
# file can coexist with `sbp_poisson_bvp_2d_lowrank.jl` in one session.

mutable struct XOperator3         # per subdomain (x-direction)
    Dx
    x::Vector{Float64}
    mx
    sx::Vector{Float64}
    A1S::Matrix{Float64}
    dRx::Vector{Float64}          # right-derivative functional (∂x at x_right)
    eigA1                         # eigen(Symmetric(A1S))
end

mutable struct YOperator3         # shared (y-direction)
    D1y
    y::Vector{Float64}
    my
    sy::Vector{Float64}
    A2S::Matrix{Float64}
    eigA2                         # eigen(Symmetric(A2S))
end

mutable struct ZOperator3         # shared (z-direction)
    D1z
    z::Vector{Float64}
    mz
    sz::Vector{Float64}
    A3S::Matrix{Float64}
    eigA3                         # eigen(Symmetric(A3S))
end

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

# Unsymmetrized physical operator  A1·₁U + U·₂A2 + U·₃A3  (A2,A3 not assumed
# symmetric, so modes 2,3 transpose).  Used to evaluate the SBP-SAT residual.
function applyL3_A(A1, A2, A3, U)
    nx, ny, nz = size(U)
    V = reshape(A1 * reshape(U, nx, ny * nz), nx, ny, nz)         # mode 1
    @inbounds for c in 1:nz                                       # mode 2
        @views V[:, :, c] .+= U[:, :, c] * A2'
    end
    V .+= reshape(reshape(U, nx * ny, nz) * A3', nx, ny, nz)      # mode 3
    return V
end

# Solve  L(X) = B  with L = A1S·₁· + A2S·₂· + A3S·₃· symmetric negative definite
# (M = -L is SPD).  Works directly on 3D arrays, warm-started with `X0`; returns
# (X, iters).  3D analogue of `cg2d`.
function cg3d(xb::XOperator3, yb::YOperator3, zb::ZOperator3, B;
              X0 = zero(B), tol = 1e-11, maxiter = 10 * length(B))
    A1S, A2S, A3S = xb.A1S, yb.A2S, zb.A3S
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

# Direct 3D solver by symmetric eigen-diagonalization — 3D analogue of `eig2d`.
# With A1S = Q1 Λ1 Q1ᵀ, A2S = Q2 Λ2 Q2ᵀ, A3S = Q3 Λ3 Q3ᵀ, the equation
# A1S·₁X + A2S·₂X + A3S·₃X = B diagonalizes to (λ1[i]+λ2[j]+λ3[k]) Y[i,j,k] = Ĝ[i,j,k]
# with Ĝ = B ×₁Q1ᵀ ×₂Q2ᵀ ×₃Q3ᵀ, so Y = Ĝ ./ D and X = Y ×₁Q1 ×₂Q2 ×₃Q3.  Negative
# definiteness makes D = λ1⊕λ2⊕λ3 < 0 everywhere (never zero).  `X0`/`tol` ignored.
function eig3d(xb::XOperator3, yb::YOperator3, zb::ZOperator3, B; X0 = nothing, tol = 0)
    λ1, Q1 = xb.eigA1.values, xb.eigA1.vectors
    λ2, Q2 = yb.eigA2.values, yb.eigA2.vectors
    λ3, Q3 = zb.eigA3.values, zb.eigA3.vectors
    Ghat = modemult(modemult(modemult(B, Q1', 1), Q2', 2), Q3', 3)
    D = reshape(λ1, :, 1, 1) .+ reshape(λ2, 1, :, 1) .+ reshape(λ3, 1, 1, :)
    Y = Ghat ./ D
    X = modemult(modemult(modemult(Y, Q1, 1), Q2, 2), Q3, 3)
    return X, 0
end

# --- low-rank Tucker solvers -----------------------------------------------

# Genuine low-rank counterpart of `eig3d`, the 3D analogue of `eig2d_lr`: the RHS
# `B` is a Tucker3 and the eigenbasis solve
#   Y[i,j,k] = (B ×₁Q1ᵀ ×₂Q2ᵀ ×₃Q3ᵀ)[i,j,k] / (λ1[i] + λ2[j] + λ3[k])
# is built as a Tucker tensor by the matrix-free Tucker `crossDEIM`, sampling
# entries of Y.  The rotated numerator Ĝ = B ×₁Q1ᵀ ×₂Q2ᵀ ×₃Q3ᵀ stays low-rank
# (rotate B's factors), so each entry is cheap via `tucker_eval`.  Cross-DEIM uses
# the solver tolerance `opts.tol_domain_solver`; no nx×ny×nz dense tensor is formed
# in the solve.  Returns X = Y ×₁Q1 ×₂Q2 ×₃Q3 as a Tucker3 (and the iteration count).
function eig3d_lr(xb::XOperator3, yb::YOperator3, zb::ZOperator3, B::Tucker3; opts = (;))
    λ1, Q1 = xb.eigA1.values, xb.eigA1.vectors
    λ2, Q2 = yb.eigA2.values, yb.eigA2.vectors
    λ3, Q3 = zb.eigA3.values, zb.eigA3.vectors
    nx, ny, nz = length(λ1), length(λ2), length(λ3)
    # Ĝ = B rotated into the eigenbasis, kept low-rank by rotating the factors.
    P = (Q1' * B.U[1], Q2' * B.U[2], Q3' * B.U[3])   # factors of Ĝ (n_l × r_l); core B.G
    gfun = (i, j, k) -> tucker_eval(P, B.G, i, j, k) / (λ1[i] + λ2[j] + λ3[k])
    tol  = get(opts, :tol_domain_solver, get(opts, :tol_trunc, 1e-11))
    # rmax leaves headroom for Cross-DEIM's oversampling (gpode needs n_l − r ≳ os).
    rmax = max(1, min(nx, ny, nz) - 4)
    cdopts = (; tol = tol, rmax = rmax, max_iter = 30, increment = 4,
              cache = get(opts, :cache, true))
    # seed with a rank-1 guess from the numerator's leading directions — Cross-DEIM
    # grows the rank itself; a near-full-rank seed would overflow the oversampling.
    seed = Tucker3(ntuple(l -> P[l][:, 1:1], 3), reshape([1.0], 1, 1, 1))
    Y, info = crossDEIM(gfun, seed, cdopts)
    X = Tucker3((Q1 * Y.U[1], Q2 * Y.U[2], Q3 * Y.U[3]), Y.G)   # rotate back
    return X, size(info, 1)
end

# Spectrally-truncated variant (3D analogue of eig2d_lr_p) — still a placeholder.
function eig3d_lr_p(xb::XOperator3, yb::YOperator3, zb::ZOperator3, B::Tucker3; opts = (;))
    error("eig3d_lr_p is a placeholder — the spectrally-truncated 3D low-rank (Tucker) " *
          "eigen-solver is not implemented yet.  Use solver = cg3d, eig3d, or eig3d_lr.")
end

# Truncate / re-orthonormalize a Tucker3 to relative tolerance `tol` without
# densifying: QR each factor (U_l = Q_l R_l), MLSVD the small transformed core
# C = G ×_l R_l, recombine.  3D analogue of `truncsum`/`trunclr` on an LRSVD.
function recompress_tucker(F::Tucker3, tol)
    QR = ntuple(l -> qr(F.U[l]), 3)
    Qs = ntuple(l -> Matrix(QR[l].Q)[:, 1:size(F.U[l], 2)], 3)   # thin Q (n_l × r_l)
    C  = modemult(modemult(modemult(F.G, QR[1].R, 1), QR[2].R, 2), QR[3].R, 3)
    Uc, Gc = LRDD.mlsvd3(C, tol)
    return Tucker3(ntuple(l -> Qs[l] * Uc[l], 3), Gc)
end

# --- subdomain solve --------------------------------------------------------

# Densify a Tucker3 to a full 3D array.
full3(F::Tucker3) = lmlragen3(F)

# A boundary-injection term  bvec ⊗ plane  as a low-rank Tucker3, without
# densifying: `bvec` (length n_vmode) sits in mode `vmode`; `plane` is the matrix
# over the other two modes (ascending order), SVD-truncated at relative `tol` so
# the term stays low rank.  The σ's go on the "diagonal" of a core of size 1 in
# `vmode` and r in the other two modes.  Used to build the RHS matrix-free.
function inj_tucker(bvec, vmode::Int, plane::AbstractMatrix; tol = 1e-12)
    sv = svd(plane)
    r  = max(1, count(>=(tol * (isempty(sv.S) ? 1.0 : sv.S[1])), sv.S))
    Up, σ, Vp = sv.U[:, 1:r], sv.S[1:r], sv.V[:, 1:r]
    a, b = vmode == 1 ? (2, 3) : (vmode == 2 ? (1, 3) : (1, 2))
    U = Vector{Matrix{Float64}}(undef, 3)
    U[vmode] = reshape(collect(Float64, bvec), :, 1)
    U[a] = Matrix(Up); U[b] = Matrix(Vp)
    G = zeros(Float64, ntuple(l -> l == vmode ? 1 : r, 3))
    for p in 1:r
        G[ntuple(l -> l == vmode ? 1 : p, 3)...] = σ[p]
    end
    return Tucker3((U[1], U[2], U[3]), G)
end

# Negate a Tucker3 (flip the core) — for forming differences in `tucker_cross_sum`.
negtucker(F::Tucker3) = Tucker3(F.U, -F.G)

# --- low-rank residual / diagnostic helpers (no nx×ny×nz array) -------------

# Low-rank apply of the unsymmetrized physical operator to a Tucker3:
# A1·₁W + W·₂A2 + W·₃A3 as three Tucker3 terms sharing the core W.G (cf. applyL3_A;
# each mode-product just pre-multiplies one factor).  Used by the low-rank residual.
applyL3_A_lr(A1, A2, A3, W::Tucker3) = (
    Tucker3((A1 * W.U[1], W.U[2], W.U[3]), W.G),
    Tucker3((W.U[1], A2 * W.U[2], W.U[3]), W.G),
    Tucker3((W.U[1], W.U[2], A3 * W.U[3]), W.G))

# Contract mode 1 of a Tucker3 with a length-nx vector v → ny×nz plane
# P[j,k] = Σ_a v[a] W[a,j,k] = U2 * (G ×₁ vᵀU1) * U3ᵀ.  No nx×ny×nz array; used for
# the interface trace (v = e₁) and the right-face flux (v = -dRx) in the residual.
function face_contract(W::Tucker3, v::AbstractVector)
    r2, r3 = size(W.G, 2), size(W.G, 3)
    Gc = reshape((v' * W.U[1]) * unfold(W.G, 1), r2, r3)   # r2×r3
    return W.U[2] * Gc * W.U[3]'
end

# Frobenius norm ‖Σ_t terms[t]‖ of a sum of Tucker3 terms, without densifying and
# without recompression: stack the cores block-diagonally, orthogonalize each mode by
# a column-pivoted QR (absorbing R·Pᵀ into the core), then `norm` the small core
# (orthonormal factors ⇒ ‖tensor‖ = ‖core‖).  Same QR phase as `tucker_sum` but skips
# the `mlsvd3` rounding; QR-stable for a residual that is a small difference of larger
# terms.  No nx×ny×nz array is formed; the only sizable temporary is the (Σr)³ core.
function tucker_terms_fnorm(terms::Vector{<:Tucker3})
    nmat = length(terms)
    szg = [size(terms[i].G, k) for i in 1:nmat, k in 1:3]
    sz  = ntuple(k -> sum(@view szg[:, k]), 3)
    off = zeros(Int, nmat + 1, 3)
    for k in 1:3, i in 1:nmat
        off[i + 1, k] = off[i, k] + szg[i, k]
    end
    G = zeros(Float64, sz)                          # block-diagonal core
    for i in 1:nmat
        G[ntuple(k -> (off[i, k] + 1):off[i + 1, k], 3)...] = terms[i].G
    end
    n = ntuple(k -> size(terms[1].U[k], 1), 3)
    for k in 1:3                                     # orthogonalize each mode
        bigU = zeros(Float64, n[k], sz[k])
        for i in 1:nmat
            bigU[:, (off[i, k] + 1):off[i + 1, k]] = terms[i].U[k]
        end
        F = qr(bigU, ColumnNorm())
        rkk = size(F.R, 1)
        newsz = ntuple(l -> l == k ? rkk : size(G, l), 3)
        G = fold(F.R * unfold(G, k)[F.p, :], k, newsz)
    end
    return norm(G)
end

# max|E| of a Tucker3, streamed one x-slice (ny×nz) at a time — never densifies the
# full nx×ny×nz tensor.  Used by the low-rank max-error diagnostic.
function maxabs_tucker(E::Tucker3)
    r2, r3 = size(E.G, 2), size(E.G, 3)
    GU = unfold(E.G, 1)                                   # r1 × (r2 r3)
    m = 0.0
    @views for a in 1:size(E.U[1], 1)
        S = E.U[2] * reshape(E.U[1][a, :]' * GU, r2, r3) * E.U[3]'
        m = max(m, maximum(abs, S))
    end
    return m
end

# Solve one 3D subdomain  w_xx+w_yy+w_zz = F  with x-face data gL,gR (ny×nz),
# y-face data gyB,gyT (nx×nz) and z-face data gzB,gzT (nx×ny).  The forcing `F`
# and the warm start `W0` are Tucker3.  `opts.solver` selects the solver; the
# dense solvers (cg3d, eig3d) use a densify → solve → re-compress bridge and the
# returned solution is a Tucker3 truncated to `opts.tol_trunc`.  Returns
# (Wout::Tucker3, iters).
function solve_subdomain_3d_lr(xb::XOperator3, yb::YOperator3, zb::ZOperator3,
                               gL, gR, gyB, gyT, gzB, gzT, F::Tucker3;
                               W0 = nothing, opts = (; solver = cg3d, tol_domain_solver = 1e-11, tol_trunc = 1e-11))
    nx, ny, nz = length(xb.sx), length(yb.sy), length(zb.sz)
    tol_trunc = get(opts, :tol_trunc, get(opts, :tol_domain_solver, 1e-11))
    # symmetrizer S = Hx^{1/2}⊗Hy^{1/2}⊗Hz^{1/2}, separable (acts on each factor).
    sx, sy, sz = xb.sx, yb.sy, zb.sz

    # Dense symmetrized RHS  F_S = S .* (F − x/y/z injections)  and the dense S, as
    # full arrays (the dense solvers and the default eig3d_lr RHS path consume these).
    dense_FS() = begin
        Ffull = full3(F)
        xinj = reshape(xb.mx.BL, nx, 1, 1) .* reshape(gL, 1, ny, nz) .+
               reshape(xb.mx.BR, nx, 1, 1) .* reshape(gR, 1, ny, nz)
        yinj = reshape(gyB, nx, 1, nz) .* reshape(yb.my.BL, 1, ny, 1) .+
               reshape(gyT, nx, 1, nz) .* reshape(yb.my.BR, 1, ny, 1)
        zinj = reshape(gzB, nx, ny, 1) .* reshape(zb.mz.BL, 1, 1, nz) .+
               reshape(gzT, nx, ny, 1) .* reshape(zb.mz.BR, 1, 1, nz)
        Sloc = sx .* reshape(sy, 1, ny) .* reshape(sz, 1, 1, nz)
        (Sloc .* (Ffull .- xinj .- yinj .- zinj), Sloc)
    end

    if opts.solver === cg3d || opts.solver === eig3d
        # dense bridge: solve dense, then re-compress the solution to Tucker3.
        F_S, S = @timeit TO3D "rhs assembly (dense)" dense_FS()
        X0  = W0 === nothing ? zero(F_S) : (S .* full3(W0))                # warm start in Û space
        What, iters = @timeit TO3D "solve (dense)" opts.solver(xb, yb, zb, F_S; X0 = X0, tol = get(opts, :tol_domain_solver, 0.0))
        Wout = @timeit TO3D "compress solution (mlsvd3)" begin
            W = What ./ S                                                  # physical solution (dense)
            U, G = LRDD.mlsvd3(W, tol_trunc)                               # re-compress to Tucker3
            Tucker3(U, G)
        end
    elseif opts.solver === eig3d_lr
        # symmetrized Tucker3 RHS B = S .* (F − injections), then a Cross-DEIM solve.
        # `opts.rhs` selects how B is built:
        #   :dense (default) — densify → mlsvd3 (fast: one HOSVD, but forms nx×ny×nz);
        #   :crosssum        — matrix-free: each boundary injection is a low-rank
        #                      Tucker3 (`inj_tucker`) and they are combined with
        #                      `tucker_sum` (exact block-diagonal low-rank sum, then
        #                      rounded via mlsvd3; no dense nx×ny×nz array is formed).
        B = if get(opts, :rhs, :dense) === :crosssum
            @timeit TO3D "rhs assembly (tucker_sum)" begin
                terms = [F,
                         negtucker(inj_tucker(xb.mx.BL, 1, gL)),  negtucker(inj_tucker(xb.mx.BR, 1, gR)),
                         negtucker(inj_tucker(yb.my.BL, 2, gyB)), negtucker(inj_tucker(yb.my.BR, 2, gyT)),
                         negtucker(inj_tucker(zb.mz.BL, 3, gzB)), negtucker(inj_tucker(zb.mz.BR, 3, gzT))]
                RHS = tucker_sum(terms, 1e-12)                     # RHS = F − injections (exact low-rank sum)
                Tucker3((sx .* RHS.U[1], sy .* RHS.U[2], sz .* RHS.U[3]), RHS.G)   # symmetrize factors
            end
        else
            @timeit TO3D "rhs assembly (dense mlsvd3)" Tucker3(LRDD.mlsvd3(dense_FS()[1], 1e-12)...)
        end
        X, iters = @timeit TO3D "solve (Cross-DEIM)" eig3d_lr(xb, yb, zb, B; opts = opts)
        Wout = @timeit TO3D "compress solution (recompress)" begin
            Wlr = Tucker3((X.U[1] ./ sx, X.U[2] ./ sy, X.U[3] ./ sz), X.G)   # W = X ./ S
            recompress_tucker(Wlr, tol_trunc)
        end
    elseif opts.solver === eig3d_lr_p
        Wout, iters = eig3d_lr_p(xb, yb, zb, F; opts = opts)               # placeholder (errors)
    else
        error("solver $(opts.solver) is not supported; use cg3d, eig3d, eig3d_lr, or eig3d_lr_p")
    end

    return Wout, iters
end

# --- driver ----------------------------------------------------------------

# Keyword arguments
#   M           : number of x-subdomains (slabs).
#   nx, ny, nz  : grid points per subdomain (x, y, z).
#   accuracy    : SBP accuracy order.
#   θ           : Dirichlet–Neumann interface relaxation factor in (0, 1].
#   xL..zT      : domain bounds.
#   maxiter     : maximum number of DN sweeps.
#   opts        : NamedTuple of solver / DD options:
#                   opts.solver            — cg3d / eig3d (implemented), or
#                                            eig3d_lr / eig3d_lr_p (placeholders).
#                   opts.tol_domain_solver — subdomain CG tolerance (ignored by eig3d).
#                   opts.tol_trunc         — Tucker truncation tolerance of the solution
#                                            (defaults to opts.tol_domain_solver).
#                   opts.tol_DD            — DN convergence tolerance on ‖Δλ‖.
#                   opts.rhs               — (eig3d_lr only) how the Tucker RHS is built:
#                                            :dense (default) densify → mlsvd3, or
#                                            :crosssum matrix-free via tucker_sum.
#   doplot      : if true (default), save a 2-panel figure of the DN convergence
#                 history and per-subdomain Tucker ranks to `plotfile`.
#   plotfile    : output PNG path for that figure.
#
# Example calls:
#   run_poisson_3d_lr(; M=2, nx=16, ny=24, nz=24, accuracy=4,opts=(; solver=cg3d,  tol_domain_solver=1e-11, tol_trunc=1e-10, tol_DD=1e-7))
#   run_poisson_3d_lr(; M=2, nx=16, ny=24, nz=24, accuracy=4,opts=(; solver=eig3d, tol_trunc=1e-10, tol_DD=1e-7))
#
# Returns (; ws, xs, y, z, sweeps, hist, errhist, rankhist): the per-subdomain
# Tucker3 solutions `ws`, the grids, the number of DN sweeps, and the per-sweep
# interface residual / max error / subdomain core-size histories.
function run_poisson_3d_lr(; M = 2, nx = 16, ny = 24, nz = 24, accuracy = 4, θ = 0.5,
                           xL = -1.0, xR = 1.0, yB = 0.0, yT = 1.0, zB = 0.0, zT = 1.0,
                           maxiter = 1000,
                           doplot = true,
                           show_timer = false,   # print the TimerOutputs bottleneck report
                           plotfile = joinpath(@__DIR__, "sbp_poisson_bvp_3d_lowrank.png"),
                           opts = (; solver = cg3d, tol_domain_solver = 1e-11,
                                     tol_trunc = 1e-11, tol_DD = 1e-9))
    M >= 1 || throw(ArgumentError("need at least one subdomain"))
    reset_timer!(TO3D)

    # Manufactured solution for Poisson  Δw = f  (harmonic part exp(√2 x)cos y cos z,
    # plus x³ giving Δ = 6x).
    w_exact(x, y, z)  = exp(sqrt(2) * x) * cos(y) * cos(z) + x^3
    wy_exact(x, y, z) = -exp(sqrt(2) * x) * sin(y) * cos(z)   # ∂y w, for the y-top Neumann
    f_force(x, y, z)  = 6 * x                                 # forcing = Δw

    xnodes = collect(range(xL, xR; length = M + 1))

    # Shared y-operator (Dir bottom / Neu top) and z-operator (Dir both); diagonalized once.
    D1y = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = yB, xmax = yT, N = ny)
    D1z = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                              accuracy_order = accuracy, xmin = zB, xmax = zT, N = nz)
    my  = sbp_sat_matrices(D1y, [1.0, 0.0, 0.0, 1.0]; symmetric = true)
    mz  = sbp_sat_matrices(D1z, [1.0, 0.0, 1.0, 0.0]; symmetric = true)
    sy  = sqrt.(diag(my.H));  sz = sqrt.(diag(mz.H))
    A2S = (sy .* my.A) ./ sy';  A3S = (sz .* mz.A) ./ sz'
    yb  = YOperator3(D1y, grid(D1y), my, sy, A2S, eigen(Symmetric(A2S)))
    zb  = ZOperator3(D1z, grid(D1z), mz, sz, A3S, eigen(Symmetric(A3S)))
    y   = yb.y;  z = zb.z

    # Per-subdomain x-operators (DN roles), symmetrizations, eigendecompositions.
    xblocks = Vector{XOperator3}(undef, M)
    for i in 1:M
        Dx = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                 accuracy_order = accuracy,
                                 xmin = xnodes[i], xmax = xnodes[i + 1], N = nx)
        bc = i == 1 ? [1.0, 0.0, 1.0, 0.0] : [0.0, 1.0, 1.0, 0.0]
        mx = sbp_sat_matrices(Dx, bc; symmetric = true)
        sx = sqrt.(diag(mx.H))
        A1S = (sx .* mx.A) ./ sx'
        t = zeros(nx); mul_transpose_derivative_right!(t, Dx, Val(1), 1.0, false)
        xblocks[i] = XOperator3(Dx, grid(Dx), mx, sx, A1S, t, eigen(Symmetric(A1S)))
    end

    # Low-rank (Tucker3) forcing per subdomain, from an HOSVD of the dense forcing.
    Fs = Vector{Tucker3{Float64}}(undef, M)
    @timeit TO3D "setup: forcing (mlsvd3)" for i in 1:M
        Ffull = [f_force(xblocks[i].x[a], y[b], z[c]) for a in 1:nx, b in 1:ny, c in 1:nz]
        U, G = LRDD.mlsvd3(Ffull, 1e-12)
        Fs[i] = Tucker3(U, G)
    end

    # Transverse boundary data per subdomain (2D arrays over the other two dims).
    gyBs = [[w_exact(xblocks[i].x[a], yB, z[c])  for a in 1:nx, c in 1:nz] for i in 1:M]
    gyTs = [[wy_exact(xblocks[i].x[a], yT, z[c]) for a in 1:nx, c in 1:nz] for i in 1:M]
    gzBs = [[w_exact(xblocks[i].x[a], y[b], zB)  for a in 1:nx, b in 1:ny] for i in 1:M]
    gzTs = [[w_exact(xblocks[i].x[a], y[b], zT)  for a in 1:nx, b in 1:ny] for i in 1:M]
    # x-face Dirichlet data (ny×nz) on the outer boundaries.
    gxL = [w_exact(xL, y[b], z[c]) for b in 1:ny, c in 1:nz]
    gxR = [w_exact(xR, y[b], z[c]) for b in 1:ny, c in 1:nz]

    # Exact solution as a rank-2 Tucker3 per subdomain (it is separable:
    # w_exact = exp(√2 x)·cos y·cos z + x³·1·1), so the max-error diagnostic can be
    # evaluated low-rank without densifying any subdomain solution.
    Wexs = Vector{Tucker3{Float64}}(undef, M)
    for i in 1:M
        xi = xblocks[i].x
        Uxe = hcat(exp.(sqrt(2) .* xi), xi .^ 3)   # nx×2
        Uye = hcat(cos.(y), ones(ny))              # ny×2
        Uze = hcat(cos.(z), ones(nz))              # nz×2
        Ge  = zeros(2, 2, 2); Ge[1, 1, 1] = 1.0; Ge[2, 2, 2] = 1.0
        Wexs[i] = Tucker3((Uxe, Uye, Uze), Ge)
    end

    ws = Vector{Tucker3{Float64}}(undef, M)        # physical subdomain solutions (Tucker3)

    # One forward DN sweep (interface unknowns λ_i are ny×nz planes).
    function forward_sweep!(λ, sopts)
        newλ = [copy(λ[i]) for i in 1:M - 1]
        q = zeros(ny, nz)                          # interface flux plane ∂x W from the left
        its = 0
        for i in 1:M
            gL = i == 1 ? gxL : -q                 # outer Dirichlet plane or Neumann flux
            gR = i == M ? gxR : λ[i]               # outer Dirichlet plane or interface λ_i
            W0 = isassigned(ws, i) ? ws[i] : nothing
            ws[i], it = solve_subdomain_3d_lr(xblocks[i], yb, zb, gL, gR,
                                              gyBs[i], gyTs[i], gzBs[i], gzTs[i], Fs[i];
                                              W0 = W0, opts = sopts)
            its += it
            Wf = @timeit TO3D "flux/trace densify" full3(ws[i])   # densify once for flux/trace
            if i < M       # flux ∂x W at the right face: contract dim 1 with dRx → ny×nz
                q = reshape(xblocks[i].dRx' * reshape(Wf, nx, ny * nz), ny, nz)
            end
            i >= 2 && (newλ[i - 1] = Wf[1, :, :])  # left-face trace at the interface
        end
        return newλ, its
    end

    # Relative SBP-SAT operator residual evaluated from the subdomain solutions only:
    # r_i = A1·₁W_i + W_i·₂A2 + W_i·₃A3 + (x,y,z injections) − F_i (3D analogue of the
    # 2D residual).  Outer faces use the exact BCs; at interfaces the Dirichlet value
    # comes from the neighbour's left-face trace and the Neumann flux from the
    # neighbour's ∂x trace.  `zerosol=true` evaluates it at W=0 (≈ the load).
    # Low-rank assembly: each subdomain residual r_i is built as a sum of Tucker3
    # terms (operator applied low-rank + boundary injections − F) and Frobenius-normed
    # via `tucker_sum` (orthonormal factors ⇒ ‖r_i‖ = ‖core‖), never forming an
    # nx×ny×nz array.  Interface trace/flux planes come from `face_contract`.
    e1 = zeros(nx); e1[1] = 1.0
    function operator_residual_abs(zerosol::Bool)
        r = 0.0
        for i in 1:M
            gL = i == 1 ? gxL :
                 (zerosol ? zeros(ny, nz) : face_contract(ws[i-1], -xblocks[i-1].dRx))
            gR = i == M ? gxR :
                 (zerosol ? zeros(ny, nz) : face_contract(ws[i+1], e1))
            terms = Tucker3{Float64}[]
            zerosol || append!(terms, applyL3_A_lr(xblocks[i].mx.A, yb.my.A, zb.mz.A, ws[i]))
            # boundary injections as low-rank Tucker3 (skip identically-zero planes so
            # `inj_tucker` does not emit a full-rank zero term for empty interfaces).
            for (bvec, vmode, plane) in (
                    (xblocks[i].mx.BL, 1, gL), (xblocks[i].mx.BR, 1, gR),
                    (yb.my.BL, 2, gyBs[i]),    (yb.my.BR, 2, gyTs[i]),
                    (zb.mz.BL, 3, gzBs[i]),    (zb.mz.BR, 3, gzTs[i]))
                any(!iszero, plane) && push!(terms, inj_tucker(bvec, vmode, plane))
            end
            push!(terms, negtucker(Fs[i]))             # − F  (load)
            r = max(r, tucker_terms_fnorm(terms))       # ‖r_i‖_F, no nx×ny×nz array
        end
        return r
    end
    res0 = operator_residual_abs(true)             # zero-solution residual (the load)
    res0 = res0 == 0 ? 1.0 : res0
    solution_residual() = operator_residual_abs(false) / res0   # relative operator residual

    # Optional per-sweep tolerance schedule (cf. the 2D file): set the solver AND Tucker
    # truncation tolerance from a residual of the previous sweep via opts.tol_schedule(res).
    # opts.sched_on selects the driving residual: :lambda (default) — the interface
    # increment ‖Δλ‖; :residual — the relative SBP-SAT operator residual.
    tol_sched = get(opts, :tol_schedule, nothing)
    sched_on  = get(opts, :sched_on, :lambda)
    opts_for(res) = tol_sched === nothing ? opts :
                    merge(opts, (; tol_domain_solver = tol_sched(res), tol_trunc = tol_sched(res)))

    λ = [zeros(ny, nz) for _ in 1:M - 1]           # interface planes (wrong guess)
    hist = Float64[]                                # interface residual ‖Δλ‖ per sweep
    errhist = Float64[]                             # max|W − w_exact| over all subdomains per sweep
    solreshist = Float64[]                          # relative SBP-SAT operator residual per sweep
    rankhist = Vector{Vector{NTuple{3, Int}}}()    # Tucker core size per subdomain per sweep
    sweeps = 0
    sched_prev = Inf                                # drives the schedule (loosest on sweep 1)
    for k in 1:maxiter
        sweeps = k
        newλ, _ = forward_sweep!(λ, opts_for(sched_prev))
        res = M > 1 ? maximum(maximum(abs, newλ[i] .- λ[i]) for i in 1:M - 1) : 0.0
        err = @timeit TO3D "error diagnostic" maximum(
                  maxabs_tucker(tucker_sum([ws[i], negtucker(Wexs[i])], 1e-12)) for i in 1:M)
        sres = @timeit TO3D "op-residual diagnostic" solution_residual()
        # divergence guard: bail out before a cryptic overflow/LAPACK error.
        (isfinite(res) && isfinite(err) && err < 1e8) ||
            error("DN iteration diverged at sweep $k (max error = $err); the solver/options " *
                  "are unstable for this configuration.")
        converged = M == 1 || res < opts.tol_DD
        for i in 1:M - 1
            @. λ[i] = θ * newλ[i] + (1 - θ) * λ[i]
        end
        push!(hist, res)
        push!(errhist, err)
        push!(solreshist, sres)
        push!(rankhist, [size(ws[i].G) for i in 1:M])
        sched_prev = sched_on === :residual ? sres : res   # convergence still tested on ‖Δλ‖
        converged && break
    end

    forward_sweep!(λ, opts_for(sched_prev))        # final consistent sweep

    show_timer && print_timer(TO3D; sortby = :firstexec)

    if doplot
        # Two panels: DN convergence history and per-subdomain Tucker ranks.
        fig = Figure(size = (1100, 450))
        sw  = 1:length(hist)
        ax1 = Axis(fig[1, 1]; xlabel = "DN sweep", ylabel = "residual / error",
                   yscale = log10, title = "convergence (M = $M)")
        scatterlines!(ax1, sw, max.(hist, 1e-16);       label = "‖Δλ‖",                  marker = :circle,  color = :red)
        scatterlines!(ax1, sw, max.(errhist, 1e-16);    label = "max|w − w_exact|",      marker = :rect,    color = :blue)
        scatterlines!(ax1, sw, max.(solreshist, 1e-16); label = "rel. op. residual (SAT)", marker = :diamond, color = :green)
        axislegend(ax1; position = :rt)

        # Tucker rank per subdomain = the largest core mode dimension max(r1,r2,r3).
        ax2 = Axis(fig[1, 2]; xlabel = "DN sweep", ylabel = "max Tucker core rank",
                   title = "subdomain ranks")
        for i in 1:M
            scatterlines!(ax2, sw, [maximum(rankhist[k][i]) for k in sw]; label = "Ω$i", marker = :circle)
        end
        axislegend(ax2; position = :rb)

        save(plotfile, fig)
        println("  saved plot to ", plotfile)
    end

    return (; ws, xs = [xb.x for xb in xblocks], y, z, sweeps, hist, errhist, solreshist, rankhist)
end

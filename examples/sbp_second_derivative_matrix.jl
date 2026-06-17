# Matrix form of the SBP–SAT second-derivative operator built in
# `sbp_second_derivative.jl`.
#
# The matrix-free routine `apply_D2_sat(D2, u, bc, gL, gR)` is linear in the
# solution `u` and in the boundary data `gL, gR`, so it can be written as
#
#     u_xx ≈ A * u + B_L * gL + B_R * gR
#
# where
#   * `A`        is the n×n matrix of all SBP volume terms *and* the SAT terms
#                that multiply the solution vector,
#   * `BL`, `BR` are the boundary-data injection (lifting) columns.
#
# In addition we build two boundary "restriction" matrices `EL`, `ER` that pull
# the *correct* boundary condition out of the solution in the neighbouring
# regions.  Let `wl` be the solution on the grid immediately to the LEFT of the
# domain (so it shares the interface `xmin` at its right-most node, `wl[end]`)
# and `wr` the solution immediately to the RIGHT (sharing `xmax` at its
# left-most node, `wr[1]`).  Then
#
#     gL = EL * wl,   gR = ER * wr
#
# For a Dirichlet side `E` extracts the interface *value*; for a Neumann side it
# extracts the outward *normal derivative* `u_n` (`-∂x u` at `xmin`, `+∂x u`
# at `xmax`).  The boundary forcing is then `BL*(EL*wl) + BR*(ER*wr)`, so the
# full update reads
#
#     u_xx ≈ A*w + BL*(EL*wl) + BR*(ER*wr).
#
# With `symmetric = true` the energy-stable Dirichlet SAT is used so that `H*A`
# is symmetric and negative definite, where `H = mass_matrix(D2)` is the SBP
# norm (also returned).  `symmetrize(m)` then returns the genuinely symmetric
# `Ã = H^{1/2} A H^{-1/2}` (same real eigenvalues as `A`), suitable for a
# symmetric/CG solver or eigenanalysis.
#
# Run from the package root:
#   julia --project=. examples/SBP/sbp_second_derivative_matrix.jl
#
# Required packages:
#   ] add SummationByPartsOperators LinearAlgebra

using LinearAlgebra
using SummationByPartsOperators

# Reuse the operator setup, `apply_D2_sat`, and `boundary_data` from the
# matrix-free file.  Its demo is guarded by `abspath(PROGRAM_FILE) == @__FILE__`,
# so including it here only brings in the function definitions.
include(joinpath(@__DIR__, "sbp_second_derivative.jl"))

# --- assemble the SBP+SAT matrices -----------------------------------------

"""
    sbp_sat_matrices(D2, bc; α = accuracy_order(D2), symmetric = false, τ = 2.0)
        -> (; A, EL, ER, BL, BR, H)

Build the explicit matrix form of the SBP–SAT approximation of `u_xx` produced
by [`apply_D2_sat`](@ref), for the boundary configuration

    bc = [Dirichlet-left, Neumann-left, Dirichlet-right, Neumann-right]

(nonzero entry = active; exactly one active per side).  With `n` grid points:

  * `A  :: Matrix` (n×n) — SBP volume term `Matrix(D2)` plus every SAT term that
    multiplies the solution vector.
  * `EL, ER :: Matrix` (1×n) — boundary restriction operators that read the
    boundary datum from the solution in the neighbouring region.  `EL*wl` returns
    the left datum from `wl`, the solution on the grid just to the LEFT of the
    domain (interface `xmin` at its right-most node); `ER*wr` returns the right
    datum from `wr`, the solution just to the RIGHT (interface `xmax` at its
    left-most node).  Dirichlet → the interface value; Neumann → the outward
    normal derivative `u_n`.
  * `BL, BR :: Vector` (n) — boundary-data injection columns.
  * `H  :: Diagonal` — the SBP norm (mass) matrix, `mass_matrix(D2)`.

These satisfy, for all `u, gL, gR`,

    apply_D2_sat(D2, u, bc, gL, gR; symmetric, τ) == A*u + BL*gL + BR*gR,

and `gL = EL*wl`, `gR = ER*wr` reproduce `boundary_data` from the neighbouring
solutions.

With `symmetric = true` the energy-stable Dirichlet SAT is used, for which
`H*A` is symmetric and negative definite (`τ ≥ 2` ensures definiteness); see
[`symmetrize`](@ref).  With `symmetric = false` (default) the simpler
wave-equation penalty `α` is used and `A` is not symmetrisable in the SBP norm.
The Neumann SAT is energy-stable in both cases.
"""
function sbp_sat_matrices(D2, bc; α = accuracy_order(D2),
                          symmetric::Bool = false, τ = 2.0)
    length(bc) == 4 || throw(ArgumentError("bc must have 4 elements [DL, NL, DR, NR]"))
    DL, NL, DR, NR = (bc[1] != 0), (bc[2] != 0), (bc[3] != 0), (bc[4] != 0)
    (DL ⊻ NL) || throw(ArgumentError("left boundary needs exactly one of Dirichlet/Neumann active"))
    (DR ⊻ NR) || throw(ArgumentError("right boundary needs exactly one of Dirichlet/Neumann active"))

    n  = length(grid(D2))
    hL = left_boundary_weight(D2)
    hR = right_boundary_weight(D2)
    H  = mass_matrix(D2)            # diagonal SBP norm
    hv = diag(H)                    # its diagonal, for the H⁻¹-weighted SAT

    # SBP volume term (banded); `sparse(D2)` is available if a sparse A is wanted.
    A = Matrix(D2)

    # Boundary first-derivative functionals as vectors.
    #   dL·u = derivative_left(D2, u, Val(1)),   dR·u = derivative_right(...)
    #   tL, tR are the transpose-derivative ("lifting") vectors; tL == dL, tR == dR.
    tL = zeros(n); mul_transpose_derivative_left!(tL, D2, Val(1), 1.0, false)
    tR = zeros(n); mul_transpose_derivative_right!(tR, D2, Val(1), 1.0, false)
    dL = zeros(n); dR = zeros(n)
    e  = zeros(n)
    for j in 1:n
        e[j] = 1.0
        dL[j] = derivative_left(D2, e, Val(1))
        dR[j] = derivative_right(D2, e, Val(1))
        e[j] = 0.0
    end

    BL = zeros(n); BR = zeros(n)
    EL = zeros(1, n); ER = zeros(1, n)

    # --- left boundary ---
    if DL
        if symmetric
            # Energy-stable Dirichlet: A += -H⁻¹ dL e₁ᵀ - τ/hL² e₁ e₁ᵀ.
            @views A[:, 1] .+= (-1) .* (tL ./ hv)
            A[1, 1]        -= τ / hL^2
            @. BL = tL / hv
            BL[1] += τ / hL^2
        else
            # Wave-equation penalty: A += -(1/hL) dL e₁ᵀ - α/hL² e₁ e₁ᵀ.
            @views A[:, 1] .+= (-1 / hL) .* tL
            A[1, 1]        -= α / hL^2
            @. BL = (1 / hL) * tL
            BL[1] += α / hL^2
        end
        EL[1, n] = 1.0                          # EL*wl = wl[end] = value at interface xmin
    else
        # Neumann (already energy-stable): A[1,:] += dL/hL.
        @views A[1, :] .+= dL ./ hL
        BL[1] = 1 / hL
        @. EL[1, :] = -dR                       # EL*wl = -∂x wl |_right = u_n at xmin
    end

    # --- right boundary ---
    if DR
        if symmetric
            # Energy-stable Dirichlet: A += +H⁻¹ dR e_nᵀ - τ/hR² e_n e_nᵀ.
            @views A[:, n] .+= (tR ./ hv)
            A[n, n]        -= τ / hR^2
            @. BR = -(tR / hv)
            BR[n] += τ / hR^2
        else
            # Wave-equation penalty: A += +(1/hR) dR e_nᵀ - α/hR² e_n e_nᵀ.
            @views A[:, n] .+= (1 / hR) .* tR
            A[n, n]        -= α / hR^2
            @. BR = (-1 / hR) * tR
            BR[n] += α / hR^2
        end
        ER[1, 1] = 1.0                           # ER*wr = wr[1] = value at interface xmax
    else
        # Neumann (already energy-stable): A[n,:] -= dR/hR.
        @views A[n, :] .-= dR ./ hR
        BR[n] = 1 / hR
        @. ER[1, :] = dL                         # ER*wr = +∂x wr |_left = u_n at xmax
    end

    return (; A, EL, ER, BL, BR, H)
end

"""
    symmetrize(m) -> Symmetric

Return the genuinely symmetric matrix `Ã = H^{1/2} A H^{-1/2}` similar to `m.A`
(same real eigenvalues).  When `m` was built with `symmetric = true`, `Ã` is
symmetric negative definite, so `-Ã` is SPD and usable with Cholesky / CG, and
`eigvals` are guaranteed real.  Equivalently, `m.H * m.A` is symmetric.
"""
function symmetrize(m)
    s = sqrt.(diag(m.H))
    return Symmetric((s .* m.A) ./ s')   # H^{1/2} A H^{-1/2}, H diagonal
end

# --- demonstration / verification ------------------------------------------

function run_demo()
    xmin, xmax = 0.0, 1.0
    u_exact(x)   = exp(sin(1.3x + 0.4))
    ux_exact(x)  = 1.3cos(1.3x + 0.4) * exp(sin(1.3x + 0.4))
    uxx_exact(x) = (1.69 * (cos(1.3x + 0.4)^2 - sin(1.3x + 0.4))) * exp(sin(1.3x + 0.4))

    accuracy = 6
    Ns = (25, 50, 100, 200, 400, 800)

    # All four boundary-condition combinations: each side independently Dirichlet
    # or Neumann.  bc = [Dirichlet-left, Neumann-left, Dirichlet-right, Neumann-right].
    combos = [
        ("Dirichlet-left & Dirichlet-right", [1.0, 0.0, 1.0, 0.0]),
        ("Dirichlet-left & Neumann-right",   [1.0, 0.0, 0.0, 1.0]),
        ("Neumann-left & Dirichlet-right",   [0.0, 1.0, 1.0, 0.0]),
        ("Neumann-left & Neumann-right",     [0.0, 1.0, 0.0, 1.0]),
    ]

    symmetric = true             # use the energy-stable (symmetrisable) SAT

    println("Matrix form of the SBP–SAT u_xx operator")
    println("  domain        : [$xmin, $xmax]")
    println("  operator      : MattssonSvärdShoeybi2008, accuracy_order = $accuracy")
    println("  u_xx ≈ A*w + BL*(EL*wl) + BR*(ER*wr), BCs pulled from neighbour solutions")
    println("  building with symmetric = $symmetric (H*A symmetric negative definite)")
    println()

    # relative asymmetry of H*A in the SBP norm
    asym(m) = (HA = m.H * m.A; opnorm(HA - HA') / opnorm(HA))

    for (name, bc) in combos
        println("Boundary condition: ", name)

        # Structural check at one resolution: A*u + BL*gL + BR*gR == apply_D2_sat.
        Dchk = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                   accuracy_order = accuracy, xmin = xmin, xmax = xmax, N = 60)
        msym = sbp_sat_matrices(Dchk, bc; symmetric = true)
        mwav = sbp_sat_matrices(Dchk, bc; symmetric = false)
        ur = randn(60); gLr = randn(); gRr = randn()
        resid = maximum(abs, msym.A * ur .+ msym.BL .* gLr .+ msym.BR .* gRr
                             .- apply_D2_sat(Dchk, ur, bc, gLr, gRr; symmetric = true))
        println("  consistency max|A*u + BL*gL + BR*gR − apply_D2_sat| = ", fmt(resid))

        # Symmetry of H*A: symmetric SAT vs wave-equation SAT, plus the spectrum.
        ev = eigvals(symmetrize(msym))
        defin = maximum(ev) < -1e-8 * abs(minimum(ev)) ? "negative definite" :
                maximum(ev) <  1e-8 * abs(minimum(ev)) ? "negative semidefinite (singular)" :
                "indefinite"
        println("  ‖HA−(HA)ᵀ‖/‖HA‖ :  symmetric = ", fmt(asym(msym)),
                " ,  wave-eq = ", fmt(asym(mwav)))
        println("  eig(H^{1/2} A H^{-1/2}) ∈ [", fmt(minimum(ev)), ", ", fmt(maximum(ev)),
                "]  (", defin, ")")

        println("  ", rpar("N", 6), rpar("h", 14), rpar("max-err", 16), rpar("rate", 8))
        prev_err = NaN; prev_h = NaN
        for Nc in Ns
            D2c = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                      accuracy_order = accuracy, xmin = xmin, xmax = xmax, N = Nc)
            xc  = grid(D2c)
            h   = step(xc)
            wc  = u_exact.(xc)
            wl  = u_exact.(xc .- (xmax-xmin))
            wr  = u_exact.(xc .+ (xmax-xmin))

            mc  = sbp_sat_matrices(D2c, bc; symmetric = symmetric)

            # Apply the matrix form, pulling the boundary data out of the solution.
            uxx_h = mc.A * wc .+ mc.BL .* (mc.EL * wl) .+ mc.BR .* (mc.ER * wr)
            emax  = maximum(abs, uxx_h .- uxx_exact.(xc))

            rate = isnan(prev_err) ? NaN : log(prev_err / emax) / log(prev_h / h)
            println("  ", rpar(string(Nc), 6), rpar(fmt(h), 14), rpar(fmt(emax), 16),
                    rpar(isnan(rate) ? "--" : fmt(rate, 2), 8))
            prev_err, prev_h = emax, h
        end
        println()
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end

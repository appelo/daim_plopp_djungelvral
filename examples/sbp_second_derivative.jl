# Approximate the second derivative u_xx on a non-periodic interval [xmin, xmax]
# with a summation-by-parts (SBP) finite-difference operator, enforcing the
# boundary conditions *weakly* through simultaneous-approximation-term (SAT)
# penalties.
#
# Each boundary can carry either a Dirichlet condition (the solution value
# u(x_b) is prescribed) or a Neumann condition (the outward normal derivative
# u_n(x_b) is prescribed).  The configuration is chosen with a 4-element array
#
#     bc = [Dirichlet-left, Neumann-left, Dirichlet-right, Neumann-right]
#
# where a nonzero flag activates the corresponding condition.  Exactly one of
# {bc[1], bc[2]} (left) and one of {bc[3], bc[4]} (right) must be active.  For
# example, Dirichlet on the left and Neumann on the right is
#
#     bc = [1.0, 0.0, 0.0, 1.0]
#
# The SAT formulation here generalizes the homogeneous SATs used by this
# package's own `WaveEquationNonperiodicSemidiscretization`
# (src/second_order_eqs/wave_eq.jl) to inhomogeneous boundary data.
#
# Run from the package root:
#   julia --project=. examples/SBP/sbp_second_derivative.jl
#
# Required packages:
#   ] add SummationByPartsOperators LinearAlgebra   (Plots only if DO_PLOT)

using LinearAlgebra
using SummationByPartsOperators

# Set to `true` to plot the approximate vs. exact u_xx and the pointwise error.
# Requires Plots.jl in the active environment.
const DO_PLOT = false
if DO_PLOT
    using Plots
end

# --- small printing helpers ------------------------------------------------

# right-pad a string into a field of width `w`
rpar(s::AbstractString, w::Integer) = rpad(s, w)
fmt(v, digits::Integer = 4) = string(round(v; sigdigits = digits))

# --- core: SBP–SAT approximation of u_xx -----------------------------------

"""
    apply_D2_sat(D2, u, bc, gL, gR; α = accuracy_order(D2),
                 symmetric = false, τ = 2.0) -> Vector

Approximate `u_xx` from the grid values `u` using the second-derivative SBP
operator `D2`, weakly imposing the boundary conditions selected by

    bc = [Dirichlet-left, Neumann-left, Dirichlet-right, Neumann-right]

(nonzero entry = active; exactly one active per side).  The boundary data are

  * `gL`, `gR` = prescribed solution value           on a Dirichlet side, or
  * `gL`, `gR` = prescribed outward normal derivative `u_n` on a Neumann side.

Sign convention for Neumann data: the outward normal derivative is
`u_n = -∂_x u` at the left boundary and `u_n = +∂_x u` at the right boundary.
The left sign flip is handled internally, so `gL`/`gR` are always `u_n`.

Two Dirichlet SAT variants are available:

  * `symmetric = false` (default) — the simple penalty used by the package's
    wave-equation demo, with strength `α` (default `accuracy_order(D2)`).  The
    matrix-form operator `A` is *not* symmetric in the SBP norm.
  * `symmetric = true` — the energy-stable SAT for which `H*A` is symmetric and
    negative definite (`H = mass_matrix(D2)`).  The Neumann SAT is already of
    this form; only the Dirichlet term changes.  `τ` is the (dimensionless)
    Dirichlet penalty factor; `τ ≥ 2` guarantees definiteness for these
    operators.

The result reduces to `D2 * u` plus the homogeneous SATs when `gL = gR = 0`.
"""
function apply_D2_sat(D2, u, bc, gL, gR; α = accuracy_order(D2),
                      symmetric::Bool = false, τ = 2.0)
    length(bc) == 4 || throw(ArgumentError("bc must have 4 elements [DL, NL, DR, NR]"))
    DL, NL, DR, NR = (bc[1] != 0), (bc[2] != 0), (bc[3] != 0), (bc[4] != 0)
    (DL ⊻ NL) || throw(ArgumentError("left boundary needs exactly one of Dirichlet/Neumann active"))
    (DR ⊻ NR) || throw(ArgumentError("right boundary needs exactly one of Dirichlet/Neumann active"))

    hL = left_boundary_weight(D2)
    hR = right_boundary_weight(D2)

    # volume term: ddu ← D2 * u
    ddu = D2 * u

    # H⁻¹-weighting of the transpose-derivative SAT term is what makes `H*A`
    # symmetric; only needed for a symmetric Dirichlet boundary.
    hvec = (symmetric && (DL || DR)) ? diag(mass_matrix(D2)) : nothing

    # --- left boundary SAT ---
    if DL
        # Dirichlet: u(x_L) = gL.  Residual r = u[1] - gL.
        r = u[1] - gL
        if symmetric
            tL = zero(ddu); mul_transpose_derivative_left!(tL, D2, Val(1), 1.0, false)
            @. ddu -= r * (tL / hvec)            # -H⁻¹ d_L (u[1] - gL)
            @inbounds ddu[1] -= τ * r / hL^2     # -τ H⁻¹ e_L (u[1] - gL)
        else
            mul_transpose_derivative_left!(ddu, D2, Val(1), -r / hL, true)
            @inbounds ddu[1] -= α * r / hL^2
        end
    else
        # Neumann: u_n(x_L) = -∂_x u(x_L) = gL  ⟹  ∂_x u(x_L) = -gL.
        # Residual r = (∂_x u)_L - (-gL) = derivative_left + gL.
        r = derivative_left(D2, u, Val(1)) + gL
        @inbounds ddu[1] += r / hL
    end

    # --- right boundary SAT ---
    if DR
        # Dirichlet: u(x_R) = gR.  Residual r = u[end] - gR.
        r = u[end] - gR
        if symmetric
            tR = zero(ddu); mul_transpose_derivative_right!(tR, D2, Val(1), 1.0, false)
            @. ddu += r * (tR / hvec)            # +H⁻¹ d_R (u[end] - gR)
            @inbounds ddu[end] -= τ * r / hR^2   # -τ H⁻¹ e_R (u[end] - gR)
        else
            mul_transpose_derivative_right!(ddu, D2, Val(1), r / hR, true)
            @inbounds ddu[end] -= α * r / hR^2
        end
    else
        # Neumann: u_n(x_R) = +∂_x u(x_R) = gR.
        # Residual r = (∂_x u)_R - gR = derivative_right - gR.
        r = derivative_right(D2, u, Val(1)) - gR
        @inbounds ddu[end] -= r / hR
    end

    return ddu
end

# Convenience: pull the correct boundary datum (value or normal derivative) for
# a manufactured solution, given the active flags.
function boundary_data(bc, uL, uxL, uR, uxR)
    DL = bc[1] != 0
    DR = bc[3] != 0
    gL = DL ? uL : -uxL      # Neumann-left datum is u_n = -∂_x u
    gR = DR ? uR :  uxR      # Neumann-right datum is u_n = +∂_x u
    return gL, gR
end

# --- demonstration: manufactured solution + convergence study --------------

function run_demo()
    # Domain and manufactured solution (non-periodic, nontrivial at both ends).
    xmin, xmax = -1.0, 2.0
    u_exact(x)   = exp(sin(1.3x + 0.4))
    ux_exact(x)  = 1.3cos(1.3x + 0.4) * exp(sin(1.3x + 0.4))
    uxx_exact(x) = (1.69 * (cos(1.3x + 0.4)^2 - sin(1.3x + 0.4))) * exp(sin(1.3x + 0.4))

    accuracy = 8                      # interior accuracy order of the SBP operator

    # Boundary configuration: [Dirichlet-left, Neumann-left, Dirichlet-right, Neumann-right]
    bc = [0.0, 1.0, 1.0, 0.0]         # Dirichlet left, Neumann right (the example in the instructions)

    bc_names = ["Dirichlet-left", "Neumann-left", "Dirichlet-right", "Neumann-right"]
    active   = bc_names[findall(!=(0), bc)]
    println("SBP–SAT approximation of u_xx")
    println("  domain        : [$xmin, $xmax]")
    println("  operator      : MattssonSvärdShoeybi2008, accuracy_order = $accuracy")
    println("  boundary cond.: ", join(active, " & "))
    println()

    Ns = [25, 50, 100, 200, 400, 800]
    println(rpar("N", 6), rpar("h", 14), rpar("max-err", 16), rpar("L2_H-err", 16), rpar("rate", 8))

    prev_err = NaN
    prev_h   = NaN
    for N in Ns
        D2 = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                 accuracy_order = accuracy, xmin = xmin, xmax = xmax, N = N)
        x  = grid(D2)
        h  = step(x)
        u  = u_exact.(x)

        gL, gR = boundary_data(bc, u_exact(xmin), ux_exact(xmin), u_exact(xmax), ux_exact(xmax))
        uxx_h  = apply_D2_sat(D2, u, bc, gL, gR)

        err  = uxx_h .- uxx_exact.(x)
        emax = maximum(abs, err)
        # H-weighted (SBP) L2 norm of the error.
        H    = mass_matrix(D2)
        eL2  = sqrt(dot(err, H, err))

        rate = isnan(prev_err) ? NaN : log(prev_err / emax) / log(prev_h / h)
        println(rpar(string(N), 6), rpar(fmt(h), 14), rpar(fmt(emax), 16),
                rpar(fmt(eL2), 16), rpar(isnan(rate) ? "--" : fmt(rate, 2), 8))
        prev_err, prev_h = emax, h
    end

    if DO_PLOT
        N  = 200
        D2 = derivative_operator(MattssonSvärdShoeybi2008(); derivative_order = 2,
                                 accuracy_order = accuracy, xmin = xmin, xmax = xmax, N = N)
        x  = grid(D2)
        u  = u_exact.(x)
        gL, gR = boundary_data(bc, u_exact(xmin), ux_exact(xmin), u_exact(xmax), ux_exact(xmax))
        uxx_h  = apply_D2_sat(D2, u, bc, gL, gR)

        p1 = plot(x, uxx_exact.(x); label = "exact u_xx", lw = 2, xlabel = "x", ylabel = "u_xx",
                  title = "SBP–SAT u_xx ($(join(active, " & ")))")
        plot!(p1, x, uxx_h; label = "SBP–SAT", lw = 2, ls = :dash)
        p2 = plot(x, uxx_h .- uxx_exact.(x); label = "error", lw = 2, xlabel = "x",
                  ylabel = "u_xx error", color = :red)
        display(plot(p1, p2; layout = (2, 1), size = (900, 700)))
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end

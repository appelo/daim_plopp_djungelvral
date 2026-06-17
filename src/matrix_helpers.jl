# Low-rank matrix helpers: truncated sums of LRSVDs, single-LRSVD truncation, and
# the Frobenius-norm residual of a sum of LRSVD terms.

# --- truncated sum of LRSVD terms ------------------------------------------

function truncsum(C::Vector{LRSVD{T}}, tol::Real, rmax::Int) where T
    n_mat = length(C)
    r     = [length(c.S) for c in C]
    r_tot = sum(r)
    n1    = size(C[1].U, 1)
    n2    = size(C[1].V, 1)

    big_U = zeros(T, n1, r_tot)
    big_V = zeros(T, n2, r_tot)
    big_S = zeros(T, r_tot, r_tot)

    rc = [0; cumsum(r)]
    for i = 1:n_mat
        range = (rc[i]+1):(rc[i+1])
        big_U[:, range] .= C[i].U[:, 1:r[i]]
        big_V[:, range] .= C[i].V[:, 1:r[i]]
        big_S[range, range] .= diagm(C[i].S)  # block diagonal of singular values
    end

    # QR
    FU = qr(big_U, ColumnNorm())
    FV = qr(big_V, ColumnNorm())

    # Build the small core matrix and take its SVD
    core = (FU.R * FU.P') * big_S * (FV.P * FV.R')
    Fst = svd(core)

    energy = cumsum(Fst.S[end:-1:1].^2)
    r_st = length(energy) - length(findall(x -> x < tol^2, energy))
    r_st = max(min(r_st, rmax), 1)  # make sure rank is at least 1

    U = FU.Q * Fst.U[:,1:r_st]
    V = FV.Q * Fst.V[:,1:r_st]
    S = Fst.S[1:r_st]

    return U, S, V
end

function truncsum(C::Vector{LRSVD{T}}, tol::Real, rmax::Int, ::Type{LRSVD}) where T
    U, S, V = truncsum(C, tol, rmax)
    return LRSVD(U, S, V)
end

# --- truncation of a single LRSVD ------------------------------------------

"""
    trunclr(F::LRSVD, tol::Real, rmax::Int = length(F.S))

Frobenius-norm truncation of a low-rank `LRSVD` whose singular values `F.S` are
non-increasing.  Drops the smallest singular values whose cumulative squared
energy is below `tol^2`, keeping at least one and at most `rmax` of them
(same criterion as [`truncsum`](@ref)).
"""
function trunclr(F::LRSVD, tol::Real, rmax::Int = length(F.S))
    energy = cumsum(F.S[end:-1:1] .^ 2)
    r_st = length(energy) - length(findall(x -> x < tol^2, energy))
    r_st = max(min(r_st, rmax), 1)
    return LRSVD(F.U[:, 1:r_st], F.S[1:r_st], F.V[:, 1:r_st])
end

# --- Frobenius residual of a sum of LRSVD terms ----------------------------

function compute_residual(C::Vector{LRSVD{T}}) where T

    n_mat = length(C)
    r = [size(c.U, 2) for c in C]
    r_tot = sum(r)
    n1 = size(C[1].U, 1)
    n2 = size(C[1].V, 1)

    big_U = zeros(T, n1, r_tot)
    big_V = zeros(T, n2, r_tot)
    big_S = zeros(T, r_tot, r_tot)

    rc = [0; cumsum(r, dims=1)]

    for i = 1:n_mat
        big_U[:, rc[i]+1:rc[i+1]] .= C[i].U
        big_V[:, rc[i]+1:rc[i+1]] .= C[i].V
        big_S[rc[i]+1:rc[i+1], rc[i]+1:rc[i+1]] .= Diagonal(C[i].S)
    end

    FU = qr(big_U, ColumnNorm())
    FV = qr(big_V, ColumnNorm())
    rS = norm(FU.R * (big_S[FU.p, FV.p]) * FV.R')

    return rS
end

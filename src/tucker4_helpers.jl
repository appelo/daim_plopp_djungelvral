# Order-4 twins of the 3D Tucker helpers (src/tucker_helpers.jl).
# `gpode` is dimension independent (operates on a single basis matrix) and is
# reused as-is from tucker_helpers.jl — not duplicated here.

# Frobenius norm of the difference of two 4D Tucker tensors, computed without
# forming either full tensor. Order-4 twin of `residual3`.
function residual4(U1::NTuple{4, AbstractMatrix}, G1::AbstractArray{<:Any, 4},
                   U2::NTuple{4, AbstractMatrix}, G2::AbstractArray{<:Any, 4})
    sz1 = size(G1)
    sz2 = size(G2)
    T = promote_type(eltype(G1), eltype(G2), eltype(U1[1]), eltype(U2[1]))
    sz = sz1 .+ sz2
    G = zeros(T, sz)
    G[ntuple(l -> 1:sz1[l], 4)...] .= G1
    G[ntuple(l -> sz1[l]+1:sz[l], 4)...] .= -G2
    szc = collect(sz)
    for k in 1:4
        F = qr(hcat(U1[k], U2[k]), ColumnNorm())
        M = unfold(G, k)
        newM = F.R * M[F.p, :]
        szc[k] = size(F.R, 1)
        G = fold(newM, k, (szc[1], szc[2], szc[3], szc[4]))
    end
    return norm(G)
end

# Sequentially-truncated MLSVD with a relative-Frobenius tolerance tol ∈ [0,1].
# Order-4 twin of `mlsvd3`. Returns (U::NTuple{4,Matrix}, core).
function mlsvd4(G::AbstractArray{T, 4}, tol::Real) where T
    N = 4
    sz = collect(size(G))
    perm = sortperm(sz)                # heuristic: ascending mode size
    U = Vector{Matrix{T}}(undef, N)
    S = Array{T, 4}(G)
    sizetens = copy(sz)
    relerr = 0.0
    T2 = 0.0
    for ni in 1:N
        n = perm[ni]
        F = svd(unfold(S, n))
        sv = F.S
        Un = F.U
        if ni == 1
            T2 = sum(abs2, sv)
        end
        cs = cumsum(reverse(sv).^2)
        thresh = tol / (N - ni + 1)
        idx = 0
        for i in 1:length(cs)
            if relerr + sqrt(cs[i] / T2) < thresh
                idx = i
            else
                break
            end
        end
        idx = min(idx, length(cs) - 1)   # keep at least one column
        if idx > 0
            relerr += sqrt(cs[idx] / T2)
            Un = Un[:, 1:end-idx]
        end
        U[n] = Matrix(Un)
        bnd = size(Un, 2)
        Vmat = F.V[:, 1:bnd] * Diagonal(sv[1:bnd])
        sizetens[n] = bnd
        S = fold(Matrix(transpose(Vmat)), n, (sizetens[1], sizetens[2], sizetens[3], sizetens[4]))
    end
    return ntuple(l -> U[l], N), S
end

# Sequentially-truncated MLSVD to a fixed core size. Order-4 twin of mlsvd3(G,size_core).
function mlsvd4(G::AbstractArray{T, 4}, size_core::AbstractVector{<:Integer}) where T
    sz = collect(size(G))
    perm = sortperm(sz)
    U = Vector{Matrix{T}}(undef, 4)
    S = Array{T, 4}(G)
    sizetens = copy(sz)
    for ni in 1:4
        n = perm[ni]
        F = svd(unfold(S, n))
        r = min(size_core[n], length(F.S))
        U[n] = Matrix(F.U[:, 1:r])
        Vmat = F.V[:, 1:r] * Diagonal(F.S[1:r])
        sizetens[n] = r
        S = fold(Matrix(transpose(Vmat)), n, (sizetens[1], sizetens[2], sizetens[3], sizetens[4]))
    end
    return ntuple(l -> U[l], 4), S
end

# Exact low-rank sum of a vector of 4D Tucker tensors, then rounding. Order-4 twin
# of `tucker_sum`: stack factor matrices, build a block-diagonal core, orthogonalize
# each mode by a column-pivoted QR (absorbing R·Pᵀ into the core), then truncate with
# `mlsvd4` to absolute Frobenius tolerance `tol`. Deterministic (no sampling).
# For differences, pass a negated-core tensor `Tucker4(F.U, -F.G)`.
function tucker_sum(Fs::AbstractVector{<:Tucker4}, tol::Real = 1e-10)
    isempty(Fs) && throw(ArgumentError("tucker_sum: need at least one tensor"))
    nmat = length(Fs)
    n = ntuple(l -> size(Fs[1].U[l], 1), 4)
    for F in Fs
        ntuple(l -> size(F.U[l], 1), 4) == n ||
            throw(DimensionMismatch("tucker_sum: all tensors must share the outer dimensions $n"))
    end
    T = promote_type((eltype(F.G) for F in Fs)...,
                     (eltype(F.U[l]) for F in Fs for l in 1:4)...)

    # per-tensor core sizes, total block-diagonal size, cumulative offsets
    szg = [size(Fs[i].G, k) for i in 1:nmat, k in 1:4]
    sz  = ntuple(k -> sum(@view szg[:, k]), 4)
    off = zeros(Int, nmat + 1, 4)
    for k in 1:4, i in 1:nmat
        off[i + 1, k] = off[i, k] + szg[i, k]
    end

    # block-diagonal core
    G = zeros(T, sz)
    for i in 1:nmat
        G[ntuple(k -> (off[i, k] + 1):off[i + 1, k], 4)...] = Fs[i].G
    end

    # per mode: concatenate the factors, pivoted QR, absorb R·Pᵀ into the core
    U = Vector{Matrix{T}}(undef, 4)
    for k in 1:4
        bigU = zeros(T, n[k], sz[k])
        for i in 1:nmat
            bigU[:, (off[i, k] + 1):off[i + 1, k]] = Fs[i].U[k]
        end
        F = qr(bigU, ColumnNorm())
        r = size(F.R, 1)                                  # = min(n[k], sz[k])
        M = unfold(G, k)
        newsz = ntuple(l -> l == k ? r : size(G, l), 4)
        G = fold(F.R * M[F.p, :], k, newsz)               # new core: R·Pᵀ applied to mode k
        U[k] = Matrix(F.Q)[:, 1:r]
    end

    # round to absolute tolerance `tol`
    Usvd, Gt = mlsvd4(G, min(tol / norm(G), 0.999))
    return Tucker4(ntuple(l -> U[l] * Usvd[l], 4), Gt)
end

# Gappy-POD oversampling point selection (Peherstorfer, Drmac & Gugercin).
# Port of matlab/service/gpode.m; operates on a single basis matrix so it is
# dimension independent. Returns `m` row indices of `U`.
function gpode(U::AbstractMatrix, m::Int)
    n, k = size(U)
    @assert k <= m <= n "gpode: need size(U,2) <= m <= size(U,1)"
    p = qdeim(U)                       # first k points via Q-DEIM
    Ut = transpose(U)
    for _ in (length(p) + 1):m
        F = svd(U[p, :])
        sv = F.S
        g = length(sv) > 1 ? sv[end-1]^2 - sv[end]^2 : sv[end]^2
        Ub = F.Vt * Ut                 # k × n
        q = vec(sum(abs2, Ub; dims = 1))
        r = (g .+ q) .- sqrt.(max.((g .+ q).^2 .- 4 .* g .* vec(Ub[end, :]).^2, 0))
        order = sortperm(r; rev = true)
        e = 1
        while order[e] in p
            e += 1
        end
        push!(p, order[e])
    end
    return p
end

# Frobenius norm of the difference of two 3D Tucker tensors, computed without
# forming either full tensor. Port of matlab/service/residual.m (d = 3).
function residual3(U1::NTuple{3, AbstractMatrix}, G1::AbstractArray{<:Any, 3},
                   U2::NTuple{3, AbstractMatrix}, G2::AbstractArray{<:Any, 3})
    sz1 = size(G1)
    sz2 = size(G2)
    T = promote_type(eltype(G1), eltype(G2), eltype(U1[1]), eltype(U2[1]))
    sz = sz1 .+ sz2
    G = zeros(T, sz)
    G[1:sz1[1], 1:sz1[2], 1:sz1[3]] .= G1
    G[sz1[1]+1:end, sz1[2]+1:end, sz1[3]+1:end] .= -G2
    szc = collect(sz)
    for k in 1:3
        F = qr(hcat(U1[k], U2[k]), ColumnNorm())
        M = unfold(G, k)
        newM = F.R * M[F.p, :]
        szc[k] = size(F.R, 1)
        G = fold(newM, k, (szc[1], szc[2], szc[3]))
    end
    return norm(G)
end

# Sequentially-truncated MLSVD with a relative-Frobenius tolerance tol ∈ [0,1].
# Port of the scalar-tol branch of matlab/tensorlab/mlsvd.m for N = 3.
# Returns (U::NTuple{3,Matrix}, core).
function mlsvd3(G::AbstractArray{T, 3}, tol::Real) where T
    N = 3
    sz = collect(size(G))
    perm = sortperm(sz)                # heuristic: ascending mode size
    U = Vector{Matrix{T}}(undef, 3)
    S = Array{T, 3}(G)
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
        S = fold(Matrix(transpose(Vmat)), n, (sizetens[1], sizetens[2], sizetens[3]))
    end
    return ntuple(l -> U[l], 3), S
end

# Sequentially-truncated MLSVD to a fixed core size (tensorlab mlsvd(G,size_core)).
function mlsvd3(G::AbstractArray{T, 3}, size_core::AbstractVector{<:Integer}) where T
    sz = collect(size(G))
    perm = sortperm(sz)
    U = Vector{Matrix{T}}(undef, 3)
    S = Array{T, 3}(G)
    sizetens = copy(sz)
    for ni in 1:3
        n = perm[ni]
        F = svd(unfold(S, n))
        r = min(size_core[n], length(F.S))
        U[n] = Matrix(F.U[:, 1:r])
        Vmat = F.V[:, 1:r] * Diagonal(F.S[1:r])
        sizetens[n] = r
        S = fold(Matrix(transpose(Vmat)), n, (sizetens[1], sizetens[2], sizetens[3]))
    end
    return ntuple(l -> U[l], 3), S
end

# Exact low-rank sum of a vector of 3D Tucker tensors, then rounding.  Port of
# matlab/service/Tucker_multi_sum.m (d = 3): stack the factor matrices and build a
# block-diagonal core, orthogonalize each mode by a column-pivoted QR (absorbing
# R·Pᵀ into the core), then truncate with `mlsvd3` to absolute Frobenius tolerance
# `tol`.  Deterministic (no sampling) — the Tucker analogue of the LRSVD `truncsum`.
# Returns a `Tucker3`.  For differences, pass a negated-core tensor `Tucker3(F.U, -F.G)`.
function tucker_sum(Fs::AbstractVector{<:Tucker3}, tol::Real = 1e-10)
    isempty(Fs) && throw(ArgumentError("tucker_sum: need at least one tensor"))
    nmat = length(Fs)
    n = ntuple(l -> size(Fs[1].U[l], 1), 3)
    for F in Fs
        ntuple(l -> size(F.U[l], 1), 3) == n ||
            throw(DimensionMismatch("tucker_sum: all tensors must share the outer dimensions $n"))
    end
    T = promote_type((eltype(F.G) for F in Fs)...,
                     (eltype(F.U[l]) for F in Fs for l in 1:3)...)

    # per-tensor core sizes, total block-diagonal size, cumulative offsets
    szg = [size(Fs[i].G, k) for i in 1:nmat, k in 1:3]
    sz  = ntuple(k -> sum(@view szg[:, k]), 3)
    off = zeros(Int, nmat + 1, 3)
    for k in 1:3, i in 1:nmat
        off[i + 1, k] = off[i, k] + szg[i, k]
    end

    # block-diagonal core
    G = zeros(T, sz)
    for i in 1:nmat
        G[ntuple(k -> (off[i, k] + 1):off[i + 1, k], 3)...] = Fs[i].G
    end

    # per mode: concatenate the factors, pivoted QR, absorb R·Pᵀ into the core
    U = Vector{Matrix{T}}(undef, 3)
    for k in 1:3
        bigU = zeros(T, n[k], sz[k])
        for i in 1:nmat
            bigU[:, (off[i, k] + 1):off[i + 1, k]] = Fs[i].U[k]
        end
        F = qr(bigU, ColumnNorm())
        r = size(F.R, 1)                                  # = min(n[k], sz[k])
        M = unfold(G, k)
        newsz = ntuple(l -> l == k ? r : size(G, l), 3)
        G = fold(F.R * M[F.p, :], k, newsz)               # new core: R·Pᵀ applied to mode k
        U[k] = Matrix(F.Q)[:, 1:r]
    end

    # round to absolute tolerance `tol`
    Usvd, Gt = mlsvd3(G, min(tol / norm(G), 0.999))
    return Tucker3(ntuple(l -> U[l] * Usvd[l], 3), Gt)
end

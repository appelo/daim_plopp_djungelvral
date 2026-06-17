# 3D Tucker cross-approximation with DEIM-guided index selection.
# Port of matlab/service/cross2DEIM3D.m, specialized to three modes.
#
# `gfun(i,j,k)` evaluates the target tensor at a single entry (matrix-free,
# same convention as the 2D `crossDEIM`/`scross`). `U0`/`G0` are the starting
# Tucker factors and core. Returns the updated factors, core and a debug
# `result` history matrix with rows `[iter res σ_min(1:3) size(G,1:3)]`.

# the two modes other than l, ascending
@inline _others(l) = l == 1 ? (2, 3) : (l == 2 ? (1, 3) : (1, 2))
@inline _scalar(x) = x isa AbstractArray ? x[1] : x

function crossDEIM(gfun, U0::NTuple{3, AbstractMatrix}, G0::AbstractArray{<:Any, 3},
                   opts = nothing)
    if !isnothing(opts)
        tol       = opts.tol
        rmax      = opts.rmax
        max_iter  = opts.max_iter
        increment = opts.increment
    else
        tol = 1e-10; rmax = 100; max_iter = 20; increment = 4
    end
    use_cache = (!isnothing(opts) && hasproperty(opts, :cache)) ? opts.cache : true

    T = promote_type(eltype(U0[1]), eltype(U0[2]), eltype(U0[3]),
                     typeof(float(_scalar(gfun(1, 1, 1)))))

    # Memoize gfun across iterations: the DEIM-guided index sets overlap heavily
    # between sweeps (I0 = previous I, oversampling reuses I), so the same (i,j,k)
    # entries are queried many times. Caching returns the identical stored value, so
    # results are bit-identical to the uncached path. Opt out via opts.cache=false
    # (e.g. when gfun is a trivial array lookup and the Dict overhead isn't worth it).
    g = if use_cache
        cache = Dict{NTuple{3, Int}, T}()
        (i, j, k) -> get!(() -> T(_scalar(gfun(i, j, k))), cache, (i, j, k))
    else
        (i, j, k) -> T(_scalar(gfun(i, j, k)))
    end

    n = ntuple(l -> size(U0[l], 1), 3)
    U  = Matrix{T}[Matrix{T}(U0[l]) for l in 1:3]
    Up = Matrix{T}[Matrix{T}(U0[l]) for l in 1:3]   # previous-iterate factors
    Gp = Array{T, 3}(G0)                            # previous-iterate core
    G  = Array{T, 3}(undef, 0, 0, 0)
    I  = [Int[], Int[], Int[]]
    I0 = [Int[], Int[], Int[]]
    sarr = zeros(Float64, 3)

    result = Vector{Vector{Float64}}()
    n_iters = 0
    for iter in 1:max_iter
        n_iters = iter

        # ---- index selection per mode -------------------------------------
        for l in 1:3
            Il = qdeim(U[l])
            Itmp = unique(vcat(Il, I0[l]))
            Il = Itmp[1:min(length(Il) + increment, length(Itmp))]
            if length(I0[l]) == length(Il)
                Iadd = setdiff(1:n[l], Il)
                if !isempty(Iadd)
                    Il = vcat(Il, Iadd[randperm(length(Iadd))[1:min(1, length(Iadd))]])
                end
            end
            if length(Il) > rmax
                Il = Il[1:rmax]
            end
            I[l] = Il
        end

        # ---- factor update per mode ---------------------------------------
        for l in 1:3
            a, b = _others(l)
            Ia, Ib, Il = I[a], I[b], I[l]
            na, nb, sl = length(Ia), length(Ib), length(Il)

            C = Matrix{T}(undef, sl, na * nb)
            for i1 in 1:sl
                col = 1
                for ja in 1:na, jb in 1:nb
                    idx = _idx(l, Il[i1], a, Ia[ja], b, Ib[jb])
                    C[i1, col] = g(idx[1], idx[2], idx[3])
                    col += 1
                end
            end
            utmp = svd(Matrix(transpose(C))).U
            Inl = qdeim(utmp)

            C2 = Matrix{T}(undef, n[l], length(Inl))
            for i2 in eachindex(Inl)
                lin = Inl[i2]
                ja = div(lin - 1, nb) + 1
                jb = mod(lin - 1, nb) + 1
                va, vb = Ia[ja], Ib[jb]
                for i1 in 1:n[l]
                    idx = _idx(l, i1, a, va, b, vb)
                    C2[i1, i2] = g(idx[1], idx[2], idx[3])
                end
            end
            U[l] = Matrix(svd(C2).U)
        end

        # ---- core construction via oversampling ---------------------------
        osm = 3
        Iover = Vector{Vector{Int}}(undef, 3)
        for l in 1:3
            os = min(osm, n[l] - length(I[l]))
            pts = gpode(Up[l], length(I[l]) + os)      # oversample on old factors
            Iov = unique(vcat(I[l], pts))
            if length(Iov) - length(I[l]) < os
                Iadd = setdiff(1:n[l], Iov)
                add = Iadd[randperm(length(Iadd))[1:min(os, length(Iadd))]]
                Iov = vcat(Iov, add)[1:length(I[l]) + os]
            end
            Iover[l] = Iov
        end

        s1, s2, s3 = length(Iover[1]), length(Iover[2]), length(Iover[3])
        W = Array{T, 3}(undef, s1, s2, s3)
        for c in 1:s3, b in 1:s2, a in 1:s1
            W[a, b, c] = g(Iover[1][a], Iover[2][b], Iover[3][c])
        end
        UI = ntuple(l -> pinv(U[l][Iover[l], :]), 3)
        G = lmlragen3(UI, W)

        for l in 1:3
            sarr[l] = svd(unfold(G, l)).S[end]
        end
        res = residual3(ntuple(l -> U[l], 3), G, ntuple(l -> Up[l], 3), Gp)

        Gp = G
        Up = Matrix{T}[copy(U[l]) for l in 1:3]
        I0 = [copy(I[l]) for l in 1:3]

        push!(result, vcat(Float64(iter), res, sarr..., Float64.(size(G))...))
        if res < tol && all(sarr .< tol)
            break
        end
    end

    # ---- final truncation ------------------------------------------------
    if n_iters > 1
        Usvd, Gt = mlsvd3(G, min(tol / norm(G), 0.999))
        G = Gt
        for l in 1:3
            U[l] = U[l] * Usvd[l]
        end
        for l in 1:3
            sarr[l] = svd(unfold(G, l)).S[end]
        end
        res = residual3(ntuple(l -> U[l], 3), G, ntuple(l -> Up[l], 3), Gp)
        push!(result, vcat(Float64(n_iters), res, sarr..., Float64.(size(G))...))
    end

    Umat = reduce(vcat, (reshape(r, 1, :) for r in result))
    return ntuple(l -> U[l], 3), G, Umat
end

# Place (vl, va, vb) at positions (l, a, b) of a 3-tuple. Returns an NTuple{3,Int}
# (no heap allocation) so the inner-loop gfun calls don't allocate per entry.
@inline function _idx(l, vl, a, va, b, vb)
    return ntuple(p -> p == l ? vl : (p == a ? va : vb), 3)
end

function crossDEIM(gfun, F0::Tucker3, opts = nothing)
    U, G, result = crossDEIM(gfun, F0.U, F0.G, opts)
    return Tucker3(U, G), result
end

# Matrix-free low-rank sum of a vector of Tucker3 tensors: returns Σ Fs[t] as a
# single Tucker3, computed by applying the Tucker `crossDEIM` to the entrywise sum
#   gfun(i,j,k) = Σ_t Fs[t][i,j,k]   (each term via `tucker_eval`, no densification).
# The Tucker analogue of the LRSVD `truncsum`.  `tol` is the Cross-DEIM accuracy;
# the result is truncated to it.  For differences, pass a tensor with a negated
# core, `Tucker3(F.U, -F.G)` (cf. `truncsum`'s `LRSVD(U, -S, V)`).
function tucker_cross_sum(Fs::AbstractVector{<:Tucker3};
                          tol = 1e-10, rmax = nothing, max_iter = 20, increment = 4,
                          cache = true)
    isempty(Fs) && throw(ArgumentError("tucker_cross_sum: need at least one tensor"))
    n = ntuple(l -> size(Fs[1].U[l], 1), 3)
    for F in Fs
        ntuple(l -> size(F.U[l], 1), 3) == n ||
            throw(DimensionMismatch("tucker_cross_sum: all tensors must share the outer dimensions $n"))
    end
    length(Fs) == 1 && return Fs[1]

    T = promote_type((eltype(F.G) for F in Fs)...,
                     (eltype(F.U[l]) for F in Fs for l in 1:3)...)
    gfun = (i, j, k) -> sum(tucker_eval(F, i, j, k) for F in Fs)

    # rank-1 seed (Cross-DEIM grows the rank; a near-full-rank seed overflows the
    # gpode oversampling) + oversampling headroom in rmax.
    seed = Tucker3(ntuple(l -> Fs[1].U[l][:, 1:1], 3), reshape([one(T)], 1, 1, 1))
    rmx  = rmax === nothing ? max(1, minimum(n) - 4) : rmax
    F, _ = crossDEIM(gfun, seed,
                     (; tol = tol, rmax = rmx, max_iter = max_iter, increment = increment, cache = cache))

    # Recompress to the requested relative tolerance: Cross-DEIM can return many more
    # modes than the true sum rank (its final truncation uses a tiny absolute tol and
    # its cross-approximation noise floor leaves spurious small singular values).  QR
    # the factors, MLSVD the small transformed core — no n³ densification.
    QR = ntuple(l -> qr(F.U[l]), 3)
    Q  = ntuple(l -> Matrix(QR[l].Q)[:, 1:size(F.U[l], 2)], 3)
    C  = modemult(modemult(modemult(F.G, QR[1].R, 1), QR[2].R, 2), QR[3].R, 3)
    Uc, Gc = mlsvd3(C, tol)
    return Tucker3(ntuple(l -> Q[l] * Uc[l], 3), Gc)
end

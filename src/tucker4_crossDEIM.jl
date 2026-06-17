# 4D Tucker cross-approximation with DEIM-guided index selection.
# Order-4 twin of src/tucker_crossDEIM.jl (which is specialized to three modes).
#
# `gfun(i,j,k,l)` evaluates the target tensor at a single entry (matrix-free).
# `U0`/`G0` are the starting Tucker factors and core. Returns the updated factors,
# core and a debug `result` history matrix with rows
# `[iter res σ_min(1:4) size(G,1:4)]`.

# the three modes other than l, ascending
@inline _others4(l) = l == 1 ? (2, 3, 4) :
                      l == 2 ? (1, 3, 4) :
                      l == 3 ? (1, 2, 4) : (1, 2, 3)

# Place (vl, va, vb, vc) at positions (l, a, b, c) of a 4-tuple. Returns an
# NTuple{4,Int} (no heap allocation) so the inner-loop gfun calls don't allocate.
@inline function _idx4(l, vl, a, va, b, vb, c, vc)
    return ntuple(p -> p == l ? vl : (p == a ? va : (p == b ? vb : vc)), 4)
end

function crossDEIM(gfun, U0::NTuple{4, AbstractMatrix}, G0::AbstractArray{<:Any, 4},
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

    T = promote_type(eltype(U0[1]), eltype(U0[2]), eltype(U0[3]), eltype(U0[4]),
                     typeof(float(_scalar(gfun(1, 1, 1, 1)))))

    # Memoize gfun across iterations (see tucker_crossDEIM.jl for rationale). Caching
    # returns the identical stored value, so results are bit-identical to the uncached
    # path. Opt out via opts.cache=false.
    g = if use_cache
        cache = Dict{NTuple{4, Int}, T}()
        (i, j, k, l) -> get!(() -> T(_scalar(gfun(i, j, k, l))), cache, (i, j, k, l))
    else
        (i, j, k, l) -> T(_scalar(gfun(i, j, k, l)))
    end

    n = ntuple(l -> size(U0[l], 1), 4)
    U  = Matrix{T}[Matrix{T}(U0[l]) for l in 1:4]
    Up = Matrix{T}[Matrix{T}(U0[l]) for l in 1:4]   # previous-iterate factors
    Gp = Array{T, 4}(G0)                            # previous-iterate core
    G  = Array{T, 4}(undef, 0, 0, 0, 0)
    I  = [Int[], Int[], Int[], Int[]]
    I0 = [Int[], Int[], Int[], Int[]]
    sarr = zeros(Float64, 4)

    result = Vector{Vector{Float64}}()
    n_iters = 0
    for iter in 1:max_iter
        n_iters = iter

        # ---- index selection per mode -------------------------------------
        for l in 1:4
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
        for l in 1:4
            a, b, c = _others4(l)
            Ia, Ib, Ic, Il = I[a], I[b], I[c], I[l]
            na, nb, nc, sl = length(Ia), length(Ib), length(Ic), length(Il)

            C = Matrix{T}(undef, sl, na * nb * nc)
            for i1 in 1:sl
                col = 1
                for ja in 1:na, jb in 1:nb, jc in 1:nc
                    idx = _idx4(l, Il[i1], a, Ia[ja], b, Ib[jb], c, Ic[jc])
                    C[i1, col] = g(idx[1], idx[2], idx[3], idx[4])
                    col += 1
                end
            end
            utmp = svd(Matrix(transpose(C))).U
            Inl = qdeim(utmp)

            C2 = Matrix{T}(undef, n[l], length(Inl))
            for i2 in eachindex(Inl)
                lin = Inl[i2] - 1                       # 0-based; columns iterate jc fastest, then jb, then ja
                jc = mod(lin, nc) + 1
                rem = div(lin, nc)
                jb = mod(rem, nb) + 1
                ja = div(rem, nb) + 1
                va, vb, vc = Ia[ja], Ib[jb], Ic[jc]
                for i1 in 1:n[l]
                    idx = _idx4(l, i1, a, va, b, vb, c, vc)
                    C2[i1, i2] = g(idx[1], idx[2], idx[3], idx[4])
                end
            end
            U[l] = Matrix(svd(C2).U)
        end

        # ---- core construction via oversampling ---------------------------
        osm = 3
        Iover = Vector{Vector{Int}}(undef, 4)
        for l in 1:4
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

        s1, s2, s3, s4 = length(Iover[1]), length(Iover[2]), length(Iover[3]), length(Iover[4])
        W = Array{T, 4}(undef, s1, s2, s3, s4)
        for d in 1:s4, c in 1:s3, b in 1:s2, a in 1:s1
            W[a, b, c, d] = g(Iover[1][a], Iover[2][b], Iover[3][c], Iover[4][d])
        end
        UI = ntuple(l -> pinv(U[l][Iover[l], :]), 4)
        G = lmlragen4(UI, W)

        for l in 1:4
            sarr[l] = svd(unfold(G, l)).S[end]
        end
        res = residual4(ntuple(l -> U[l], 4), G, ntuple(l -> Up[l], 4), Gp)

        Gp = G
        Up = Matrix{T}[copy(U[l]) for l in 1:4]
        I0 = [copy(I[l]) for l in 1:4]

        push!(result, vcat(Float64(iter), res, sarr..., Float64.(size(G))...))
        if res < tol && all(sarr .< tol)
            break
        end
    end

    # ---- final truncation ------------------------------------------------
    if n_iters > 1
        Usvd, Gt = mlsvd4(G, min(tol / norm(G), 0.999))
        G = Gt
        for l in 1:4
            U[l] = U[l] * Usvd[l]
        end
        for l in 1:4
            sarr[l] = svd(unfold(G, l)).S[end]
        end
        res = residual4(ntuple(l -> U[l], 4), G, ntuple(l -> Up[l], 4), Gp)
        push!(result, vcat(Float64(n_iters), res, sarr..., Float64.(size(G))...))
    end

    Umat = reduce(vcat, (reshape(r, 1, :) for r in result))
    return ntuple(l -> U[l], 4), G, Umat
end

function crossDEIM(gfun, F0::Tucker4, opts = nothing)
    U, G, result = crossDEIM(gfun, F0.U, F0.G, opts)
    return Tucker4(U, G), result
end

# Matrix-free low-rank sum of a vector of Tucker4 tensors: returns Σ Fs[t] as a single
# Tucker4 via the Tucker `crossDEIM` on the entrywise sum. Order-4 twin of
# `tucker_cross_sum`. For differences, pass a tensor with a negated core,
# `Tucker4(F.U, -F.G)`.
function tucker_cross_sum(Fs::AbstractVector{<:Tucker4};
                          tol = 1e-10, rmax = nothing, max_iter = 20, increment = 4,
                          cache = true)
    isempty(Fs) && throw(ArgumentError("tucker_cross_sum: need at least one tensor"))
    n = ntuple(l -> size(Fs[1].U[l], 1), 4)
    for F in Fs
        ntuple(l -> size(F.U[l], 1), 4) == n ||
            throw(DimensionMismatch("tucker_cross_sum: all tensors must share the outer dimensions $n"))
    end
    length(Fs) == 1 && return Fs[1]

    T = promote_type((eltype(F.G) for F in Fs)...,
                     (eltype(F.U[l]) for F in Fs for l in 1:4)...)
    gfun = (i, j, k, l) -> sum(tucker_eval(F, i, j, k, l) for F in Fs)

    # rank-1 seed (Cross-DEIM grows the rank) + oversampling headroom in rmax.
    seed = Tucker4(ntuple(l -> Fs[1].U[l][:, 1:1], 4), reshape([one(T)], 1, 1, 1, 1))
    rmx  = rmax === nothing ? max(1, minimum(n) - 4) : rmax
    F, _ = crossDEIM(gfun, seed,
                     (; tol = tol, rmax = rmx, max_iter = max_iter, increment = increment, cache = cache))

    # Recompress to the requested relative tolerance (see tucker_cross_sum, d=3).
    QR = ntuple(l -> qr(F.U[l]), 4)
    Q  = ntuple(l -> Matrix(QR[l].Q)[:, 1:size(F.U[l], 2)], 4)
    C  = modemult(modemult(modemult(modemult(F.G, QR[1].R, 1), QR[2].R, 2), QR[3].R, 3), QR[4].R, 4)
    Uc, Gc = mlsvd4(C, tol)
    return Tucker4(ntuple(l -> Q[l] * Uc[l], 4), Gc)
end

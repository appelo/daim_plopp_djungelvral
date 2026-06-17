# kv_leverage_demo.jl
# Demonstration of leverage score–based KV cache pruning

using LinearAlgebra, Random, Statistics
using NPZ

# --- Load a real GPT-2 inference KV cache produced by extract_kv.py ---
# K, V are the cached keys/values of one (layer, head): seq_len × head_dim.
# q is the query of the last position (what a decode step attends with).
function load_kv(path="kv_cache.npz")
    data = npzread(path)
    K = Float64.(data["K"])   # seq × head_dim
    V = Float64.(data["V"])   # seq × head_dim
    q = Float64.(data["q"])   # head_dim
    return K, V, q
end

# --- Random projection of K onto an r-dimensional orthonormal basis ---
# Computed once and shared by both the leverage-score and Q-DEIM paths.
function random_projection(K; r=32)
    n, d = size(K)
    Ω = randn(d, r)
    Y = K * Ω
    Q, _ = qr(Y)
    return Matrix(Q[:, 1:r])
end

# --- Approximate leverage scores from a projected orthonormal basis ---
function leverage_scores(Q)
    lev = sum(abs2, Q; dims=2)
    return vec(lev)
end

# --- Prune KV cache using top-k leverage ---
function prune_kv(K, V, k, Q)
    lev = leverage_scores(Q)
    idx = partialsortperm(lev, rev=true, 1:k)
    return K[idx, :], V[idx, :], lev
end

function qdeim(U)
    # Input : U n−by−m with orthonormal columns
    # Output : S selection of m row indices with guaranteed upper bound
    # norm(inv(U(S,:))) <= sqrt(n−m+1) * O(2ˆm).
    # : M the matrix U*inv(U(S,:));
    # The Q−DEIM projection of an n−by−1 vector f is M*f(S).
    # Coded by Zlatko Drmac, April 2015.
    F = qr(transpose(U), ColumnNorm())
    if ndims(U) == 1
        I = zeros(Int64,1)
        I[1] = F.p[1]
        return I
    else
        n, m = size(U)
        return F.p[1:m]
    end
end

# --- Prune KV cache using Q-DEIM on the projected basis ---
function prune_kv_qdeim(K, V, k, Q)
    idx = qdeim(Q[:, 1:k])
    return K[idx, :], V[idx, :], idx
end


# --- Attention computation (scaled dot-product) ---
function attention(q, K, V)
    scores = (K * q) ./ sqrt(size(K, 2))   # 1/sqrt(d) scaling avoids a near-argmax softmax
    weights = exp.(scores .- maximum(scores))
    weights /= sum(weights)
    return weights' * V
end

# --- Main demo ---
function main()
    Random.seed!(0)

    K, V, q = load_kv()      # real GPT-2 KV cache: K,V are seq × d
    n, d = size(K)           # n = seq_len tokens, d = key dimension (all heads → 768)
    k = min(256, d)          # keep only top-k tokens

    # Q-DEIM selects k rows from a rank-k basis, and the projected basis K*Ω can
    # only reach rank d. Requiring k <= d keeps that basis full rank, so no
    # arbitrary null-space columns leak into the selection.
    @assert k <= d "projection rank k=$k must not exceed head dimension d=$d"

    full_out = attention(q, K, V)

    # Shared random projection: qdeim needs k orthonormal columns, so project to rank k.
    Q = random_projection(K; r=k)

    Kp, Vp, lev = prune_kv(K, V, k, Q)
    pruned_out = attention(q, Kp, Vp)

    Kp_qdeim, Vp_qdeim,idx = prune_kv_qdeim(K, V, k, Q)
    pruned_out_qdeim = attention(q, Kp_qdeim, Vp_qdeim)

    err = norm(full_out - pruned_out) / norm(full_out)
    err_qdeim = norm(full_out - pruned_out_qdeim) / norm(full_out)

    println("Original tokens: ", n)
    println("Pruned tokens:   ", k)
    println("Relative error:  ", err)
    println("Relative error q_deim:  ", err_qdeim)
    return lev,idx,Kp,Kp_qdeim,K
end

lev,idx,Kp,Kp_qdeim,K = main()

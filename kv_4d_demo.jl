# kv_4d_demo.jl
# End-to-end 4D KV-cache demo on top of LRDD's Tucker4 tools.
#
# The cache produced by extract_kv_4d.py is a 4D tensor
#     K : [n_layers, n_heads, seq, head_dim]
# i.e. the genuine GPT-2 cached keys for every (layer, head). This script shows three
# things you can do with it via LRDD:
#   1. Tucker-compress the whole cache (sequentially-truncated MLSVD) -> mlsvd4
#   2. Recover it matrix-free, sampling only individual entries          -> crossDEIM
#   3. Prune the *token* axis with Q-DEIM (the 4D analogue of kv_leverage_demo.jl)
#
# Run:  julia --project=. kv_4d_demo.jl
# (Loads kv_cache_4d.npz if NPZ is installed and the file exists; otherwise it falls
#  back to a synthetic low-rank cache so the demo runs with no Python/torch involved.)

using LRDD
using LinearAlgebra, Random

# NPZ is optional: only needed to read a real kv_cache_4d.npz. The demo runs without it.
const HAS_NPZ = try
    @eval import NPZ
    true
catch
    false
end

# Load the real 4D KV cache, or synthesize one with low multilinear rank + small noise.
function load_or_synth_kv4d(path = "kv_cache_4d.npz")
    if HAS_NPZ && isfile(path)
        K = Float64.(NPZ.npzread(path)["K"])     # [layers, heads, seq, head_dim]
        println("Loaded real GPT-2 KV cache from $path   size = ", size(K))
        return K
    end
    Random.seed!(0)
    L, H, S, D = 12, 12, 256, 64
    r = (4, 4, 6, 5)                              # true multilinear rank
    U = ntuple(m -> Matrix(qr(randn((L, H, S, D)[m], r[m])).Q)[:, 1:r[m]], 4)
    K = lmlragen4(Tucker4(U, randn(r...)))
    K .+= 1e-3 * (norm(K) / sqrt(length(K))) .* randn(size(K))   # ~1e-3 relative noise
    println("No npz found; using synthetic low-rank KV tensor      size = ", size(K),
            "   (true multilinear rank ", r, ")")
    return K
end

function main()
    K = load_or_synth_kv4d()
    L, H, S, D = size(K)
    nrmK = norm(K)
    full_params = length(K)
    println("Cache entries: ", full_params, "  ‖K‖_F = ", round(nrmK; digits = 4))
    println()

    # 1) Tucker compression of the entire cache via sequentially-truncated MLSVD.
    tol = 1e-2
    U, G = LRDD.mlsvd4(K, tol)
    err_mlsvd = norm(lmlragen4(U, G) - K) / nrmK
    tucker_params = sum(length, U) + length(G)
    println("[1] MLSVD Tucker compression (tol = $tol)")
    println("    core size     : ", size(G))
    println("    parameters    : ", tucker_params, "  (", round(full_params / tucker_params; digits = 1), "x smaller)")
    println("    relative error: ", err_mlsvd)
    println()

    # 2) Matrix-free recovery with Cross-DEIM: only individual entries are queried.
    nq = Ref(0)
    gfun = (i, j, k, l) -> (nq[] += 1; K[i, j, k, l])
    seed = Tucker4(ntuple(m -> Matrix(qr(randn(size(K, m), 1)).Q)[:, 1:1], 4),
                   reshape([1.0], 1, 1, 1, 1))
    Fr, _ = crossDEIM(gfun, seed,
                      (; tol = 1e-3, rmax = 30, max_iter = 25, increment = 4, cache = true))
    err_cross = norm(lmlragen4(Fr) - K) / nrmK
    println("[2] Cross-DEIM matrix-free recovery")
    println("    core size     : ", size(Fr.G))
    println("    entries sampled: ", nq[], "  (", round(100 * nq[] / full_params; digits = 2), "% of the cache)")
    println("    relative error: ", err_cross)
    println()

    # 3) Q-DEIM token pruning: pick k token positions that best span the cache's token
    #    mode (mode 3), then reconstruct the full cache from only those token-slices.
    #    This is the 4D analogue of the leverage/Q-DEIM pruning in kv_leverage_demo.jl.
    k = min(32, S)
    U3 = svd(unfold(K, 3)).U[:, 1:k]                 # orthonormal token basis
    idx = qdeim(U3)                                  # k representative token positions
    M = U3 * inv(U3[idx, :])                         # DEIM interpolation operator (S×k)
    Krec = modemult(K[:, :, idx, :], M, 3)           # rebuild all S tokens from k slices
    err_tokens = norm(Krec - K) / nrmK
    println("[3] Q-DEIM token pruning ($S tokens -> $k)")
    println("    kept tokens    : ", sort(idx))
    println("    relative error : ", err_tokens)

    return (; K, mlsvd = (U, G), cross = Fr, idx)
end

main()

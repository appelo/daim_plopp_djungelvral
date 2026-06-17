using LRDD
using LinearAlgebra, Random
using Test

# Brute-force dense Tucker contraction T[i,j,k,l] = Σ G[p,q,r,s] U1[i,p] U2[j,q] U3[k,r] U4[l,s].
function dense4(U, G)
    n = ntuple(m -> size(U[m], 1), 4)
    r = size(G)
    T = zeros(promote_type(eltype(G), eltype(U[1])), n)
    for i in 1:n[1], j in 1:n[2], k in 1:n[3], l in 1:n[4]
        s = 0.0
        for p in 1:r[1], q in 1:r[2], rr in 1:r[3], ss in 1:r[4]
            s += G[p, q, rr, ss] * U[1][i, p] * U[2][j, q] * U[3][k, rr] * U[4][l, ss]
        end
        T[i, j, k, l] = s
    end
    return T
end

# Random Tucker4 with orthonormal factors and the given outer/core sizes.
function rand_tucker4(n::NTuple{4,Int}, r::NTuple{4,Int})
    U = ntuple(m -> Matrix(qr(randn(n[m], r[m])).Q)[:, 1:r[m]], 4)
    G = randn(r...)
    return Tucker4(U, G)
end

@testset "Tucker4 (4D LRDD)" begin
    Random.seed!(0)

    @testset "unfold/fold roundtrip" begin
        G = randn(3, 4, 5, 6)
        for n in 1:4
            @test fold(unfold(G, n), n, size(G)) ≈ G
            @test size(unfold(G, n), 1) == size(G, n)
        end
    end

    @testset "lmlragen4 vs tucker_eval vs dense" begin
        F = rand_tucker4((4, 5, 3, 6), (2, 3, 2, 4))
        T = lmlragen4(F)
        @test size(T) == (4, 5, 3, 6)
        @test T ≈ dense4(F.U, F.G)
        # spot-check single-entry evaluation against the full contraction
        for _ in 1:20
            i, j, k, l = rand(1:4), rand(1:5), rand(1:3), rand(1:6)
            @test tucker_eval(F, i, j, k, l) ≈ T[i, j, k, l]
        end
    end

    @testset "modemult matches dense mode-n product" begin
        G = randn(3, 4, 5, 2)
        M = randn(7, size(G, 3))            # act on mode 3
        out = modemult(G, M, 3)
        @test size(out) == (3, 4, 7, 2)
        @test fold(M * unfold(G, 3), 3, (3, 4, 7, 2)) ≈ out
    end

    @testset "mlsvd4 reconstruction" begin
        # exactly low-rank tensor: MLSVD must reproduce it
        F = rand_tucker4((6, 5, 4, 7), (2, 3, 2, 3))
        T = lmlragen4(F)
        U, core = LRDD.mlsvd4(T, 1e-12)
        @test LRDD.lmlragen4(U, core) ≈ T
        # fixed core size recovers the underlying multilinear rank
        U2, core2 = LRDD.mlsvd4(T, [2, 3, 2, 3])
        @test LRDD.lmlragen4(U2, core2) ≈ T
        # truncating a full-rank tensor stays within the requested tolerance
        Gf = randn(5, 5, 5, 5)
        Uf, cf = LRDD.mlsvd4(Gf, 1e-2)
        @test norm(LRDD.lmlragen4(Uf, cf) - Gf) / norm(Gf) < 1e-2
    end

    @testset "residual4 equals dense Frobenius difference" begin
        F1 = rand_tucker4((5, 4, 6, 3), (2, 2, 3, 2))
        F2 = rand_tucker4((5, 4, 6, 3), (3, 2, 2, 3))
        r = LRDD.residual4(F1.U, F1.G, F2.U, F2.G)
        @test r ≈ norm(lmlragen4(F1) - lmlragen4(F2))
    end

    @testset "tucker_sum equals dense sum" begin
        n = (5, 4, 6, 3)
        Fs = [rand_tucker4(n, (2, 2, 2, 2)) for _ in 1:3]
        S = tucker_sum(Fs, 1e-12)
        dense = sum(lmlragen4(F) for F in Fs)
        @test lmlragen4(S) ≈ dense
    end

    @testset "crossDEIM recovers a low-rank tensor" begin
        F = rand_tucker4((10, 9, 8, 7), (2, 3, 2, 2))
        T = lmlragen4(F)
        gfun = (i, j, k, l) -> T[i, j, k, l]
        seed = Tucker4(ntuple(m -> F.U[m][:, 1:1], 4), reshape([1.0], 1, 1, 1, 1))
        Fr, _ = crossDEIM(gfun, seed,
                          (; tol = 1e-10, rmax = 20, max_iter = 20, increment = 4, cache = true))
        @test norm(lmlragen4(Fr) - T) / norm(T) < 1e-8
    end

    @testset "tucker_cross_sum equals dense sum" begin
        # n large enough that the default rmax = minimum(n)-4 exceeds the true sum
        # rank (three rank-2 tensors → multilinear rank ≤ 6 per mode).
        n = (16, 15, 14, 13)
        Fs = [rand_tucker4(n, (2, 2, 2, 2)) for _ in 1:3]
        S = tucker_cross_sum(Fs; tol = 1e-10)
        dense = sum(lmlragen4(F) for F in Fs)
        @test norm(lmlragen4(S) - dense) / norm(dense) < 1e-7
    end
end

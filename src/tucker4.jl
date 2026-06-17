# 4D Tucker tensor: four factor matrices U[1..4] and a 4D core G.
# The represented tensor is  G ×₁ U[1] ×₂ U[2] ×₃ U[3] ×₄ U[4].
# Dedicated order-4 twin of `Tucker3` (src/tucker.jl); the 3D paths are untouched.
mutable struct Tucker4{T <: Number}
    U::NTuple{4, Matrix{T}}
    G::Array{T, 4}
end

function Tucker4(U::NTuple{4, AbstractMatrix}, G::AbstractArray{<:Any, 4})
    T = promote_type(eltype(U[1]), eltype(U[2]), eltype(U[3]), eltype(U[4]), eltype(G))
    Tucker4{T}(ntuple(l -> Matrix{T}(U[l]), 4), Array{T, 4}(G))
end

Tucker4(U1, U2, U3, U4, G) = Tucker4((U1, U2, U3, U4), G)

# Mode-n unfolding (matricization). Columns iterate the remaining modes in
# ascending order with the lower mode fastest, matching tensorlab's tens2mat and
# the 3D `unfold`. A single permutation handles all four modes.
function unfold(G::AbstractArray{T, 4}, n::Int) where T
    others = Tuple(setdiff(1:4, n))
    return reshape(permutedims(G, (n, others...)), size(G, n), :)
end

# Inverse of `unfold`: rebuild a 4D array of size `sz` from its mode-n unfolding.
function fold(M::AbstractMatrix, n::Int, sz::NTuple{4, Int})
    others = Tuple(setdiff(1:4, n))
    p = (n, others...)                                   # permutation used by `unfold`
    shaped = reshape(M, ntuple(i -> sz[p[i]], 4))
    invp = ntuple(i -> findfirst(==(i), p), 4)           # inverse permutation
    return permutedims(shaped, invp)
end

# Mode-n product T ×ₙ M : replaces every mode-n fiber f by M*f.
function modemult(T4::AbstractArray{<:Any, 4}, M::AbstractMatrix, n::Int)
    Mn = M * unfold(T4, n)
    sz = ntuple(l -> l == n ? size(M, 1) : size(T4, l), 4)
    return fold(Mn, n, sz)
end

# Full Tucker contraction G ×₁ U[1] ×₂ U[2] ×₃ U[3] ×₄ U[4] (tensorlab lmlragen, d=4).
function lmlragen4(U::NTuple{4, AbstractMatrix}, G::AbstractArray{<:Any, 4})
    return modemult(modemult(modemult(modemult(G, U[1], 1), U[2], 2), U[3], 3), U[4], 4)
end

lmlragen4(F::Tucker4) = lmlragen4(F.U, F.G)

# Single-entry evaluation of a 4D Tucker tensor:
# T[i,j,k,l] = Σ_{p,q,r,s} G[p,q,r,s]·U[1][i,p]·U[2][j,q]·U[3][k,r]·U[4][l,s].
function tucker_eval(U::NTuple{4, AbstractMatrix}, G::AbstractArray{<:Any, 4}, i, j, k, l)
    r1, r2, r3, r4 = size(G)
    s = zero(promote_type(eltype(G), eltype(U[1]), eltype(U[2]), eltype(U[3]), eltype(U[4])))
    @inbounds for sidx in 1:r4
        c4 = U[4][l, sidx]
        for r in 1:r3
            c34 = c4 * U[3][k, r]
            for q in 1:r2
                c234 = c34 * U[2][j, q]
                for p in 1:r1
                    s += G[p, q, r, sidx] * U[1][i, p] * c234
                end
            end
        end
    end
    return s
end

tucker_eval(F::Tucker4, i, j, k, l) = tucker_eval(F.U, F.G, i, j, k, l)

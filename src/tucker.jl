# 3D Tucker tensor: three factor matrices U[1],U[2],U[3] and a 3D core G.
# The represented tensor is  G ×₁ U[1] ×₂ U[2] ×₃ U[3].
mutable struct Tucker3{T <: Number}
    U::NTuple{3, Matrix{T}}
    G::Array{T, 3}
end

function Tucker3(U::NTuple{3, AbstractMatrix}, G::AbstractArray{<:Any, 3})
    T = promote_type(eltype(U[1]), eltype(U[2]), eltype(U[3]), eltype(G))
    Tucker3{T}(ntuple(l -> Matrix{T}(U[l]), 3), Array{T, 3}(G))
end

Tucker3(U1, U2, U3, G) = Tucker3((U1, U2, U3), G)

# Mode-n unfolding (matricization). Columns iterate the remaining modes in
# ascending order with the lower mode fastest, matching tensorlab's tens2mat.
function unfold(G::AbstractArray{T, 3}, n::Int) where T
    if n == 1
        return reshape(G, size(G, 1), :)
    elseif n == 2
        return reshape(permutedims(G, (2, 1, 3)), size(G, 2), :)
    else
        return reshape(permutedims(G, (3, 1, 2)), size(G, 3), :)
    end
end

# Inverse of `unfold`: rebuild a 3D array of size `sz` from its mode-n unfolding.
function fold(M::AbstractMatrix, n::Int, sz::NTuple{3, Int})
    if n == 1
        return reshape(M, sz)
    elseif n == 2
        return permutedims(reshape(M, (sz[2], sz[1], sz[3])), (2, 1, 3))
    else
        return permutedims(reshape(M, (sz[3], sz[1], sz[2])), (2, 3, 1))
    end
end

# Mode-n product T ×ₙ M : replaces every mode-n fiber f by M*f.
function modemult(T3::AbstractArray{<:Any, 3}, M::AbstractMatrix, n::Int)
    Mn = M * unfold(T3, n)
    sz = ntuple(l -> l == n ? size(M, 1) : size(T3, l), 3)
    return fold(Mn, n, sz)
end

# Full Tucker contraction G ×₁ U[1] ×₂ U[2] ×₃ U[3] (tensorlab lmlragen, d=3).
function lmlragen3(U::NTuple{3, AbstractMatrix}, G::AbstractArray{<:Any, 3})
    return modemult(modemult(modemult(G, U[1], 1), U[2], 2), U[3], 3)
end

lmlragen3(F::Tucker3) = lmlragen3(F.U, F.G)

# Single-entry evaluation of a Tucker tensor (analogue of tensorlab's ful at a
# point): T[i,j,k] = Σ_{p,q,r} G[p,q,r]·U[1][i,p]·U[2][j,q]·U[3][k,r].
function tucker_eval(U::NTuple{3, AbstractMatrix}, G::AbstractArray{<:Any, 3}, i, j, k)
    r1, r2, r3 = size(G)
    s = zero(promote_type(eltype(G), eltype(U[1]), eltype(U[2]), eltype(U[3])))
    @inbounds for r in 1:r3
        c3 = U[3][k, r]
        for q in 1:r2
            c23 = c3 * U[2][j, q]
            for p in 1:r1
                s += G[p, q, r] * U[1][i, p] * c23
            end
        end
    end
    return s
end

tucker_eval(F::Tucker3, i, j, k) = tucker_eval(F.U, F.G, i, j, k)

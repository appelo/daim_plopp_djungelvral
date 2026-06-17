module LRDD

using LinearAlgebra, Random
mutable struct LRSVD{T <: Number}
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
end

function LRSVD(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix)
    T = promote_type(eltype(U), eltype(S), eltype(V))
    LRSVD{T}(Matrix{T}(U), Vector{T}(S), Matrix{T}(V))
end

export greet
export qdeim, crossDEIM, scross, compute_residual, truncsum, trunclr, LRSVD
export Tucker3, fold, unfold, modemult, lmlragen3, tucker_eval, tucker_cross_sum, tucker_sum
export Tucker4, lmlragen4

include("scross.jl")
include("crossDEIM.jl")        # includes qdeim
include("matrix_helpers.jl")   # truncsum, trunclr, compute_residual
include("tucker.jl")
include("tucker_helpers.jl")
include("tucker_crossDEIM.jl")
include("tucker4.jl")              # Tucker4 + 4D unfold/fold/modemult/lmlragen4/tucker_eval
include("tucker4_helpers.jl")      # residual4, mlsvd4, tucker_sum(Tucker4)
include("tucker4_crossDEIM.jl")    # crossDEIM(4D), tucker_cross_sum(Tucker4)

"""
    greet()

Print a friendly greeting. Placeholder for the package's public API.
"""
greet() = println("Hello from LRDD.jl — Low rank Domain Decomposition")

end # module LRDD

# --- Q-DEIM index selection ------------------------------------------------

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

# --- DEIM-guided cross approximation ---------------------------------------

function crossDEIM(gfun, U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix, opts = nothing)
    if !isnothing(opts)
        tol_LR = opts.tol
        tol_RES = opts.tol
        tol_E = opts.tol
        r_max = opts.r_max
        r_in = opts.r_in
        max_iter = opts.max_iter
    else
        tol_LR = 1e-10
        tol_RES = 1e-10
        tol_E = 1e-10
        r_max = min(size(U, 1), size(V, 1))
        r_in = r_max
        max_iter = 20
    end

    
    n1 = size(U, 1)
    n2 = size(V, 1)
    I_all = Array{Any, 1}(1:n1)
    J_all = Array{Any, 1}(1:n2)
    T = eltype(U)
    C_res = Vector{LRSVD{T}}(undef, 2)
    C_res[1] = LRSVD(U, S, V)
    I0 = []
    J0 = []
    F = LRSVD(U, S, V)
    n_iters = 0
    for iter = 1:max_iter
        n_iters = iter
        # find index selection using QDEIM
        J = qdeim(F.V)
        I = qdeim(F.U)
        len_I = length(I)
        len_J = length(J)
        I = unique(cat(I, I0, dims=1))
        J = unique(cat(J, J0, dims=1))

        # oversample by at least one
        if length(I0) == length(I) || (iter == 1)
            I_add = copy(I_all)
            deleteat!(I_add, sort(I))
            I = cat(I, I_add[randperm(length(I_add))[1:min(1, length(I_add))]], dims=1)
        end

        if length(J0) == length(J) || iter == 1
            J_add = copy(J_all)
            deleteat!(J_add, sort(J))
            J = cat(J, J_add[randperm(length(J_add))[1:min(1, length(J_add))]], dims=1)
        end

        if length(I) > r_in
            I = I[1:r_in]
        end

        if length(J) > r_in
            J = J[1:r_in]
        end

        F, RC, RR = scross(gfun, I, J, I_all, J_all)
        I0 = copy(I)
        J0 = copy(J)
        # prune redundant information
        if !isempty(I0)
            I_dep0 = findall(abs.(diag(RR)) .< 1e-14 * maximum(abs.(diag(RR))))
            deleteat!(I0, I_dep0)
        end
        if !isempty(J0)
            J_dep0 = findall(abs.(diag(RC)) .< 1e-14 * maximum(abs.(diag(RR))))
            deleteat!(J0, J_dep0)
        end
        C_res[2] = LRSVD(F.U, -F.S, F.V)
        residual = compute_residual(C_res)
        C_res[1] = LRSVD(F.U, F.S, F.V)
        eta1 = 1 / norm(F.U[I, :], 2)
        eta2 = 1 / norm(F.V[J, :], 2)
        if residual < tol_RES && min(eta1 * (1 + eta2), eta2 * (1 + eta1) * F.S[end]) < tol_LR
            break
        end
    end
    
    rank_F = length(F.S)
    sd = F.S
    energy = cumsum(sd[end:-1:1].^2)
    r_st = length(energy) - length(findall(x -> x < tol_E^2, energy))
    r_st = max(min(r_st, r_max), 1)
    U = F.U[:, 1:r_st]
    S = F.S[1:r_st]
    V = F.V[:, 1:r_st]

    return U, S, V, (n_iters, rank_F, r_st)
end

function crossDEIM(gfun, F0::LRSVD, opts = nothing)
    U, S, V, info = crossDEIM(gfun, F0.U, F0.S, F0.V, opts)
    return LRSVD(U, S, V), info
end



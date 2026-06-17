function scross(gfun, I, J, I_all, J_all)
    tol_cond = 1e10
    tol_pinv = 1e-5
    sample_val = sum(gfun(I_all[1], J[1]))
    T = typeof(sample_val)

    C = zeros(T, length(I_all), length(J))
    for j in 1:length(J)
        for i in 1:length(I_all)
            C[i,j] = gfun(I_all[i], J[j])[1]
        end
    end

    R = zeros(T, length(I), length(J_all))
    for j in 1:length(J_all)
        for i in 1:length(I)
            R[i,j] = gfun(I[i], J_all[j])[1]
        end
    end
    
    if size(C, 2) <= size(R, 1)
        QRC = qr(C, ColumnNorm())
        RC = zeros(T, size(C,2), size(C,2))
        for i = 1:size(C,2)
            RC[i,i] = QRC.R[QRC.p[i],QRC.p[i]]
        end

        QRR = qr(Matrix(R'), ColumnNorm())
        RR = zeros(T, size(R,1), size(R,1))
        for i = 1:size(R,1)
            RR[i,i] = QRR.R[QRR.p[i],QRR.p[i]]
        end
        Q = Matrix(QRC.Q)
        if cond(Q[I, :]) > tol_cond
           U_R = pinv(Q[I, :], tol_pinv) * R
        else
           U_R = Q[I, :]\R
        end
        
        F1 = svd(U_R)
        U = Q * Matrix(F1.U)
        F = LRSVD(U, F1.S, Matrix(F1.V))
    else
        QRC = qr(C, ColumnNorm())
        RC = zeros(T, size(C,2), size(C,2))
        for i = 1:size(C,2)
            RC[i,i] = QRC.R[QRC.p[i],QRC.p[i]]
        end
        QRR = qr(Matrix(R'), ColumnNorm())
        RR = zeros(T, size(R,1), size(R,1))
        for i = 1:size(R,1)
            RR[i,i] = QRR.R[QRR.p[i],QRR.p[i]]
        end
        Z = Matrix(QRR.Q)
        if cond(Z[J, :]) > tol_cond
            C_U = pinv(Z[J, :], tol_pinv) * Matrix(C')
        else
            C_U = Z[J, :]\Matrix(C')
        end
        F1 = svd(Matrix(C_U'))
        V = Z * Matrix(F1.V)
        F = LRSVD(F1.U, F1.S, V)
    end
    return F, RC, RR
end

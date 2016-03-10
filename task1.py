import numpy as np

#LU factorization
#Ax = b
#LUx = b
#Lz = b
#Ux = z
def lu(A,b):
    #define matrix L, U, x, z
    L = [[0 for x in range (0,n)] for x in range (0,n)]
    U = [[0 for x in range (0,n)] for x in range (0,n)]
    x = [0 for x in range (0,n)]
    z = [0 for x in range (0,n)]

    for k in range(0, n):
        #set matrix L as with diagonal = 1
        L[k][k] = 1
        #compute upper triangular matrix, U
        for j in range (0, k+1):
            s1 = sum(U[s][k] * L[j][s] for s in range (0, j))
            U[j][k] = A[j][k] - s1
        
        #compute lower triangular matrix, L
        for i in range (k, n):
            s2 = sum(U[s][k] * L[i][s] for s in range(0, k))
            L[i][k] = (A[i][k] - s2)/U[k][k]
    
    #compute matrix z where z = (inverse of L) * b
    for i in range (0, n):
        u1 = sum(L[i][j] * z[j] for j in range (0, i))
        z[i] = b[i] - u1            
    
    #compute matrix x where x = (inverse of U) * z
    for i in range (n-1, -1, -1):
        u2 = sum(U[i][j] * x[j] for j in range (i, n))
        x[i] = (z[i] - u2)/U[i][i]
    
    sol = x
    
    return list(sol)
    
 #SOR method
 #A = D - L - U
 #x(k+1) = (inverse of Q) * (Q - A) * x(k) + (inverse of Q) * b
 #OR can be written as
 #x(k+1) = K * x(k) + s1
def sor(A, b):
    #Define the following matrices
    D = [[0 for x in range (0,n)] for x in range (0,n)]
    L = [[0 for x in range (0,n)] for x in range (0,n)]
    U = [[0 for x in range (0,n)] for x in range (0,n)]
    invD = [[0 for x in range (0,n)] for x in range (0,n)]
    s1 = [0 for x in range (0,n)]
    T = [[0 for x in range (0,n)] for x in range (0,n)]
    Q = [[0 for x in range (0,n)] for x in range (0,n)]
    K = [[0 for x in range (0,n)] for x in range (0,n)]
    x = [0 for x in range (0,n)]
    
    #compute diagonal matrix D
    for i in range (0, n):
        D[i][i] = A[i][i]
    
    #compute lower triangular matrix L    
    for i in range (1, n):
        for j in range (0, i):
            L[i][j] = -A[i][j]
    
    #compute upper triangular U        
    for i in range (0, n):
        for j in range(i+1, n):
            U[i][j] = -A[i][j]
    
    #compute inverse matrix of D, named invD
    for i in range (0, n):
        invD[i][i] = 1/D[i][i]
    
    #compute matrix T, where T = invD * (L + U)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                T[i][j] += invD[i][k]*(L[k][j] + U[k][j])
                
    #compute the spectral radius of matrix T, named pT, where pT = max(all eigenvalues)
    pT = max(abs(np.linalg.eigvals(T)))
    
    #compute optimual omega for SOR method
    omega = 2*(1 - np.sqrt(1 - ((pT)**2)))/((pT)**2)
    
    iteration = 0
    maxN = 10
    
    #compute matrix Q, where Q = 1/omega * (D - omega * L)
    for i in range (0, n):
        for j in range (0, n):
            Q[i][j] = (1/omega)*(D[i][j] - omega*L[i][j])
     
    #compute inverse of Q, named invQ        
    invQ = np.linalg.inv(Q)
    
    #compute matrix K, where K = invQ * (Q - A)
    for i in range (0, n):
        for j in range (0,n):
            for k in range (0,n):
                K[i][j] += invQ[i][k]*(Q[k][j] - A[k][j])
    
    #compute matrix s2, where s1 = invQ * b
    for i in range (0,n):
        for j in range (0,n):
            s1[i] += invQ[i][j]*b[j]
    
    #calculate each iteration of x           
    while (iteration < maxN):
        #define a matrix s2
        s2 = [0 for x in range (0,n)]
        #compute matrix s2, where s2 = K * x
        for i in range (0, n):
            for j in range (0, n):
                s2[i] += K[i][j]*x[j]
        #compute matrix x
        for k in range (0, n):
            x[k] = s2[k] + s1[k]    
        print("iteration %1d: x(%1d)" %(iteration,iteration+1), x)
        iteration += 1
    
    sol = x
    
    return list(sol)

#choose a suitable method to solve Ax = b
def solve(A, b):
    for i in range (0, n-1):
        #check whether the matrix A is a tridiagonal matrix
        if (A[i][i+1] != A[i+1][i]):
            #if it is not
            print('Solve by lu(A,b)')
            return lu(A,b)
            break
        else:
            #if it is
            print('Solve by sor(A,b)')
            return sor(A,b)

if __name__ == "__main__":
    ## import checker
    ## checker.test(lu, sor, solve)
    
    A = [[2,1,6], [8,3,2], [1,5,1]]
    b = [9, 13, 7]
    n = len(A)
    sol = solve(A,b)
    print(sol)
    
    A = [[6566, -5202, -4040, -5224, 1420, 6229],
         [4104, 7449, -2518, -4588,-8841, 4040],
         [5266,-4008,6803, -4702, 1240, 5060],
         [-9306, 7213,5723, 7961, -1981,-8834],
         [-3782, 3840, 2464, -8389, 9781,-3334],
         [-6903, 5610, 4306, 5548, -1380, 3539.]]
    b = [ 17603,  -63286,   56563,  -26523.5, 103396.5, -27906]
    n = len(A)
    sol = solve(A,b)
    print(sol)
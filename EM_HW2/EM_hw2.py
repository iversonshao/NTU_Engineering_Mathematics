import numpy as np
import math

#Gram-Schmidt process
def gram_schmidt(V):
    # V is a matrix by (m, n) m->number of vectors, n->dimension of vector

    n = V.shape[0] # Number of vectors
    for g in range(n):
        for p in range(g): # Subtract the projection of V[g] onto V[p] from V[g]
            V[g, :] -= (V[p, :] @ V[g, :]) * V[p, :] 
        V[g, :] = V[g, :] / np.linalg.norm(V[g, :])  # Normalize the orthogonalized vector V[g]
    return V

#Gram-Schmidt process with weights
def w_gram_schmidt(V, w):
    n = V.shape[0]
    for g in range(n):
        for p in range(g):
            V[g, :] -= (V[p, :] @ V[g, :]) * V[p, :] * w[g]
        V[g, :] =(V[g, :] / np.linalg.norm(V[g, :])) * (1 / np.sqrt(w[g]))
    return V

if __name__ == '__main__':
    w = []
    for n in range(-6, 7):
        w.append(1-abs(n)/7)
    b_k = []
    for k in range(5):
        temp = []
        for n in range(-6, 7):
            temp.append(math.pow(n, k))
        b_k.append(temp)

    a = np.array(b_k)
    print("a,GS:")
    print(gram_schmidt(a))
    print("b,wGS:")
    print(w_gram_schmidt(a, w))


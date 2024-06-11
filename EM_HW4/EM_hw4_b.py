import numpy as np

# Calculate the probability mass function Pxy(m, n)
def Pxy(m, n):
    m, n = np.meshgrid(m, n)
    return (100 - np.abs(m - n)) / 666700

# Calculate the marginal PMF Px(n) and Py(m)
def calc_marginals(Pxy):
    Px = np.sum(Pxy, axis=0)  # sum over columns to get Px
    Py = np.sum(Pxy, axis=1)  # sum over rows to get Py
    return Px, Py

# Calculate the conditional probability P(Y|X)
def conditional_prob(Pxy, Px):
    Pylx = np.zeros(Px.shape)
    for n in range(len(Px)):
        Pylx[n] = Pxy[n,n] / Px[n]
    return Pylx

# Calculate the conditional cross entropy H(X, Y)
def conditional_cross_entropy(Px, Pylx):
    Hxy = 0
    for n in range(len(Px)):
        if Px[n] != 0:
            Hxy += Px[n] * np.log(Pylx[n])
    return -Hxy

if __name__ == '__main__':
    m = np.arange(1, 101, 1)
    n = np.arange(1, 101, 1)
    Pxy = Pxy(m, n)
    Px, Py = calc_marginals(Pxy)
    Pylx = conditional_prob(Pxy, Px)
    cross_entropy = conditional_cross_entropy(Px, Pylx)
    
    print(f"Cross entropy H(X, Y): {cross_entropy:.6f}")

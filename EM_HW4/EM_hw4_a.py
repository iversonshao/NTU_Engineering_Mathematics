import numpy as np


# Calculate the probability mass function Pxy(m, n)
def Pxy(m, n):
    m, n = np.meshgrid(m, n)
    return (100 - np.abs(m - n)) / 666700

# Calculate the PMF Px(n) and Py(m)
def calc_marginals(Pxy):
    Px = np.sum(Pxy, axis=0)
    Py = np.sum(Pxy, axis=1)
    return Px, Py

# Calculate the marginal probability mass functions Px(m) and Py(n)
def cross_entropy(Px, Py):
    Hxy = 0
    for n in range(len(Px)):
        if Px[n] != 0 and Py[n] > 0:
            Hxy += Px[n] * np.log(Py[n])
    return -Hxy

#check the entropy of a random variable X
# def entropy_X(Px):
#     Hx = 0
#     for p in Px:
#         if p > 0:
#             Hx += p * np.log(p)
#     return -Hx

# def cross_YX(Px, Py):
#     Hyx = 0
#     for n in range(100):
#         if Py[n] > 0:
#             Hyx += np.log(Px[n]) * Py[n]
#     return -Hyx

if __name__ == '__main__':
    m = np.arange(1, 101, 1)
    n = np.arange(1, 101, 1)
    Pxy = Pxy(m, n)
    Px, Py = calc_marginals(Pxy)
    cross_entropy = cross_entropy(Px, Py)
    print(f"Cross entropy H(X, Y): {cross_entropy:.16f}")
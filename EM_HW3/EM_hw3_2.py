import numpy as np
import matplotlib.pyplot as plt


#g(x) = sin((pi*x**2)/9) 0<x<3 delta_x=0.1
#0-3 10*3+1=31
dx = 0.1
N = int((3 - 0) / dx)
df = (1 / N) / dx
fs = 1 / dx

# sampling
f = np.linspace(0, 3, N + 1)
g1 = np.sin((np.pi / 9) * (f**2))

# Compute the Fourier transform of g(x)

Gd = np.fft.fft(g1)
m = np.linspace(0, 31, 31)

# Mapping
G1 = []
f_coord = []
for m in range(N):
    if m <= N / 2:
        f_coord.append(m * df)
        G1.append(Gd[m] * dx)
    else:
        f_coord.append(m * df - fs)
        G1.append(Gd[m] * dx)

# Modulation
#fs/N 0.3226
f_coord = np.array(f_coord)

#x1 = -4
G = []
for i in range(len(f_coord)):
    fp = 1j * 2 * np.pi * f_coord[i]
    t = np.exp(fp)
    G.append(t * G1[i])

pts = []
for i in range(len(G)):
    temp = []
    temp = tuple([f_coord[i], G[i]])

    pts.append(temp)
G = np.array(G)

plt.plot(f_coord, G.real, "b*", label="Real Part")
plt.plot(f_coord, G.imag, "r*", label="Imaginary Part")
plt.xlabel("f")
plt.ylabel("G(f)")
plt.legend()
plt.savefig("5_2.png")
plt.show()

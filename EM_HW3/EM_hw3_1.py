import numpy as np
import matplotlib.pyplot as plt


#g(x) = exp(-|x|**0.5) - exp(-2) -4<x<4 delta_x=0.05

#0-8 20*8+1=161
dx = 0.05
N = int((4 - (-4)) / dx)
df = (1 / N) / dx
fs = 1 / dx

# sampling
f = []
for i in range(N):
    f.append(i * dx)
f = np.array(f)
g1 = np.exp(-np.abs(f - 4)**0.5) - np.exp(-2)

# Compute the Fourier transform of g(x)

Gd = np.fft.fft(g1)
m = np.linspace(0, 161, 161)

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
#fs/N 0.1242
f_coord = np.array(f_coord)

#x1 = -4
G = []
for i in range(len(f_coord)):
    fp = 1j * 8 * np.pi * f_coord[i]
    t = np.exp(fp)
    G.append(t * G1[i])

pts = []
for i in range(len(G)):
    temp = []
    temp = tuple([f_coord[i], G[i]])

    pts.append(temp)
G = np.array(G)

plt.plot(f_coord, G.real, "b*", label="Real Part")
plt.plot(f_coord, G.imag, "r*--", label="Imaginary Part")
plt.xlabel("f")
plt.ylabel("G(f)")
plt.legend()
plt.savefig("5_1.png")
plt.show()

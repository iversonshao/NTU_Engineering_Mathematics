import numpy as np
import matplotlib.pyplot as plt
from formula import Euler, EulerModify, RK4
from scipy.integrate import odeint

# y'(x) = 5cos(-1/5|xy|)
def func(x, y):
    return 5 * np.cos(-1/5 * np.abs(x*y))

#Define parameters
h = 0.1 #step size
x = np.linspace(0, 10, 100) #x range
x0 = 0 #initial condition

#Euler method
euler = Euler(func, x0, x, h)
euler.solve()
y_euler = euler._y

#Euler modified method
euler_modify = EulerModify(func, x0, x, h)
euler_modify.solve()
y_euler_modify = euler_modify._y

#Runge-Kutta method
rk4 = RK4(func, x0, x, h)
rk4.solve()
y_rk4 = rk4._y

#use scipy to Prove the result
y_odeint = odeint(func, x0, x)

#euler method
plt.plot(x, y_odeint, label='odeint', color='k')
plt.plot(x, y_euler, label='Euler', color='m')
plt.title("Euler's method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('euler.png')
plt.show()


#euler modified method
plt.plot(x, y_odeint, label='odeint', color='k')
plt.plot(x, y_euler_modify, label='Euler modified', color='c')
plt.title("Modified Euler's method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('euler_mod.png')
plt.show()


#RK4 method
plt.plot(x, y_odeint, label='odeint', color='k')
plt.plot(x, y_rk4, label='RK4', color='DeepSkyBlue')
plt.title("RK4 method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('RK4.png')
plt.show()


plt.plot(x, y_odeint, label='odeint', color='k')
plt.plot(x, y_euler, label='Euler', color='m')
plt.plot(x, y_euler_modify, label='Euler modified', color='c')
plt.plot(x, y_rk4, label='RK4', color='DeepSkyBlue')
plt.title("Total comparison of methods")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('overlapping.png')
plt.show()

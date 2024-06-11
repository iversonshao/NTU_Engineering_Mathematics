import numpy as np

class NumericalSolver:
    def __init__(self, func, x0, x, h):
        self._func = func
        self._x0 = x0
        self._x = x
        self._h = h
        self._y = np.zeros(len(x))
        self._y[0] = x0

    def solve(self):
        for i in range(0, len(self._x) - 1):
            raise NotImplementedError("Subclasses must implement the solve method.")

class Euler(NumericalSolver):
    def solve(self):
        for i in range(0, len(self._x) - 1):
            self._y[i + 1] = self._y[i] + self._func(self._x[i], self._y[i]) * self._h

class EulerModify(NumericalSolver):
    def solve(self):
        for i in range(0, len(self._x) - 1):
            temp = self._y[i] + self._func(self._x[i], self._y[i]) * self._h
            self._y[i + 1] = self._y[i] + (self._func(self._x[i], self._y[i]) + self._func(self._x[i + 1], temp)) * self._h / 2

class RK4(NumericalSolver):
    def solve(self):
        for i in range(0, len(self._x) - 1):
            k1 = self._func(self._x[i], self._y[i])
            k2 = self._func(self._x[i] + self._h / 2, self._y[i] + self._h / 2 * k1)
            k3 = self._func(self._x[i] + self._h / 2, self._y[i] + self._h / 2 * k2)
            k4 = self._func(self._x[i] + self._h, self._y[i] + self._h * k3)
            self._y[i + 1] = self._y[i] + self._h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# file: gradient_descent.py

import numpy as np

methods = {'fixed': 0, 'linear': 1, 'quadratic': 2, 'BB': 3}

# IT MIGHT NOT WORK
class DualGDOptimizer:
    def __init__(self,
                 objfun: callable, x0: float or np.ndarray,
                 step_size: float, method: str = 'fixed', buffer: int = 5):
        # algorithm
        self.objfun = objfun
        self.step_size = step_size
        self.method = methods[method]
        self.buffer = buffer
        self.iter = 0
        self.i = 0
        # variables
        size = len(x0)
        self.x = np.zeros((buffer, size))
        self.g = np.zeros((buffer, size))
        self.f = np.zeros((buffer, size))
        # ini
        self.x[-1] = x0
        self.g[-1] = np.zeros(size)
        self.f[-1] = self.objfun(x0)

    def step(self, gradient):
        self.x[self.i] = self.x[self.i - 1] - self.step_size * gradient
        self.g[self.i] = gradient
        self.f[self.i] = self.objfun(self.x[self.i])
        self.update()
        return self.x[self.i], self.g[self.i], self.f[self.i]

    def update(self):
        # update the iter so the step size can be computed
        self.iter += 1
        if self.method == 0:
            pass
        elif self.method == 1:
            self.step_size = self.step_size / self.iter
        elif self.method == 2:
            self.step_size = self.step_size / self.iter ** 2
        elif self.method == 3:
            # if self.iter == 1:
            a = (self.x[self.i] - self.x[self.i - 1]).T @ (self.g[self.i] - self.g[self.i - 1]) / ((self.g[self.i] - self.g[self.i - 1]).T @ (self.g[self.i] - self.g[self.i - 1]))
            self.step_size = max(1e-6, min(a, 1e3))
        # update index after BB
        self.i = np.mod(self.iter, self.buffer)


def f(x):
    return x ** 2 + x

def g(x):
    return 2 * x + 1

if __name__ == '__main__':
    max_iter = int(1e6)
    buf = max_iter
    x0 = np.array([2])
    gd = DualGDOptimizer(f, x0, step_size=1e-1, buffer=buf, method='fixed')


    g_tmp = g(x0)
    for ij in range(max_iter):
        gd.step(g_tmp)
        g_tmp = g(gd.x[gd.i])
        if np.abs(g_tmp) <= 1e-5:
            break

    import matplotlib.pyplot as plt
    plt.scatter(gd.x, gd.f, label='opt')
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, f(x))
    plt.grid()
    plt.show()

    print




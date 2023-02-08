# eval_sinc.py: estimate x for sinc(x) / x
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scenario.common import printplot
from os import path


def sinc(x):
    with warnings.catch_warnings(): # avoid division by zero warning
        warnings.simplefilter("ignore")
        y = np.sin(x) / x
    y[np.isnan(y)] = 1
    return y

# param
DATADIR = path.join(path.dirname(__file__), 'data')
precision = 1e-4
tol = 1e-3
tau = np.arange(0.1, 1.01, 0.01)
x = np.arange(0, 5, precision)
s = sinc(x)
S = np.abs(s) ** 2

# cycle
x_star = np.zeros(tau.shape)
for i, t in enumerate(tau[::-1]):
    x_star[i] = x[np.argmin(np.abs(S - t))]

print(x_star[np.newaxis])
np.save(path.join(DATADIR, 'sinc_argument'), np.vstack((x_star, tau[::-1])))


plt.plot(x, s, label=r'$\mathrm{sinc}(x)$')
plt.plot(x, S, label=r'$|\mathrm{sinc}(x)|^2$')
plt.yticks(tau)
plt.xticks(x_star[1:], [f'{v:.2f}' for v in x_star[1:]])
plt.xlim(x[0], x[np.argmin(np.abs(s))])
plt.ylim(0, 1)
printplot(labels=[r'$x$', r'$f(x)$'])

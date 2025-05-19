import numpy as np
from bayes_optimization2 import BayesianOptimizer
def f(x):
    return 0.5 * np.sin(0.5 * x) * np.sin(0.2 * x) + 0.5


# mock criticality function for numerical parameters
def f2(params):
    result = 0
    for i, p in enumerate(params):
        if i % 3 == 0:
            result += np.sin(p)
        elif i % 3 == 1:
            result += np.cos(p)
        else:
            result += np.exp(-abs(p))
    # normalize using sigmoid to constrain to (0, 1)
    return 1 / (1 + np.exp(-result))

import numpy as np


def f(x):
    return 0.5 * np.sin(0.5 * x) * np.sin(0.2 * x) + 0.5


# mock sigmoidal criticality function for (numerical) parameters
# can be used for ex7_1D_float.osc, ex7_1D_int.osc, ex7_1D_enum.osc, ex9_2d_enum.osc
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


# 5-degree polynom mocking criticality function on a single float x ∈ [40,80].
# can be used for ex7_1D_float.osc
def f3(params):
    if not isinstance(params, (list, tuple)) or len(params) != 1:
        raise ValueError(f"f3 expects a single-element list or tuple, got {params!r}")

    x = params[0]
    if x < 40.0 or x > 80.0:
        raise ValueError(f"f3 input must be in [40,80], got {x}")

    def _g(t):
        return - (t - 40) * (t - 50) * (t - 60) * (t - 70) * (t - 80)

    grid = np.linspace(40.0, 80.0, 10001)
    vals = _g(grid)
    g_min, g_max = vals.min(), vals.max()

    raw = _g(x)
    y = (raw - g_min) / (g_max - g_min)

    return float(y)

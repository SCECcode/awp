import material
import numpy as np

def test_rocksoil1d():
    x = np.linspace(-10, 0, 40)
    y = np.linspace(-10, 0, 40)
    X, Y = np.meshgrid(x, y)

    mat = material.material(Y.T)
    rho = mat[0]
    assert np.max(rho) == 2.8
    assert np.min(rho) == 1.0

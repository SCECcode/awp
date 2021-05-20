"""usage: topography.py refine=int plot=int save=int
Generate Gaussian hill topography file.

Args:
        refine int      Level of grid refinement

Optional args:
    plot int        Show plot
    save int        Save figure to file
    peaks int       Use peaks elevation map data instead of Gaussian hill
    """
import numpy as np
import sys
import pyawp

plot = 0

filename = sys.argv[1]
nx = int(sys.argv[2])
ny = int(sys.argv[3])
h = float(sys.argv[4])


ngsl = 8
T = pyawp.Topography(nx, ny, h, ngsl)

a = 0e3
b = 1
xc = 64
yc = 64
gaussian = lambda x, y: a * np.exp(-b**-2*(x - xc) ** 2 -b**-2*(y - yc) ** 2)
z = T.map(gaussian)
Z = T.reshape(z)


T.write(Z, filename)
print("Wrote topography file: %s" % filename)

if plot:
    import matplotlib.pyplot as plt
    T.imshow(Z)
    plt.show()


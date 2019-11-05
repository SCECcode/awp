# Write curvilinear grid

This tool reads in a topography binary file and produces a binary file that
contains the grid coordinates `(x_i, y_j, z_k)` for each grid point in the
curvilinear grid. 
Another task is reading in an external regular grid property file containing 
(vp, vs, rho), then assigning the properties at the curvilinear grids 
with the values at the nearest regular grid point.

## Usage

```
write_grid <input> <output> <property> <mesh> <nx> <ny> <nz> <mz> <h> <px> <py> 
```
---------------------------------------------------------------
|  Argument   |  Description                                  |
|-------------|-----------------------------------------------|
| input       |   Topography binary file                      |
| output      |   Binary file to write containing curvilinear grid coordinates           |
| property    |   Property binary file to read (regular grid) |
| mesh        |   Mesh binary file to write containing properties at curvilinear grids   | 
| nx `int`    |   Number of grid points in the x-direction    |
| ny `int`    |   Number of grid points in the y-direction    |
| nz `int`    |   Number of grid points in the z-direction    |
| mz `int`    |   Number of grid points in the z-direction of the regular property grid  |
| h `float`   |   Grid spacing                                |
| px `int`    |   Number of MPI partitions in the x-direction |
| py `int`    |   Number of MPI partitions in the y-direction |
| rpt `int`   |   Whether repeating the top layer twice       |

See
[awp-benchmarks](https://github.com/SCECcode/awp-benchmarks/tree/master/tests/topography/write_grid)
for an example that demonstrates how to use this tool and also show its expected
output.

## Binary file format


The binary file is of size `3 x nx x ny x nz x sizeof(float)` and contains the
data entries

```
x[0,0,0] y[0,0,0] z[0,0,0] 
x[1,0,0] y[1,0,0] z[1,0,0] 
x[2,0,0] y[2,0,0] z[2,0,0] 
.
.
.
x[nx-1,ny-1,nz-1] y[nx-1,ny-1,nz-1] z[nx-1,ny-1,nz-1] 
```

### output
`dimensions = (nz, ny, nx, 3)`

The first entry is the x-coordinate at the front left corner (0,0,0) on the top
boundary of the grid. The next value is the y-coordinate at the same position.
The fastest direction is the x-direction, followed by the y-direction, and
z-direction. The final three data entries contain the coordinate at the back
right corner (nx-1, ny-1, nz-1) on the bottom boundary of the grid.

When `rpt == 1`, the top layer will be repeated twice.

### property
`dimensions = (mz, ny, nx, 3)`

This contains contains the properties that will be assigned at the curvilinear grids. Since the 
original regular grids are stretched in Z direction, the height of the domain will increase from
`nz * h` to some value larger. It is therefore necessary to prepare a larger domain that is able 
to hold the stretched domain, which determines the value of `mz`.

### mesh
`dimensions = (nz, ny, nx, 3)`

This file is quite similar with the `output` file, except that the three entries in `output` are 
(x, y, z) coordinates, while `mesh` contains three entries (vp, vs, rho).

When `rpt == 1`, the top layer will be repeated twice.

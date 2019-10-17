# Write curvilinear grid

This tool reads in a topography binary file and produces a binary file that
contains the grid coordinates `(x_i, y_j, z_k)` for each grid point in the
curvilinear grid.

## Usage

```
write_grid <input> <output> <nx> <ny> <nz> <h> <px> <py> 
```
---------------------------------------------------------------
|  Argument   |  Description                                  |
|-------------|-----------------------------------------------|
| input       |   Topography binary file                      |
| output      |   Binary file to write                        |
| nx `int`    |   Number of grid points in the x-direction    |
| ny `int`    |   Number of grid points in the y-direction    |
| nz `int`    |   Number of grid points in the z-direction    |
| h `float`   |   Grid spacing                                |
| px `int`    |   Number of MPI partitions in the x-direction |
| py `int`    |   Number of MPI partitions in the y-direction |

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

The first entry is the x-coordinate at the front left corner (0,0,0) on the top
boundary of the grid. The next value is the y-coordinate at the same position.
The fastest direction is the x-direction, followed by the y-direction, and
z-direction. The final three data entries contain the coordinate at the back
left corner (nx-1, ny-1, nz-1) on the bottom boundary of the grid.

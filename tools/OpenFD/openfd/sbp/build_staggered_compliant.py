"""
build_staggered_compliant.py <input_file> <output_file> <shift>

This script reads in SBP staggered grid operators and adjusts them so that the
number of grid points is the same on both grids.

<input_file>            read JSON file that contains operator data.
<output_file>           write JSON file with modified operator data. 
<shift>                 modify operator for regular grid (shift = 0),
                        or shifted grid (shift = 1).
<sign>                  Apply a sign shift to the interior stencil that gets
                        packed into the boundary region. Use `-1` for
                        differentation and `1` for interpolation.

"""
import sys
import json
import tools

if len(sys.argv)  == 5:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    shift = int(sys.argv[3])
    sign = int(sys.argv[4])

else:
    print("""Not enough input arguments.
build_staggered_compliant.py <input_file> <output_file> <shift> <sign>""")
    exit(1)

assert input_file != output_file

def load(filename):
    print("Reading %s" % filename)
    res = open(filename).read()
    D = json.loads(res)
    return D

def write(filename, D):
    print("Writing %s" % filename)
    with open(filename, 'w') as out:
        json.dump(D, out, sort_keys = True, indent = 4)

if shift:
    Dh = load(input_file)
    tools.add_interior(Dh, 'right', sign)
    write(output_file, Dh)
else:
    D = load(input_file)
    tools.add_zeros(D, 'right')
    write(output_file, D)

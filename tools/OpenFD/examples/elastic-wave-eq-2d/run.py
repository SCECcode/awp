"""
Fourth order staggered grid elastic wave equation solver

Usage: run.py source [--moment=<mxx,mzz,mxz> --force=<fx,fz>] [--bc=<bc>] 
                [--acoustic] [--src=<x,z> --recv=<x,z> --FS2-x=<opt>]
                [ --het=<model>]
                [ --refine=<num>] [--output=<out> --path=<dir>]
       run.py mu [--acoustic]
       run.py -h | --help
    
Options:
  -h --help
  --moment=<mxx,mzz,mxz> Components of moment rate tensor (10^12 N/s).
  --force=<fx,fz>        Components of point force (10^9 N).
  --bc=<bc>              Boundary condition implementation (SAT|FS2|SYM).
                         [default: SAT].
  --acoustic             Run in acoustic mode.
  --src=<x,z>            Source location (x, z) (km) [default: 0,0].
  --recv=<x1,z1,x2,z2>   Receiver locations (x, z) (km) [default: 0,0].
  --FS2-x=<opt>          Set to 'src' or 'recv' to shift source/receiver
                         location by one grid point. This option should be used
                         if 'bc=FS2' when either placing a point force acting in
                         the horizontal direction half a grid point below the
                         free surface, or when having a receiver half a grid
                         point below the free surface. This receiver will be
                         monitoring the particle velocity in the horizontal
                         direction. This setting does not apply to 'bc=SAT'.
  --het=<model>          Select heteregenous material model. 
  --output=<out>         Numpy outputfile.
  --path=<dir>           Path to output directory [default: ''].
  --refine=<num>         Number of grid refinements to perform.

"""
import solver
import sys
from docopt import docopt

def refine(h, nx, ny, nt):
    """
    Half the grid spacing and double the number of time steps.

    Arguments:
        h : Grid spacing.
        nx, ny : Number of grid points in the regular grid in each direction.
        nt : Number of time steps to take.

    """
    return h/2, 2*(nx - 1) + 1, 2*(ny - 1) + 1, 2*nt

def get_problem(lamb, garvin, depth):
    """
    Get problem name
    """
    if lamb and depth == 0.0:
        problem = 'Lamb-Surface'
    elif lamb and depth != 0.0:
        problem = 'Lamb-Interior'
    else:
        problem = 'Garvin'

    return problem

def run_source(r0=0, 
               rn=3, 
               moment=None,
               force=None,
               bc='SAT',
               path='', 
               output='lamb_surf', 
               src=None, 
               pos=None,
               acoustic=False,
               fs2x=None,
               hetfcn=0,
               use_vtk=0):
    """
    Run a convergence test using a source that has been setup to solve either
    Lamb's or Garvin's problems.

    Arguments:
        r0 : First grid number.
        rn : Last grid number.
        bc : Boundary condition implementation.
        moment : Components of moment rate tensor.
        force : Components of point force.
        path : Path to output file directory.
        output : Name of .npz output file.
        src : Source coordinate (xs, zs)
        pos : Receiver coordinates (x1, z1, x2, z2, ...)
        hetfcn : Function pointer to hetereogeneous material properties (see
            material.py).
        use_vtk : Write vtk files.
        direction : Direction in which to apply point force for Lamb's problem.
        acoustic : Run in acoustic mode. Defaults to `False`.

    """

    h=0.2
    nx=301
    ny=301
    nt=1001
    xs=src[0]
    zs=src[1]
    A=1.0
    fp=0.96
    t0=2.0
    cs=1.2
    cp=3.0
    rho=2.8
    cfl=0.25

    if acoustic:
        cs = 0.0 
        fp = 0.96
        t0 = 3.0
        rho = 2.8
        cp = 3.0
        nx = 401
        ny = 401
        h = 0.25 

    seismfile = lambda ref, ext : '%s%s_%d%s'%(path, output, ref, ext)
    vtkfile = lambda ref : '%s%s_%d'%(path, output, ref)

    if hetfcn:
        mat = None 
    else:
        mat = solver.init_material(cs=cs, cp=cp, rho=rho, het=het)

    for ref in range(0, r0):
        h, nx, ny, nt = refine(h, nx, ny, nt)

    for ref in range(r0, rn):
        src = solver.init_source(xs, zs, h, bc, moment=moment,
                                 force=force,
                                 fp=fp, t0=t0)
        recv = solver.init_receivers(pos)
        solver.evaluate(nx=nx, ny=ny, 
                        nt=nt, cfl=cfl, 
                        st=round(nt/10),
                        h=h,
                        mat=mat,
                        hetfcn=hetfcn,
                        use_sponge=0,
                        vst=80*2**ref,
                        use_vtk=use_vtk,
                        run_solver=1,
                        recv=recv,
                        bc=bc,
                        src=src,
                        fs2x=fs2x,
                        vtkfile=vtkfile(ref),
                        seismfile=seismfile(ref, ''))
        h, nx, ny, nt = refine(h, nx, ny, nt)

if __name__ == '__main__':

    def parse_comps(arg):
        if arg:
            return [float(argi) for argi in arg.strip(' ').split(',')]
        return []

    arguments = docopt(__doc__)
    source = arguments['source']
    moment = parse_comps(arguments['--moment'])
    force = parse_comps(arguments['--force'])
    src = parse_comps(arguments['--src'])
    recv = parse_comps(arguments['--recv'])
    output = arguments['--output']
    acoustic = arguments['--acoustic']
    bc = arguments['--bc']
    ref = int(arguments['--refine'])
    path = arguments['--path']
    fs2x = arguments['--FS2-x']
    src_type = None
    het = arguments['--het']

    if moment and force:
        raise ValueError("Only one source type supported. Source must be"\
                         "either 'moment' or 'force' ")
    if moment and len(moment) != 3:
        raise ValueError("Expected three moment tensor components. Got:",
                          moment)
    if force and len(force) != 2:
        raise ValueError("Expected two point force components. Got: ", force)
    if len(src) != 2:
        raise ValueError("Expected two source coordinates (x, z). Got: ", src)
    
    if bc not in ['FS2', 'SAT', 'SYM', 'BNDOPT']:
        raise ValueError('Unknown boundary condition implementation %s'\
                         ' selected' % bc)
    if not output:
        output ='%s'%(bc.lower())

    if not het:
        het = None
    elif het == 'rock-soil-1d':
        import material
        het = material.rocksoil1d
    else:
        raise NotImplementedError("Unknown material model selected.")


    if source:
        run_source(r0=0, rn=ref, path=path, output=output, 
                   bc=bc, moment=moment, force=force,
                   src=src, pos=recv,
                   hetfcn=het,
                   acoustic=acoustic, fs2x=fs2x)

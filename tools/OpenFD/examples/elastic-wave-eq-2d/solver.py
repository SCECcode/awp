import numpy
import openfd
import pycuda.autoinit
from openfd.cuda import make_kernel, write_kernels, load_kernels, init, \
                        copy_htod, copy_dtoh, append
from openfd import gridfunctions, Bounds, Struct, Memory
from helper import operator, output_vtk, init_block2d, Writer, vtk_2d, \
                   norm_coef, init_block_sponge
from openfd import grids, CArray, sponge
from sympy import  symbols, exp
from helper import shifts
import time

prec = numpy.float32
openfd.prec = prec
debug=0
openfd.debug=0

def init_debug(cpu, gpu, shape, use_debug=0):
    """
    This function initializes a the stress variables using the grid coordinates.
    It is used to check that the boundary conditions are applied correctly.
    Set `run_solver = 0` and `use_sponge=0` and only run using the sat boundary
    term.
    """
    if not use_debug:
        return
    print("Initializing debugging.")
    X0 = Struct()
    Y0 = Struct()
    X0.v1,  Y0.v1  = grids.xy(shape, (0, 1), extend=1)
    X0.v2,  Y0.v2  = grids.xy(shape, (1, 0), extend=1)
    X0.s11, Y0.s11 = grids.xy(shape, (1, 1), extend=1)
    X0.s22, Y0.s22 = grids.xy(shape, (1, 1), extend=1)
    X0.s12, Y0.s12 = grids.xy(shape, (0, 0), extend=1)

    cpu.s11 = prec(1  + 0*X0.s11)
    cpu.s22 = prec(1  + 0*X0.s22)
    cpu.s12 = prec(1  + 0*X0.s12)
    copy_htod(gpu, cpu, 's12 s11 s22')

def init_grids(shape):
    """
    Initalize grids for each variable

    Arguments:
        shape : tuple (nx, ny), giving the size of a regular grid

    Returns:
        X : x-coordinates (size nx x ny) 
        Y : y-coordinates (size nx x ny) 
        The size nx x ny changes depending on the gridfunction stored.
    """
    # Grids
    X = Struct()
    Y = Struct()
    G = shifts()
    shape_ = shape
    shape = (shape_[1], shape_[0])
    X.v1,  Y.v1  = grids.xy(shape, G.v1)
    X.v2,  Y.v2  = grids.xy(shape, G.v2)
    X.s11, Y.s11 = grids.xy(shape, G.snn)
    X.s22, Y.s22 = grids.xy(shape, G.snn)
    X.s12, Y.s12 = grids.xy(shape, G.s12)
    return X, Y

def init_seismogram(xr, zr, xs, h):
    """
    Save the velocity field at selected indices. The indices are determined by
    reading receiver coordinates (x, y) where x=0 is the source location and y=0
    is the free surface. Positive x values are placed to the right of the
    source.

    Arguments:
        xs : Source x-coordinate in kilometers
           h : Grid spacing in kilometers
        
    """
    x = [numpy.int32(numpy.round(r/h)) for r in xr] 
    z = [numpy.int32(numpy.round(r/h)) for r in zr] 
    x = numpy.array(x) 
    z = numpy.array(z) 
    return x, z

def init_source(xs, zs, h, bc, moment=None, force=None,
                fp=0.96, t0=2.4):
    """
    Initialize source

    Arguments:
        xs, zs : Source coordinate (x, z) in kilometers
        h : Grid spacing in meters
        type : Type of source: 'moment tensor` or 'point force'
        moment : Moment rate tensor components: [mxx, mzz, mxz]
        force : Point force components: [fx, fz]

    """

    if moment and force:
        raise ValueError('Both moment rate tensor and point force cannot be'\
                         'specified.')

    import warnings

    src = Struct()
    if zs <= 0:
        src.on_boundary = True
    else:
        src.on_boundary = False
    src.xs = prec(xs)
    src.zs = prec(zs)
    src.xi = numpy.int32(round(xs/h))
    src.zi = numpy.int32(round(zs/h))
    src.src_type = 'none'

    if moment:
        src.src_type = 'moment'
        src.Mxx = prec(moment[0])
        src.Mzz = prec(moment[1])
        src.Mxz = prec(moment[2])
    else:
        src.Mxx = prec(0.0)
        src.Mzz = prec(0.0)
        src.Mxz = prec(0.0)

    if force:
        src.src_type = 'force'
        src.Fx = prec(force[0])
        src.Fz = prec(force[1])
    else:
        src.Fx = prec(0.0)
        src.Fz = prec(0.0)

    src.fp = prec(fp)
    src.t0 = prec(t0)
    return src

def init_receivers(pos):
    """
    Initialize receivers

    Arguments:
        pos : List of receiver coordinates (x1, z1, x2, z2, ... ).

    Returns:
        Struct containing:
        xr : List of receiver x-coordinates.
        zr : List of receiver x-coordinates.

    """
    recv = Struct()
    recv.x = []
    recv.z = []
    lpos = int(len(pos)/2)
    for i in range(lpos):
        recv.x.append(pos[2*i])
        recv.z.append(pos[2*i+1])

    return recv

def set_grid_positions(src, recv, h):
    """
    Convert the (x, z) coordinates to indices (xi, zi).

    Arguments:
        src: Source object.
        recv : Receiver.

    """

    src.xi = numpy.int32(numpy.round(src.xs/h))
    src.zi = numpy.int32(numpy.round(src.zs/h))

    recv.xi = numpy.array([numpy.int32(numpy.round(r/h)) for r in recv.x])
    recv.zi = numpy.array([numpy.int32(numpy.round(r/h)) for r in recv.z]) 

def add_offset(src, recv, bc, fs2x):
    """
    Add offset to adjust grid indices so that zi=0 falls on the free surface for
    the shear stress when using FS2.

                            Fz Fx       
                            |  |
    o--x--o--x--o--x--o--x--o--x--o ---> z
                            |        
                            FS
    o : vz
    x : vx
    FS : Free surface (z=0, zi=4)

    When FS2 is used, the free surface is placed at the fourth grid point.  If
    the force is in the horizontal direction, then it is applied at the fifth
    grid point, half a grid point below the free surface. This option needs to
    be toggled by passing `fs2x='src'`. If the receiver for the horizontal
    component of velocity should be positioned half a grid point below the free
    surface, then `fs2x='recv'`.

    Arguments:
        src: Source object.
        recv : Receiver object.
        bc : Free surface boundary implementation.
        fs2x : Shift of grid point for FS2 free surface boundary implementation
            when either source (force acting in x-direction) or receiver
            (x-component of particle velocity) is half a grid point below the
            free surface.

    """

    if bc != 'FS2' and bc != 'SYM':
        return

    snum = 4
    rnum = 4

    if fs2x == 'src':
        snum = 5
        src.on_boundary = False
    if fs2x == 'recv':
        rnum = 5

    src.zi = src.zi + numpy.int32(snum)
    recv.zi = numpy.array([zi + numpy.int32(rnum) for zi in recv.zi])

def ppw(mat, src, h):
    """
    Compute minimum number of grid points per wavelength for a Ricker wavelet
    source. Use threshold 5% of peak amplitude of frequency content.

    fmax = 2.5*fp

    """
    fmax = 2.5*src.fp
    return mat.cs/(fmax*h)

def init_material(cs=1.2, cp=3.0, rho=1.8):
    """
    Initialize material properties.

    Arguments:
        cs : Shear wave speed
        cp : Dilational wave speed
        rho : Density

    """
    # Material properties
    mat = Struct()
    mat.cs = prec(cs)
    mat.cp = prec(cp)
    mat.rho = prec(rho)
    mat.mu = prec(mat.cs**2*mat.rho)
    mat.lam = prec(mat.rho*mat.cp**2 - 2*mat.mu)
    mat.rhoi = prec(1.0/mat.rho)
    mat.zp = mat.cp*mat.rho
    mat.zs = mat.cs*mat.rho
    mat.lam_mu = prec(mat.lam/(mat.lam + 2*mat.mu))
    return mat

def init_het_material(hetfcn, X, Y, bc):
    """
    Initialize heterogeneous material properties using function

    """
    import numpy as np
    mat = Struct()
    rho, cs, cp  = hetfcn(X.s11, Y.s11)
    mu = prec(cs**2*rho)
    mat.lam = prec(rho*cp**2 - 2*mu)
    mat.cp = np.max(cp)
    mat.cs = np.min(cs)
    mat.mu_11 = mu 
    mat.rho = np.mean(rho)

    rho, cs, cp  = hetfcn(X.v1, Y.v1)
    mat.rhoi_v1 = prec(1.0/rho)

    rho, cs, cp  = hetfcn(X.v2, Y.v2)
    mat.rhoi_v2 = prec(1.0/rho)

    rho, cs, cp  = hetfcn(X.s12, Y.s12)
    mat.mu_12 = prec(cs**2*rho)

    return mat



def init_info(seismfile, src, mat, recv, cpu_ts, nx, ny, h, cfl, dt,
              nt):
    """
    Show simulation settings
    """
    print('--------------------------------------------------------------')
    print('Output: %s' % seismfile)
    print('nx = %d ny = %d h = %2.2f m dt = %2.5f s nt = %d '
           %(nx, ny, 1e3*h, dt, nt))
    print('Source : x = %2.3f, z = %2.3f km [%d][%d]'\
          ' fp = %2.3f Hz t0 = %2.3f s'
           %(src.xs, src.zs, src.xi, src.zi, src.fp, src.t0))
    print('On boundary : ', src.on_boundary)
    if src.src_type == 'moment':
        print('Mxx = %2.3f Mzz = %2.3f Mxz = %2.3f'%( 
               src.Mxx, src.Mzz, src.Mxz))
    if src.src_type == 'force':
        print('Fx = %2.3f Fz = %2.3f'% (src.Fx, src.Fz))
    print('Receivers : x = ', recv.x, 'km' , 'z = ', recv.z, 'km',
            cpu_ts.xi, cpu_ts.zi)
    print('minimum grid points per wavelength: %2.2f ' % ppw(mat, src, h))
    print('--------------------------------------------------------------')

def evaluate(nx=256, ny=256, cfl=0.25, nt=100, st=10, h=10, vst=1000,
             recv=None,         
             mat=None,
             src=None,
             use_seism=1,
             use_vtk=0,
             ds=4,
             use_sponge=1,
             run_solver=1,
             hetfcn=0,
             bc='SAT',
             fs2x=None,
             seismfile='out/seism',
             vtkfile='vtk/'):
    """
    Run simulation

    Solve the P-SV elastic wave equation in 2D using a specific source.

    bc: Free surface boundary condition implementation:
        `SAT` : Weakly enforced BC using the SAT penalty technique
        `FS2` : Strongly enforced BC using the stress imaging technique
        `BNDOPT` : Weakly enforced BC using boundary-optimized scheme

    Arguments:
        nx, ny : Number of grid points in the regular grid.
        cfl : CFL number defined as the scaling dt = cfl*h/cp.
        st : How often do display information.
        h : Grid spacing (m).
        vst : How often to write vtk output (if enabled).
        ds : Stride factor for vtk output.
        recv : A tuple of tuples that define receiver locations relative to the
            source. Example: ((x0, x1), (y0, y1)).
        mat : Material parameters (see init_material, or init_het_material).
        src : Source parameters (see init_source).
        use_seism : Save seismograms at each time step.
        use_vtk : Save vtk output at time step `vst`.
        use_sponge : Use Cerjan's sponge layer. Disabled by default.
        run_solver: Solve the elastic wave equation. Can be useful to disable
            for debugging.
        hetfcn : Function pointer to function for generating heterogeneous
            material properties.
        bc : Select boundary condition implementation ('SAT', 'FS2').
        seismfile : File to write seismograms to.
        vtkfile : Path to write vtk files to.

    """

    nx = numpy.int32(nx)
    ny = numpy.int32(ny)
    shape = (nx, ny)
    X, Y = init_grids(shape)

    if bc == 'BNDOPT': 
        velocity = load_kernels('kernels/bndopt', 'update_velocity_*')
        stress = load_kernels('kernels/bndopt', 'update_stress_*')
    else:
        velocity = load_kernels('kernels/elastic', 'update_velocity_*')
        stress = load_kernels('kernels/elastic', 'update_stress_*')
    source_moment, = load_kernels('kernels/source', 'src_moment_tensor')
    source_force_sat, = load_kernels('kernels/source', 'src_force_sat')
    source_force_bndopt, = load_kernels('kernels/source', 'src_force_bndopt_v2')
    source_force, = load_kernels('kernels/source', 'src_force')
    source_force_fs2, = load_kernels('kernels/source', 'src_force_fs2')
    sat = load_kernels('kernels/sat', 'bnd_*')
    #bndopt_v1, = load_kernels('kernels/bndopt_sat', 'v1_*')
    #bndopt_v2, = load_kernels('kernels/bndopt_sat', 'v2_*')
    sponge_vel = load_kernels('kernels/sponge', 'sponge_vel_*')
    sponge_stress = load_kernels('kernels/sponge', 'sponge_stress_*')
    timeseries, = load_kernels('timeseries', 'save_seism')
    fs2_vel, = load_kernels('kernels/fs2', 'fs2_velocity_*')
    sym_vel, = load_kernels('kernels/fs2', 'sym_velocity_*')
    fs2_stress, = load_kernels('kernels/fs2', 'fs2_stress_*')
    sgt, = load_kernels('timeseries', 'sgt')

    if hetfcn:
        het_velocity = load_kernels('kernels/elastic_heterogeneous', 'update_velocity_*')
        het_stress = load_kernels('kernels/elastic_heterogeneous', 'update_stress_*')
        het_sat = load_kernels('kernels/sat_heterogeneous', 'bnd_*')
        mat = init_het_material(hetfcn, X, Y, bc)


    if not mat:
        mat = init_material()

    if not src:
        src = init_source(5, 5, h, bc)
    
    if not recv:
        recv = init_receivers((0, 0))

    # Grid sizes, time step etc
    mem = (nx + 1, ny + 1)
    mx = numpy.int32(nx + 1)
    my = numpy.int32(ny + 1)
    h = prec(h)
    hi = prec(1.0/h)
    Vi = prec(hi**2)
    dt = prec(cfl*h/mat.cp)

    set_grid_positions(src, recv, h)
    add_offset(src, recv, bc, fs2x)

    cpu, gpu = init('v1 v2 s11 s22 s12', mem)
    if hetfcn:
        append(cpu, gpu, 'rhoi_v1 rhoi_v2 mu_11 mu_12 lam', mem)
        cpu.rhoi_v1 = mat.rhoi_v1
        cpu.rhoi_v2 = mat.rhoi_v2
        cpu.mu_11 = mat.mu_11
        cpu.mu_12 = mat.mu_12
        cpu.lam = mat.lam
        copy_htod(gpu, cpu, 'rhoi_v1 rhoi_v2 mu_11 mu_12 lam')

    init_debug(cpu, gpu, shape, debug)

    # Vtk output
    vtk = Struct()
    vtk.s22 = Writer('%s/s22'%vtkfile, vtk_2d)
    vtk.s12 = Writer('%s/s12'%vtkfile, vtk_2d)
    vtk.v1 = Writer('%s/v1'%vtkfile, vtk_2d)
    vtk.v2 = Writer('%s/v2'%vtkfile, vtk_2d)

    # Seismogram and SGT output
    nidx = numpy.int32(len(recv.x))
    cpu_ts, gpu_ts = init('v1 v2 e11 e22 e12 s11 s22 s12', (nidx*nt,))
    append(cpu_ts, gpu_ts, 'xi zi', (nidx,), precision=numpy.int32)
    cpu_ts.xi = recv.xi
    cpu_ts.zi =  recv.zi
    copy_htod(gpu_ts, cpu_ts, 'xi zi')
    cpu_ts.t = numpy.zeros((nt,))

    # Cuda grids
    cublock, cugrids = init_block2d(mem[0], mem[1], threads=16)
    block, spgrids = init_block_sponge(mem[0], mem[1], threads=32, width=64)
    bndgrid = [cugrids[1], cugrids[7], cugrids[3], cugrids[5]]
    block, fs2grids = init_block_sponge(mem[0], mem[1], threads=32, width=1)
    block, bndoptgrids = init_block_sponge(mem[0], mem[1], threads=32, width=32)
    numgrids=9

    def update_vel(i, fcn, block, grid):
            if not run_solver:
                return
            if (bc == 'FS2' or bc == 'SYM') and i != 4:
                return
            fcn(gpu.v1, gpu.v2, gpu.s11, gpu.s12, gpu.s22, dt, hi, mat.rhoi, 
                mx, my, block=block, grid=grid)

    def update_het_vel(i, fcn, block, grid):
            if not run_solver:
                return
            if (bc == 'FS2' or bc == 'SYM') and i != 4:
                return
            fcn(gpu.v1, gpu.v2, gpu.rhoi_v1, gpu.rhoi_v2, gpu.s11, gpu.s12,
                gpu.s22, dt, hi, mx, my, block=block, grid=grid)

    def update_het_stress(i, fcn, block, grid):
            if not run_solver:
                return
            if (bc == 'FS2' or bc == 'SYM') and i != 4:
                return
            fcn(gpu.s11, gpu.s12, gpu.s22, gpu.lam, gpu.mu_11, gpu.mu_12,
                gpu.v1, gpu.v2, dt, hi, mx, my, block=block, grid=grid)

    def update_stress(i, fcn, block, grid):
            if not run_solver:
                return
            if (bc == 'FS2' or bc == 'SYM') and i != 4:
                return
            fcn(gpu.s11, gpu.s12, gpu.s22, gpu.v1, gpu.v2, dt, hi, 
                mat.lam, mat.mu, mx, my,
                 block=block, grid=grid)

    def apply_sat(fcn, block, grid):
            if not run_solver:
                return
            fcn(gpu.v1, gpu.v2, gpu.s11, gpu.s12, gpu.s22, dt, hi, mat.rhoi, 
                nx, ny,
                block=block, grid=grid)

    def apply_het_sat(fcn, block, grid):
            if not run_solver:
                return
            fcn(gpu.v1, gpu.v2, gpu.rhoi_v1, gpu.rhoi_v2, gpu.s11, gpu.s12,
                gpu.s22, dt, hi,
                nx, ny,
                block=block, grid=grid)

    def apply_bndopt(i, fcn_v1, fcn_v2, block, grid):
            #if not run_solver:
            #    return
            if i != 3:
                return
            fcn_v1(gpu.v1, gpu.s11, gpu.s12, gpu.s22, dt, hi, mat.rhoi, 
                nx, ny,
                block=block, grid=grid)
            fcn_v2(gpu.v2, gpu.s11, gpu.s12, gpu.s22, dt, hi, mat.rhoi, 
                nx, ny,
                block=block, grid=grid)

    def apply_fs2_stress(i, fcn, block, grid):
            if not run_solver:
                return
            if i != 3:
                return
            fcn(gpu.s12, gpu.s22, mx, my, block=block, grid=grid)
    
    def apply_fs2_velocity(i, fcn, block, grid):
            if not run_solver:
                return
            if i != 3:
                return
            fcn(gpu.v1, gpu.v2, mat.lam_mu, mx, my, block=block, grid=grid)

    def apply_sym_velocity(i, fcn, block, grid):
            if not run_solver:
                return
            if i != 3:
                return
            fcn(gpu.v1, gpu.v2, mx, my, block=block, grid=grid)

    def apply_sponge_vel(i, fcn, block, grid):
            if not use_sponge:
                return
            # Skip free surface and interior
            if i == 3 or i == 4:
                return
            fcn(gpu.v1, gpu.v2, mx, my,
                block=block, grid=grid)

    def apply_sponge_stress(i, fcn, block, grid):
            if not use_sponge:
                return
            # Skip free surface and interior
            if i == 3 or i == 4:
                return
            fcn(gpu.s11, gpu.s12, gpu.s22, mx, my,
                block=block, grid=grid)

    def apply_source(bc):
        if not run_solver:
            return
        if src.src_type == 'moment':
            source_moment(gpu.s11, gpu.s12, gpu.s22, src.Mxx, src.Mxz, src.Mzz, 
                          Vi, dt, src.fp, t, src.t0,
                          mx, my, src.xi, src.zi, block=block, grid=(1, 1))
        elif src.src_type == 'force':
            if not src.on_boundary:
                source_force(gpu.v1, gpu.v2, src.Fx, src.Fz, Vi, dt, src.fp,
                             mat.rhoi, t, src.t0, mx, my, src.xi, src.zi,
                             block=block, grid=(1, 1))
            elif bc == 'SAT':
                source_force_sat(gpu.v1, gpu.v2, src.Fx, src.Fz, dt, src.fp, hi,
                                 mat.rhoi, t, src.t0, mx, my, src.xi,
                                 src.zi, block=block, grid=(1, 1))
            elif bc == 'BNDOPT':
                source_force_bndopt(gpu.v2, src.Fz, dt, src.fp, hi,
                                 mat.rhoi, t, src.t0, mx, my, src.xi,
                                 src.zi, block=block, grid=(1, 1))
            elif bc == 'FS2' or bc == 'SYM':
                source_force_fs2(gpu.v1, gpu.v2, src.Fx, src.Fz, Vi, dt, src.fp,
                                 mat.rhoi, t, src.t0, mx, my, src.xi, src.zi,
                                 block=block, grid=(1, 1))

    def save_vtk(use_vtk):
        if not use_vtk:
            return
        copy_dtoh(cpu, gpu, 'v1 v2')
        vtk.v1.write('v1', X.v1[::ds, ::ds], 
                           Y.v1[::ds, ::ds], 
                           [cpu.v1[::ds, ::ds]])
        vtk.v2.write('v2', X.v2[::ds, ::ds], 
                           Y.v2[::ds, ::ds], 
                           [cpu.v2[::ds, ::ds]])
    
    def save_seism(step, t, save_sgt=0):
        cpu_ts.t[step] = t
        timeseries(gpu_ts.v1, gpu_ts.v2, gpu_ts.s11, gpu_ts.s12, gpu_ts.s22,
                   gpu.v1, gpu.v2,
                   gpu.s11, gpu.s12, gpu.s22,
                   gpu_ts.xi, gpu_ts.zi, 
                   mx, my,
                   nidx,
                   numpy.int32(step), block=(256, 1, 1), grid=(1, 1))
        if hetfcn and save_sgt:
            raise not ImplementedError("""SGT for heterogeneous material
            properties is not implemented""")
        if save_sgt:
            sgt(gpu_ts.e11, gpu_ts.e12, gpu_ts.e22, gpu.s11, gpu.s12, gpu.s22,
            mat.lam, mat.mu, gpu_ts.xi, gpu_ts.zi, 
            mx, my,
            nidx,
            numpy.int32(step), block=(256, 1, 1), grid=(1, 1))

    def show_info(step):
            end = time.time()
            print('step: %06d \t'% step, 
                   't = %3.2f \t'% t, 
                  'time elapsed: %3.2f s'%(end-start))

    init_info(seismfile, src, mat, recv, cpu_ts, nx, ny, h, cfl, dt,
              nt)
                  
    start = time.time()
    for step in range(nt):
        t = prec(step*dt)
        save_seism(step, t)

        if (step % st == 0):
            show_info(step)
        if (step % vst == 0):
            save_vtk(use_vtk)

        if hetfcn:
            for i, vi, grid in zip(range(numgrids), het_velocity, cugrids):
                update_het_vel(i, vi, cublock, grid)
    
        else:
            for i, vi, grid in zip(range(numgrids), velocity, cugrids):
                update_vel(i, vi, cublock, grid)

        #FIXME: Sponge layer disabled
        #for i, si, grid in zip(range(numgrids), sponge_vel, spgrids):
        #    apply_sponge_vel(i, si, block, grid)

        apply_source(bc)

        if bc == 'FS2':
            for i, grid in zip(range(numgrids), cugrids):
                apply_fs2_velocity(i, fs2_vel, block, grid)
        elif bc == 'SYM':
            for i, grid in zip(range(numgrids), cugrids):
                apply_sym_velocity(i, sym_vel, block, grid)

        if hetfcn:
            for i, si, grid in zip(range(numgrids), het_stress, cugrids):
                update_het_stress(i, si, cublock, grid)
        else:
            for i, si, grid in zip(range(numgrids), stress, cugrids):
                update_stress(i, si, cublock, grid)

        if bc == 'SAT' and hetfcn:
            for i, si, grid in zip(range(numgrids), het_sat, bndgrid):
                apply_het_sat(si, block, grid)

        if bc == 'SAT' and not hetfcn:
            for i, si, grid in zip(range(numgrids), sat, bndgrid):
                apply_sat(si, block, grid)

        if bc == 'BNDOPT':
            for i, grid in zip(range(numgrids), 
                               bndoptgrids):
                apply_bndopt(i, bndopt_v1, bndopt_v2, block, grid)

        elif bc == 'FS2' or bc == 'SYM':
            for i, grid in zip(range(numgrids), cugrids):
                apply_fs2_stress(i, fs2_stress, block, grid)

        for i, si, grid in zip(range(numgrids), sponge_stress, spgrids):
            apply_sponge_stress(i, si, block, grid)

    save_vtk(use_vtk)
    copy_dtoh(cpu_ts, gpu_ts, 'v1 v2 e11 e12 e22 s11 s12 s22')
    assert(not numpy.any(numpy.isnan(cpu_ts.v1)))
    assert(not numpy.any(numpy.isnan(cpu_ts.v2)))
    numpy.savez(seismfile, 
                v1=cpu_ts.v1, v2=cpu_ts.v2, 
                s11=cpu_ts.s11, s12=cpu_ts.s12, s22=cpu_ts.s22,
                e11=cpu_ts.e11, e12=cpu_ts.e12, e22=cpu_ts.e22,
                t=cpu_ts.t, 
                xr=recv.x, zr=recv.z,
                Mxx=src.Mxx, Mzz=src.Mzz, Mxz=src.Mxz,
                Fx=src.Fx, Fz=src.Fz,
                cp=mat.cp, cs=mat.cs, rho=mat.rho, 
                xs=src.xs, zs=src.zs, t0=src.t0,
                fp=src.fp,
                ppw=ppw(mat, src, h),
                bc=bc)

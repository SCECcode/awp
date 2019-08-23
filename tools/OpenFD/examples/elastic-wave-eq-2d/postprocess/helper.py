import numpy
import matplotlib.pyplot as plt
from openfd import Struct
import sys

def load_data(filename):
    s = Struct()
    data = numpy.load(filename)
    s.xr = data['xr']
    s.zr = data['zr']
    n = len(s.xr)
    s.t = data['t']
    s.v1 = data['v1'].reshape(len(s.t), n)
    s.v2 = data['v2'].reshape(len(s.t), n)
    s.e11 = data['e11'].reshape(len(s.t), n)
    s.e12 = data['e12'].reshape(len(s.t), n)
    s.e22 = data['e22'].reshape(len(s.t), n)
    s.sgt = [s.e11, s.e22, s.e12]
    s.cp = data['cp']
    s.cs = data['cs']
    s.rho = data['rho']
    s.zs = data['zs']
    s.t0 = data['t0']
    s.fp = data['fp']
    s.Mxx = data['Mxx']
    s.Mzz = data['Mzz']
    s.Mxz = data['Mxz']
    s.M = [s.Mxx, s.Mzz, s.Mxz]
    s.Fx = data['Fx']
    s.Fz = data['Fz']
    return s

def representation_theorem(sgt, moment_tensor):
    """
    Compute the particle velocity field using representation theorem

    Arguments :
        sgt : Strain Green's tensor (e11, e22, e12) 
    tensor source

    """
    s = sgt
    m = moment_tensor
    # Since deviatoric component of the moment
    # tensor is only given once, we need to scale this term by a factor of two
    return s[0]*m[0] + s[1]*m[1] + 2*s[2]*m[2]

def load_source(src='out/source.exact', dec=10):
    """
    Load the source used as input for producing the analytic data for
    * EX2DDIR
    * EX2DVAEL

    Arguments:
        dec : Decimation rate. The sample rate of the source is 10 times higher
            than the analytic solution. 
    """
    data = numpy.loadtxt(src, unpack=True)
    t = data[0][::dec]
    f = data[1][::dec]
    return t, f

def load_lamb_exact(time_shift=0, src='out/source.exact', 
                    v1='out/lamb_surf_v1.exact', 
                    v2='out/lamb_surf_v2.exact',
                    nudge=0,
                    nrecv=3):
    """
    Load semi-analytic solution for Lamb's problem. 
    Must specify the correct number of records
    `nrecv`.
    
    This code adjusts the time vector to match the output from 
    EX2DDIR. 
    
    The amplitude is also adjusted by a factor of 0.5 to address a difference in
    how the Ricker wavelet is defined. The Ricker wavelet is
    defined as 0.5*(1 - 2*a2*tau)*exp(a2*tau), a2 = fp**pi**2, tau=t-t0 in
    EX2DDIR.

    """
    s = Struct()
    v1 = numpy.loadtxt(v1, unpack=True)
    v2 = numpy.loadtxt(v2, unpack=True)
    nstep = int(len(v1)/nrecv)
    # Adjust amplitude and change sign convention on vertical velocity
    s.v1 = 0.5*v1.reshape(nstep, nrecv)
    s.v2 = -0.5*v2.reshape(nstep, nrecv)
    src = load_source()
    s.t = src[0]
    # Fix time vector
    s.t = 3*s.t - time_shift  +nudge*s.t[1]
    return s

def load_garvin_exact(time_shift=0, src='out/source_garvin.exact',
                    v1='out/garvin_v1.exact', 
                    v2='out/garvin_v2.exact',
                    nrecv=3, rho=1.0):
    """
    Load semi-analytic solution for Lamb's problem. 
    Must specify the correct number of records
    `nrecv`.
    
    This code adjusts the amplitude to match the output from `EX2DVAEL`. Since
    `EX2DVAEL` assumes unit density, the amplitude of the analytic solution
    needs to be multiplied by density. There is also an additional
    multiplication by a mysterious factor of `1/14` that is needed to match the
    numerical code.

    """

    v1 = numpy.loadtxt(v1, unpack=True)
    v2 = numpy.loadtxt(v2, unpack=True)

    s = Struct()
    s.v1 = -0.5*v1[1:, :]/14/rho
    s.v2 =  0.5*v2[1:, :]/14/rho
    s.t, f = load_source(src)
    s.t = s.t - time_shift
    s.v1 = s.v1.T
    s.v2 = s.v2.T

    return s

def load_exact(v1, v2):
    s = Struct()
    v1 = numpy.loadtxt(v1, unpack=True)
    v2 = numpy.loadtxt(v2, unpack=True)
    s.t = v1[0]
    s.v1 = numpy.array([v1[1], v1[2], v1[3]]).T
    s.v2 = numpy.array([v2[1], v2[2], v2[3]]).T
    return s

def plot(num, exact=None, comp='v1', recv=0, offset=1.2):
    """
    Accuracy plot showing the 1D solution under subsequent grid refinement. Each
    refinement is offset by some amount computed from the maximum absolute
    amplitude of the numerical solution.

    Arguments:
        num : Numerical solution (struct)
        exact : Exact solution (struct)
        comp : Component to plot
        recv : Receiver index
        offset : Minimum scale factor 

    """
    vmax = 0.0
    colors = ['C0', 'C3', 'C2', 'C4']
    fig,ax = plt.subplots(1)
    for numi, ci in zip(num, colors):
        curr_max = numpy.max(numpy.abs(numi[comp][:, recv]))
        if curr_max > vmax:
            vmax = curr_max

    if exact:
        v = exact[comp][:, recv]
        nans = numpy.isnan(v)
        infs = numpy.isinf(v)
        v[nans] = 0.0
        v[infs] = 0.0
        vmax = numpy.max(numpy.abs(v))

    i = 0
    for numi, ci in zip(num, colors):
        if exact:
            ax.plot(exact.t, -i*vmax*offset + exact[comp][:, recv],'k')
        ax.plot(numi.t, -i*vmax*offset + numi[comp][:, recv], color=colors[0])
        i += 1

    plt.xlabel('Time (s)')
    if comp == 'v1':
        plt.ylabel('$v_x$ (m/s)')
    else:
        plt.ylabel('$v_z$ (m/s)')
    plt.xlim((0, 10))
    ax.set_yticklabels([])
    return ax

import numpy
import matplotlib.pyplot as plt
from openfd import Struct
import sys

def load_data(filename):
    s = Struct()
    data = numpy.load(filename)
    s.recv = data['recv']
    n = len(s.recv[0])
    s.t = data['t']
    s.v1 = data['v1'].reshape(len(s.t), n)
    s.v2 = data['v2'].reshape(len(s.t), n)
    s.cp = data['cp']
    s.cs = data['cs']
    s.rho = data['cs']
    s.ys = data['ys']
    s.t0 = data['t0']
    s.fp = data['fp']
    return s

def load_garvin_exact(params, v1='out/garvin_v1.exact', v2='out/garvin_v2.exact'):
    return load_exact(v1, v2)

def load_garvin_exact_kausel(params):
    import analytical as an
    cs = params.cs
    cp = params.cp
    rho = params.rho
    fp = params.fp
    t0 = params.t0
    ys = params.ys
    nt = len(params.t)
    tend = params.t[-1]
    nstation = len(params.recv[0])
    s = Struct()
    s.v1 = numpy.zeros((nt, nstation))
    s.v2 = numpy.zeros((nt, nstation))
    for i, x in enumerate(params.recv[0]):
        s.t, s.v1[:, i], s.v2[:, i] = an.garvin_exact(x, ys, rho, cs, 
                                                     cp, fp, t0, 10*tend, nt)
    return s

def load_lamb_exact_kausel(params):
    import analytical as an
    cs = params.cs
    cp = params.cp
    rho = params.rho
    fp = params.fp
    t0 = params.t0
    ys = params.ys
    nt = len(params.t)
    tend = params.t[-1]
    nstation = len(params.recv[0])
    s = Struct()
    s.v1 = []
    s.v2 = []
    s.t  = []

    nt = len(params.t)

    for i, x in enumerate(params.recv[0]):
        t, v1, v2 = an.lamb_exact(x, ys, rho, cs, 
                                  cp, fp, t0, 2*tend, 2*nt)
        print(v1[-1], v2[-1])
        # Flip components
        s.v1.append(-v1)
        s.v2.append(v2)
        s.t.append(t)
    return s

def load_lamb_exact(v1='out/lamb_v1.exact', v2='out/lamb_v2.exact', Tmax=30,
                    nrecv=3):
    """
    Load semi-analytic solution for Lamb's problem. Must specify the final time
    `Tmax` to construct the correct time vector and the number of records
    `nrecv`. The number of records  
    """
    s = Struct()
    v1 = numpy.loadtxt(v1, unpack=True)
    v2 = numpy.loadtxt(v2, unpack=True)
    nstep = int(len(v1)/nrecv)
    s.v1 = v1.reshape(nstep, nrecv)
    s.v2 = v2.reshape(nstep, nrecv)
    s.t = numpy.linspace(0, 20, nstep)
    return s

def load_exact(v1, v2):
    s = Struct()
    v1 = numpy.loadtxt(v1, unpack=True)
    v2 = numpy.loadtxt(v2, unpack=True)
    s.t = v1[0]
    s.v1 = numpy.array([v1[1], v1[2], v1[3]]).T
    s.v2 = numpy.array([v2[1], v2[2], v2[3]]).T
    return s

def save_data(label, d, exact=None, fmt='png', xlim=None):

    for station in range(d[0].v1.shape[1]):
        plt.clf()
        for di in d:
            plt.plot(di.t, di.v1[:,station])
        if exact:
            plt.plot(exact.t[station], exact.v1[station], 'k')
        if xlim:
            plt.xlim(xlim)
        plt.xlabel('t (s)')
        plt.ylabel('v_1 (m/s)')
        plt.savefig('figures/%s-v1-%d.%s'%(label, station, fmt), 
                    bbox_inches='tight', dpi=300)
        plt.clf()
        for di in d:
            plt.plot(di.t, di.v2[:,station])
        if exact:
            # The sign convention for the vertical displacement is reversed
            plt.plot(exact.t[station], exact.v2[station], 'k')
        if xlim:
            plt.xlim(xlim)
        plt.xlabel('t (s)')
        plt.ylabel('v_2 (m/s)')
        plt.savefig('figures/%s-v2-%d.%s'%(label, station, fmt), 
                    bbox_inches='tight', dpi=300)

import matplotlib.pyplot as plt
import analytical as an
def test_garvin():
    x=5.0
    z=-1.0
    cs=1.0
    cp=3.0
    rho=1.0
    pois=an.poisson(rho, cs, cp)
    t, ux, uz = an.garvin(x, z, pois, cs=cs, rho=rho, tend=16, np=1e5)
    dt = t[-1] - t[-2]
    t0 = 5.0
    fp = 1.0
    r = an.ricker(t, t0, fp)
    du = an.convolve(uz, r)
    #print(T)
    #print(Ux)
    plt.plot(t, -du)
    plt.show()

#test_garvin()

def lamb(tend, nt):
    x=1.0
    z=0.0
    cs=1.0
    cp=3.0
    rho=1.0
    t0 = 4.0
    fp = 1.0
    pois=an.poisson(rho, cs, cp)
    t, uxx, uzz, uxz, uzx = an.lamb2D(x, z, pois, cs=cs, rho=rho, tend=tend,
            nt=nt)
    r = an.ricker(t, t0, fp)
    du = an.convolve(uzz, r, t)
    return t, du, uzz


def test_lamb():
    t, du, uzz = lamb(2.0, 2e3)
    print(uzz[-1])
    plt.plot(t, uzz)
    #plt.plot(t, du)
    #t, du, uzz = lamb(45.0, 8e5)
    ##t = t[::10]
    ##du = du[::10]
    #plt.plot(t, du)
    plt.xlim((0, 2.5))
    plt.show()

test_lamb()

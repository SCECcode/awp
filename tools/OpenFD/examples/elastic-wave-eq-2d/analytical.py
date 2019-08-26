import numpy

def poisson(rho, cs, cp):
    G = cs**2*rho
    lam = -2*G + cp**2*rho
    return lam/(2*(lam + G))


def garvin_exact(x, z, rho, cs, cp, fp, t0, tend, nt):
    """
    Compute solution to Garvin's problem using a Ricker wavelet as source.

    Arguments:
        x : Receiver position on surface
        z : Source depth
        rho : Density (g/cm^3)
        cs : S-wave speed (km/s)
        cp : P-wave speed (km/s)
        fp : Peak frequency for Ricker wavelet (Hz)
        t0 : Time delay (s)

    """
    pois = poisson(rho, cs, cp)
    t, ux, uz = garvin(x, -z, pois, cs=cs, rho=rho, tend=tend, np=nt)
    dt = t[-1] - t[-2]
    r = ricker(t, t0, fp)
    out_ux = convolve(ux, r)
    out_uz = convolve(uz, r)
    return t, out_ux, out_uz 

def garvin(x, z, pois, cs=1.0, rho=1.0, np=1e4, tend=2.0):
    """
    Garvin's line blast problem, plane strain (2-D)
    Step blast line load applied at depth z below surface of elastic halfspace
    
    Written by Eduardo Kausel, MIT, Room 1-271, Cambridge, MA
    
    Arguments:
           x : range of receiver on surface (z = 0)
           z : depth of source at x = 0
           pois : Poisson's ratio
           tend : Final time (s)
    Unit soil properties are assumed (Cs=1, rho=1)

    Returns: 
           T : Time vector (s)
           ux :  Horizontal displacement at surface
           zz :  Vertical displacement at surface
    
    Sign convention:
      x from left to right, z=0 at the surface, z points up.
      Displacements are positive up and to the right.
      If z > 0 ==> an upper halfspace is assumed, z=0 is the lower boundary
      If z < 0 ==>  a lower halfspace is assumed, z=0 is the upper boundary
    Reference:
      W.W. Garvin, Exact transient solution of the buried line source problem,
                   Proceedings of the Royal Society of London, Series A
                   Vol. 234, No. 1199, March 1956, pp. 528-541
    """

    
    r = numpy.sqrt(x**2+z**2)	# Source-receiver distance
    mu = rho*cs**2              # Shear modulus
    a2 = (1-2*pois)/(2-2*pois)
    a = numpy.sqrt(a2 + 0j)          #  Cs/Cp
    tmax = tend*cs/r
    dt = (tmax-a)/np            # time step

    c = numpy.abs(z)/r		# direction cosine w.r.t. z axis
    s = numpy.abs(x)/r		#     "        "     "    x axis
    theta = numpy.arcsin(s)	# Source-receiver angle w.r.t. vertical
    fac = 1/(numpy.pi*mu*r)	# scaling factor for displacements
    T0 = numpy.array([0, a]) 	# time interval before arrival of P waves 
    Ux0 = [0, 0]
    Uz0 = [0, 0]
    T = numpy.linspace(a+dt, tmax, np)    # Time vector
    T2 = T**2;
    T1 = numpy.conj(numpy.sqrt(T2-a2 +0j))	# make the imaginary part negative
    q1 = c*T1+1j*s*T		        # Complex horizontal slowness, P waves
    p1 = c*T+1j*s*T1		        #   "     vertical      "      "   "
    Q1 = q1**2
    s1 = numpy.sqrt(Q1+1 + 0j)
    S1 = 2*Q1+1
    R1 = S1**2 - 4*Q1*p1*s1	        # Rayleigh function, P waves
    D1 = p1/T1/R1	        	# Derivative of q1 divided by R1
    Ux1 = 2*fac*numpy.sign(x)*numpy.imag(q1*s1*D1)
    Uz1 = -fac*numpy.sign(z)*numpy.real(S1*D1)
    #T = numpy.concatenate((T0, T))
    Ux = numpy.concatenate((Ux0, Ux1))
    Uz = numpy.concatenate((Uz0,Uz1))

    t = T*r/cs

    ux = Ux1/r
    uz = Uz1/r

    return t, ux, uz

def convolve(u, v, t, dt=1.0):
    import numpy as np
    from scipy import integrate

    n = len(u)
    dt = t[1] - t[0]
    freq = numpy.fft.fftfreq(n, 1.0)
    om = 2*numpy.pi*freq
    do = 1j*om 

    uhat = numpy.fft.fft(u)
    vhat = numpy.fft.fft(v)
    #uhat = uhat[0:-1]
    #N = int(n)
    #print(n)
    #what = 1j*np.zeros(N)
    #what[0:int(N/2)] = 1j*np.arange(0, int(N/2), 1)
    #what[int(N/2)+1:] = 1j*np.arange(-int(N/2) + 1, 0, 1)
    #what = what*uhat
    #w = np.real(np.fft.ifft(what))
    #f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    #f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    if n % 2 == 0:
        m = int(n/2)
        do[m-1] = 0.0
        do[m] = 0.0
    elif n % 2 == 1:
        m = int((n-1)/2)
        do[m] = 0.0
        do[m+1] = 0.0

    
    #print(len(uhat), len(om))
    # Differentiate function
    what = uhat*vhat*do
    res = numpy.real(numpy.fft.ifft(what))

    t = np.arange(0, n, 1)
    return res
    #return integrate.cumtrapz(res, t, initial=0) 

def ricker(t, t0, fp):
    import numpy as np
    a = np.pi**2*fp**2
    return (1 - 2*a*(t-t0)**2)*np.exp(-a*(t-t0)**2)

def dricker(t, t0, fp):
    import numpy as np
    a = -np.pi**2*fp**2
    tau = t - t0
    return 2*a*tau*(2*a*tau**2 + 3)*np.exp(a*tau**2)

def lamb_exact(x, z, rho, cs, cp, fp, t0, tend, nt):
    """
    Compute solution to Garvin's problem using a Ricker wavelet as source.

    Arguments:
        x : Receiver position on surface
        z : Source depth
        rho : Density (g/cm^3)
        cs : S-wave speed (km/s)
        cp : P-wave speed (km/s)
        fp : Peak frequency for Ricker wavelet (Hz)
        t0 : Time delay (s)

    """
    pois = poisson(rho, cs, cp)
    t, uxx, uzz, uxz, uzx = lamb2D(x, z, pois, cs=cs, rho=rho, tend=tend, nt=nt)
    dt = t[-1] - t[-2]
    r = ricker(t, t0, fp)
    out_ux = convolve(uxz, r, t)
    out_uz = convolve(uzz, r, t)
    return t, out_ux, out_uz 

def lamb2D(x,z,pois, cs=1.0, rho=1.0, nt=200, tend=20):
    """
     Lamb's problem in plane strain (2-D)
     Implusive line load applied onto the surface of an elastic
     half-space and receiver at depth z
    
     To obtain the solution for an interior source and receiver at the
     surface, exchange the coupling terms, and reverse their signs.
    
     Written by Eduardo Kausel, MIT, Room 1-271, Cambridge, MA
    
     [T, Uxx, Uxz, Uzz] = lamb2(x,z,pois)
      Input arguments:
            x, z = coordinates of receiver relative to source at (0,0)
            pois = Poisson's ratio

            cs = 1 : Shear wave velocity
            rho = 1 : Mass density
            np = 200 : Number of time intervals

      Output arguments
            T   = Dimensionless time vector, tau = t*Cs/r
            Uxx = Horizontal displacement caused by a horizontal load
            Uxz = Horizontal displacement caused by a vertical load
            Uzx = Vertical displacement caused by a horizontal load
            Uzz = Vertical displacement caused by a vertical load
    
     Unit soil properties are assumed (Cs=1, rho=1)
    
     Sign convention:
       x from left to right, z=0 at the surface, z points up.
       Displacements are positive up and to the right.
       If z > 0 ==> an upper halfspace is assumed, z=0 is the lower boundary
                    Vertical impulse at z=0 is compressive, i.e. up)
       If z < 0 ==>  a lower halfspace is assumed, z=0 is the upper boundary
                    Vertical impulse at z=0 is tensile, i.e. up)
       If z = 0 ==>  a lower halfspace is assumed, z=0 is the upper boundary
                     both the load and the displacements are at the surface
    
     References:
       Walter L. Pilant, Elastic Waves in the Earth, Elsevier, 1979
       Eringen & Suhubi, Elastodynamics, Vol. 2, page 615, eq. 7.16.5,
                         Pages 606/607, eqs. 7.14.22 and 7.14.27
                         Page 612, eq. 7.15.9, page 617, eqs. 7.16.8-10
    """
    import numpy as np
    
    # Basic data
    r = np.sqrt(x**2+z**2);	# Source-receiver distance
    tmax = tend*cs/r
    mang = 0.4999*np.pi	# Max. angle for interior solution
    
    mu = rho*cs**2;	# Shear modulus
    a2 = (1-2*pois)/(2-2*pois);
    a = np.sqrt(a2);		# Cs/Cp
    print("cp", cs/a)
    dt = (tmax-a)/nt;	# time step

    
    c = np.abs(z)/r;		# direction cosine w.r.t. z axis
    s = np.abs(x)/r;		#     "        "     "    x axis
    theta = np.arcsin(s);	# Source-receiver angle w.r.t. vertical
    crang = np.arcsin(a);	# Critical angle w.r.t. vertical
    fac = cs/(np.pi*mu*r);	# scaling factor for displacements
    
    # t < tp=a (two points suffice)
    T0 = [0, a];
    Uxx0 = [0, 0];
    Uxz0 = [0, 0];
    Uzx0 = [0, 0];
    Uzz0 = [0, 0];
    T = np.arange(a+dt,tmax,dt);		# Time vector
    T2 = T**2;
    jl = len(T);
    if theta > mang:
        # Displacements on surface
        T1 = np.arange(a,1, dt);		# interval from tp to ts
        T2 = np.arange(1+dt,tmax,dt);	# t > ts=1
        # a=tp <= t <= ts=1
        t2 = T1**2;
        p = t2-a2;
        s = 1-t2;
        q = 2*t2-1;
        d = q**4 + 16*p*s*t2**2;
        p = np.sqrt(p + 0j);
        Uxx1 = 4*fac*t2*s*p/d;
        Uzz1 = -fac*q**2*p/d;
        s = np.sqrt(s + 0j);
        Uxz1 = 2*fac*T1*q*s*p/d;
        # t > ts=1
        t2 = T2**2;
        p = np.sqrt(t2-a2 + 0j);
        s = np.sqrt(t2-1 + 0j);
        q = 2*t2-1;
        d = q**2-4*t2*p*s;
        Uxx2 = -fac*s/d;
        Uzz2 = -fac*p/d;
        Uxz2 = np.zeros(Uxx2.shape);
        cr = cs*(0.874+(0.197-(0.056+0.0276*pois)*pois)*pois);
        tr = cs/cr;		# dimensionless arrival time of R waves
        xr = (cr/cs)**2;
        wr = 0.25*fac*np.pi*(1-0.5*xr)**3/(1-0.5*xr**2+0.125*xr**3-a2);
        jr =  np.floor((tr-a)/dt+1);
        # Combine solutions
        Uxx = np.concatenate((Uxx1, Uxx2));
        Uzz = np.concatenate((Uzz1, Uzz2));
        Uxz = np.concatenate((Uxz1, Uxz2));
        Uzx = -Uxz;
        T = np.concatenate((T1, T2));   
    elif theta < 1e-3:
        # Displacements on epicentral line
        D1 = (T2-a2+0.5)**2 - T*(T2-a2)*np.sqrt(T2-a2+1 +0j);
        Uxx = -T*np.sqrt(T2-a2 + 0j)*np.sqrt(T2-a2+1 + 0j)/D1;
        Uzz = T2*(T2-a2+0.5)/np.sqrt(T2-a2 + 0j)/D1;
        js = 1+np.floor((1-a)/dt+1); # first element after arrival of S waves
        T2 = T2[js:jl];
        T1 = T[js:jl];
        D2 = (T2-0.5)**2 - T1*(T2-1)*np.sqrt(T2+a2-1 + 0j);
        Uxx[js:jl] = Uxx[js:jl] + T2*(T2-0.5)/D2/np.sqrt(T2-1 + 0j);
        Uzz[js:jl] = Uzz[js:jl] - T1*np.sqrt(T2-1)*np.sqrt(T2-1+a2 + 0j)/D2;
        T1 = [];
        f = 0.5*fac;
        Uxx = f*Uxx;
        Uzz = f*Uzz;
        Uxz = zeros(T.shape);
        Uzx = Uxz;
    else:
        # Displacements in the interior
        T1 = np.conj(np.sqrt(T2-a2 + 0j));# make the imaginary part negative
        T2 = np.conj(np.sqrt(T2-1 + 0j));
        q1 = c*T1+1j*s*T;	# Complex slowness, P waves (Cagniard - De Hoop path)
        q2 = c*T2+1j*s*T;	#    "        "     S   "
        p1 = c*T+1j*s*T1;
        p2 = c*T+1j*s*T2;
        Q1 = q1**2;
        Q2 = q2**2;
        s1 = np.sqrt(Q1+1 + 0j);
        s2 = np.sqrt(Q2+a2 + 0j);
        S1 = 2*Q1+1;
        S2 = 2*Q2+1;     
        R1 = S1**2 - 4*Q1*p1*s1;	# Rayleigh function, P waves
        R2 = S2**2 - 4*Q2*p2*s2;	# Rayleigh function, S waves
        D1 = p1/T1/R1;		# Derivative of q1 divided by R1
        D2 = p2/T2/R2;		#     "         q2    "     " R2
        # Check critical angle
        if (theta > crang):
            tcrit = np.cos(theta-crang);
        else:
            tcrit = 1;
        # Apply Heaviside step function
        k = np.floor(np.real((tcrit-a)/dt+1));
        if k >=1: 
            D2[1:int(k)]=0; 
        S1 = S1*D1;
        S2 = S2*D2;
        D1 = 2*D1*s1; 
        D2 = 2*D2*s2;
        # Displacements due to impulsive line load
        sgn = fac*np.sign(x)*np.sign(z);
        Uxx = fac*np.real(p2*S2-Q1*D1);
        Uzz = fac*np.real(p1*S1-Q2*D2);
        Uzx = sgn*np.imag(q1*p1*D1-q2*S2);
        Uxz = sgn*np.imag(q1*S1-q2*p2*D2);
    #T = [T0,T];
    #Uxx = [Uxx0,Uxx];
    #Uzz = [Uzz0,Uzz];
    #Uxz = [Uxz0,Uxz];
    #Uzx = [Uzx0,Uzx];

    T0 = np.arange(0,T[0], dt);
    n0 = len(T0)
    z = numpy.zeros((n0,))

    Uxx = np.real(np.concatenate((z, Uxx)))
    Uzz = np.real(np.concatenate((z, Uzz)))
    Uzx = np.real(np.concatenate((z, Uzx)))
    Uxz = np.real(np.concatenate((z, Uxz)))
    T = np.concatenate((T0, T))

    t = T*r/cs
    #Uxx = Uxx/r
    #Uzz = Uzz/r
    #Uxz = Uxz/r
    #Uzx = Uzx/r


    return t, Uxx, Uzz, Uxz, Uzx

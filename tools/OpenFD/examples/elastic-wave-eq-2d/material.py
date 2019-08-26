import numpy as np

def rocksoil1d(x, y):
    """
    Generatate 1D velocity model
    
    Arguments:

        x  : X-coordinate (not used) (nx x ny)
        y  : Y-coordinate of grid (nx x ny)
    
    """
    rho = np.zeros(y.shape)
    cs = np.zeros(y.shape)
    cp = np.zeros(y.shape)

    for j in range(y.shape[1]):
        for i in range(y.shape[0]):
            if y[i,j]<-7:
                rho[i,j] = 2.8 
                cs[i,j] = 3.464 
                cp[i,j] = 6
            elif y[i,j]<-3:
                rho[i,j] = 2.3 
                cs[i,j] = 2.7
                cp[i,j] = 5
            elif y[i,j]<-1.5:
                rho[i,j] = 1.5 
                cs[i,j] = 1.0
                cp[i,j] = 4
            else:
                rho[i,j] = 1.0 
                cs[i,j] = 0.8
                cp[i,j] = 3.6
    
    return rho, cs, cp

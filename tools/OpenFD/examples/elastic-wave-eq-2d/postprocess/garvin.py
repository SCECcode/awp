import numpy as np
import matplotlib.pyplot as plt
datafile = 'out/garvin/sat_3.npz'

data = np.load(datafile)
print(data.files)
t = data['t']
vx = data['v1']
vz = data['v2']

plt.figure()
plt.subplot(1,2,1)
plt.plot(t, -vx)
plt.xlabel('t (s)')
plt.ylabel('$v_x$ (m/s)')
plt.subplot(1,2,2)
plt.plot(t, vz)
plt.xlabel('t (s)')
plt.ylabel('$v_z$ (m/s)')
plt.show()

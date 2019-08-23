import helper 
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = lambda label, ref, ext : 'out/%s_%d%s'%(label, ref, ext)
output = 'test'  
label = 'garvin_comp_sat_fs2-5'
station = 1
nref = 4
fmt = 'png'

if len(sys.argv) > 1:
    fmt = sys.argv[1]

fs2 = []
for i in range(nref):
    fs2.append(helper.load_data(filename('fs2_lamb_s_vol', i, '.npz')))
    #fs2.append(helper.load_data(filename('fs2_lamb_s_vol', i, '.npz')))

sat = []
for i in range(nref):
    sat.append(helper.load_data(filename('sat_lamb_s_surf', i, '.npz')))
    #sat.append(helper.load_data(filename('sat_lamb_s_vol', i, '.npz')))

#plt.plot(fs2[0].t, fs2[0].v2[:,0])
#plt.plot(sat[0].t, (num[0].v2[:,0] -
#    sat[0].v2[:,0])/np.max(np.abs(sat[0].v2[:,0])))
plt.xlabel('time (s)')
#helper.plot(num, exact=sat[-1], comp='v1', recv=station)
plt.plot(sat[2].t, sat[2].v2[:,1], 'b')
plt.plot(sat[3].t, sat[3].v2[:,1], 'k')
#plt.plot(fs2[1].t, fs2[1].v1[:,1])
plt.plot(fs2[2].t, 2*fs2[2].v2[:,1])
plt.plot(fs2[3].t, 2*fs2[3].v2[:,1])
#plt.plot(sat[1].t, sat[1].v1[:,0])
plt.xlim((2.0, 16.5))
plt.ylim((-0.04, 0.04))

#plt.savefig('figures/%s-v1-%d.%s'%(label, station, fmt), 
#            bbox_inches='tight', dpi=300)


#helper.plot(num, exact=sat[-1], comp='v2', recv=station)
#helper.plot(num, exact=None, comp='v2', recv=station)
#plt.xlim((3.5, 7))
#plt.clf()
#plt.plot(sat[3].t, sat[3].v2[:,0], 'k')
#plt.plot(fs2[1].t, fs2[1].v2[:,1])
#plt.plot(sat[1].t, sat[1].v2[:,0])
#plt.savefig('figures/%s-v2-%d.%s'%(label, station, fmt), 
#            bbox_inches='tight', dpi=300)
plt.show()

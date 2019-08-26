import sys
import helper 
import matplotlib.pyplot as plt

filename = lambda label, ref, ext : 'out/%s_%d%s'%(label, ref, ext)
output = 'lamb_vol'  
nref = 3
delay = 9.0
label = 'lamb_vol'
station = 1
nref = 3
fmt = 'png'

if len(sys.argv) > 1:
    fmt = sys.argv[1]

num = []
for i in range(nref):
    num.append(helper.load_data(filename('lamb_vol', i, '.npz')))

t0 = delay - num[0].t0
exact1 = helper.load_lamb_exact(time_shift=t0, v1='out/lamb_vol_v1.exact',
                               v2='out/lamb_vol_v2.exact', nudge=3)
exact2 = helper.load_lamb_exact(time_shift=t0, v1='out/lamb_vol_v1.exact',
                               v2='out/lamb_vol_v2.exact', nudge=0)

helper.plot(num, exact1, 'v1', 0, offset=1.5)
plt.xlim((3.0, 9))
plt.savefig('figures/%s-v1-%d.%s'%(label, station, fmt), 
            bbox_inches='tight', dpi=300)


helper.plot(num, exact2, 'v2', 0, offset=1.5)
plt.xlim((3.0, 9))
plt.savefig('figures/%s-v2-%d.%s'%(label, station, fmt), 
            bbox_inches='tight', dpi=300)


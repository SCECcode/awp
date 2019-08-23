import sys
import helper 
import matplotlib.pyplot as plt

filename = lambda method, ref : 'out/lamb_surf/x_%s_%d.npz'%(method, ref)
output = lambda method, comp, fmt : 'figures/lamb_surf/x_%s-%s.%s'%(method,
                                    comp, fmt)
nref = 3
delay = 9.0
station = 0
nref = 4
fmt = 'png'
xlim = (2, 8.0)

if len(sys.argv) > 1:
    fmt = sys.argv[1]

sat = []
for i in range(1, nref):
    sat.append(helper.load_data(filename('sat', i)))
fs2 = []
for i in range(1, nref):
    fs2.append(helper.load_data(filename('fs2', i)))

t0 = delay - sat[0].t0
exact1 = None#helper.load_lamb_exact(time_shift=t0, nudge=3)
exact2 = None#helper.load_lamb_exact(time_shift=t0, nudge=0)
 
exact1 = helper.load_data(filename('sat', 4))
exact2 = helper.load_data(filename('sat', 4))

ax = helper.plot(fs2, exact1, 'v1', station, offset=2.0)
plt.xlim(xlim)
ylim = ax.get_ylim()
plt.savefig(output('fs2', 'v1', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(sat, exact1, 'v1', station, offset=2.0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig(output('sat', 'v1', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(fs2, exact2, 'v2', station, offset=2.0)
plt.xlim(xlim)
ylim = ax.get_ylim()
plt.savefig(output('fs2', 'v2', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(sat, exact2, 'v2', station, offset=2.0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig(output('sat', 'v2', fmt), bbox_inches='tight', dpi=300)

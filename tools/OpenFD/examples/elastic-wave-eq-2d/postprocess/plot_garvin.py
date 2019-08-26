import helper 
import matplotlib.pyplot as plt
import sys

filename = lambda method, ref : 'out/garvin/%s_%d.npz'%(method, ref)
output = lambda method, comp, fmt : 'figures/garvin/%s-%s.%s'%(method,
                                    comp, fmt)
station = 0
nref = 3
fmt = 'png'

if len(sys.argv) > 1:
    fmt = sys.argv[1]

sat = []
for i in range(0, nref):
    sat.append(helper.load_data(filename('sat', i)))
fs2 = []
for i in range(0, nref):
    fs2.append(helper.load_data(filename('fs2', i)))

exact = helper.load_garvin_exact(rho=sat[0].rho)

ax = helper.plot(fs2, exact, 'v1', station)
#plt.xlim((3.5, 7))
ylim = ax.get_ylim()
plt.savefig(output('fs2', 'v1', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(sat, exact, 'v1', station)
#plt.xlim((3.5, 7))
plt.ylim(ylim)
plt.savefig(output('sat', 'v1', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(fs2, exact, 'v2', station)
#plt.xlim((3.5, 7))
ylim = ax.get_ylim()
plt.savefig(output('fs2', 'v2', fmt), bbox_inches='tight', dpi=300)

ax = helper.plot(sat, exact, 'v2', station)
#plt.xlim((3.5, 7))
plt.ylim(ylim)
plt.savefig(output('sat', 'v2', fmt), bbox_inches='tight', dpi=300)

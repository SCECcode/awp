import sys
import helper 
import matplotlib.pyplot as plt

filename = lambda method, ref : 'out/bndopt/%s_%d.npz'%(method, ref)
output = lambda method, comp, fmt : 'figures/lamb_surf/z_%s-%s.%s'%(method,
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
for i in range(0, nref):
    sat.append(helper.load_data(filename('sat', i)))
bndopt = []
for i in range(0, nref):
    bndopt.append(helper.load_data(filename('bndopt', i)))

bndopt_new = []
for i in range(0, 3):
    bndopt_new.append(helper.load_data(filename('opt1_bndopt', i)))

fs2 = []
for i in range(0, 3):
    fs2.append(helper.load_data(filename('fs2', i)))


res=1
plt.plot(sat[3].t, sat[3].v1, 'k')
plt.plot(sat[res].t, sat[res].v1, label='sat')
plt.plot(bndopt_new[res].t, bndopt_new[res].v1, label='bndopt')
#plt.plot(fs2[res].t, fs2[res].v2, 'm')
plt.legend()

plt.show()


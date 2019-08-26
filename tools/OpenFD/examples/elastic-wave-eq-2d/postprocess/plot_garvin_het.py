import helper 
import matplotlib.pyplot as plt
import sys

filename = lambda method, ref : 'out/het-garvin/%s_%d.npz'%(method, ref)
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
#fs2 = []
#for i in range(0, nref):
#    fs2.append(helper.load_data(filename('fs2', i)))
#
plt.plot(sat[-1].t, -sat[-1].v2)
plt.show()

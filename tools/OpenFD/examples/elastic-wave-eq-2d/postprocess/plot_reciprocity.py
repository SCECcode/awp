import sys
import helper 
import matplotlib.pyplot as plt
from openfd import Struct



green_file = lambda comp, method, ref : (
           'out/reciprocity/%s_%s_%d.npz' %(comp, method, ref))
fwd_file = lambda moment, method, ref : (
            'out/reciprocity/moment_%s_%s_%d.npz' % (moment, method, ref))
output = lambda method, comp, fmt : 'figures/lamb_surf/x_%s-%s.%s'%(method,
                                    comp, fmt)
if len(sys.argv) > 1:
    fmt = sys.argv[1]

ref = range(2, 3)

def load_inverse(gx, gz):
    out = Struct()
    out.xx_vx = helper.representation_theorem(gx, [1.0, 0.0, .0])  
    out.xx_vz = helper.representation_theorem(gz, [1.0, 0.0, .0])  
    out.zz_vx = helper.representation_theorem(gx, [0.0, 1.0, 0.0])  
    out.zz_vz = helper.representation_theorem(gz, [0.0, 1.0, 0.0])  
    out.xz_vx = helper.representation_theorem(gx, [0.0, 0.0, 1.0])  
    out.xz_vz = helper.representation_theorem(gz, [0.0, 0.0, 1.0])  
    return out

def load_data(bc, comp, rng):
    out = Struct()
    out.gx = []
    out.gz = []
    out.mxx = []
    out.mzz = []
    out.mxz = []
    for i in rng:
        out.gx.append(helper.load_data(green_file('fx', bc, i)))
        out.gz.append(helper.load_data(green_file('fz', bc, i)))
        out.mxx.append(helper.load_data(green_file('mxx'+comp, bc, i)))
        out.mzz.append(helper.load_data(green_file('mzz'+comp, bc, i)))
        out.mxz.append(helper.load_data(green_file('mxz'+comp, bc, i)))
    return out

def make_plots(obj, rec, field, r0):
    if field == 'v1':
        v = 'vx'
    else:
        v = 'vz'
    #plt.plot(obj.mxx[r0].t, obj.mxx[r0][field][:,0],'C0')
    #plt.plot(obj.mxx[r0].t, rec['xx_' + v],'C2')
    
    #plt.figure()
    #plt.plot(obj.mzz[r0].t, obj.mzz[r0][field][:,0],'C0')
    #plt.plot(obj.mzz[r0].t, rec['zz_' + v],'C2')
    #
    #plt.figure()
    plt.plot(obj.mxz[r0].t, obj.mxz[r0][field][:,0],'C0')
    plt.plot(obj.mxz[r0].t, rec['xz_' + v],'C2')


r0=0
fs2x = load_data('fs2', '_x', ref)
rec_fs2x = load_inverse(fs2x.gx[r0].sgt, fs2x.gz[r0].sgt)
fs2z = load_data('fs2', '_z', ref)
rec_fs2z = load_inverse(fs2z.gx[r0].sgt, fs2z.gz[r0].sgt)
sat = load_data('sat', '', ref)
rec_sat = load_inverse(sat.gx[r0].sgt, sat.gz[r0].sgt)

make_plots(fs2x, rec_fs2x, 'v1',  0)
#make_plots(fs2z, rec_fs2z, 'v2', 0)
make_plots(sat, rec_sat, 'v1',  0)
#make_plots(sat, rec_sat, 'v2',  0)



#plt.figure()
#plt.plot(fs2.x_mxz[r0].t, fs2.x_mxz[r0].v1[:,0],'C0')
#plt.plot(fs2.x_mxz[r0].t, rep_fs2.xz_vx,'C2')
#plt.savefig('fs2-representation-vx-5', bbox_inches='tight', dpi=300)

#plt.figure()
#plt.plot(fs2.z_mxz[r0].t, fs2.z_mxz[r0].v2[:,0],'C0')
#plt.plot(fs2.z_mxz[r0].t, rep_fs2.xz_vz,'C2')
#plt.savefig('fs2-representation-vz-5', bbox_inches='tight', dpi=300)

plt.show()

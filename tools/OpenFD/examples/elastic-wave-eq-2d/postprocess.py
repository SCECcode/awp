import sys
import make_seismograms as seism

def save_garvin(r0=0, rn=4, fmt='png', xlim=None):
    seismfile = lambda ref, ext : 'out/garvin_%d%s'%(ref, ext)
    # Make figures
    d = []
    for ref in range(r0, rn):
        data = seism.load_data(seismfile(ref, '.npz')) 
        d.append(data)
        e = seism.load_garvin_exact(data)
    seism.save_data('garvin/recv', d, exact=e, fmt=fmt, xlim=xlim)

def save_lamb(r0=0, rn=4, fmt='png', xlim=None, label='lamb_surf'):
    seismfile = lambda ref, ext : 'out/%s_%d%s'%(label, ref, ext)
    # Make figures
    d = []
    for ref in range(r0, rn):
        data = seism.load_data(seismfile(ref, '.npz'))
        d.append(data)
    e = seism.load_lamb_exact_kausel(data)
    seism.save_data('%s/recv'%label, d, exact=e, fmt=fmt, xlim=xlim)

if sys.argv[1] == 'Garvin':
    save_garvin(r0=0, rn=3, fmt='png')
    #save_garvin(r0=0, rn=1, fmt='pdf')

if sys.argv[1] == 'Lamb-Surface':
    save_lamb(r0=1, rn=3, fmt='png', label='lamb_surf', xlim=(0, 20))
    #save_lamb(r0=1, rn=3, fmt='pdf', label='lamb_surf', xlim=(0, 20))

if sys.argv[1] == 'Lamb-Volume':
    save_lamb(r0=0, rn=4, fmt='png', label='lamb_vol')
    save_lamb(r0=0, rn=4, fmt='pdf', label='lamb_vol')

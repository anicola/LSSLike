#!/usr/bin/env python
import sacc
from  desclss import *
from desclss.lss_theory import LSSTheory
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

fns="../2pt_validation/theory_comparison/GaussPZ_0.02+FullSky+run0+ztrue_mean.sacc"
fn='../2pt_validation/theory_comparison/inputs/fastcats/GaussPZ_0.02+FullSky+run20+ztrue/twopoints20_ns2048.sacc'
zmin=2.
zmax=2.1
lmin=300
lmax=350
s=sacc.SACC.loadFromHDF(fn, zmin=zmin, zmax=zmax, lmin=lmin, lmax=lmax, diagonal=False,precision_filename=fns)
T=LSSTheory(s,zmin, zmax, lmin, lmax)
params={'omega_c':0.25,
        'omega_b':0.05,
        'omega_k':0.0,
        'sigma_8':0.8,
        'h':0.7,
        'n_s':0.96,
        'transfer_function':'eisenstein_hu',
        'matter_power_spectrum':'linear',
        'rsm':2.9280975*0.7}

#lmin=300
#lmax=350
bz,bb=np.loadtxt('/global/cscratch1/sd/kakoon/sims_LSST/sample_LSST/BzBlue.txt',unpack=True)
bofz=interp1d(bz,bb)
zar=np.linspace(0,1.6,16)
params['gals_z_b']=zar
params['gals_b_b']=np.array([bofz(z) for z in zar])
ierrs=s.precision.matrix.diagonal()


def chi2(x):
    params['sigma_8']=x[0]
    params['omega_c']=x[1]-params['omega_b']
    params['rsm']=x[2]
    tvec=T.get_prediction(params)
    chi2=0
    dof=0
    for cc,(i1,i2,ells,ndx) in enumerate(s.sortTracers()):
        if i1==i2:
            data=s.mean.vector[ndx]
            theo=tvec[ndx]
            ierr=ierrs[ndx]
            w=np.where((ells>lmin)&(ells<lmax))
            chi2+=((data[w]-theo[w])**2*ierr[w]).sum()
            dof+=len(w[0])
    print x,chi2
    return chi2

##
#if False:
#    for i in range(len(s.tracers)):
#        ndx=lcut_ndx(i,i,lmin,lmax)
#        data=s.mean.vector[ndx]



x=[0.8,0.3,2.9280975*0.7]
print fmin(chi2,x)



# print "Making predictions"
# tvec=T.get_prediction(params)
# print "done"
# Nx=12
# for cc,(i1,i2,ells,ndx) in enumerate(s.sortTracers()):
#     if (i1<Nx) and (i2<Nx):
#         plt.subplot(Nx,Nx,i1*Nx+i2+1)
#         plt.plot(ells,s.mean.vector[ndx],'bo')
#         plt.plot(ells,tvec[ndx])
#         plt.semilogy()
    
# plt.show()



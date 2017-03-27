#!/usr/bin/env python
import sys
import numpy as np
import sacc
from desclss import LSSTheory
import copy
import fishplot as fsh
import matplotlib.pyplot as plt

#Simulation to use
if len(sys.argv)!=4 :
    print "Usage ./sample_emcee.py fisher_file chains_file plot_out"
    exit(1)
fisher_file=sys.argv[1]
chains_file=sys.argv[2]
plot_out=sys.argv[3]

s0=sacc.SACC.loadFromHDF("sim_sample/sims/sim_mean.sacc")
errs=np.sqrt(np.diag(np.linalg.inv(s0.precision.matrix)))
n_tracers=len(s0.tracers)

#Plot N(z)
plt.figure()
for tr in s0.tracers :
    plt.plot(tr.z,tr.Nz,lw=2)
plt.xlabel('$z$',fontsize=18)
plt.ylabel('$p(z)/p_{\\rm max}$',fontsize=18)
plt.savefig("test/plots/dndz.pdf",bbox_inches='tight')

#Plot Cells
fig=plt.figure(figsize=(9,9))
plt.subplots_adjust(wspace=0.0,hspace=0.0)
for i1,i2,ells,ndx in s0.sortTracers() :
    ax=fig.add_subplot(n_tracers,n_tracers,i1+n_tracers*i2+1)
    ax.errorbar(ells,s0.mean.vector[ndx],errs[ndx],label='%d '%i1+'%d '%i2,fmt='.',color='k')
    ax.get_yaxis().set_ticks([])
    ax.set_xscale('log')
    ax.set_xlim([5,702])
    ax.set_xlabel('$\\ell$',fontsize=16)
    ax.annotate('$%d\\times'%i1+'%d$'%i2,xy=(0.7,0.85),xycoords='axes fraction',fontsize=14)
plt.savefig("test/plots/cls.pdf",bbox_inches='tight')

#Parameters
params={}
params['omh2']={'value':0.1414,'dval':0.005,'label':'$\\Omega_M\\,h^2$','isfree':True}
params['w']   ={'value':-1.0  ,'dval':0.02 ,'label':'$w$'              ,'isfree':True}
params['s8']  ={'value':0.8   ,'dval':0.02 ,'label':'$\\sigma_8$'      ,'isfree':False}
for i in np.arange(4) :
    params['b%d'%i]  ={'value':1.0   ,'dval':0.02 ,'label':'$b_%d$'%i  ,'isfree':True}
npar=len(params)
params_vary={}
for p in params :
    if params[p]['isfree'] :
        params_vary[p]=params[p]
npar_vary=len(params_vary)

#Fisher matrix
fisher=np.load(fisher_file)
covar=np.linalg.inv(fisher)
corr=covar/np.sqrt(np.diag(covar)[:,None]*np.diag(covar)[None,:])
covar_dict={}
for i,nam_i in enumerate(params_vary) :
    covar_dict[nam_i]={}
    for j,nam_j in enumerate(params_vary) :
        covar_dict[nam_i][nam_j]=covar[i,j]

#Samples
chain=np.load(chains_file)
samples_wrong=(chain)[:,300:,:].reshape((-1,npar_vary))
samples=samples_wrong.copy(); samples[:,[0,1]]=samples_wrong[:,[1,0]]
samples_dict={}
for  i,nam in enumerate(params_vary) :
    samples_dict[nam]=samples[:,i]

#Print results
print "Fisher forecast:"
for i,nam in enumerate(params_vary) :
    print params_vary[nam]['label']+" = %lE"%(params_vary[nam]['value'])+' +- %lE'%(np.sqrt(covar[i,i]))
print "\nSamples:"
for i,nam in enumerate(params_vary) :
    print params_vary[nam]['label']+" = %lE"%(np.mean(samples_dict[nam]))+" +- %lE"%(np.std(samples_dict[nam]))
print "\nCorrelation matrix:"
print corr

#Plot results
fsh.plot_fisher_all(params_vary,[covar_dict,samples_dict],['covar','chains'],
                    [{'ls':'solid','col':'blue','alpha':0.5,'lw':2},
                     {'ls':'solid','col':'red','alpha':0.5,'lw':1.5}],
                    ['Fisher','Chains'],3,'test/plots/'+plot_out,do_1D=False)

plt.show()

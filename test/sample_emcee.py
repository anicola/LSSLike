#!/usr/bin/env python
import sys
import numpy as np
import sacc
from desclss import LSSTheory,LSSLikelihood
import emcee as mc
#import corner as cn
#import matplotlib.pyplot as plt

#Simulation to use
if len(sys.argv)!=2 :
    print "Usage ./sample_emcee.py sim_num"
    exit(1)
isim=int(sys.argv[1])

#Parameter order
i_om=0; i_s8=1; i_b0=2; i_b1=3; i_b2=4; i_b3=5;
labels=['$\\Omega_M\\,h^2$','$\\sigma_8$','$b_0$','$b_1$','$b_2$','$b_3$']
npar=6


#Read file with noiseless sim and initialize theory object
s0=sacc.SACC.loadFromHDF("sim_sample/sims/sim_mean.sacc")
lt=LSSTheory("sim_sample/sims/sim_mean.sacc")


#Define log(p). This is just a wrapper around the LSSLikelihood lk
def logprob(p,lk) :
    print p
    oc=p[i_om]/0.67**2-0.045
    dic={'omega_c':oc,'omega_b':0.045,'omega_k':0.0,'omega_nu':0.0,
         'h0':0.67,'sigma_8':p[i_s8],'n_s':0.96,
         'transfer_function':'eisenstein_hu','matter_power_spectrum':'linear'}
    dic.update({'gals_z_b':[0.0,
                            0.5    ,0.7    ,0.9    ,1.1    ,
                            1.7],
                'gals_b_b':[1.0,
                            p[i_b0],p[i_b1],p[i_b2],p[i_b3],
                            1.0]})
    cls=lt.get_prediction(dic)
    return lk(cls)


#Read data and initialize likelihood object
if isim<0 :
    fn="sim_sample/sims/sim_mean.sacc"
    fno="test/chains_mean"
else :
    fn="sim_sample/sims/sim_%03d.sacc"%isim
    fno="test/chains_%03d"%isim
s=sacc.SACC.loadFromHDF(fn)
s.precision=s0.precision
lk=LSSLikelihood(s)


#First sample
p0=np.zeros(npar);
p0[i_om]=0.315*0.67**2; p0[i_s8]=0.8; p0[i_b0]=1.0; p0[i_b1]=1.0; p0[i_b2]=1.0; p0[i_b3]=1.0; 
print "This should be a small number: %lE"%(logprob(p0,lk))


#Setup sampler
nwalkers=100
nsteps_burn=100
nsteps_per_chain=1000
sampler=mc.EnsembleSampler(nwalkers,npar,logprob,args=[lk])


#First sample for each walker
par0=(p0)[None,:]*(1+0.001*np.random.randn(nwalkers,npar))#[:,None]
#Burning phase
print "Burning"
pos,prob,stat=sampler.run_mcmc(par0,nsteps_burn)
sampler.reset()
#Running
print "Running"
sampler.run_mcmc(pos,nsteps_per_chain)
print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))


#Save result and analyze it
#np.save(fno,sampler.chain)
chain=np.load(fno+".npy")
samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
    print l+'= %lE +-'%m+' %lE'%s
#fig=cn.corner(samples,labels=labels,truths=p0)
#plt.show()                 

#!/usr/bin/env python
import numpy as np
import sacc
from desclss import LSSTheory,LSSLikelihood
import emcee as mc
import corner as cn
import matplotlib.pyplot as plt

#Simulation to use
isim=-1

#Parameter order
i_oc=0; i_ob=1; i_hh=2; i_ns=3; i_s8=4;
i_b0=5; i_b1=6; i_b2=7; i_b3=8;
npar=9
labels=['$\\Omega_c$','$\\Omega_b$','$h$','$n_s$','$\\sigma_8$','$b_0$','$b_1$','$b_2$','$b_3$']
       

#Read file with noiseless sim and initialize theory object
s0=sacc.SACC.loadFromHDF("sim_sample/sims/sim_mean.sacc")
lt=LSSTheory("sim_sample/sims/sim_mean.sacc")


#Define log(p). This is just a wrapper around the LSSLikelihood lk
def logprob(p,lk) :
    print p
    dic={'omega_c':p[i_oc],'omega_b':p[i_ob],'omega_k':0.0,'omega_nu':0.0,
         'h0':p[i_hh],'sigma_8':p[i_s8],'n_s':p[i_ns],
         'transfer_function':'bbks','matter_power_spectrum':'linear'}
    dic.update({'gals_z_b':[0.0,0.1,0.3,
                            0.5    ,0.7    ,0.9    ,1.1    ,
                            1.3,1.5,1.7],
                'gals_b_b':[1.0,1.0,1.0,
                            p[i_b0],p[i_b1],p[i_b2],p[i_b3],
                            1.0,1.0,1.0]})
    cls=lt.get_prediction(dic)
    return lk(cls)


#Read data and initialize likelihood object
if isim<0 :
    fn="sim_sample/sims/sim_mean.sacc"
else :
    fn="sim_sample/sims/sim_%03d.sacc"%isim
lk=LSSLikelihood(fn)


#First sample
p0=np.zeros(npar);
p0[i_oc]=0.27; p0[i_ob]=0.045; p0[i_hh]=0.67; p0[i_ns]=0.96; p0[i_s8]=0.8;
p0[i_b0]=1.0; p0[i_b1]=1.0; p0[i_b2]=1.0; p0[i_b3]=1.0; 
print "This should be a small number: %lE"%(logprob(p0,lk))


#Setup sampler
nwalkers=28
nsteps_burn=100
nsteps_per_chain=100
sampler=mc.EnsembleSampler(nwalkers,npar,logprob,args=[lk])


#First sample for each walker
par0=(p0)[None,:]*(1+0.001*np.random.randn(nwalkers))[:,None]
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
np.save("chain",sampler.chain)
chain=np.load("chain.npy")
samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
    print l+'= %lE +-'%m+' %lE'%s
fig=cn.corner(samples,labels=labels,truths=p0)
plt.show()                 

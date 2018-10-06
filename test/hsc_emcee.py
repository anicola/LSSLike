#!/usr/bin/env python
import sys
import numpy as np
import sacc
from desclss import LSSTheory,LSSLikelihood
import emcee as mc
from ParamVec import ParamVec
import scipy.optimize
import matplotlib.pyplot as plt


#Simulation to use
if len(sys.argv)<2 :
    print ("Usage ./hsc_emcee.py saccfiles")
    exit(1)


saccs=[sacc.SACC.loadFromHDF(fn) for fn in sys.argv[1:]]
## let's rename exp_sample, because into hscgals, because these are the same gals
for s in saccs:
    assert (len(s.tracers)==len(saccs[0].tracers)) ## Let's require the same number of bins
    for t in s.tracers:
        t.name=t.exp_sample
        t.exp_sample='hscgals'
        #plt.plot(t.z,t.Nz)
        #plt.show()

Ntomo=len(saccs[0].tracers)
# lmax as a function of sample
if Ntomo==1:
    ## distance to 0.6 = 1500 Mpc/h
    ## lmax = 0.3Mpc/h*1500 so 450??

    lmax=[5000]
    lmin=[0]
elif Ntomo==4:
    lmax=[1000,2000,3000,4000]
    lmin=[0,0,0,0]
else:
    print ("weird Ntomo")
    stop()

for s in saccs:
    s.cullLminLmax(lmin,lmax)

print ("Loaded %i sacc files."%len(saccs))
lts=[LSSTheory(s) for s in saccs]

        
#Parameter order
P=ParamVec()
#P.addParam('Oc',0.25,'$\\Omega_c$',min=0.2,max=0.4)

# biases
zbias=[0.0,0.5,1.0,2.0,4.0]
[P.addParam('b_%i'%i,0.7+z/2,min=0.5,max=5) for i,z in enumerate(zbias)]

fitNoise=True
if fitNoise:
    [P.addParam('Pw_%i'%i,0.75,min=0.0,max=10) for i in range(Ntomo)] ## in units of 1e-8

addPZShifts=False
if addPZShifts:
    [P.addParam('s_%i'%i,0.0,min=-0.5,max=+0.5) for i in range(Ntomo)]


    
#Define log(p). This is just a wrapper around the LSSLikelihood lk
def logprob(p,lks,plot=False) :
    P.setValues(p)
    oc=0.25#P.value('Oc')
    dic={'Omega_c':oc,
         'transfer_function':'eisenstein_hu',
         'matter_power_spectrum':'linear',
         'has_rsd':False,'has_magnification':False}
    dic.update({'hscgals_z_b':zbias,
                'hscgals_b_b':[P.value('b_%i'%i) for i in range(len(zbias))]})
    if fitNoise:
        for i in range(Ntomo):
            dic['Pw_bin%i'%i]=P.value('Pw_%i'%i)*1e-8
        

    if addPZShifts:
        for i in range(Ntomo):
            dic['zshift_bin%i'%i]=P.value('s_%i'%i)
    cls=[lt.getPrediction(dic) for lt in lts]
    #print (cls)
    likes=np.array([lk(cl) for lk,cl in zip(lks,cls)])
    if plot:
        plt.figure()
        clrcy='rgbycmk'
        for i,s in enumerate(saccs):
            print(clrcy[i],i)
        
            print ("plotting ",i)
            #plt.subplot(3,3,i+1)
            s.plot_vector(prediction=cls[i],out_name=None,clr=clrcy[i],lofsf=1.01**i,label=s.tracers[0].name)
            #plt.title(s.tracers[0].name)
        plt.show()
        
    print (p,likes, likes.sum())
    return likes.sum()

P.setValues([0.72598453, 1.51343386, 1.76253033, 1.75832404, 2.70147275, 1.03383443])
P.setValues([0.5 ,       0.5,        2.56030839, 5. ,        5.       ,  0.78089506])
# for lmax=5000
#P.setValues([0.4,        0.5,        1.08235945 ,2.3216078,  1.87988751, 2.39857773, 0.7199586])
# for lmax=2
#P.setValues([0.24789857 ,3.60703621 ,1.92675625, 1.50583288, 0.5,        1.74654988,
# 0.61436667])

#First sample
p0=P.values()
lks=[LSSLikelihood(s) for s in saccs]

print ("These should be a small number: %lE"%(logprob(p0,lks,plot=True)))

runMinimizer=True
if runMinimizer:
    #saccs[0].plot_vector()
    print (P.values(),P.bounds())
    scipy.optimize.minimize(lambda x:-logprob(x,lks),p0,bounds=P.bounds(),method='TNC',options={'eps':1e-3})

else:
    #Setup sampler
    nwalkers=100
    nsteps_burn=100
    nsteps_per_chain=1000
    npar=len(P)
    sampler=mc.EnsembleSampler(nwalkers,npar,logprob,args=[lks])


    #First sample for each walker
    par0=(p0)[None,:]*(1+0.001*np.random.randn(nwalkers,npar))#[:,None]
    #Burning phase
    print ("Burning")
    pos,prob,stat=sampler.run_mcmc(par0,nsteps_burn)
    sampler.reset()
    #Running
    print ("Running")
    sampler.run_mcmc(pos,nsteps_per_chain)
    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))


    #Save result and analyze it
    np.save(fno,sampler.chain)
    chain=np.load(fno+".npy")
    samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
    for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
        print (l+'= %lE +-'%m+' %lE'%s)
    #fig=cn.corner(samples,labels=labels,truths=p0)
    #plt.show()                 

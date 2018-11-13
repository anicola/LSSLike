#!/usr/bin/env python
import sys
import numpy as np
import logging
import scipy.optimize
import matplotlib.pyplot as plt

import sacc
from desclss import LSSTheory,LSSLikelihood

import emcee as mc
from ParamVec import ParamVec


class HSCAnalyze:

    def __init__(self, fnames, lmin='auto', lmax='auto',
                 fitOc=True, Oc=0.25,
                 fits8=True, s8=0.8,
                 fith0=True, h0 = 0.6774,
                 fitBias=False,
                 zbias=[0.0,0.5,1.0,2.0,4.0],
                 bias=[0.7,1.5,1.8,2.0,2.5],
                 fitNoise=False, noise=0.75, ## in units of 1e-8
                 fitPZShifts=False,
                 pzshifts=[0,0,0,0],
                 log=logging.DEBUG):

        if type(log)==logging.Logger:
            self.log=log
        else:
            self.log = logging.getLogger('HSCAnalyze')
            self.log.setLevel(log)
            ch = logging.StreamHandler()
            ch.setLevel(log)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
            print (self.log)
            
        self.saccs=[sacc.SACC.loadFromHDF(fn) for fn in fnames]
        self.log.info ("Loaded %i sacc files."%len(self.saccs))

        self.Ntomo=len(self.saccs[0].tracers) ## number of tomo bins
        self.log.info ("Ntomo bins: %i"%self.Ntomo)

        self.fixnames()
        self.cutLranges(lmin,lmax)
        self.setParametes(fitOc,Oc,fits8,s8,fith0,h0,fitBias,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts)
            
        self.lts=[LSSTheory(s) for s in self.saccs]
        self.lks=[LSSLikelihood(s) for s in self.saccs]

        
    def fixnames(self):
        
        ## let's rename exp_sample, because into hscgals, because these are the same gals
        ## and at the same time also add a "name" field in case we need it later
        for s in self.saccs:
            assert (len(s.tracers)==len(self.saccs[0].tracers)) ## Let's require the same number of bins
            for t in s.tracers:
                t.name=t.exp_sample
                t.exp_sample='hscgals'


    def cutLranges(self,lmin,lmax):
        # lmax as a function of sample
        if lmax=='auto':
            if self.Ntomo==1:
                ## distance to 0.6 = 1500 Mpc/h
                ## lmax = 0.3Mpc/h*1500 so 450??
                self.lmax=[1000]
                self.lmin=[0]
            elif self.Ntomo==4:
                self.lmax=[1000,2000,3000,4000]
                self.lmin=[0,0,0,0]
            else:
                print ("weird Ntomo")
                stop()
        else:
            self.lmin=lmin
            self.lmax=lmax

        for s in self.saccs:
            s.cullLminLmax(self.lmin,self.lmax)

    def setParametes(self,fitOc,Oc,fits8,s8,fith0,h0,fitBias,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts):
        #### set up parameters
        self.fitOc=fitOc
        self.Oc=Oc
        self.fits8=fits8
        self.s8=s8
        self.h0=h0
        self.fith0=fith0
        self.fitBias=fitBias
        self.zbias=zbias
        self.bias=bias
        self.fitNoise=fitNoise
        self.noise=noise
        self.fitPZShifts=fitPZShifts
        self.pzshifts=pzshifts
            
        self.P=ParamVec()
        if self.fitOc:
            self.P.addParam('Oc',Oc,'$\\Omega_c$',min=0.2,max=0.4)
        if self.fits8:
            self.P.addParam('s8',s8, '$\sigma_8$',min=0.1,max=2.0)
        if self.h0:
            self.P.addParam('h0',h0, '$h_0$',min=0.65,max=0.75)
        if self.fitBias:
            for z,b in zip(self.zbias,self.bias):
                self.P.addParam('b_%2.1f'%z,b,min=0.5,max=5) 

        if self.fitNoise:
            for i in range(self.Ntomo):
                self.P.addParam('Pw_%i'%i,noise,min=0.0,max=10)  ## in units of 1e-8

        if self.fitPZShifts:
            for i in range(self.Ntomo):
                self.P.addParam('s_%i'%i,0.0,min=-0.5,max=+0.5)
        self.log.info ("Parameters: "+str(self.P._names))
            

    

    def predictTheory(self,p):
        P=self.P.clone()
        P.setValues(p)
        oc=P.value('Oc') if self.fitOc else self.Oc
        s8=P.value('s8') if self.fits8 else self.s8
        h0=P.value('h0') if self.fith0 else self.h0
        dic={'Omega_c':oc,
         'Omega_b':0.0486,
         'Omega_k':0.0,
         'Omega_nu':0.001436176,
         'h0':h0,
         'n_s':0.96,
         'sigma_8':s8,
         'transfer_function':'eisenstein_hu',
         'matter_power_spectrum':'linear',
         'has_rsd':False,'has_magnification':False}
        if self.fitBias:
            dic.update({'hscgals_z_b':self.zbias,
                        'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
        else:
            dic.update({'hscgals_z_b':self.zbias,
                        'hscgals_b_b':self.bias})
            
        if self.fitNoise:
            for i in range(self.Ntomo):
                dic['Pw_bin%i'%i]=P.value('Pw_%i'%i)*1e-8
        
        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)
        cls=[lt.get_prediction(dic) for lt in self.lts]
        return cls
    
    #Define log(p). This is just a wrapper around the LSSLikelihood lk
    def logprobs(self,p):
        cls=self.predictTheory(p)
        #print (cls)
        likes=np.array([lk(cl) for lk,cl in zip(self.lks,cls)])
        self.log.debug("parameters: "+str(p)+" -> chi2= "+str(-2*likes.sum()))
        return likes

    def logprob(self,p):
        return self.logprobs(p).sum()
    
    def plotDataTheory(self):
        fig = plt.figure()
        subplot=fig.add_subplot(111)
        clrcy='rgbycmk'
        cls=self.predictTheory(self.P.values())
        for i,s in enumerate(self.saccs):
            #plt.subplot(3,3,i+1)
            s.plot_vector(subplot,plot_corr = 'auto',prediction=cls[i],clr=clrcy[i],lofsf=1.01**i,weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=False)
            #plt.title(s.tracers[0].name)
        plt.show()

    def minimize(self):
        scipy.optimize.minimize(lambda x:-self.logprobs(x).sum(),
                                self.P.values(),
                                bounds=self.P.bounds(),
                                method='TNC',options={'eps':1e-3})

    def MCMCSample(self,fno='chain'):
        #Setup sampler
        nwalkers=100
        nsteps_burn=100
        nsteps_per_chain=1000
        npar=len(self.P)
        sampler=mc.EnsembleSampler(nwalkers,npar,self.logprob)

        p0=self.P.values()
        #First sample for each walker
        par0=(p0)[None,:]*(1+0.1*np.random.randn(nwalkers,npar))#[:,None]
        #Burning phase
        self.log.info ("Burning")
        pos,prob,stat=sampler.run_mcmc(par0,nsteps_burn)
        sampler.reset()
        #Running
        self.log.info ("Running")
        sampler.run_mcmc(pos,nsteps_per_chain)
        self.log.info("Mean acceptance fraction: {0:.3f}"
              .format(np.mean(sampler.acceptance_fraction)))


        #Save result and analyze it
        np.save(fno,sampler.chain)
        chain=np.load(fno+".npy")
        samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
        for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
            print (l+'= %lE +-'%m+' %lE'%s)
        #fig=cn.corner(samples,labels=labels,truths=p0)
        #plt.show()                 




if __name__=="__main__":
    #Simulation to use
    if len(sys.argv)<2 :
        print ("Usage ./hsc_driver.py saccfiles")
        exit(1)

    h=HSCAnalyze(sys.argv[1:])
    h.plotDataTheory()
    #h.minimize()
    #h.MCMCSample()

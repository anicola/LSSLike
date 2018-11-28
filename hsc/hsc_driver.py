#!/usr/bin/env python
import sys
import numpy as np
import logging
import scipy.optimize
import matplotlib.pyplot as plt
import pyccl as ccl

import sacc
from desclss import LSSTheory,LSSLikelihood

import emcee as mc
from ParamVec import ParamVec


class HSCAnalyze:

    def __init__(self, fnames, lmin='auto', lmax='auto', kmax=None, zeff=None, cosmo=None,
                 fitOc=True, Oc=0.25,
                 fits8=False, s8=0.8,
                 fith0=False, h0 = 0.6774,
                 fitBias=True, BiasMod='bz',
                 zbias=[0.0,0.5,1.0,2.0,4.0],
                 bias=[0.7,1.5,1.8,2.0,2.5],
                 fitNoise=True, noise=0.75, ## in units of 1e-8
                 fitPZShifts=False,
                 pzshifts=[0,0,0,0],
                 join_saccs=True,   ## If true, Cinverse add all saccs into one
                 cull_cross=True,  ## If true, use just auto
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

        if join_saccs:
            self.saccs=[sacc.coadd(self.saccs)]
        if cull_cross:
            for s in self.saccs:
                s.cullCross()
    
            
        self.Ntomo=len(self.saccs[0].tracers) ## number of tomo bins
        self.log.info ("Ntomo bins: %i"%self.Ntomo)

        self.fixnames()
        self.cutLranges(lmin, lmax, kmax, zeff, cosmo)
        self.setParametes(fitOc,Oc,fits8,s8,fith0,h0,fitBias,BiasMod,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts)
            
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


    def cutLranges(self, lmin, lmax, kmax, zeff, cosmo):
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
        elif lmax == 'kmax':
            assert kmax is not None, 'kmax not provided.'
            assert zeff is not None, 'zeff array not provided.'
            assert self.Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
            self.log.info('Computing lmax according to specified kmax = {}.'.format(kmax))

            self.lmax = self.kmax2lmax(kmax, zeff, cosmo)

            if self.Ntomo == 1:
                self.lmin = [0]
            elif self.Ntomo == 4:
                self.lmin=[0,0,0,0]
            else:
                print ("weird Ntomo")
                stop()

        else:
            self.lmin=lmin
            self.lmax=lmax

        self.log.info('lmin = {}, lmax = {}.'.format(self.lmin, self.lmax))

        for s in self.saccs:
            s.cullLminLmax(self.lmin,self.lmax)

    def kmax2lmax(self, kmax, zeff, cosmo=None):
        """
        Determine lmax corresponding to given kmax at an effective redshift zeff according to
        kmax = (lmax + 1/2)/chi(zeff)
        :param kmax: maximal wavevector in Mpc^-1
        :param zeff: effective redshift of sample
        :return lmax: maximal angular multipole corresponding to kmax
        """

        if cosmo is None:
            self.log.info('CCL cosmology object not supplied. Initializing with Planck 2018 cosmological parameters.')
            cosmo = ccl.Cosmology(n_s=0.9649, A_s=2.1e-9, h=0.6736, Omega_c=0.264, Omega_b=0.0493)

        # Comoving angular diameter distance in Mpc
        chi_A = ccl.comoving_angular_distance(cosmo, 1./(1.+zeff))
        lmax = kmax*chi_A - 1./2.

        return lmax

    def setParametes(self,fitOc,Oc,fits8,s8,fith0,h0,fitBias,BiasMod,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts):
        #### set up parameters
        self.fitOc=fitOc
        self.Oc=Oc
        self.fits8=fits8
        self.s8=s8
        self.h0=h0
        self.fith0=fith0
        self.fitBias=fitBias
        self.BiasMod=BiasMod
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
        if self.fith0:
            self.P.addParam('h0',h0, '$h_0$',min=0.65,max=0.75)

        if self.fitBias:
            if self.BiasMod == 'bz':
                for z,b in zip(self.zbias,self.bias):
                    self.P.addParam('b_%2.1f'%z,b,min=0.5,max=5)
            elif self.BiasMod == 'const':
                for i, b in enumerate(self.bias):
                    self.P.addParam('b_bin%i'%i, b, min=0.5, max=5.)
            else:
                raise ValueError("bias needed")

        if self.fitNoise:
            if not (type(noise)==list):
                noise=[noise]*self.Ntomo
            for i in range(self.Ntomo):
                self.P.addParam('Pw_%i'%i,noise[i],min=0.0,max=10)  ## in units of 1e-8

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
         'matter_power_spectrum':'halofit',
         'has_rsd':False,'has_magnification':None}
        if self.fitBias:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
            else:
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = P.value('b_bin%i'%i)
        else:
            dic.update({'hscgals_z_b':self.zbias,
                        'hscgals_b_b':self.bias})
            
        if self.fitNoise:
            for i in range(self.Ntomo):
                dic['Pw_bin%i'%i]=P.value('Pw_%i'%i)*1e-8
        
        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)
        print(dic)
        cls=[lt.get_prediction(dic) for lt in self.lts]
        return cls

    def predictNoSNTheory(self,p):
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
         'matter_power_spectrum':'halofit',
         'has_rsd':False,'has_magnification':None}
        if self.fitBias:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
            else:
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = P.value('b_bin%i'%i)
        else:
            dic.update({'hscgals_z_b':self.zbias,
                        'hscgals_b_b':self.bias})

        if self.fitNoise:
            for i in range(self.Ntomo):
                dic['Pw_bin%i'%i]=P.value('Pw_%i'%i)*1e-8

        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)
        cls=[lt.get_noSN_prediction(dic) for lt in self.lts]

        return cls
    
    #Define log(p). This is just a wrapper around the LSSLikelihood lk
    def logprobs(self,p):
        cls=self.predictTheory(p)
        #print (cls)
        likes=np.array([lk(cl) for lk,cl in zip(self.lks,cls)])
        dof=np.array([len(cl) for cl in cls])
        self.log.debug("parameters: "+str(p)+" -> chi2= "+str(-2*likes.sum())+" dof= "+str(dof.sum()))
        return likes

    def logprob(self,p):
        return self.logprobs(p).sum()
    
    def plotDataTheory(self, params=None, path2fig=None):

        P = self.P.clone()
        if params is not None:
            P.setValues(params)

        fig = plt.figure()
        subplot=fig.add_subplot(111)
        clrcy='rgbycmk'
        self.log.info('Best-fit parameters = {}.'.format(P.values()))
        cls=self.predictTheory(P.values())

        cls_sn_rem = self.predictNoSNTheory(P.values())

        for i,s in enumerate(self.saccs):
            #plt.subplot(3,3,i+1)
            s.plot_vector(subplot,plot_corr = 'auto',prediction=cls[i],clr=clrcy[i],lofsf=1.01**i,weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=False)
            s.plot_vector(subplot,plot_corr = 'auto',prediction=cls_sn_rem[i],clr=clrcy[2],lofsf=1.01**i,weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=False)
            #plt.title(s.tracers[0].name)
        if path2fig is not None:
            plt.savefig(path2fig)
        plt.show()


    def plot_vector (self, sacc, subplot = None, plot_corr='all', weightpow = 2, set_logx=True, set_logy=True,
                    show_axislabels = False, show_legend=True, prediction=None, clr='r', lofsf=1.0, label=None):
        """
        Plots the mean vector associated to the different tracers
        in the sacc file. The tracer correlations to plot can be selected by
        passing a list of pairs of values in plot_corr.  It can also plot
        the autocorrelation by passing 'auto', the cross-correlation by
        passng 'cross', and both by passing 'all'.  The C_ell's will be
        weighted by a factor of ell^{weightpow}.
        """
        import matplotlib.pyplot as plt
        import matplotlib


        if subplot is None:
            fig = plt.figure()
            subplot = fig.add_subplot(111)

        if sacc.precision is not None:
            errs=np.sqrt(sacc.precision.getCovarianceMatrix().diagonal())
        else:
            errs=None

        plot_cross = False
        plot_auto = False
        plot_pairs = []

        if plot_corr == 'all':
            # Plot the auto-correlation and the cross-correlation
            plot_cross = True
            plot_auto = True
        elif plot_corr == 'cross':
            # Plot ALL cross-correlations only
            plot_cross = True
        elif plot_corr == 'auto':
            # Plot the auto-correlation only
            plot_auto = True
        elif hasattr(plot_corr, '__iter__'):
            plot_pairs = plot_corr
        else:
            print('plot_corr needs to be \'all\', \'auto\',\'cross\', or a list of pairs of values.')

        tracer_array = np.arange(len(sacc.tracers))
        if plot_cross:
            for tr_i in tracer_array:
                other_tr = np.delete(tracer_array, np.where(tracer_array != tr_i))
                for tr_j in other_tr:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_i, tr_j])

        if plot_auto:
            for tr_i in tracer_array:
                plot_pairs.append([tr_i, tr_i])

        plot_pairs = np.array(plot_pairs)

        ###################################
        # Plotting routines below this line
        ###################################

        npairs = len(tracer_array)
        cmap1 = matplotlib.cm.get_cmap('plasma')
        colors = [cmap1(i) for i in np.linspace(0, 0.9, npairs)]

        i = 0
        for (tr_i, tr_j) in plot_pairs:
            tbin = np.logical_and(sacc.binning.binar['T1']==tr_i,sacc.binning.binar['T2']==tr_j)
            ell = sacc.binning.binar['ls'][tbin]
            C_ell = sacc.mean.vector[tbin]

            subplot.plot(ell,C_ell * np.power(ell,weightpow),color=colors[i])
            if errs is not None:
                subplot.errorbar(ell,C_ell * np.power(ell,weightpow),yerr=errs[tbin]*np.power(ell,weightpow), linestyle = 'None',color=colors[i])
            subplot.plot(ell,C_ell * np.power(ell,weightpow),linestyle='None', marker='o', markeredgecolor = colors[i], color = colors[i],
                label= sacc.tracers[0].exp_sample+' $C_{%i%i}$' %(tr_i,tr_j))
            i += 1

        if set_logx:
            subplot.set_xscale('log')
        if set_logy:
            subplot.set_yscale('log')
        if show_axislabels:
            subplot.set_xlabel(r'$l$')
            if weightpow == 0:
                elltext = ''
            elif weightpow == 1:
                elltext = r'$\ell$'
            else:
                elltext = r'$\ell^' + '{%i}$' % weightpow
            subplot.set_ylabel(elltext + r'$C_{l}$')
        if show_legend:
            subplot.legend(loc='best')

    def plotData(self, path2fig=None):

        fig = plt.figure()
        subplot=fig.add_subplot(111)
        clrcy='rgbycmk'

        for i,s in enumerate(self.saccs):
            #plt.subplot(3,3,i+1)
            self.plot_vector(s, subplot,plot_corr = 'auto',clr=clrcy[i],lofsf=1.01**i,weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=True)
            #plt.title(s.tracers[0].name)
        if path2fig is not None:
            plt.savefig(path2fig)
        plt.show()

    def minimize(self):

        res = scipy.optimize.minimize(lambda x:-self.logprobs(x).sum(),
                                self.P.values(),
                                bounds=self.P.bounds(),
                                method='TNC',options={'eps':1e-3, 'disp':True})

        return res

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

    if False:
        h=HSCAnalyze(sys.argv[1:], Oc=0.4,   bias=[1.36719745,  1.44648823,  1.58481978,  2.78524986,  2.48624793],
                 fitNoise=True, noise=[10.,          5.36774973,  4.14486325,  2.7542516])

        h.plotDataTheory()
    #
    # h=HSCAnalyze(sys.argv[1:], lmax='kmax', lmin='kmax', kmax=0.15, cosmo=None, BiasMod='const', bias=[0.7,1.5,1.8,2.0], \
    #              zeff=np.array([0.57, 0.70, 0.92, 1.25]))
    h=HSCAnalyze(sys.argv[1:], BiasMod='const', bias=[0.7,1.5,1.8,2.0])
    # res = h.minimize()
    # h.log.info('Optimizer message {}.'.format(res.message))
    # h.log.info('Minimum found at {}.'.format(res.x))
    # h.log.info('No of iterations {}.'.format(res.nit))
    # h.plotDataTheory(params=res.x, path2fig='/Users/Andrina/Documents/WORK/HSC-LSS/plots/spectra_eab_best_pzb4bins_bpw200_covdata_nocont/cls_data-theory+SN-rem_mPk=halofit_bz=const_Ntomo=4_kmax=0.15.pdf')
    h.plotData(path2fig='/Users/Andrina/Documents/WORK/HSC-LSS/plots/spectra_eab_best_pzb4bins_bpw200_covdata_nocont/cls_data_Ntomo=4_lmax=default.pdf')
    #h.MCMCSample()

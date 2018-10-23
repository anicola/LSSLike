#!/usr/bin/env python

import sys
import numpy as np
import sacc
import matplotlib.pyplot as plt
import scipy.linalg as la
from  scipy.stats import chi2 as chi2d
from copy import deepcopy

def main():

    fnames = []
    Lmax = -1
    crosscor = False
    for argnum in range(1, len(sys.argv)):
        if '--Lmax=' in sys.argv[argnum]:
            Lmax = int(sys.argv[argnum].split('--Lmax=')[-1])
        elif '--crosscorr' in sys.argv[argnum]:
            crosscor = True
        else:
            fnames.append(sys.argv[argnum])

    if Lmax == -1:
        Lmax = 200000

    if len(fnames)<2:
        print ("Specify at least two files on input")
        sys.exit(1)
    saccsin=[[print ("Loading %s..."%fn),
           sacc.SACC.loadFromHDF(fn)].pop() for fn in fnames]

    Ntomo=len(saccsin[0].tracers)
    
    fig = plt.figure()
    splist = []

    if Ntomo==4:
        Nx=Ny=2
    elif Ntomo==1:
        Nx=Ny=1

    for itomo in range(-1,Ntomo):
        if itomo>=0:
            print ("Tomographic bin:",itomo)
        else:
            print ("All bins together")
        saccs=[deepcopy(s) for s in saccsin]
        if itomo<0:
            lmin=[0]*Ntomo
            lmax=[Lmax]*Ntomo
        else:
            lmin=[100000]*Ntomo
            lmax=[Lmax]*Ntomo
            lmin[itomo]=0
            
        for s in saccs:
            s.cullLminLmax(lmin,lmax)
            
        mvecs=[s.mean.vector for s in saccs]
        pmats=[s.precision.getPrecisionMatrix() for s in saccs]
        cmats=[s.precision.getCovarianceMatrix() for s in saccs]
        sumws=np.array([np.dot(p,v) for p,v in zip(pmats,mvecs)]).sum(axis=0)
        sumicov=np.array(pmats).sum(axis=0)
        sumcov=la.inv(sumicov)
        mean=np.dot(sumcov,sumws)

        ## Sk tests.
        chi2=[]
        dof=len(mean)

        for m,c,s in zip(mvecs,cmats,saccs):
            diff=(m-mean)
            C=c-sumcov
            chi2=np.dot(diff,np.dot(la.inv(C),diff))
            print 
            print ("{:20s} {:7.2f} {:7.2f} {:7.4f} ".format(s.tracers[0].exp_sample.replace("'","").replace("b'",""),chi2,dof,1-chi2d(df=dof).cdf(chi2)))

        if (itomo>=0):
            if not crosscor:
                sp = fig.add_subplot(Nx,Ny,itomo+1)
                plotDataTheory_autocor(saccs,mean)
                clrcy='rgbycmk'
                for (i,s) in enumerate(saccs):
                    sp.text(0.98, 0.98 - (0.06*i), s.tracers[0].exp_sample, fontsize = 6, color = clrcy[i], transform = sp.transAxes, ha = 'right', va = 'top')
    if crosscor:
        for (i,s) in enumerate(saccs):
            fig = plt.figure()
            sp = fig.add_subplot(111)
            plotDataTheory_crosscor(s)
            clrcy='rgbycmk'
            sp.text(0.98, 0.98 - (0.06*i), s.tracers[0].exp_sample, fontsize = 6, color = clrcy[i], transform = sp.transAxes, ha = 'right', va = 'top')

                

    plt.show()
    
def plotDataTheory_autocor (saccs,mean):
    clrcy='rgbycmk'
    for i,s in enumerate(saccs):
        s.plot_vector(out_name=None,clr=clrcy[i],lofsf=1.01**i,
                      label=s.tracers[0].exp_sample, show_legend=False)
    els=saccs[0].binning.binar['ls']
    plt.plot(els,mean,'k-',lw=2)


def plotDataTheory_crosscor(saccs):
    clrcy='rgbycmk'
    for i,s in enumerate(saccs):
        s.plot_vector(out_name=None,clr=clrcy[i],lofsf=1.01**i, plot_cross = True,
                      label=s.tracers[0].exp_sample, show_legend=False)




if __name__=="__main__":
    main()
    

    


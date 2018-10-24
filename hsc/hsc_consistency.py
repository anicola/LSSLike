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
    savefig = False
    figsize = (7,7)
    for argnum in range(1, len(sys.argv)):
        if '--Lmax=' in sys.argv[argnum]:
            Lmax = int(sys.argv[argnum].split('--Lmax=')[-1])
        elif '--savefp=' in sys.argv[argnum]:
            savefig = True
            figname = sys.argv[argnum].split('--savefp=')[-1]
        elif '--bigfig' in sys.argv[argnum]:
            figsize = (14,14)
        else:
            fnames.append(sys.argv[argnum])
    
    surveynames = [f.split('/')[-2] for f in fnames]

    if Lmax == -1:
        Lmax = 200000

    if len(fnames)<2 and not crosscorr:
        print ("Specify at least two files on input")
        sys.exit(1)
    saccsin=[[print ("Loading %s..."%fn),
           sacc.SACC.loadFromHDF(fn)].pop() for fn in fnames]

    Ntomo=len(saccsin[0].tracers)

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

    fig, splist = plt.subplots(Ntomo,Ntomo, figsize = figsize)
    clrcy='rgbycmk'
    splist = np.array(splist).T.tolist()
    for (i,s) in enumerate(saccsin):
        for x in range(len(splist)):
            for y in range(x, len(splist[x])):
                s.plot_vector(splist[x][y], plot_corr = [[x, y]], clr=clrcy[i],lofsf=1.01**i,
                              label=surveynames[i], show_legend = False, show_axislabels = False)
                splist[x][y].set_ylim(10**-9, 10**-5)
                if x==0 and y!=len(splist[x])-1:
                    splist[x][y].set_xticklabels([])
                if x!=0:
                    splist[x][y].set_yticklabels([])
                if x!=y:
                    splist[y][x].set_visible(False)
                fig.text(0.9, 0.9-(i*0.05), surveynames[i], fontsize = 18, color = clrcy[i], ha = 'right', va = 'top')
                splist[x][y].text(0.98, 0.98, '$C_{%i%i}$' % (x,y), ha = 'right', va = 'top', fontsize = 18, transform = splist[x][y].transAxes)
    plt.subplots_adjust(wspace = 0, hspace = 0, top = 0.97, right = 0.97)
    fig.text(0.5, 0.0, r'$\ell$', fontsize = 18)
    fig.text(0.04, 0.5, r'$C_\ell$', fontsize = 18, ha = 'center', va = 'center', rotation = 'vertical')
    if savefig:
        plt.savefig(figname, bbox_inches = 'tight')
    else:
        plt.show()


if __name__=="__main__":
    main()
    

    


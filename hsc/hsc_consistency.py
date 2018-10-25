#!/usr/bin/env python

import sys
import numpy as np
import sacc
import matplotlib.pyplot as plt
import scipy.linalg as la
from  scipy.stats import chi2 as chi2d
from copy import deepcopy
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('fnames', help='Files to be included in the power spectrum plot.', nargs='+')
    parser.add_argument('--Lmax', type = int, default = 200000,
        help = 'Set the maximum ell for which the power spectrum will be calculated')
    parser.add_argument('--savefp', type = str, default = None, 
        help = 'Filepath for the saved figure.  Will not produce a figure on screen.')
    parser.add_argument('--bigfig', action = 'store_true', 
        help = 'Plots a much larger figure, which can be used to help differentiate the lines.')
    parser.add_argument('--weightpow', type = int, default = 2, choices = [1,2], 
        help = r'The power of \ell by which the C_\ell values will be weighted (must be 1 or 2).')

    args = parser.parse_args()

    if not args.savefp == None:
        savefig = True
    else:
        savefig = False

    if args.bigfig:
        figsize = (14,14)
    else:
        figsize = (7,7)

    surveynames = [f.split('/')[-2] for f in args.fnames]

    if len(args.fnames)<2 and not crosscorr:
        print ("Specify at least two files on input")
        sys.exit(1)
    saccsin=[[print ("Loading %s..."%fn),
           sacc.SACC.loadFromHDF(fn)].pop() for fn in args.fnames]

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
            lmax=[args.Lmax]*Ntomo
        else:
            lmin=[100000]*Ntomo
            lmax=[args.Lmax]*Ntomo
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

    fig, splist = plt.subplots(Ntomo, Ntomo, figsize = figsize)
    clrcy='rgbycmk'
    splist = np.array(splist).T.tolist()
    for (i,s) in enumerate(saccsin):
        for (x,sp_col) in enumerate(splist):
            for (y,sp_xy) in enumerate(sp_col):
                s.plot_vector(sp_xy, plot_corr = [[x, y]], weightpow = args.weightpow, clr=clrcy[i],lofsf=1.01**i,
                              label=surveynames[i], show_legend = False, show_axislabels = False, 
                              set_logx=True, set_logy = False)
                # sp_xy.set_ylim(10**-9, 10**-5)
                if x==0 and y!=len(sp_col)-1:
                    sp_xy.set_xticklabels([])
                if x!=0:
                    sp_xy.set_yticklabels([])
                if x>y:
                    sp_xy.set_visible(False)
                fig.text(0.9, 0.9-(i*0.05), surveynames[i], fontsize = 18, color = clrcy[i], ha = 'right', va = 'top')
                sp_xy.text(0.02, 0.02, '$C_{%i%i}$' % (x,y), ha = 'left', va = 'bottom', fontsize = 18, transform = sp_xy.transAxes)
    plt.subplots_adjust(wspace = 0, hspace = 0, top = 0.97, right = 0.97)
    fig.text(0.5, 0.0, r'$\ell$', fontsize = 18)
    if args.weightpow == 1:
        elltext = r'$\ell\,$'
    elif args.weightpow == 2:
        elltext = r'$\ell^2\,$'
    fig.text(0.04, 0.5, elltext + r'$C_\ell$', fontsize = 18, ha = 'center', va = 'center', rotation = 'vertical')
    if savefig:
        plt.savefig(args.savefp, bbox_inches = 'tight')
    else:
        plt.show()


if __name__=="__main__":
    main()
    

    


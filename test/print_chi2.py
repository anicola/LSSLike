#!/usr/bin/env python
import sacc
from  desclss import *
from scipy.stats import chi2

true_pred=sacc.SACC.loadFromHDF("inputs/fastcats/GaussPZ_0.02+FullSky+run0+ztrue_mean.sacc")
true_vec=true_pred.mean.vector
Ndof=true_pred.size()
precision_fn="inputs/fastcats/GaussPZ_0.02+FullSky+run0+ztrue_mean.sacc"

for u in range(1,10):
    fn="inputs/fastcats/GaussPZ_0.02+FullSky+run%i+ztrue/twopoints%i_ns2048.sacc"%(u,u)
    print "fn=",fn
    t=LSSLikelihood(sacc.SACC.loadFromHDF(fn, precision_filename=precision_fn))
    ch2=t.chi2(true_vec)
    print u,"chi2=",ch2,"/",Ndof,"p=",chi2.cdf(ch2,Ndof)

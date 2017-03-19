#!/usr/bin/env python
import sacc
from  desclss import *
from scipy.stats import chi2

true_pred=sacc.SACC.loadFromHDF("sim_sample/sims/sim_mean.sacc")
true_vec=true_pred.mean.vector
Ndof=true_pred.size()
precision_fn="sim_sample/sims/sim_mean.sacc"

for u in range(10):
    t=LSSLikelihood(sacc.SACC.loadFromHDF("sim_sample/sims/sim_%03d.sacc"%u,
                                          precision_filename=precision_fn))
    ch2=t.chi2(true_vec)
    print u,"chi2=",ch2,"/",Ndof,"p=",chi2.cdf(ch2,Ndof)

#!/usr/bin/env python
import sys
from  scipy.stats import chi2
ch2m=float(sys.argv[1])
dof=int(sys.argv[2])
o=chi2(df=dof)
print "dof    =",dof
print "chi2   =",ch2m
print "Prob   =",o.cdf(ch2m)
print "1-Prob =",1-o.cdf(ch2m)
print "5%,95% =", o.isf(0.95), o.isf(0.05)
print "1%,99% =", o.isf(0.99), o.isf(0.01)

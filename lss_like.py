import sacc
import numpy as np

#Not implemented:
# - Include ell-cuts?
# - Include cuts in the pairs of tracers to correlate?
# - Ability to return simulated scatter

class LSSLikelihood(object):
    def __init__(self,sacc_filename) :
        self.s=sacc.SACC.loadFromHDF(sacc_filename)
        if self.s.precision==None :
            raise ValueError("Precision matrix needed!")
        if self.s.mean==None :
            raise ValueError("Mean vector needed!")

    #We're assuming data_theory will come in the form of a sacc.Means object
    def __call__(self,theory_vec) :
        delta=theory_vec - self.s.mean.data['value']
        chi2=np.einsum('i,ij,j',delta,self.s.precision,delta)
        
        return -0.5*chi2

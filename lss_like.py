import sacc
import numpy as np

#Not implemented:
# - Include ell-cuts?
# - Include cuts in the pairs of tracers to correlate?
# - Ability to return simulated scatter

class LSSLikelihood(object):
    def __init__(self,sacc_filename) :
        s=sacc.SACC.loadFromHDF(sacc_filename)
        if s.precision==None :
            raise ValueError("Precision matrix needed!")
        if s.mean==None :
            raise ValueError("Mean vector needed!")
        self.data_prec=s.precision.matrix
        self.data_means=s.mean
        self.data_vector=np.concatenate([m.data['value'] for m in self.data_means])

    #We're assuming data_theory will come in the form of a sacc.Means object
    def __call__(self,data_theory) :
        theory_vector=np.concatenate([m.data['value'] for m in data_theory])
        delta=theory_vector - self.data_vector;
        chi2=np.einsum('i,ij,j',delta,self.data_prec,delta)
        
        return -0.5*chi2

import sacc
import numpy as np

#Not implemented:
# - Include ell-cuts?
# - Include cuts in the pairs of tracers to correlate?
# - Ability to return simulated scatter

class LSSLikelihood(object):
    def __init__(self,data_means,data_prec) :
        self.data_prec=data_prec
        self.n_data=np.sum([len(m.data['ls']) for m in data_means])
        if self.n_data != len(self.prec) :
            raise ValueError("Data and precision matrix must have the same size")
        self.data_means=data_means
        self.data_vector=np.concatenat([m.data['value'] for m in data_means])

    #We're assuming data_theory will come in the form of a sacc.Means object
    def __call__(self,data_theory) :
        theory_vector=np.concatenate([m.data['value'] for m in data_theory])
        delta=theory_vector - self.data_vector;
        chi2=np.einsum('i,ij,j',delta,self.data_prec,delta)
        
        return -0.5*chi2

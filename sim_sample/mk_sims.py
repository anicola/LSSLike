import numpy as np
import scipy.linalg as la
import pyccl as ccl
import matplotlib.pyplot as plt
import copy
import sacc

## settings


lmin=10
lmax=500
dl=10
zbins=[0.5,0.7,0.9,1.1]
zbin_size=0.1

def main():

    cosmo=ccl.Cosmology(ccl.Parameters(Omega_c=0.27,Omega_b=0.045,h=0.67,sigma8=0.8,n_s=0.96,),
                        transfer_function='bbks',matter_power_spectrum='linear')
    tracers,cltracers=getTracers(cosmo)
    binning=getBinning(tracers)
    binning_sacc=sacc.SACC(tracers,binning)
    theories=getTheories(cosmo,binning_sacc,cltracers)
    mean=getTheoryVec(binning_sacc,theories)
    precision, covmatrix=getPrecisionMatrix(binning_sacc,theories)
    chol=la.cholesky(covmatrix)
    csacc=sacc.SACC(tracers,binning,mean,precision)
    csacc.printInfo()
    ## generate mean sim
    generate_sim("sims/sim_mean.sacc",csacc,add_random=False,store_precision=True, cholesky=chol)
    #Generate file containing only noiseless realization and precision matrix
    nsim=10
    for i in np.arange(nsim) :
        generate_sim("sims/sim_%03d.sacc"%i,csacc, cholesky=chol)

    

def getTracers(cosmo):
    #Create SACC tracers and corresponding CCL tracers
    tracers=[]
    cltracers=[]
    for i,z in enumerate(zbins):
        zar=np.arange(z-3*zbin_size,z+3*zbin_size,0.001)
        Nz=np.exp(-(z-zar)**2/(2*zbin_size**2))
        T=sacc.Tracer("des_gals_"+str(i),"point",zar,Nz,exp_sample="gals",
                      Nz_sigma_logmean=0.01, Nz_sigma_logwidth=0.1)
        bias=np.ones_like(zar)
        T.addColumns({'b':bias})
        tracers.append(T)
        cltracers.append(ccl.ClTracerNumberCounts(cosmo,False,False,zar,Nz,zar,bias))
    return tracers, cltracers

def getBinning(tracers):
    larr=np.arange(lmin,lmax,dl)
    binning=[]
    #Array of ell values
    typ,ell,t1,q1,t2,q2=[],[],[],[],[],[]
    ntr=len(tracers)
    for i1 in np.arange(ntr) :
        for i2 in np.arange(ntr-i1)+i1 :
            for l in larr:
                typ.append('F')
                ell.append(l)
                t1.append(i1)
                t2.append(i2)
                q1.append('P')
                q2.append('P')
    return sacc.Binning(typ,ell,t1,q1,t2,q2)


def getTheories(ccl_cosmo,s,ctracers):
    theo={}
    for t1i,t2i,ells,_ in s.sortTracers():
        cls=ccl.angular_cl(ccl_cosmo,ctracers[t1i],ctracers[t2i],ells)
        theo[(t1i,t2i)]=cls
        theo[(t2i,t1i)]=cls
    return theo

def getTheoryVec(s, cls_theory):
    vec=np.zeros((s.size(),))
    for t1i,t2i,ells,ndx in s.sortTracers():
        vec[ndx]=cls_theory[(t1i,t2i)]
    return sacc.MeanVec(vec)

def getPrecisionMatrix(s,cls_theory):
    
    #Compute theory covariance matrix
    fsky=0.4
    Np=s.size()
    Nt=len(s.tracers)
    covar_theory=np.zeros((Np,Np))
    for a in np.arange(Nt):
        for b in np.arange(a,Nt):
            for c in np.arange(Nt):
                for d in np.arange(c,Nt):
                    c_ac=cls_theory[(a,c)]
                    c_ad=cls_theory[(a,d)]
                    c_bc=cls_theory[(b,c)]
                    c_bd=cls_theory[(b,d)]
                    cdiag=(c_ac*c_bd+c_ad*c_bc)/(fsky*(2.*s.lrange(a,b)+1.)*dl) ## we happen to have identical lranges everywhere
                    for l1,l2 in zip(s.ilrange(a,b),s.ilrange(c,d)):
                        covar_theory[l1,l2]=cdiag
                        covar_theory[l2,l1]=cdiag

    precision=la.inv(covar_theory)
    return sacc.Precision(precision,mode='ell_block_diagonal', binning=s.binning), covar_theory



#Generates simulated cls and stores them into a SACC file
def generate_sim(fname,s,add_random=True,store_precision=False, cholesky=None) :
    msacc=copy.deepcopy(s)                    
    if add_random :
        msacc.mean.vector+=np.dot(cholesky,np.random.normal(0.,1.,msacc.size()))
    msacc.saveToHDF(fname,save_precision=store_precision)


if __name__=="__main__":
    main()
    
                        
                        

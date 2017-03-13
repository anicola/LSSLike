import numpy as np
import sacc
import scipy.linalg as la
import pyccl as ccl
import matplotlib.pyplot as plt

cosmo=ccl.Cosmology(ccl.Parameters(Omega_c=0.27,Omega_b=0.045,h=0.67,A_s=1E-10,n_s=0.96))

#Create SACC tracers and corresponding CCL tracers
tracers=[]
cltracers=[]
sz=0.1
for i,z in enumerate([0.5,0.7,0.9,1.1]):
    zar=np.arange(z-3*sz,z+3*sz,0.001)
    Nz=np.exp(-(z-zar)**2/(2*sz**2))
    T=sacc.Tracer("des_gals_"+str(i),"point",zar,Nz,exp_sample="des_gals",
                  Nz_sigma_logmean=0.01, Nz_sigma_logwidth=0.1)
    bias=np.ones_like(zar)
    T.addColumns({'b':bias})
    tracers.append(T)
    cltracers.append(ccl.ClTracerNumberCounts(cosmo,False,False,zar,Nz,zar,bias))

#Array of ell values
dl=10
larr=np.arange(10,500,dl)

#This function denotes the flattened index of the pair (i1,i2)
ntr=len(tracers)
def n_of_pair(i1,i2) :
    j=max(i1,i2)
    i=min(i1,i2)
    return j+(i*(2*ntr-i-1))/2

#Compute theory power spectra
ncls=(ntr*(ntr+1))/2
nl=len(larr)
cls_theory=np.zeros([ncls,nl])
for i1 in np.arange(ntr) :
    for i2 in np.arange(ntr-i1)+i1 :
        cls_theory[n_of_pair(i1,i2),:]=ccl.angular_cl(cosmo,cltracers[i1],cltracers[i2],larr)

#Flatten into vector
cls_theory_vec=cls_theory.flatten()

#Compute theory covariance matrix
fsky=0.4
covar_theory=np.zeros([ncls*nl,ncls*nl])
for a in np.arange(ntr) :
    for b in np.arange(ntr-a)+a :
        id_ab=n_of_pair(a,b)
        for c in np.arange(ntr) :
            for d in np.arange(ntr-c)+c :
                id_cd=n_of_pair(c,d)
                c_ac=cls_theory[n_of_pair(a,c)]
                c_ad=cls_theory[n_of_pair(a,d)]
                c_bc=cls_theory[n_of_pair(b,c)]
                c_bd=cls_theory[n_of_pair(b,d)]
                cdiag=np.diag((c_ac*c_bd+c_ad*c_bc)/(fsky*(2.*larr+1.)*dl))
                covar_theory[id_ab*nl:(id_ab+1)*nl,id_cd*nl:(id_cd+1)*nl]=cdiag

#Cholesky decomposition for Gaussian sims
chol=la.cholesky(covar_theory)

#Precision matrix
precision=la.inv(covar_theory)

#Diagonal errors
errors=np.sqrt(np.diag(covar_theory))

#Precision matrix in SACC format
precsacc=sacc.Precision(matrix=precision)

#Generates simulated cls and stores them into a SACC file
def generate_sim(fname,add_random=True,store_precision=False) :
    clsim=cls_theory_vec.copy()
    if add_random :
        clsim+=np.dot(chol,np.random.normal(0.,1.,ncls*nl))
    clsim=np.reshape(clsim,[ncls,nl])

    prec_use=None
    if store_precision :
        prec_use=precsacc

    typ,ell,t1,q1,t2,q2,val,err=[],[],[],[],[],[],[],[]
    for i1 in np.arange(ntr) :
        for i2 in np.arange(ntr-i1)+i1 :
            i12=n_of_pair(i1,i2)
            cls=clsim[i12]
            errs=errors[nl*i12:(i12+1)*nl]
            for i in np.arange(nl) :
                typ.append('F')
                ell.append(larr[i])
                t1.append(i1)
                t2.append(i2)
                q1.append('P')
                q2.append('P')
                val.append(cls[i])
                err.append(errs[i])

    mean=sacc.MeanVec(typ,ell,t1,q1,t2,q2,val,err)
    s=sacc.SACC(tracers,mean,precision=prec_use)
    s.printInfo()
    s.saveToHDF(fname)

#Generate file containing only noiseless realization and precision matrix
nsim=10
generate_sim("sims/sim_mean.sacc",add_random=False,store_precision=True)
for i in np.arange(nsim) :
    generate_sim("sims/sim_%03d.sacc"%i)

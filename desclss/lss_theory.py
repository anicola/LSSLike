import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):
    def __init__(self,data_file, log_normal=True) :
        self.s=sacc.SACC.loadFromHDF(data_file)
        self.log_normal=log_normal
        self.ln_dz=0.05 ## fixed for now for lognorm predictions
        self.ln_kmin=1e-4
        self.ln_kmax=50.
        self.ln_Nk=65536
        
        if self.s.binning==None :
            raise ValueError("Binning needed!")
        # get zmax
        zmax=-1
        for tr in self.s.tracers :
            zmax=max(zmax,tr.z.max())
        self.zmax=zmax
        
    def bias_interpolators(self,dic_par):
        toret=[]
        for tr in self.s.tracers :
            if tr.type == 'point' : 
                try:
                    z_b_arr=dic_par[tr.exp_sample+'_z_b']
                    b_b_arr=dic_par[tr.exp_sample+'_b_b']
                except:
                    raise ValueError("bias needed for each tracer")

                # bias function of the input parameters
                bf=interp1d(z_b_arr,b_b_arr,kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                toret.append(bf)
            else :
                raise ValueError("Only \"point\" tracers supported")
        return toret

    def get_tracers(self,cosmo,dic_par) :
        tracers=self.s.tracers
        if self.log_normal:
            bias_arrs=[np.ones(len(tr.z)) for tr in tracers]
        else:
            bfuncs=self.bias_interpolators(dic_par)
            bias_arrs=[bf(tr.z) for (bf,tr) in zip(bfuncs,tracers)]
            
        ccl_tracers=[ccl.ClTracerNumberCounts(cosmo,False,False,tr.z,tr.Nz,tr.z,b_arr)
                for (tr,b_arr) in zip(tracers,bias_arrs)]
        return ccl_tracers
    
    def get_cosmo(self, dic_par, tracers) :
        omega_c = dic_par['omega_c']
        omega_b = dic_par['omega_b']
        omega_k = dic_par['omega_k']
        omega_nu = dic_par['omega_nu']
        if 'w' in dic_par :
            w = dic_par['w']
        else :
            w = -1.
        if 'wa' in dic_par :
            wa = dic_par['wa']
        else :
            wa = 0.
        h0 = dic_par['h0']
        has_sigma8 = ('sigma_8' in dic_par)
        has_A_s = ('A_s' in dic_par)
        n_s = dic_par['n_s']
        if has_sigma8 and has_A_s:
            raise ValueError("Specifying both sigma8 and A_s: pick one")
        elif has_sigma8:
            sigma8=dic_par['sigma_8']
            params=ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                  Omega_n=omega_nu,w0=w,wa=wa,sigma8=sigma8,n_s=n_s,h=h0)
        elif has_A_s:
            A_s = dic_par['A_s']
            params = ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                    Omega_n=omega_nu,w0=w,wa=wa,A_s=A_s,n_s=n_s,h=h0)
        else:
            raise ValueError("Need either sigma 8 or A_s in pyccl.")


        if not self.log_normal:
            cosmo=ccl.Cosmology(params,
                            transfer_function=dic_par['transfer_function'],
                            matter_power_spectrum=dic_par['matter_power_spectrum'])
        else:

            def fix_pk(karr,pk):
                ## fixes lognormal power spec, based on damonge hocus pocus
                idpos=np.where(pk>0)[0];
                if len(idpos)>0 :
                    idmax=idpos[-1]
                    pk=np.maximum(pk,pk[idmax])
                    w=1./karr**6
                    pk[idmax:]=pk[idmax]*w[idmax:]/w[idmax]
                return pk

            ## actually generate a cosmology for each pair of tracers
            ## master cosmology to pull out matter power spectra
            mcosmo=ccl.Cosmology(params,
                            transfer_function=dic_par['transfer_function'],
                            matter_power_spectrum=dic_par['matter_power_spectrum'])
            bfuncs=self.bias_interpolators(dic_par)
            assert (len(self.s.get_exp_sample_set())==1)
            ks=np.logspace(np.log10(self.ln_kmin),np.log10(self.ln_kmax),self.ln_Nk)
            zs=np.arange(0.,self.zmax+self.ln_dz,self.ln_dz)
            pk_lins=[ccl.linear_matter_power(mcosmo,ks,a) for a in 1./(1+zs)]
            rsm2=dic_par['rsm2']/(params['h']*params['h'])
            lntpk=[]
            for z,pk_lin in zip(zs,pk_lins):
                b=bfuncs[0](z) # here we assume that all tracers have same bias
                pkt=pk_lin*b*b*np.exp(-ks*ks*rsm2)
                r_lin,xi_lin=ccl.pk2xi(ks,pk_lin)
                xi_t=np.exp(xi_lin)-1
                _,pkt=ccl.xi2pk(r_lin,xi_t)
                pkt=fix_pk(ks,pkt)
                lntpk.append(pkt)
                    
            ccl.update_matter_power(mcosmo,ks,1./(1+zs),lntpk,is_linear=True)
            ccl.update_matter_power(mcosmo,ks,1./(1+zs),lntpk,is_linear=False)
        return mcosmo

    def get_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        tr=self.get_tracers(cosmo,dic_par)
        cosmo=self.get_cosmo(dic_par,tr)

        for cc,(i1,i2,ells,ndx) in enumerate(self.s.sortTracers()):
            cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            theory_out[ndx]=cls
            
        return theory_out    

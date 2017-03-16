import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):
    def __init__(self,sacc_filename) :
        self.s=sacc.SACC.loadFromHDF(sacc_filename)
        if self.s.mean==None :
            raise ValueError("Mean vector needed!")

    def get_tracers(self,cosmo,dic_par) :
        tr_out=[]
        for tr in self.s.tracers :
            if tr.type == 'point' :
                z_b_arr=dic_par[tr.exp_sample+'_z_b'] #Should check for the existence of this
                b_b_arr=dic_par[tr.exp_sample+'_b_b'] #Should check for the existence of this
                bf=interp1d(z_b_arr,b_b_arr,kind='linear') #Assuming linear interpolation. Decide on extrapolation.
                b_arr=bf(tr.zNz) #Assuming that tracers have this attribute
                
                #We assume no RSDs
                #We assume no magnification
                #Only linear bias implemented so far
                tr_out.append(ccl.ClTracerNumberCounts(cosmo,False,False,tr.zNz,tr.Nz,tr.zNz,b_arr))
            else :
                raise ValueError("Only \"point\" tracers supported")

        return tr_out

    def get_cosmo(self,dic_par) :
        omega_c = dic_par['omega_c']
        omega_b = dic_par['omega_b']
        omega_k = dic_par['omega_k']
        omega_nu = dic_par['omega_nu']
        if 'w' in dic_par :
            w = dic_par['w']
        else :
            w = -1.
        if 'w' in dic_par :
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
    
        cosmo=ccl.Cosmology(params)
        
        return cosmo

    def get_prediction(self,dic_par) :
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)
        theory_out=np.zeros_like(s.mean.data['value'])
        for i1,i2,ells,ndx in s.sortTracers() :
            #I'm assuming here that m.data['T1'] coincides with the index of that tracer
            #I'm not averaging over ells withing each bin
            cls=ccl.angular_cls(tr[i1],tr[i2],ells)
            theory_output[ndx]=cls

        return theory_out

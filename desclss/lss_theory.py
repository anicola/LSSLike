import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):
    def __init__(self,data_file) :
        self.s=sacc.SACC.loadFromHDF(data_file)
        if self.s.binning==None :
            raise ValueError("Binning needed!")

    def get_tracers(self,cosmo,dic_par) :
        tr_out=[]
        for (tr_index, thistracer) in enumerate(self.s.tracers) :
            if thistracer.type.__contains__('point'):
                try:
                    z_b_arr=dic_par[thistracer.exp_sample+'_z_b']
                    b_b_arr=dic_par[thistracer.exp_sample+'_b_b']
                except:
                    raise ValueError("bias needed for each tracer")

                if 'zshift_bin' + str(tr_index) in dic_par.keys:
                    zbins = thistracer.z + dic_par['zshift_bin' + str(tr_index)]
                else:
                    zbins = thistracer.z
                
                bf=interp1d(z_b_arr,b_b_arr,kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                b_arr=bf(thistracer.z) #Assuming that tracers have this attribute
                tr_out.append(ccl.ClTracerNumberCounts(cosmo,False,False,zbins,thistracer.Nz,zbins,b_arr))
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
                                  #Omega_n=omega_nu,
                                  w0=w,wa=wa,sigma8=sigma8,n_s=n_s,h=h0)
        elif has_A_s:
            A_s = dic_par['A_s']
            params = ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                    #Omega_n=omega_nu,
                                    w0=w,wa=wa,A_s=A_s,n_s=n_s,h=h0)
        else:
            raise ValueError("Need either sigma 8 or A_s in pyccl.")

        if 'transfer_function' in dic_par and 'matter_power_spectrum' in dic_par:
            cosmo=ccl.Cosmology(params,
                                transfer_function=dic_par['transfer_function'],
                                matter_power_spectrum=dic_par['matter_power_spectrum'])
        elif 'transfer_function' in dic_par and not 'matter_power_spectrum' in dic_par:
            cosmo=ccl.Cosmology(params,
                                transfer_function=dic_par['transfer_function'])
        elif 'transfer_function' not in dic_par and 'matter_power_spectrum' in dic_par:
            cosmo=ccl.Cosmology(params,
                                matter_power_spectrum=dic_par['matter_power_spectrum'])
        else:
            cosmo=ccl.Cosmology(params)

        return cosmo

    def get_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)
        for i1,i2,_,ells,ndx in self.s.sortTracers() :
            cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            theory_out[ndx]=cls
            
        return theory_out    

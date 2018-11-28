import numpy as np
import pyccl as ccl
import logging
from scipy.interpolate import interp1d

class LSSTheory(object):

    def __init__(self,sacc, log=logging.INFO):
        if  type(sacc)==str:
            sacc=sacc.SACC.loadFromHDF(sacc)
        self.s=sacc
        if self.s.binning==None :
            raise ValueError("Binning needed!")

        if type(log)==logging.Logger:
            self.log=log
        else:
            self.log = logging.getLogger('LSSTheory')
            self.log.setLevel(log)
            ch = logging.StreamHandler()
            ch.setLevel(log)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
            print (self.log)

    def get_tracers(self,cosmo,dic_par) :
        tr_out=[]
        has_rsd=dic_par.get('has_rds',False)
        has_magnification=dic_par.get('has_magnification',None)
        for (tr_index, thistracer) in enumerate(self.s.tracers) :
            if thistracer.type.__contains__('point'):
                if thistracer.exp_sample+'_b_bin' + str(tr_index) in dic_par:
                    b_b = dic_par[thistracer.exp_sample+'_b_bin' + str(tr_index)]
                    z_b_arr = thistracer.z
                    b_b_arr = b_b*np.ones_like(z_b_arr)
                elif thistracer.exp_sample+'_z_b' in dic_par:
                    z_b_arr=dic_par[thistracer.exp_sample+'_z_b']
                    b_b_arr = dic_par[thistracer.exp_sample+'_b_b']
                    bf=interp1d(z_b_arr,b_b_arr,kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                    b_arr=bf(thistracer.z) #Assuming that tracers have this attribute
                else:
                    raise ValueError("bias needed for each tracer")

                if 'zshift_bin' + str(tr_index) in dic_par:
                    zbins = thistracer.z + dic_par['zshift_bin' + str(tr_index)]
                else:
                    zbins = thistracer.z

                tr_out.append(ccl.NumberCountsTracer(cosmo, has_rsd=dic_par['has_rsd'], dndz=(zbins, thistracer.Nz), \
                                                     bias=(z_b_arr, b_b_arr), mag_bias=dic_par['has_magnification']))
            else :
                raise ValueError("Only \"point\" tracers supported")
        return tr_out

    def get_cosmo(self,dic_par) :
        Omega_c = dic_par.get('Omega_c',0.255)
        Omega_b = dic_par.get('Omega_b', 0.045)
        Omega_k = dic_par.get('Omega_k', 0.0)
        mnu = dic_par.get('mnu', 0.06)
        w  = dic_par.get('w', -1.0)
        wa = dic_par.get('wa', 0.0)
        h0 = dic_par.get('h0', 0.67)
        n_s = dic_par.get('n_s', 0.96)
        has_sigma8 = ('sigma_8' in dic_par)
        has_A_s = ('A_s' in dic_par)

        transfer_function=dic_par.get('transfer_function','boltzmann_class')
        matter_power_spectrum=dic_par.get('matter_power_spectrum','halofit')

        if has_sigma8 and has_A_s:
            raise ValueError("Specifying both sigma8 and A_s: pick one")
        elif has_A_s:
            A_s = dic_par['A_s']
            cosmo=ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k, w0=w, wa=wa, A_s=A_s, n_s=n_s, h=h0,
                                transfer_function=dic_par['transfer_function'],
                                matter_power_spectrum=dic_par['matter_power_spectrum'])

        else:
            sigma8=dic_par.get('sigma_8',0.8)
            cosmo=ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k, w0=w, wa=wa, sigma8=sigma8, n_s=n_s, h=h0,
                                transfer_function=dic_par['transfer_function'],
                                matter_power_spectrum=dic_par['matter_power_spectrum'])

        self.log.info('CCL called with cosmology = {}.'.format(cosmo))

        return cosmo

    def get_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)
        for i1,i2,_,ells,ndx in self.s.sortTracers() :
            cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            if (i1==i2) and ('Pw_bin%i'%i1 in dic_par):
                cls+=dic_par['Pw_bin%i'%i1]
            theory_out[ndx]=cls
            
        return theory_out

    def get_noSN_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)
        for i1,i2,_,ells,ndx in self.s.sortTracers() :
            cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            theory_out[ndx]=cls

        return theory_out

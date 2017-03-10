import numpy as np
import pyccl as ccl
import sacc

def get_cosmo(block) :
    omega_c = block['cosmological_parameters','omega_c']
    omega_b = block['cosmological_parameters','omega_b']
    omega_k = block['cosmological_parameters','omega_k']
    omega_nu = block['cosmological_parameters','omega_nu']
    w = block.get_double('cosmological_parameters','w', -1.0)
    wa = block.get_double('cosmological_parameters','wa', 0.0)
    h0 = block['cosmological_parameters','h0']
    has_sigma8 = block.has_value('cosmological_parameters','sigma_8')
    has_A_s = block.has_value('cosmological_parameters','A_s')
    n_s = block['cosmological_parameters','n_s']
    if has_sigma8 and has_A_s:
        raise ValueError("Specifying both sigma8 and A_s: pick one")
    elif has_sigma8:
        sigma8=block['cosmological_parameters','sigma_8']
        params=ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                  Omega_n=omega_nu,w0=w,wa=wa,sigma8=sigma8,n_s=n_s,h=h0)    
    elif has_A_s:
        A_s = block['cosmological_parameters','A_s']
        params = ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                  Omega_n=omega_nu,w0=w,wa=wa,A_s=A_s,n_s=n_s,h=h0)
    else:
        raise ValueError("Need either sigma 8 or A_s in pyccl.")
    
    cosmo=ccl.Cosmology(params)
    
    return cosmo

def get_nuisance_arrays(block,section,name_n,name_z) :
    z_arr=[]; n_arr=[];

    inode=0
    while block.has_value(section,name_n+'_%d'%inode) :
        n_arr.append(block[section,name_n+'_%d'%inode])
        inode+=1
    inode=0
    while block.has_value(section,name_z+'_%d'%inode) :
        z_arr.append(block[section,name_z+'_%d'%inode])
        inode+=1

    if len(z_arr)!=len(n_arr) :
        raise KeyError("Can't form nodes for nuisance "+section+" "+name_n)

    return z_arr,n_arr

def get_tracers(block,cosmo,tracers) :
    tr_out={}
    for tr in tracers :
        sec_params=tr.exp_sample+'_parameters'
        if tr.type == 'point' :
            z_b_arr,b_b_arr=get_nuisance_arrays(block,sec_params,'b_b','z_b')
            bf=interp1d(z_b_arr,b_b_arr,kind='linear') #We assume linear interpolation everywhere
            b_arr=bf(tr.zNz) #We're assuming that tracers have this attribute

            #We assume no RSDs
            #We assume no magnification
            #Only linear bias implemented so far
            tr_out[tr.name]=ccl.ClTracerNumberCounts(cosmo,False,False,tr.zNz,tr.Nz,tr.zNz,b_arr)
        else :
            raise ValueError("Only \"point\" tracers supported")
                          
def setup(options) :
    #These functions are completely made up
    tracers=sacc.read_means(options['lss_like','data_file'])
    #These functions are completely made up
    means=sacc.read_means(options['lss_like','data_file'])

    return {'tracers': tracers,'means': means}

def execute(block,config) :
    cosmo=get_cosmo(block)
    tr=get_tracers(block,cosmo,config['tracers'])
    theory_out=[]
    for m in config['means'] :
        #I'm assuming here that m.data['T1'] coincides with the name of that tracer
        #I'm not averaging over ells within each bin right now
        cls=ccl.angular_cls(tr[m.data['T1']],tr[m.data['T2']],m.data['ls'])
        #I guess one could use copy here and just fill in value
        theory_out.append(sacc.MeanVec(typ='F',ls=m.data['ls'],
                                       T1=m.data['T1'],Q1=m.data['Q1'],
                                       T2=m.data['T2'],Q2=m.data['Q2'],
                                       value=cls,error=np.zeros_like(cls)))

    block['LSStheory',theory_out]

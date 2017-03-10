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
    tr_out=[]
    for tr in tracers :
        sec_params=tr.exp_sample+'_parameters'
        if tr.type == 'point' :
            z_b_arr,b_b_arr=get_nuisance_arrays(block,sec_params,'b_b','z_b')
            bf=interp1d(z_b_arr,b_b_arr,kind='linear') #We assume linear interpolation everywhere
            b_arr=bf(tr.zNz) #We're assuming that tracers have this attribute

            #We assume no RSDs
            #We assume no magnification
            tr_out.append(ccl.ClTracerNumberCounts(cosmo,False,False

def setup(options) :
    tracers=sacc.read_means(options['lss_like','data_file'])
    means=sacc.read_means(options['lss_like','data_file'])

    return {'tracers': tracers,'means': means}

def execute(block,config)
#....


def ccl_lss_logprior(params):
    """Prototype for LSS priors
    
    Args:
        ccl_params (`obj`:ccl.Parameters) Cosmological parameters of
        the model to compute.
        
    Returns: Value of the prior
    """
    return 0


def ccl_lss_logp_cl(params,ldata,cl_data,cl_cov,z_n,n,
                    has_rsd=True,has_magnification=False,z_n2=None,
                    n2=None,z_s=None,s=None,z_s2=None,s2=None,
                    invert_cov=False,lmfit=True):
    """Prototype for LSS likelihood
    
    Args:
          params (`dict`:Parameters or `obj` lmfit parameters) Cosmological parameters
          of the model to compute.
          cl_data (`np.array`) Data vector containing the angular power-spectrum.
          cl_cov (`np.array`) Covariance matrix.
          z_n (`np.array`) redshift at which dN/dz is evaluated.
          n (`np.array`) values of dN/dz.
          has_rsd (`bool`, optional) if True include RSD.
          has_magnification (`bool`, optional) if True include magnification
          z_n2 (`np.array`, optional) redshift at which dN/dz of the second tracer
          is evaluated, if None then auto-power spectrum is computed.
          n2 (`np.array`) dN/dz for cross-power spectrum calculation, if None
          auto-power spectrum is computed.
          z_b2 (`np.array`, optional) redshift at which b(z) of the second tracer
          is computed, if None, then it will be equal to the one in the first tracer
          b2 (`np.array`, optional) bias of the second tracer, if None it will be
          equal to the first tracer.
          z_s (`np.array`, optional) if has_magnification=True then this is the
          redshift at which the dn/dz of the sources is provided.
          s (`np.array`, optional) if has_magnification=True this is the dn/dz of 
          the sources for magnification.
          z_s2 (`np.array`, optional) same as z_s for the second tracers, if not
          provided, then they will be equal to the z_s.
          s2 (`np.array`, optional) same as s for the second tracers, if not provided
          they will be equal to s.
          invert_cov (`bool`, optional) If True invert the covariance matrix, if False,
          it means that the input is the inverse covariance matrix (faster).
          lmfit (`bool`, optional) If True it assumes that params is a lmfit.Parameters
          object. Requires lmfit.
    Returns:
          logp (`double`) Log-posterior value for the model evaluated
    """
    if lmfit:
        oc = params['Omega_c'].value
        ob = params['Omega_b'].value
        h = params['h'].value
        A_s = params['A_s'].value
        n_s = params['n_s'].value
        ok = params['Omega_k'].value
        on = params['Omega_n'].value
        w0 = params['w0'].value
        wa = params['wa'].value
        b = params['b'].value
        z_b = params['z_b'].value
        if 'b_2' in params.keys():
            b2 = params['b_2'].value
            z_b2 = params['z_b_2'].value
    else:
        oc = params['Omega_c']
        ob = params['Omega_b']
        h = params['h']
        A_s = params['A_s']
        n_s = params['n_s']
        ok = params['Omega_k']
        on = params['Omega_n']
        w0 = params['w0']
        wa = params['wa']
        b = params['b']
        z_b = params['z_b']
        if 'b_2' in params.keys():
            b2 = params['b_2']
            z_b2 = params['z_b_2']
    ccl_params = ccl.Parameters(Omega_c=oc,
                                Omega_b=ob,
                                h=h,A_s=A_s,
                                n_s=n_s,Omega_k=ok,
                               Omega_n=on,w0=w0,
                               wa=wa)
    ccl_cosmo = ccl.Cosmology(ccl_params)
    
    if(z_n2==None):
        z_n2 = z_n
        n2 = n
        b2 = [b]
        z_b2 = [z_b]
        if(has_magnification):
            z_s2 = z_s
            s2 = s
    if('b_2' in params.keys()):
        b2 = [b2]
        z_b2 = [z_b2]

    if(has_magnification):
        cltracer1 = ccl.ClTracerNumberCounts(ccl_cosmo,has_rsd=has_rsd,
                                             has_magnification=has_magnification
                                             ,z_n=z_n,n=n,z_b=z_b
                                             ,b=b,
                                             z_s=z_s,s=s)
        cltracer2 = ccl.ClTracerNumberCounts(ccl_cosmo,has_rsd=has_rsd,
                                         has_magnification=has_magnification
                                         ,z_n=z_n2,n=n2,z_b=z_b2,b=b2,z_s=z_s2,s=s2)
    else:
        cltracer1 = ccl.ClTracerNumberCounts(ccl_cosmo,has_rsd=has_rsd,
                                             has_magnification=has_magnification
                                             ,z_n=z_n,n=n,z_b=z_b,
                                             b=b,
                                             )
        cltracer2 = ccl.ClTracerNumberCounts(ccl_cosmo,has_rsd=has_rsd,
                                         has_magnification=has_magnification
                                         ,z_n=z_n2,n=n2,z_b=z_b2,b=b2)
    
    cl_model = ccl.angular_cl(ccl_cosmo,cltracer1,cltracer1,ldata)
    if(invert_cov):
        covinv = np.linalg.inv(cl_cov)
    else:
        covinv = cl_cov
    #We calculate the log-posterior
    logp = -0.5*(np.linalg.dot(cl_model-cl_data),np.matmult(covinv,(cl_model-cl_data)))+ccl_lss_logprior(params)
    return logp


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

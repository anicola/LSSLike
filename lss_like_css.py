from cosmosis.datablock import option_section, names as section_names
from lss_like import LSSLikelihood
import numpy as np

def setup(options) :
    ll=LSSLikelihood(options['lss_like','data_file'])
    return ll

def execute(block,config) :
    ll=config
    
    like=ll(block['lss_data','theory_means'])
    block[section_names.likelihoods,'lss_like']=like

    return 0

def cleanup(config) :
    return 0

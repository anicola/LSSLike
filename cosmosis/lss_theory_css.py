from cosmosis.datablock import option_section, names as section_names
import numpy as np
from desclss import LSSTheory

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


def setup(options) :
    lt=LSSTheory(options['lss_like','data_file'])
    return lt


def execute(block,config) :
    lssth=config

    #Parse cosmological parameters
    dic_par={}
    dic_par['omega_c']=block['cosmological_parameters','omega_c']
    dic_par['omega_b']=block['cosmological_parameters','omega_b']
    dic_par['omega_k']=block['cosmological_parameters','omega_k']
    dic_par['omega_nu']=block['cosmological_parameters','omega_nu']
    dic_par['w']=block.get_double('cosmological_parameters','w',-1.0)
    dic_par['wa']=block.get_double('cosmological_parameters','wa',-1.0)
    dic_par['h0']=block['cosmological_parameters','h0']
    dic_par['n_s']=block['cosmological_parameters','n_s']
    if block.has_value('cosmological_parameters','sigma_8') :
        dic_par['sigma_8']=block['cosmological_parameters','sigma_8']
    if block.has_value('cosmological_parameters','A_s') :
        dic_par['A_s']=block['cosmological_parameters','A_s']

    #Parse nuisance parameters
    exp_list=[]
    for tr in lssth.tracers :
        if tr.exp_sample in exp_list :
            continue
        else :
            z_arr,bz_arr=get_nuisance_arrays(block,tr.exp_sample+'_parameters','b_b','z_b')
            dic_par[tr.exp_sample+'_z_b']=z_arr
            dic_par[tr.exp_sample+'_b_b']=bz_arr
            exp_list.append(tr.exp_sample)

    #Compute theory
    theory=lssth.get_prediction(dic_par)

    #Store theory
    block['lss_data','theory_means']=theory

    return 0


def cleanup(config) :
    return 0

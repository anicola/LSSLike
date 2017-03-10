#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ccl.h>

//Structure containing all info needed to compute the LSS likelihood
typedef struct {
  int nbins; //Number of redshift bins

  int *num_z; //Number of z-values for the n(z) of each bin (dims -> [nbins])
  double **z_arr; //z-values for the n(z) of each bin       (dims -> [nbins,num_z])
  double **nz_arr; //n(z) of each bin                       (dims -> [nbins,num_z])

  int n_bpw;     //Number of multipole bandpowers
  int *lmin_arr; //Minimum multipole for bandpower (dims->[n_bpw])
  int *lmax_arr; //Minimum multipole for bandpower (dims->[n_bpw])

  double **inv_covar; //Inverse covariance matrix of the power spectrum measurements
                      //(dims->[n_bpw*(nbins*(nbins+1))/2,n_bpw*(nbins*(nbins+1))/2])
                      //Symmetric matrix

  int flag_include_rsd; //Do we include RSDs in the calculation?
  int flag_include_magnification; //Do we include lensing magnification?
  int flag_include_gr; //Do we include relativistic effects?
  int flag_nonlinear_method; //What method to use for non-linearities.
  int flag_bias_method; //What bias prescription to use.
} ParamsLSS;
//Caveats:
// - We assume same bandpowers for all bins
// - We assume single tracer
// - Should we include cosmology dependence on inv_covar?
// - Still missing some structure describing the priors on cosmological parameters.
// ...

//Structure containing all LSS nuisance parameters
typedef struct {
  int n_nodes_bias; //Number of nuisance bias parameters
  double *z_arr_bias; //z-values for the different bias parameters (dims->[n_nodes_bias])
  double *b_arr_bias; //Values for the bias parameters (dims->[n_nodes_bias])
  PriorStruct *p_arr_bias; //Priors for the bias parameters (dims->[n_nodes_bias])
  int flag_interpolate_type_bias; //Interpolation scheme for bias parameters

  int n_nodes_dz; //Number of nuisance photo-z bias parameters
  double *z_arr_dz; //z-values for the different photo-z bias parameters (dims->[n_nodes_dz])
  double *dz_arr_dz; //Values for the photo-z bias parameters (dims->[n_nodes_dz])
  PriorStruct *p_arr_dz; //Priors for the photo-z bias parameters (dims->[n_nodes_dz])
  int flag_interpolate_type_dz; //Interpolation scheme for photo-z bias

  //...
} NuisanceLSS;
//Caveats:
// - Over simplistic example! Tons of other nuisance functions.
// - Non-linear bias? Scale-invariant bias? 
// - Better parametrization of photo-z uncertainties
// - PriorStruct TBD
// ...

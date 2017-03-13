#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ccl.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

//Structure containing info about a model parameter
typedef struct {
  int index;
  double value;
} LSS_param;

//Structure defining one of the tracers being correlated
//Note that each LSS tracer contains many redshift bins
//(whereas a CCL_ClTracer corresponds to a single bin)
typedef struct {
  int n_bins; //Number of redshift bins

  int *num_z; //Number of z-values for the n(z) of each bin (dims -> [nbins])
  double **z_nz_arr; //z-values for the n(z) of each bin    (dims -> [nbins,num_z])
  double **n_nz_arr; //n(z) of each bin                     (dims -> [nbins,num_z])

  int n_nodes_b; //Number of nodes defining b(z)
  double *z_bz_arr; //Redshift values for each node (dims -> [n_nodes_b])
  LSS_param *b_bz_arr; //Bias values for each node  (dims -> [n_nodes_b])
  int interp_scheme_b; //Interpolation scheme for b(z)

  //Add similar stuff for magnification bias, 2nd-order bias etc.,

  //Add similar stuff for photo-z uncertainties?

  //Flags describing theoretical calculation
  int flag_w_rsd; //Include RSDs?
  int flag_w_mag; //Include magnification?
  int non_linear_type; //Scheme for non-linearities?

  //Array of CCL_ClTracers for each redshift bin (stored here to avoid recomputation?)
  CCL_ClTracer *t; // (dims -> [n_bins])
} LSS_tracer_info;


//Structure containing info needed to compute the LSS likelihood
typedef struct {
  //Cosmological parameters
  LSS_param par_Oc;
  LSS_param par_Ob;
  LSS_param par_Ok;
  LSS_param par_h;
  LSS_param par_s8;
  LSS_param par_ns;
  LSS_param par_w0;
  LSS_param par_wa;

  //Tracers to correlate
  LSS_tracer_info *tr1;
  LSS_tracer_info *tr2;

  //Probably a lot of stuff missing here
  //...
} LSS_likelihood_workspace;

//Structure defining a set of power spectra
typedef struct {
  //Number of bins for each tracer
  int nbin_1;
  int nbin_2;

  //Number of bandpowers for each cross-power spectrum
  int **n_bpw; // (dim -> [nbin_1,nbin_2])
  //Effective ell for each bandpower
  double ***l_bpw; // (dim -> [nbin_1,nbin_2,n_bpw[nbin_1,nbin_2]])
  //Effective C_ell for each bandpower
  gsl_vector *cl_bpw; // (flattened [nbin_1,nbin_2,n_bpw[nbin_1,nbin_2]])
  //This will probably have to be more general, with a weight-per-ell
  //array for each bandpower (e.g. to account for pseudo-Cl binning).
} LSS_2pt_Cell;

//Something like the above should be defined for the precision matrix,
//or it could be read directly into a 2D matrix


//////////
// Functions defined in lss_cell_io.c
//
//Read power spectrum
LSS_2pt_Cell *lss_read_cell(char *fname);
//Write power spectrum
void lss_write_cell(char *fname,LSS_2pt_Cell *cl);
//Read precision matrix
gsl_matrix *lss_read_precision(char *fname);
//Write precision matrix
void lss_write_precision(char *fname,gsl_matrix *prec,LSS_2pt_Cell *cl);
//Destructor
void lss_free_cell(LSS_2pt_Cell *cl);

//////////
// Functions defined in lss_tracers.c
//
//Creates new tracer
LSS_tracer_info *lss_tracer_new(int n_bins,
				int *num_z,double **z_nz_arr,double **n_nz_arr,
				int n_nodes_b,double *z_bz_arr,double *b_bz_arr,
				int flag_w_rsd,int flag_w_mag,int non_linear_type);
//Destructor
void lss_tracer_free(LSS_tracer_info *tr);

//////////
// Functions defined in lss_cell_th.c
//
//Compute theory power spectrum (and return as gsl_vector)
gsl_vector *lss_cell_theory(ccl_cosmology *cosmo,LSS_likelihood_workspace *w,LSS_2pt_Cell *cl);

//////////
// Functions defined in lss_cell_like.c
//
//Computes likelihood for a set of cosmological and nuisance parameters (encoded in params)
double lss_likelihood(int n_par,double *params,LSS_2pt_Cell *data,gsl_matrix *prec,LSS_likelihood_workspace *w);
//Creates a new LSS_likelihood_workspace
LSS_likelihood_workspace *lss_like_workspace_new(ccl_cosmology *cosmo,LSS_tracer_info *tr1,LSS_tracer_info *tr2);
//Destructor
void lss_like_workspace_free(LSS_likelihood_workspace *w);

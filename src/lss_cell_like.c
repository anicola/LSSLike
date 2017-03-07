#include "lss_common.h"

LSS_likelihood_workspace *lss_like_workspace_new(ccl_cosmology *cosmo,LSS_tracer_info *tr1,LSS_tracer_info *tr2)
{
  //Creates a new LSS_likelihood_workspace
  //TODO
}

void lss_like_workspace_free(LSS_likelihood_workspace *w)
{
  //Destroy w
  //TODO
}

static ccl_cosmology *get_cosmology(int n_par,double *params,LSS_likelihood_workspace *w)
{
  //Create ccl_cosmology objects from params (using information in w)
  //TODO
}

double lss_likelihood(int n_par,double *params,LSS_2pt_Cell *data,gsl_matrix *prec,LSS_likelihood_workspace *w)
{
  //Compute likelihood for:
  //  - Set of cosmological and nuisance parameters encoded in params
  //  - Data vector in data
  //  - Precision matrix prec
  //  - Information in w (used to transform params into a theory vector)

  ccl_cosmology *cosmo=get_cosmology(n_par,params,w);
  gsl_vector *theory=lss_cell_theory(cosmo,w,data);
  double chi2=compute_chi2(data->cl_bpw,theory,prec); //Write simple functions that performs (d-t)^T * C^-1 * (d-t)

  gsl_vector_free(theory);
  ccl_cosmology_free(cosmo);

  return chi2;
}

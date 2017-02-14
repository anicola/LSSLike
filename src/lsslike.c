#include "common.h"

//chi_LSS This will compute chi2 for LSS observables
//Input:
// - par      -> LSS parameters: describes data layout and provides parameters needed to guide
//               the theory calculation
// - vec_data -> Data vector: in this case, a set of angular cross-power spectra
// - cosmo    -> CCL cosmology structure containing all cosmological information
// - nui      -> LSS nuisance parameters: nuisance parameters needed to model vec_data
//               including their priors
//Output:
// - chi2     -> On output, it contains the value of the log-posterior
//On output, return 0 if everything went OK, error code otherwise
int chi2_LSS(ParamsLSS *par,double *vec_data,ccl_cosmology *cosmo,NuisanceLSS *nui,double *chi2)
{
  int status;
  double chi2_like,chi2_prior;
  double *vec_theory=NULL;

  *chi2=0;

  //Start by adding priors
  status=compute_priors(par,cosmo,nui,&chi2_prior);
  *chi2+=chi2_prior
  if(status)
    return status;

  //Compute the theoretical prediction for vec_data
  status=compute_theory_LSS(par,cosmo,nui,&vec_theory);
  if(status) {
    if(vec_theory!=NULL)
      free(vec_theory);
    return status;
  }

  //Compute log-likelihood and add it to the current value of chi2
  status=compute_chi2(vec_data,vec_theory,par->inv_covar,&chi2_like);
  *chi2+=chi2_like;

  //Cleanup
  free(vec_theory);

  return status;
}
//Caveats
// - Again, too many!
// - Assuming Gaussian distribution
// - A more sophisticated structure for vec_data is probably needed
// - Better status handling necessary, error codes TBD
// - Several functions TBD

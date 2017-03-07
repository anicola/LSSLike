#include "lss_common.h"

LSS_2pt_Cell *lss_read_cell(char *fname)
{
  //Read power spectrum
  //TODO
}

void lss_write_cell(char *fname,LSS_2pt_Cell *cl)
{
  //Write power spectrum
  //TODO
}

gsl_matrix *lss_read_precision(char *fname)
{
  //Read precision matrix
  //TODO
}

void lss_write_precision(char *fname,gsl_matrix *prec,LSS_2pt_Cell *cl)
{
  //Write precision matrix
  //TODO
}

void lss_free_cell(LSS_2pt_Cell *cl)
{
  //Destroy cl
}

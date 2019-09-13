#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <test/grid_check.h>

double check_fl1err(const _prec *u, const _prec *v, 
                    const int i0, const int in, 
                    const int j0, const int jn, 
                    const int k0, const int kn, 
                    const int line, const int slice)
{
        double err = 0.0;
        int num = 0;
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                err += fabs(u[pos] - v[pos]);
                num++;
        }
        }
        }
        return err;

}

#include <test/test.h>
double check_fl2err(const _prec *u, const _prec *v, 
                    const int i0, const int in, 
                    const int j0, const int jn, 
                    const int k0, const int kn, 
                    const int line, const int slice)
{
        double err = 0.0;
        int num = 0;
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                err += pow(u[pos] - v[pos], 2);
                num++;
        }
        }
        }
        return sqrt(err);

}

double check_flinferr(const _prec *u, const _prec *v, 
                      const int i0, const int in, 
                      const int j0, const int jn, 
                      const int k0, const int kn, 
                      const int line, const int slice)
{
        double err = 0.0;
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                double diff = fabs(u[pos] - v[pos]);
                err = diff >  err ? diff : err;
                if (diff > 1e-5 ){
                        printf("Error at: %d %d %d: %g %g \n", i, j, k, u[pos], v[pos]);
                }
        }
        }
        }
        return err;

}

double check_flinfprint(const _prec *u, const _prec *v, 
                      const int i0, const int in, 
                      const int j0, const int jn, 
                      const int k0, const int kn, 
                      const int line, const int slice)
{
        double err = 0.0;
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                err = err > fabs(u[pos] - v[pos]) ? err : fabs(u[pos] - v[pos]);
                if (err > 0) {
                        printf("error[%d %d %d] = %g \n", i, j, k, err);
                }
        }
        }
        }
        return err;
}

int check_all(check_fun fp, 
              const _prec *field, const _prec *result, 
              const int *off_x, const int *off_y, const int *off_z, 
              const int nx, const int ny,
              const int line, const int slice, 
              const _prec tol,
              const int *regions,
              _prec *regions_out
              )
{
        int err = 0;
        double errs[25] = {0};

        for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
                int pos = i + nx * j;
                if (!regions[pos]) {
                        continue;
                }
                errs[pos] =
                    fp(field, result, off_x[i], off_x[i + 1], off_y[j],
                       off_y[j + 1], off_z[1], off_z[2], line, slice);
                if (errs[pos] > tol) {
                      err = 1;
                }
                if(regions_out) regions_out[pos] = errs[pos];
        }
        }

        return err;
}

void check_printerr(const char *fcn, const int rank, const char *field_str, 
                    const _prec *err)
{

        char buf[512];
        fflush(stdout);
        sprintf(buf, 
                "%s(%d) Errors for %s.\n"
                "%e \t %e \t %e \n"
                "%e \t %e \t %e \n"
                "%e \t %e \t %e \n",
                fcn, rank, field_str,
                err[6], err[7], err[8],
                err[3], err[4], err[5],
                err[0], err[1], err[2]
                );
        fprintf(stdout,"%s",buf);
        fflush(stdout);
}

void check_printerr53(const char *fcn, const int rank, const char *field_str, 
                    const _prec *err)
{

        char buf[512];
        fflush(stdout);
        sprintf(buf, 
                "%s(%d) Errors for %s.\n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n",
                fcn, rank, field_str,
                err[10], err[11], err[12], err[13], err[14],
                err[5], err[6], err[7], err[8], err[9],
                err[0], err[1], err[2], err[3], err[4]
                );
        fprintf(stdout,"%s",buf);
        fflush(stdout);
}

void check_printerr55(const char *fcn, const int rank, const char *field_str, 
                    const _prec *err)
{

        char buf[512];
        fflush(stdout);
        sprintf(buf, 
                "%s(%d) Errors for %s.\n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n"
                "%e \t %e \t %e \t %e \t %e \n",
                fcn, rank, field_str,
                err[20], err[21], err[22], err[23], err[24],
                err[15], err[16], err[17], err[18], err[19],
                err[10], err[11], err[12], err[13], err[14],
                err[5], err[6], err[7], err[8], err[9],
                err[0], err[1], err[2], err[3], err[4]
                );
        fprintf(stdout,"%s",buf);
        fflush(stdout);
}


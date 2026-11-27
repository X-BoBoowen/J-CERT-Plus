#include <math.h>
#include <stdint.h>
#include <string.h>

/* Jacobi PCG with independent coefficients and stopping tests.  Both
   products traverse the same CSR row while their search directions are live. */
int paired_pcg(int32_t n, const int32_t *row, const int32_t *col,
               const double *a, const double *diag_inv, const double *b,
               double *x, double *workspace, int32_t *iterations,
               double rtol, double atol, int32_t maxiter) {
    double (*r)[2] = (double (*)[2])workspace;
    double (*p)[2] = r + n, (*ap)[2] = p + n;
    double rho[2] = {0}, tolerance[2] = {0};
    int live[2];
    memset(x, 0, 2*(size_t)n*sizeof(double));
    for (int i=0; i<n; ++i) for (int k=0; k<2; ++k) {
        r[i][k] = b[2*i+k];
        p[i][k] = diag_inv[i]*r[i][k];
        rho[k] += r[i][k]*p[i][k];
        tolerance[k] += r[i][k]*r[i][k];
    }
    for (int k=0; k<2; ++k) {
        double norm2 = tolerance[k];
        tolerance[k] = fmax(atol*atol,rtol*rtol*norm2);
        live[k] = norm2 > tolerance[k];
        iterations[k] = 0;
    }
    for (int it=0; it<maxiter && (live[0] || live[1]); ++it) {
        double pap[2] = {0}, next[2] = {0}, rr[2] = {0}, alpha[2], beta[2];
        for (int i=0; i<n; ++i) {
            double v0=0.,v1=0.;
            for (int j=row[i]; j<row[i+1]; ++j) {
                v0 += a[j]*p[col[j]][0];
                v1 += a[j]*p[col[j]][1];
            }
            ap[i][0]=v0; ap[i][1]=v1;
            pap[0]+=p[i][0]*v0; pap[1]+=p[i][1]*v1;
        }
        for (int k=0; k<2; ++k) {
            if (live[k] && (!(pap[k]>0.) || !isfinite(pap[k]))) return -1;
            alpha[k] = live[k] ? rho[k]/pap[k] : 0.;
            iterations[k] += live[k];
        }
        for (int i=0; i<n; ++i) for (int k=0; k<2; ++k) {
            x[2*i+k] += alpha[k]*p[i][k];
            r[i][k] -= alpha[k]*ap[i][k];
            double square = r[i][k]*r[i][k];
            rr[k] += square;
            next[k] += square*diag_inv[i];
        }
        for (int k=0; k<2; ++k) {
            if (!isfinite(rr[k])) return -1;
            live[k] = rr[k]>tolerance[k];
            beta[k] = live[k] ? next[k]/rho[k] : 0.;
            rho[k]=next[k];
        }
        for (int i=0; i<n; ++i) for (int k=0; k<2; ++k)
            p[i][k] = diag_inv[i]*r[i][k]+beta[k]*p[i][k];
    }
    return live[0] || live[1];
}

/* Single-column version for state and rank-one endpoint solves. */
int scalar_pcg(int32_t n, const int32_t *row, const int32_t *col,
               const double *a, const double *diag_inv, const double *b,
               double *x, double *workspace, int32_t *iterations,
               double rtol, double atol, int32_t maxiter) {
    double *r = workspace, *p = r+n, *ap = p+n;
    double rho = 0., rr = 0.;
    memset(x, 0, (size_t)n*sizeof(double));
    for (int i=0; i<n; ++i) {
        r[i] = b[i];
        p[i] = diag_inv[i]*r[i];
        rho += r[i]*p[i];
        rr += r[i]*r[i];
    }
    double tolerance = fmax(atol*atol, rtol*rtol*rr);
    *iterations = 0;
    while (rr > tolerance && *iterations < maxiter) {
        double pap=0., next=0.;
        for (int i=0; i<n; ++i) {
            double v=0.;
            for (int j=row[i]; j<row[i+1]; ++j) v += a[j]*p[col[j]];
            ap[i]=v;
            pap += p[i]*v;
        }
        if (!(pap>0.) || !isfinite(pap)) return -1;
        double alpha = rho/pap;
        rr = 0.;
        for (int i=0; i<n; ++i) {
            x[i] += alpha*p[i];
            r[i] -= alpha*ap[i];
            double square = r[i]*r[i];
            rr += square;
            next += square*diag_inv[i];
        }
        if (!isfinite(rr)) return -1;
        double beta = next/rho;
        for (int i=0; i<n; ++i) p[i] = diag_inv[i]*r[i]+beta*p[i];
        rho=next;
        ++*iterations;
    }
    return rr>tolerance;
}

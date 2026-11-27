/* Strict binary64 CSR residual certificate for the currently bound operator. */
#include <math.h>
#include <float.h>
#include <stdint.h>

double residual6(int n, const int32_t *p, const int32_t *j, const double *a,
                 const double *rhs, const double *x, double *cap) {
    double scale = 0.;
    for (int i = 0; i < n; ++i) {
        double product = 0., magnitude = 0.;
        for (int k = p[i]; k < p[i+1]; ++k) {
            double term = a[k] * x[j[k]];
            product += term;
            magnitude += fabs(term);
        }
        double gamma = (2. * (p[i+1]-p[i]) + 4.) * DBL_EPSILON;
        cap[i] = fabs(rhs[i]-product) + 4.*gamma*(fabs(rhs[i])+magnitude) + 1e-300;
        if (!isfinite(cap[i])) return NAN;
        if (cap[i] > scale) scale = cap[i];
    }
    double sum = 0.;
    for (int i = 0; i < n; ++i) {
        double scaled = cap[i]/scale;
        sum += scaled * scaled;
    }
    return nextafter(scale * sqrt(sum) * (1. + 8.*n*DBL_EPSILON), INFINITY);
}

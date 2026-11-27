/* Contiguous broad-cap pass for the full-scan control. Same integer expression
   as long_promotion.cap(tighten=False). Input range is checked in Python. */
#include <stdint.h>
#include <stdlib.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
API int32_t broad_pass(int32_t begin, int32_t end, const int32_t *a,
    const int32_t *b, const uint8_t *flip, const int64_t *x,
    int64_t r, int64_t scale, int64_t target, int32_t *ids) {
    int32_t count = 0;
    for (int32_t e = begin; e < end; ++e) {
        int64_t u = x[a[e]], v = x[b[e]], upper;
        if (flip[e]) {
            upper = scale*(-4*u*v + 4*r*(llabs(u)+llabs(v)+r));
        } else {
            int64_t d = llabs(u-v)+2*r;
            upper = d*d;
        }
        if (upper >= target) ids[count++] = e;
    }
    return count;
}

/* Full endpoint sandwich in integer proof-grid squared units.  A is the
   integer-scaled operator, A >= scale I; an addition has weight 1 and a
   flip replaces the negative edge of weight scale by a positive edge.
   No contracted operations or unsafe reassociation are necessary. */
#include <math.h>
#include <float.h>

API int32_t endpoint_pass(int32_t begin, int32_t end, int32_t n, int32_t m,
    const int32_t *ids, const int32_t *excluded, int32_t nexcluded,
    const int32_t *left, const int32_t *right, const uint8_t *flip,
    const double *x, const double *radius, const double *diag, double scale,
    double threshold, int32_t *out, double *highs, double *lows,
    double *maxlow, int32_t *visited) {
    int32_t count = 0, skip = 0, rows = 0;
    double largest = -INFINITY;
    for (int32_t k = begin; k < end; ++k) {
        int32_t e = ids ? ids[k] : k;
        if (skip < nexcluded && excluded[skip] == e) { ++skip; continue; }
        if (e < 0 || e >= m) return -1;
        int32_t i = left[e], j = right[e];
        if ((uint32_t)i >= (uint32_t)n || (uint32_t)j >= (uint32_t)n || i==j)
            return -1;
        ++rows;
        double u=x[i], v=x[j], ri=radius[i], rj=radius[j], a=diag[i], b=diag[j];
        double lo, hi;
        if (!flip[e]) {
            /* Difference and radius sum are exact integral binary64 values.
               Both (10) and (11) reduce to these same fractions. */
            double d=fabs(u-v), err=ri+rj, dl=fmax(0.,d-err), dh=d+err;
            double ab=a*b;
            lo=dl*dl*(scale/(scale+2.))*(1.-16.*DBL_EPSILON);
            hi=dh*dh*(ab/(ab+a+b))*(1.+16.*DBL_EPSILON);
        } else {
            /* v_e=-2s(y,x); det(ab-s^2) is exact.  F has eigenvalues
               1/s, 1/(3s).  Both inverse blocks have norm <=1/s. */
            double sum=u+v, diff=u-v, au=fabs(u), av=fabs(v);
            double gain=-4.*scale*u*v;
            double qf=2.*scale*(sum*sum+diff*diff/3.);
            double det=a*b-scale*scale;
            if (a<2.*scale || b<2.*scale) return -1;
            double qc=(4.*scale*scale/det)*(b*v*v+2.*scale*u*v+a*u*u);
            double err=4.*scale*(au*rj+av*ri+ri*rj
                         +2.*(au+av)*(ri+rj)+(ri+rj)*(ri+rj));
            /* Absolute roundoff bound: a,b>=2s gives det>=3ab/4;
               the unsigned terms of qc are <=8s(|u|+|v|)^2.
               The gain, qf, and box-error terms are respectively bounded
               by 4, 4, 12 times s(|u|+|v|+ri+rj)^2.  Each term, including
               its final accumulation, has <=16 rounded operations along
               any path.  512 eps covers gamma_16 times their sum and
               rounding of this positive allowance and the final endpoints. */
            double mag=au+av+ri+rj;
            double pad=(512.*DBL_EPSILON)*scale*mag*mag;
            lo=(gain-qf)-err-pad;
            hi=(gain-qc)+err+pad;
        }
        largest=fmax(largest,lo);
        if (hi>=threshold) {
            out[count]=e;
            lows[count]=lo;
            highs[count++]=hi;
        }
    }
    *maxlow=largest;
    *visited=rows;
    return count;
}

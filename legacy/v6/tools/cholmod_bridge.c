/* Thin C ABI for SuiteSparse CHOLMOD. Caller supplies packed lower CSC. */
#include <cholmod.h>
#include <stdint.h>
#ifdef JCERT_SCALE_DIAGNOSTICS
#include <stdio.h>
#endif
#include <stdlib.h>
#include <string.h>

typedef struct {
    cholmod_common common;
    cholmod_factor *factor;
    int *inverse_perm;
    int n;
} jc_factor;

void jc_free(jc_factor *f) {
    if (!f) return;
    cholmod_free_factor(&f->factor, &f->common);
    free(f->inverse_perm);
    cholmod_finish(&f->common);
    free(f);
}

static cholmod_sparse packed(int rows, int cols, int *p, int *i, double *x, int symmetry) {
    cholmod_sparse a = {0};
    a.nrow = rows; a.ncol = cols; a.nzmax = p[cols];
    a.p = p; a.i = i; a.x = x; a.stype = symmetry;
    a.itype = CHOLMOD_INT; a.xtype = CHOLMOD_REAL; a.dtype = CHOLMOD_DOUBLE;
    a.sorted = 1; a.packed = 1;
    return a;
}

jc_factor *jc_new(int n, int *p, int *i, double *x) {
    jc_factor *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    cholmod_start(&f->common);
    /* CHOLMOD chooses the root factorization according to estimated fill.
       Hand a simplicial LDL' to the update stage regardless of its choice. */
#ifdef JCERT_FORCE_SIMPLICIAL
    f->common.supernodal = CHOLMOD_SIMPLICIAL;
#else
    f->common.supernodal = CHOLMOD_AUTO;
#endif
    f->common.final_asis = 0;
    f->common.final_super = 0;
    f->common.final_ll = 0;
#ifdef JCERT_UPDATE_READY
    /* CHOLMOD recommends unpacked columns for subsequent up/down operations.
       Drop padding introduced by relaxed supernodal amalgamation. */
    f->common.final_pack = 0;
    f->common.final_resymbol = 1;
#endif
    f->common.print = 0;
    f->n = n;
    cholmod_sparse a = packed(n, n, p, i, x, -1);
    f->factor = cholmod_analyze(&a, &f->common);
    if (!f->factor || !cholmod_factorize(&a, f->factor, &f->common)
        || f->common.status != CHOLMOD_OK || f->factor->minor != (size_t)n) goto fail;
    f->inverse_perm = malloc((size_t)n * sizeof(int));
    if (!f->inverse_perm) goto fail;
    const int *perm = (const int *)f->factor->Perm;
    for (int k = 0; k < n; ++k) f->inverse_perm[perm ? perm[k] : k] = k;
    return f;
fail:
#ifdef JCERT_SCALE_DIAGNOSTICS
    fprintf(stderr, "CHOLMOD native status %d after root analysis/factorization\n",
            f->common.status);
#endif
    jc_free(f); return NULL;
}

jc_factor *jc_clone(const jc_factor *parent) {
    jc_factor *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    cholmod_start(&f->common);
    f->common.print = 0;
    f->n = parent->n;
    f->factor = cholmod_copy_factor(parent->factor, &f->common);
    f->inverse_perm = malloc((size_t)f->n * sizeof(int));
    if (!f->factor || !f->inverse_perm) {
        jc_free(f);
        return NULL;
    }
    memcpy(f->inverse_perm, parent->inverse_perm, (size_t)f->n * sizeof(int));
    return f;
}

int jc_solve(jc_factor *f, int columns, double *x, double *out) {
    cholmod_dense b = {0};
    b.nrow = f->n; b.ncol = columns; b.d = f->n;
    b.nzmax = (size_t)f->n * columns; b.x = x;
    b.xtype = CHOLMOD_REAL; b.dtype = CHOLMOD_DOUBLE;
    cholmod_dense *y = cholmod_solve(CHOLMOD_A, f->factor, &b, &f->common);
    if (!y) return 0;
    memcpy(out, y->x, b.nzmax * sizeof(double));
    cholmod_free_dense(&y, &f->common);
    return f->common.status == CHOLMOD_OK;
}

/* CHOLMOD updown works on P*A*P'. Supply the same permutation on the RHS. */
int jc_modify(jc_factor *f, int a, int b, double x1, double x2, int add) {
    int p[2] = {0, 2}, i[2] = {f->inverse_perm[a], f->inverse_perm[b]};
    double x[2] = {x1, x2};
    if (i[0] > i[1]) {
        int index = i[0]; i[0] = i[1]; i[1] = index;
        x[0] = x2; x[1] = x1;
    }
    cholmod_sparse c = packed(f->n, 1, p, i, x, 0);
    int ok = cholmod_updown(add, &c, f->factor, &f->common)
             && f->common.status == CHOLMOD_OK;
#ifdef JCERT_SCALE_DIAGNOSTICS
    if (!ok) fprintf(stderr, "CHOLMOD native status %d after factor modification; "
                     "native peak %zu bytes\n", f->common.status, f->common.memory_usage);
#endif
    return ok;
}

size_t jc_memory(jc_factor *f, int peak) {
    return (peak ? f->common.memory_usage : f->common.memory_inuse)
           + sizeof(*f) + (size_t)f->n * sizeof(int);
}

/* Count occupied factor entries, excluding spare column capacity. */
size_t jc_factor_entries(jc_factor *f) {
    if (!f || !f->factor || f->factor->is_super || !f->factor->nz) return 0;
    const int *counts = (const int *)f->factor->nz;
    size_t entries = 0;
    for (int j = 0; j < f->n; ++j) entries += (size_t)counts[j];
    return entries;
}

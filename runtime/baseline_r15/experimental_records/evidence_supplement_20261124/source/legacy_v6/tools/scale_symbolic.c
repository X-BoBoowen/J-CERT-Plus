/* Root symbolic analysis using the same default ordering policy as jc_new. */
#include <cholmod.h>
#include <stdint.h>
#include <string.h>

int scale_symbolic(int n, int *p, int *i, double *x, double *out) {
    cholmod_common c;
    if (!cholmod_start(&c)) return 0;
    c.print = 0;
    cholmod_sparse a = {0};
    a.nrow = a.ncol = n;
    a.nzmax = p[n];
    a.p = p; a.i = i; a.x = x; a.stype = -1;
    a.itype = CHOLMOD_INT; a.xtype = CHOLMOD_REAL; a.dtype = CHOLMOD_DOUBLE;
    a.sorted = a.packed = 1;
    cholmod_factor *f = cholmod_analyze(&a, &c);
    int success = f && c.status == CHOLMOD_OK;
    if (success) {
        out[0] = c.lnz; out[1] = c.fl; out[2] = c.memory_usage;
        out[3] = f->ordering;
        out[4] = (double)f->xsize;
        out[5] = (double)f->maxcsize;
    }
    cholmod_free_factor(&f, &c);
    cholmod_finish(&c);
    return success;
}


#ifdef _WIN32
#include <windows.h>
/* A kernel-enforced private-commit ceiling confines a large numeric trial. */
int scale_limit(size_t bytes) {
    HANDLE job = CreateJobObjectW(NULL, NULL);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = {0};
    if (!job) return 0;
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    info.ProcessMemoryLimit = bytes;
    int ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                    &info, sizeof(info)) &&
             AssignProcessToJobObject(job, GetCurrentProcess());
    if (!ok) CloseHandle(job);
    return ok;
}
#endif

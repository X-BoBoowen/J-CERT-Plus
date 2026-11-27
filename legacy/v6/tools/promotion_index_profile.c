/* Counting-sort outward root-key bins and deduplicate two posting directions. */
#include <stdint.h>
#include <stdlib.h>

static inline int32_t bucket(int32_t i, int32_t j, uint8_t flip,
                             const int64_t *x, int64_t r,
                             int64_t offset, int64_t width, int32_t bins) {
    int64_t a=x[i], b=x[j], d=llabs(a-b)+2*r;
    int64_t key = flip ? -4*a*b + 4*r*(llabs(a)+llabs(b)+r) : d*d;
    return flip*bins + (int32_t)((key+offset)/width);
}

void index_pass(int32_t m, const int32_t *i, const int32_t *j,
                const uint8_t *flip, const int64_t *x, int64_t r,
                int64_t offset, int64_t width, int32_t bins,
                int32_t *position, int32_t *order) {
    for (int32_t e=0;e<m;e++) {
        int32_t k=bucket(i[e],j[e],flip[e],x,r,offset,width,bins);
        int32_t slot=position[k]++;
        if (order) order[slot]=e;
    }
}

static inline void take(int32_t e, uint8_t stamp, uint8_t *marks,
                        int32_t *out, int32_t *count) {
    if (marks[e]!=stamp) {
        marks[e]=stamp;
        out[(*count)++]=e;
    }
}

int32_t gather(int32_t n, const int32_t *vertices, const int64_t *left,
               const int64_t *right, const int32_t *right_ids,
               const int32_t *order, const int32_t *bounds, uint8_t stamp,
               uint8_t *marks, int32_t *out) {
    int32_t count=0;
    for (int32_t i=0;i<n;i++) {
        int32_t v=vertices[i];
        for (int32_t k=(int32_t)left[v];k<left[v+1];k++) take(k,stamp,marks,out,&count);
        for (int32_t k=(int32_t)right[v];k<right[v+1];k++) take(right_ids[k],stamp,marks,out,&count);
    }
    for (int c=0;c<2;c++)
        for (int32_t k=bounds[2*c];k<bounds[2*c+1];k++) take(order[k],stamp,marks,out,&count);
    return count;
}

/* h[p] counts posting visits with ceil(log2(drift)) = p. For a dyadic
   choice 2^p, the suffix above p is precisely the cost of moving vertices.
   The supported drift 4*SCREEN bounds every signed shift product below 2^51. */
int32_t dyadic_plan(int32_t n, const int64_t *drift, const int64_t *degree,
                    int64_t target, int64_t screen, int64_t scale,
                    int64_t root_norm, int64_t offset, int64_t width,
                    int32_t bins, const int32_t *offsets, int64_t fixed,
                    int64_t *choices, int64_t *costs, int32_t *bounds,
                    int32_t *best_index) {
    int64_t hist[32]={0}, total=0;
    int32_t largest=0;
    for (int32_t v=0;v<n;v++) {
        int64_t d=drift[v];
        if (d<0 || d>4*screen) return -1;
        if (fixed) {
            if (d>fixed) total+=degree[v];
        } else if (d>1) {
            int32_t p=64-__builtin_clzll((unsigned long long)(d-1));
            hist[p]+=degree[v];
            total+=degree[v];
            if (p>largest) largest=p;
        }
    }
    int64_t best=INT64_MAX;
    for (int32_t p=0;p<=largest;p++) {
        int64_t tau=fixed ? fixed : INT64_C(1)<<p;
        int64_t linear=8*(screen/scale)*tau;
        int64_t quadratic=8*root_norm*tau+4*tau*tau;
        int64_t shift=linear<quadratic ? linear : quadratic;
        total-=hist[p];
        int64_t cost=total;
        int32_t limits[4];
        for (int c=0;c<2;c++) {
            int64_t w=c ? scale : 1;
            int64_t cut=target/w + (target%w>0) - shift + offset;
            int64_t bucket=cut<=0 ? 0 : cut/width;
            if (bucket>bins) bucket=bins;
            limits[2*c]=offsets[c*bins+bucket];
            limits[2*c+1]=offsets[(c+1)*bins];
            cost+=(int64_t)limits[2*c+1]-limits[2*c];
        }
        choices[p]=tau;
        costs[p]=cost;
        if (cost<best) {
            best=cost;
            *best_index=p;
            for (int k=0;k<4;k++) bounds[k]=limits[k];
        }
    }
    return largest+1;
}

/* A flat threshold scan avoids the moving-vertex allocation. Each posting
   uses the existing cursor-local stamp for both catalog classes. */
int32_t gather_at(int32_t n, const int64_t *drift, int64_t tau,
                  const int64_t *left, const int64_t *right,
                  const int32_t *right_ids, const int32_t *order,
                  const int32_t *bounds, uint8_t stamp, uint8_t *marks,
                  int32_t *out) {
    int32_t count=0;
    for (int32_t v=0;v<n;v++) {
        if (drift[v]<=tau) continue;
        for (int32_t k=(int32_t)left[v];k<left[v+1];k++) take(k,stamp,marks,out,&count);
        for (int32_t k=(int32_t)right[v];k<right[v+1];k++) take(right_ids[k],stamp,marks,out,&count);
    }
    for (int c=0;c<2;c++)
        for (int32_t k=bounds[2*c];k<bounds[2*c+1];k++) take(order[k],stamp,marks,out,&count);
    return count;
}

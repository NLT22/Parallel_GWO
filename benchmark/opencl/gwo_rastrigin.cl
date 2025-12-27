// gwo_rastrigin.cl
// build with: -DREAL_IS_DOUBLE=1 or 0

#if defined(REAL_IS_DOUBLE) && (REAL_IS_DOUBLE == 1)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real;
#else
  typedef float  real;
#endif

typedef ulong u64;

// ---------- deterministic RNG (stateless) ----------
inline u64 splitmix64(u64 x)
{
    x += (u64)0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * (u64)0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * (u64)0x94D049BB133111EBUL;
    x = x ^ (x >> 31);
    return x;
}

inline real indexed_random(int iter, int wid, int dim, int channel, u64 global_seed)
{
    u64 mix =
        ((u64)iter    * (u64)0x9E3779B97F4A7C15UL) ^
        ((u64)wid     * (u64)0xBF58476D1CE4E5B9UL) ^
        ((u64)dim     * (u64)0x94D049BB133111EBUL) ^
        ((u64)channel * (u64)0x123456789ABCDEFUL) ^
        global_seed;

    u64 r = splitmix64(mix);

#if defined(REAL_IS_DOUBLE) && (REAL_IS_DOUBLE == 1)
    return (real)((r >> 11) * 0x1.0p-53); // double [0,1)
#else
    uint rf = (uint)(r >> 40);            // top 24 bits
    return (real)rf * (real)(1.0f / 16777216.0f); // /2^24
#endif
}

inline real clamp_real(real x, real lo, real hi) {
    return fmin(fmax(x, lo), hi);
}

inline real rastrigin_term(real x)
{
    const real A = (real)10.0;
#if defined(REAL_IS_DOUBLE) && (REAL_IS_DOUBLE == 1)
    const real two_pi = (real)6.283185307179586;
#else
    const real two_pi = (real)6.283185307179586f;
#endif
    return x * x - A * cos(two_pi * x);
}

// ---------- deterministic compare for top3 ----------
inline void consider_pair(real v, int idx,
                          __private real* b0, __private int* i0,
                          __private real* b1, __private int* i1,
                          __private real* b2, __private int* i2)
{
    // tie-break: lower idx wins
    if (v < *b0 || (v == *b0 && idx < *i0)) {
        *b2 = *b1; *i2 = *i1;
        *b1 = *b0; *i1 = *i0;
        *b0 = v;   *i0 = idx;
    } else if (v < *b1 || (v == *b1 && idx < *i1)) {
        *b2 = *b1; *i2 = *i1;
        *b1 = v;   *i1 = idx;
    } else if (v < *b2 || (v == *b2 && idx < *i2)) {
        *b2 = v;   *i2 = idx;
    }
}

inline int bad_val(real v) {
    return isnan(v) || isinf(v);
}

// ---------- Kernel 1: update X + compute term buffer ----------
__kernel void gwo_update_and_term(
    __global real* X,                  // POP*N
    __global real* term,               // POP*N
    __global const real* X_alpha,      // N
    __global const real* X_beta,       // N
    __global const real* X_delta,      // N
    int N,
    int POP,
    int iter,
    int max_iters,
    u64 global_seed,
    real minX,
    real maxX
) {
    int gid = (int)get_global_id(0);
    int total = POP * N;
    if (gid >= total) return;

    int wid = gid / N;
    int dim = gid - wid * N;

    real x = X[gid];

    // a: 2 -> 0
    real a = (real)2.0 * ((real)1.0 - (real)iter / (real)max_iters);

    // alpha
    real r1 = indexed_random(iter, wid, dim, 0, global_seed);
    real r2 = indexed_random(iter, wid, dim, 1, global_seed);
    real A1 = (real)2.0 * a * r1 - a;
    real C1 = (real)2.0 * r2;
    real D_alpha = fabs(C1 * X_alpha[dim] - x);
    real X1 = X_alpha[dim] - A1 * D_alpha;

    // beta
    real r3 = indexed_random(iter, wid, dim, 2, global_seed);
    real r4 = indexed_random(iter, wid, dim, 3, global_seed);
    real A2 = (real)2.0 * a * r3 - a;
    real C2 = (real)2.0 * r4;
    real D_beta = fabs(C2 * X_beta[dim] - x);
    real X2 = X_beta[dim] - A2 * D_beta;

    // delta
    real r5 = indexed_random(iter, wid, dim, 4, global_seed);
    real r6 = indexed_random(iter, wid, dim, 5, global_seed);
    real A3 = (real)2.0 * a * r5 - a;
    real C3 = (real)2.0 * r6;
    real D_delta = fabs(C3 * X_delta[dim] - x);
    real X3 = X_delta[dim] - A3 * D_delta;

    real newx = (X1 + X2 + X3) / (real)3.0;
    newx = clamp_real(newx, minX, maxX);

    X[gid] = newx;
    term[gid] = rastrigin_term(newx);
}

// ---------- Kernel 2: reduction stage1 (term -> partial[POP*blocks]) ----------
// Each work-group handles one (wolf, block) and reduces LR elements.
// LR = get_local_size(0). Local mem is passed dynamically from host.
__kernel void reduce_stage1(
    __global const real* term,   // POP*N
    __global real* partial,      // POP*blocks
    int N,
    int POP,
    int blocks,
    __local real* sdata          // size = LR
) {
    size_t lid = get_local_id(0);
    size_t group = get_group_id(0);
    int LR = (int)get_local_size(0);

    int wolf = (int)(group / (size_t)blocks);
    int blk  = (int)(group - (size_t)wolf * (size_t)blocks);
    if (wolf >= POP) return;

    int dim0 = blk * LR + (int)lid;
    int idx  = wolf * N + dim0;

    real v = (real)0;
    if (dim0 < N) v = term[idx];

    sdata[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = LR / 2; stride > 0; stride >>= 1) {
        if ((int)lid < stride) {
            sdata[lid] += sdata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[wolf * blocks + blk] = sdata[0];
    }
}

// ---------- Kernel 3: reduction stage2 (partial -> fitness[POP]) ----------
// One work-group per wolf. LR = get_local_size(0).
__kernel void reduce_stage2(
    __global const real* partial, // POP*blocks
    __global real* fitness,       // POP
    int POP,
    int blocks,
    int N,
    __local real* sdata           // size = LR
) {
    size_t lid = get_local_id(0);
    int LR = (int)get_local_size(0);

    int wolf = (int)get_group_id(0);
    if (wolf >= POP) return;

    real sum = (real)0;
    for (int i = (int)lid; i < blocks; i += LR) {
        sum += partial[wolf * blocks + i];
    }

    sdata[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = LR / 2; stride > 0; stride >>= 1) {
        if ((int)lid < stride) {
            sdata[lid] += sdata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        sdata[0] += (real)10.0 * (real)N;
        fitness[wolf] = sdata[0];
    }
}

// ---------- Kernel 4: find global top3 (single work-item) ----------
// This replaces top3_stage1 + top3_stage2. Much simpler.
__kernel void top3_find_global(
    __global const real* fitness, // POP
    int POP,
    __global int* outTop3Idx      // 3
) {
    if (get_global_id(0) != 0) return;

    real b0=(real)INFINITY, b1=(real)INFINITY, b2=(real)INFINITY;
    int  i0=0x7fffffff,     i1=0x7fffffff,     i2=0x7fffffff;

    for (int i = 0; i < POP; i++) {
        real v = fitness[i];
        if (!bad_val(v)) {
            consider_pair(v, i, &b0,&i0,&b1,&i1,&b2,&i2);
        }
    }

    // fallback
    if (i0 < 0 || i0 >= POP) i0 = 0;
    if (i1 < 0 || i1 >= POP) i1 = (POP > 1 ? 1 : 0);
    if (i2 < 0 || i2 >= POP) i2 = (POP > 2 ? 2 : i1);

    outTop3Idx[0] = i0;
    outTop3Idx[1] = i1;
    outTop3Idx[2] = i2;
}

// ---------- Kernel 5: gather leaders (safe) ----------
__kernel void gather_leaders(
    __global const real* X,      // POP*N
    int N,
    int POP,
    __global const int* top3Idx, // 3
    __global real* X_alpha,      // N
    __global real* X_beta,       // N
    __global real* X_delta       // N
) {
    int dim = (int)get_global_id(0);
    if (dim >= N) return;

    int a = top3Idx[0];
    int b = top3Idx[1];
    int d = top3Idx[2];

    if (a < 0 || a >= POP) a = 0;
    if (b < 0 || b >= POP) b = (POP > 1 ? 1 : 0);
    if (d < 0 || d >= POP) d = (POP > 2 ? 2 : b);

    X_alpha[dim] = X[a * N + dim];
    X_beta[dim]  = X[b * N + dim];
    X_delta[dim] = X[d * N + dim];
}

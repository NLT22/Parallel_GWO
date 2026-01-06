// gwo_kmeans.cl
// Build options from host:
//   -D USE_FP64=1 (if cl_khr_fp64 supported and allowed) else 0
//   -D K_CONST=10

typedef float real;
typedef ulong u64;

// ---------------- RNG: splitmix64 + indexed_random ----------------
static inline u64 splitmix64(u64 x)
{
    x += (u64)0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * (u64)0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * (u64)0x94D049BB133111EBUL;
    x = x ^ (x >> 31);
    return x;
}

static inline real u01_from_u64(u64 r)
{
    // lấy 24 bit cao -> float mantissa ~24-bit
    uint x = (uint)(r >> 40);                 // 0 .. 2^24-1
    return (real)x * (real)(1.0f / 16777216.0f); // / 2^24 -> [0,1)
}

static inline real indexed_random(int iter, int wid, int dim, int channel, u64 global_seed)
{
    u64 mix =
        ((u64)iter) * (u64)0x9E3779B97F4A7C15UL ^
        ((u64)wid ) * (u64)0xBF58476D1CE4E5B9UL ^
        ((u64)dim ) * (u64)0x94D049BB133111EBUL ^
        ((u64)channel) * (u64)0x123456789ABCDEFUL ^
        global_seed;

    u64 r = splitmix64(mix);
    return u01_from_u64(r);
}

static inline real clamp01(real x) {
    if (x < (real)0) return (real)0;
    if (x > (real)1) return (real)1;
    return x;
}

// ---------------- Kernel 1: init population pos in [0,1] ----------------
__kernel void init_pos(__global real* pos,
                       int POP, int NDIM,
                       u64 seed)
{
    int gid = (int)get_global_id(0);
    int total = POP * NDIM;
    if (gid >= total) return;

    int wolf = gid / NDIM;
    int dim  = gid - wolf * NDIM;

    // match OpenMP init: indexed_random(0, wolf, dim, 77)
    real r = indexed_random(0, wolf, dim, 77, seed);
    pos[gid] = r;
}

// ---------------- Kernel 2a: compute partial SSE per (wolf, groupX) ----------------
__kernel void sse_partial(__global const float* X,
                          int Ndata, int D,
                          __global const real* pos,
                          int K,
                          int POP,
                          int nGroupsX,
                          __global real* partial_out,
                          __local real* smem) // dynamic local
{
    int i    = (int)get_global_id(0);
    int wolf = (int)get_global_id(1);

    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int groupX = (int)get_group_id(0);

    real val = (real)0;

    if (wolf < POP && i < Ndata)
    {
        const __global float* xi = X + ((size_t)i * (size_t)D);
        const __global real* cent = pos + ((size_t)wolf * (size_t)(K * D));

        real best = (real)1e30;

        for (int k = 0; k < K; ++k) {
            const __global real* ck = cent + ((size_t)k * (size_t)D);
            real dist = (real)0;
            for (int d = 0; d < D; ++d) {
                real diff = (real)xi[d] - ck[d];
                dist += diff * diff;
            }
            if (dist < best) best = dist;
        }
        val = best;
    }

    smem[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) smem[lid] += smem[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_out[(size_t)wolf * (size_t)nGroupsX + (size_t)groupX] = smem[0];
    }
}

// ---------------- Kernel 2b: reduce partial SSE to final SSE per wolf ----------------
__kernel void sse_reduce(__global const real* partial,
                         int POP, int nGroupsX,
                         __global real* sse_out)
{
    int wolf = (int)get_global_id(0);
    if (wolf >= POP) return;

    real sum = (real)0;
    size_t base = (size_t)wolf * (size_t)nGroupsX;
    for (int g = 0; g < nGroupsX; ++g) sum += partial[base + (size_t)g];
    sse_out[wolf] = sum;
}

// ---------------- Top-3 reduction ----------------
typedef struct { real f; int i; } PairFI;

static inline int better_smallest(PairFI a, PairFI b) {
    if (a.f < b.f) return 1;
    if (a.f > b.f) return 0;
    return (a.i < b.i) ? 1 : 0;
}

static inline void top3_init(__private PairFI* v0, __private PairFI* v1, __private PairFI* v2) {
    v0->f = (real)1e30; v0->i = -1;
    v1->f = (real)1e30; v1->i = -1;
    v2->f = (real)1e30; v2->i = -1;
}

// NOTE: accept __local pointers to avoid address-space errors
static inline void top3_push(PairFI x, __local PairFI* v0, __local PairFI* v1, __local PairFI* v2) {
    if (better_smallest(x, *v0)) {
        *v2 = *v1; *v1 = *v0; *v0 = x;
    } else if (better_smallest(x, *v1)) {
        *v2 = *v1; *v1 = x;
    } else if (better_smallest(x, *v2)) {
        *v2 = x;
    }
}

// stage1: each workgroup returns 3 smallest
__kernel void reduce_top3_stage1(__global const real* sse,
                                 int POP,
                                 __global PairFI* cand,
                                 __local PairFI* s0,
                                 __local PairFI* s1,
                                 __local PairFI* s2)
{
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int grp = (int)get_group_id(0);

    // init private best3 then store into local arrays
    PairFI p0, p1, p2;
    // private init
    p0.f = (real)1e30; p0.i = -1;
    p1.f = (real)1e30; p1.i = -1;
    p2.f = (real)1e30; p2.i = -1;

    for (int idx = gid; idx < POP; idx += (int)get_global_size(0)) {
        PairFI x; x.f = sse[idx]; x.i = idx;

        // push into private (manual, no address-space issue)
        if (better_smallest(x, p0)) { p2 = p1; p1 = p0; p0 = x; }
        else if (better_smallest(x, p1)) { p2 = p1; p1 = x; }
        else if (better_smallest(x, p2)) { p2 = x; }
    }

    s0[lid] = p0; s1[lid] = p1; s2[lid] = p2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) {
            top3_push(s0[lid + off], &s0[lid], &s1[lid], &s2[lid]);
            top3_push(s1[lid + off], &s0[lid], &s1[lid], &s2[lid]);
            top3_push(s2[lid + off], &s0[lid], &s1[lid], &s2[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int out = grp * 3;
        cand[out + 0] = s0[0];
        cand[out + 1] = s1[0];
        cand[out + 2] = s2[0];
    }
}

// stage2: reduce cand to final 3, then output OpenMP order [v2,v1,v0]
__kernel void reduce_top3_stage2(__global const PairFI* cand,
                                 int N3,
                                 __global int* best3_out,
                                 __local PairFI* s0,
                                 __local PairFI* s1,
                                 __local PairFI* s2)
{
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);

    // private accumulation
    PairFI p0, p1, p2;
    p0.f = (real)1e30; p0.i = -1;
    p1.f = (real)1e30; p1.i = -1;
    p2.f = (real)1e30; p2.i = -1;

    for (int idx = lid; idx < N3; idx += lsz) {
        PairFI x = cand[idx];
        if (x.i < 0) continue;

        if (better_smallest(x, p0)) { p2 = p1; p1 = p0; p0 = x; }
        else if (better_smallest(x, p1)) { p2 = p1; p1 = x; }
        else if (better_smallest(x, p2)) { p2 = x; }
    }

    s0[lid] = p0; s1[lid] = p1; s2[lid] = p2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) {
            top3_push(s0[lid + off], &s0[lid], &s1[lid], &s2[lid]);
            top3_push(s1[lid + off], &s0[lid], &s1[lid], &s2[lid]);
            top3_push(s2[lid + off], &s0[lid], &s1[lid], &s2[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // s0=smallest (best), s1=2nd, s2=3rd
        best3_out[0] = s0[0].i; // alpha
        best3_out[1] = s1[0].i; // beta
        best3_out[2] = s2[0].i; // delta
    }

}

__kernel void gather_leaders(__global const real* pos,
                             int NDIM,
                             __global const int* best3,
                             __global real* L0,
                             __global real* L1,
                             __global real* L2)
{
    int d = (int)get_global_id(0);
    if (d >= NDIM) return;

    int i0 = best3[0];
    int i1 = best3[1];
    int i2 = best3[2];

    L0[d] = pos[(size_t)i0 * (size_t)NDIM + (size_t)d];
    L1[d] = pos[(size_t)i1 * (size_t)NDIM + (size_t)d];
    L2[d] = pos[(size_t)i2 * (size_t)NDIM + (size_t)d];
}

__kernel void gwo_update(__global real* pos,
                         int POP, int NDIM,
                         __global const real* L0,
                         __global const real* L1,
                         __global const real* L2,
                         real a,
                         int iter,
                         u64 seed)
{
    int wolf = (int)get_global_id(0);
    int d    = (int)get_global_id(1);
    if (wolf >= POP || d >= NDIM) return;

    size_t idx = (size_t)wolf * (size_t)NDIM + (size_t)d;
    real x = pos[idx];

    // j=0
    real r1 = indexed_random(iter, wolf, d, 0, seed);
    real r2 = indexed_random(iter, wolf, d, 1, seed);
    real A1 = (real)2 * a * r1 - a;
    real C1 = (real)2 * r2;
    real D1 = fabs(C1 * L0[d] - x);
    real X1 = L0[d] - A1 * D1;

    // j=1
    r1 = indexed_random(iter, wolf, d, 2, seed);
    r2 = indexed_random(iter, wolf, d, 3, seed);
    real A2 = (real)2 * a * r1 - a;
    real C2 = (real)2 * r2;
    real D2 = fabs(C2 * L1[d] - x);
    real X2 = L1[d] - A2 * D2;

    // j=2
    r1 = indexed_random(iter, wolf, d, 4, seed);
    r2 = indexed_random(iter, wolf, d, 5, seed);
    real A3 = (real)2 * a * r1 - a;
    real C3 = (real)2 * r2;
    real D3 = fabs(C3 * L2[d] - x);
    real X3 = L2[d] - A3 * D3;

    real newx = (X1 + X2 + X3) / (real)3;
    pos[idx] = clamp01(newx);
}

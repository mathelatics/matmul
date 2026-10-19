/*
 * matmul.h - single-file matrix multiplication library built on top of ggml (https://github.com/ggml-org/ggml)
 *
 * This library exposes the *full* capability set of the ggml compute graph engine through a tiny,
 * dependency-free C API:
 *
 *   - Real matrix products via ggml's own `ggml_mul_mat()` kernel (the same one llama.cpp uses),
 *     which means you automatically get every ggml backend and every ggml optimization:
 *         CPU   : SIMD dispatch (AVX/AVX2/AVX-512/AMX, NEON/SVE/SME, WASM simd128), OpenMP threads,
 *                 optional cuBLAS-style BLAS (OpenBLAS / Accelerate) when ggml is built with it,
 *                 runtime repacking (Q4_0 -> Q4_K etc.), flash-attn style fused kernels, ...
 *         GPU   : CUDA / Metal / Vulkan / SYCL / HIP / ... whenever the corresponding ggml backend
 *                 shared library is present (ggml's dynamic backend loader picks them up).
 *   - Automatic quantization of operands: F32 / F16 / BF16 / Q8_0 / Q4_K / Q4_0 / IQ4_XS / ...
 *     ggml dequantizes + accumulates in fp32 internally, so quantized matmuls stay numerically sane.
 *   - Row-gather matmul (embedding-table lookups, MoE experts) via `ggml_get_rows` + `ggml_mul_mat_id`.
 *   - Batched matmul (3D/4D broadcast semantics identical to ggml/torch).
 *   - Graph fusion: arbitrary chains C = alpha*(A@B) + beta*C, A@B@C, ... are built as a lazy ggml
 *     computation graph and executed in a single `compute()` call. Weights that don't change across
 *     calls are kept in backend memory (e.g. VRAM) and are never re-uploaded.
 *   - Backend selection policy (auto / cpu / gpu / cuda / metal / vulkan / sycl / openblas / any name),
 *     multi-backend scheduling through `ggml_backend_sched`, tensor placement overrides,
 *     device enumeration & reporting, timing (GFLOPS), error handling, logging hooks.
 *   - No extensions required: pure C99 public API, works from C and C++ translation units.
 *
 * HOW TO BUILD (amalgamation mode - default):
 *   matmul.c embeds a complete copy of the ggml core (ggml.c, ggml-quants.c, ggml-alloc.c,
 *   ggml-backend*.cpp, ggml-threading.cpp) inside itself, so you literally only need these two
 *   files to compile an application:
 *
 *       gcc  -O3 -std=gnu11 -c matmul.c -o matmul.o -lm -lpthread -ldl -lstdc++ -latomic
 *       # or link against a system ggml instead of the embedded copy:
 *       gcc  -O3 -DMATMUL_USE_SYSTEM_GGML -std=gnu11 matmul.c -lggml -o app
 *
 * LICENSE: this wrapper is MIT. It embeds ggml, which is MIT as well (see the bundled notice at
 * the bottom of matmul.c).
 */

#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* export macro (static single-file use by default; define MATMUL_SHARED to build a DLL) */
#ifndef MATMUL_API
#  if defined(_WIN32) && defined(MATMUL_SHARED)
#    ifdef MATMUL_BUILD
#      define MATMUL_API __declspec(dllexport)
#    else
#      define MATMUL_API __declspec(dllimport)
#    endif
#  elif defined(__GNUC__) && defined(MATMUL_SHARED)
#    define MATMUL_API __attribute__((visibility("default")))
#  else
#    define MATMUL_API extern
#  endif
#endif

/* ------------------------------------------------------------------ */
/* version                                                             */
/* ------------------------------------------------------------------ */

#define MATMUL_VERSION_MAJOR 1
#define MATMUL_VERSION_MINOR 0
#define MATMUL_VERSION_PATCH 0

MATMUL_API const char * matmul_version(void);   /* e.g. "1.0.0" */
MATMUL_API int          matmul_version_int(void);
MATMUL_API const char * matmul_ggml_version(void);  /* embedded ggml version string */
MATMUL_API const char * matmul_ggml_commit(void);

/* ------------------------------------------------------------------ */
/* error handling                                                      */
/* ------------------------------------------------------------------ */

typedef enum matmul_error {
    MATMUL_OK                    =  0,
    MATMUL_ERR_INVALID_ARG       = -1,   /* bad shape / NULL pointer */
    MATMUL_ERR_UNSUPPORTED       = -2,   /* unsupported dtype/backend combo */
    MATMUL_ERR_NO_BACKEND        = -3,   /* requested backend/device not available */
    MATMUL_ERR_OUT_OF_MEMORY     = -4,
    MATMUL_ERR_GRAPH_TOO_SMALL   = -5,   /* more nodes than the reserved graph size */
    MATMUL_ERR_NOT_COMPUTED      = -6,   /* result requested before compute() */
    MATMUL_ERR_COMPUTE_FAILED    = -7,   /* ggml returned a non-success status */
    MATMUL_ERR_INTERNAL          = -8,
} matmul_error;

MATMUL_API const char * matmul_strerror(matmul_error err);

/* last error message for the calling thread (never NULL, static buffer) */
MATMUL_API const char * matmul_last_error(void);

/* optional log hook: receives ggml log messages (level: 0..5 like ggml_log_level) */
typedef void (*matmul_log_func)(int level, const char * msg, void * user_data);
MATMUL_API void matmul_set_logger(matmul_log_func cb, void * user_data);

/* ------------------------------------------------------------------ */
/* data types (mirrors a useful subset of ggml_type)                   */
/* ------------------------------------------------------------------ */

typedef enum matmul_dtype {
    MATMUL_F32    = 0,
    MATMUL_F16    = 1,
    MATMUL_BF16   = 2,
    MATMUL_Q8_0   = 3,
    MATMUL_Q4_0   = 4,
    MATMUL_Q4_K   = 5,
    MATMUL_IQ4_XS = 6,
    MATMUL_DTYPE_COUNT
} matmul_dtype;

MATMUL_API const char * matmul_dtype_name(matmul_dtype dt);
/* parse "f32","q4_k",... case-insensitive; returns true on success */
MATMUL_API bool matmul_dtype_from_name(const char * name, matmul_dtype * out);
/* bytes per element averaged over a full block (quant types are sub-byte) */
MATMUL_API double matmul_dtype_bpp(matmul_dtype dt);
MATMUL_API bool   matmul_dtype_is_quantized(matmul_dtype dt);

/* ------------------------------------------------------------------ */
/* backend / device policies                                           */
/* ------------------------------------------------------------------ */

typedef enum matmul_device_policy {
    MATMUL_DEVICE_AUTO = 0,   /* best device: first GPU if any, else CPU   */
    MATMUL_DEVICE_CPU,        /* force CPU                                 */
    MATMUL_DEVICE_GPU,        /* require a GPU, fail if none               */
    MATMUL_DEVICE_CUDA,       /* require a specific backend by name        */
    MATMUL_DEVICE_METAL,
    MATMUL_DEVICE_VULKAN,
    MATMUL_DEVICE_SYCL,
    MATMUL_DEVICE_OPENCL,
    MATMUL_DEVICE_CUSTOM,     /* use matmul_params::device_name            */
} matmul_device_policy;

typedef enum matmul_order {
    MATMUL_ORDER_ROW_MAJOR = 0,  /* C-style (default)                       */
    MATMUL_ORDER_COL_MAJOR,      /* Fortran/BLAS/ggml-native order          */
} matmul_order;

/* operand storage layout for matmul_run() convenience calls */
typedef enum matmul_layout {
    MATMUL_LAYOUT_NK = 0,  /* B is [N x K] row-major, i.e. plain A @ B^T    */
    MATMUL_LAYOUT_KN,      /* B is [K x N] row-major, i.e. plain A @ B      */
} matmul_layout;

/* ------------------------------------------------------------------ */
/* opaque handles                                                      */
/* ------------------------------------------------------------------ */

typedef struct matmul_ctx      matmul_ctx;      /* owns ggml context+backend */
typedef struct matmul_tensor   matmul_tensor;   /* cached device tensor      */
typedef struct matmul_plan     matmul_plan;     /* reusable fused graph      */

/* ------------------------------------------------------------------ */
/* library / context init                                              */
/* ------------------------------------------------------------------ */

typedef struct matmul_params {
    matmul_device_policy device;      /* MATMUL_DEVICE_AUTO by default      */
    const char *         device_name; /* for MATMUL_DEVICE_CUSTOM           */
    int                  n_threads;   /* <=0: let ggml decide               */
    size_t               arena_bytes; /* ggml host arena, 0 => default 1GiB */
    bool                 verbose;     /* forward ggml logs to logger        */
} matmul_params;

MATMUL_API void matmul_params_default(matmul_params * p);

/* One-time process-wide initialization (loads ggml backends, sets up logging).
 * Safe to call multiple times. Call once before creating contexts. */
MATMUL_API matmul_error matmul_init(void);
MATMUL_API void         matmul_shutdown(void);

MATMUL_API matmul_ctx * matmul_ctx_new(const matmul_params * params); /* NULL on failure */
MATMUL_API void         matmul_ctx_free(matmul_ctx * ctx);

/* info about the chosen backend/devices */
MATMUL_API const char * matmul_ctx_backend_name(const matmul_ctx * ctx);
MATMUL_API const char * matmul_ctx_device_description(const matmul_ctx * ctx);
MATMUL_API int          matmul_ctx_n_threads(const matmul_ctx * ctx);
MATMUL_API matmul_error matmul_ctx_set_n_threads(matmul_ctx * ctx, int n_threads);

/* enumerate all devices known to ggml (after matmul_init()) */
MATMUL_API size_t matmul_device_count(void);
MATMUL_API bool   matmul_device_info(size_t idx, const char ** name, const char ** desc, bool * is_gpu);

/* ------------------------------------------------------------------ */
/* tensors (cached operands)                                           */
/* ------------------------------------------------------------------ */

/* Create a persistent operand. `data` may be NULL (then fill later with
 * matmul_tensor_upload()). dtype selects quantization for weights.
 * rows/cols are LOGICAL dims (always interpreted as a row-major logical
 * [rows x cols] matrix unless col_major=true). */
MATMUL_API matmul_tensor * matmul_tensor_new(matmul_ctx * ctx,
                                                int64_t rows, int64_t cols,
                                                matmul_dtype dtype,
                                                const void * data /* f32 row-major, may be NULL */,
                                                bool col_major);
MATMUL_API void matmul_tensor_free(matmul_tensor * t);

MATMUL_API int64_t      matmul_tensor_rows(const matmul_tensor * t);
MATMUL_API int64_t      matmul_tensor_cols(const matmul_tensor * t);
MATMUL_API matmul_dtype matmul_tensor_dtype(const matmul_tensor * t);
MATMUL_API bool         matmul_tensor_is_col_major(const matmul_tensor * t);
MATMUL_API const char * matmul_tensor_backend(const matmul_tensor * t);

/* Re-upload new contents into an existing tensor (keeps device allocation).
 * src is fp32 row-major logical [rows x cols]; quantized targets requantize. */
MATMUL_API matmul_error matmul_tensor_upload(matmul_tensor * t, const float * src);

/* Download current contents to fp32 row-major host buffer (size >= rows*cols floats). */
MATMUL_API matmul_error matmul_tensor_download(const matmul_tensor * t, float * dst);

/* ------------------------------------------------------------------ */
/* graph plans (fused matmul expressions)                              */
/* ------------------------------------------------------------------ */

MATMUL_API matmul_plan * matmul_plan_new(matmul_ctx * ctx, size_t max_nodes);
MATMUL_API void           matmul_plan_free(matmul_plan * p);
MATMUL_API void           matmul_plan_reset(matmul_plan * p);   /* clear nodes, keep allocations */
MATMUL_API size_t         matmul_plan_node_count(const matmul_plan * p);

/* Append: OUT += alpha * op(A) @ op(B)   (accumulates into `out`)
 *   - A must be a cached tensor, B may be a cached tensor OR a live input pointer.
 *   - `a_transposed` flips A's orientation (op(A)=A^T), same for B.
 *   - shapes are checked against the logical dims.
 * Returns node id (>=0) or negative matmul_error. */
MATMUL_API int matmul_plan_add_mm(matmul_plan * p,
                                  matmul_tensor * out,
                                  matmul_tensor * a, bool a_transposed,
                                  matmul_tensor * b, bool b_transposed,
                                  float alpha);

/* Same but B is a caller-provided fp32 matrix that lives in RAM at compute() time
 * (row-major [K x N] if !b_transposed, else [N x K]). Use for activations/batches. */
MATMUL_API int matmul_plan_add_mm_src(matmul_plan * p,
                                      matmul_tensor * out,
                                      matmul_tensor * a, bool a_transposed,
                                      const float * b, int64_t b_rows, int64_t b_cols,
                                      bool b_transposed, float alpha);

/* Chain of three: OUT += alpha * A @ B @ C  (fused as a single ggml graph) */
MATMUL_API int matmul_plan_add_mmm(matmul_plan * p,
                                   matmul_tensor * out,
                                   matmul_tensor * a, bool a_t,
                                   matmul_tensor * b, bool b_t,
                                   matmul_tensor * c, bool c_t,
                                   float alpha);

/* Row-gather matmul: OUT[i,:] += alpha * A[ids[i],:] @ B
 * ids is an int64 index array of length m provided at compute time. */
MATMUL_API int matmul_plan_add_gather(matmul_plan * p,
                                      matmul_tensor * out,
                                      matmul_tensor * a,
                                      matmul_tensor * b,
                                      float alpha);
MATMUL_API matmul_error matmul_plan_set_ids(matmul_plan * p, int node,
                                            const int64_t * ids, size_t n);

/* Set the live fp32 source pointer for an *_src node before compute(). */
MATMUL_API matmul_error matmul_plan_set_src(matmul_plan * p, int node, const float * data);

/* Execute the whole plan. If `ids`/`src` were bound they are consumed now.
 * Parallelism: uses the ctx thread count / ggml thread pool. */
MATMUL_API matmul_error matmul_plan_compute(matmul_plan * p);

/* Read result out of a tensor used as `out` in a plan (fp32 row-major). */
MATMUL_API matmul_error matmul_plan_read_out(const matmul_plan * p,
                                             const matmul_tensor * out, float * dst);

/* ------------------------------------------------------------------ */
/* one-shot convenience API                                            */
/* ------------------------------------------------------------------ */

/* C[m x n](fp32, row-major, ldc free) = alpha * op(A) @ op(B) + beta * C_in
 *  - A: row-major [m x k] (or [k x m] if a_transposed)
 *  - B: depends on `layout`:
 *       MATMUL_LAYOUT_KN : row-major [k x n]   (plain A@B)
 *       MATMUL_LAYOUT_NK : row-major [n x k]   (A @ B^T, classic NN-of-rows)
 *  - weight_dtype != MATMUL_F32  => B is quantized once then multiplied
 *    (great for repeated inference against a fixed weight matrix).
 *  - reuse != NULL: cache the internal context/graph for the next call with
 *    the same shapes (weights stay resident, no re-alloc, no re-upload). */
MATMUL_API matmul_error matmul_run(matmul_ctx * ctx,
                                   int64_t m, int64_t n, int64_t k,
                                   const float * A, bool a_transposed,
                                   const float * B, matmul_layout layout,
                                   float * C,
                                   float alpha, float beta,
                                   matmul_dtype weight_dtype,
                                   matmul_order c_order,
                                   matmul_plan ** reuse);

/* batched: C[b] = op(A[b]) @ op(B[b]), leading-dimension strides given.
 * nb can be >1 broadcasting either operand (nb==1 reuses the same matrix). */
MATMUL_API matmul_error matmul_run_batched(matmul_ctx * ctx,
                                           int64_t m, int64_t n, int64_t k, int64_t nb,
                                           const float * A, int64_t stride_a, bool a_transposed,
                                           const float * B, int64_t stride_b, matmul_layout layout,
                                           float * C, int64_t stride_c,
                                           matmul_dtype weight_dtype);

/* ------------------------------------------------------------------ */
/* benchmarking helpers                                                */
/* ------------------------------------------------------------------ */

MATMUL_API int64_t matmul_time_us(void);   /* monotonic clock */

/* GFLOP count for an m x n x k product (= 2*m*n*k / 1e9) */
MATMUL_API double matmul_gflops(int64_t m, int64_t n, int64_t k);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_H */

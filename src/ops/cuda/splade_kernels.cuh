/*
 * SPLADE CUDA Kernels - Header File
 *
 * High-performance CUDA implementations for SPLADE operations:
 * - Fused masked SPLADE aggregation: log1p(relu(x)) * mask -> max_pool
 * - FLOPS regularization for sparsity enforcement
 * - Top-k sparse extraction for interpretability
 *
 * Optimizations:
 * - Warp-level primitives for fast reductions
 * - Vectorized memory access (float4) for bandwidth
 * - Shared memory tiling for large vocabulary
 * - Fused operations to minimize kernel launches
 *
 * Hardware Requirements:
 * - CUDA Compute Capability >= 7.0 (Volta+)
 * - Optimized for A100/H100 GPUs
 */

#ifndef SPLADE_KERNELS_CUH
#define SPLADE_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Constants and Configuration
// ============================================================================

// Block sizes tuned for modern GPUs (A100/H100)
constexpr int BLOCK_SIZE_AGGREGATE = 256;
constexpr int BLOCK_SIZE_FLOPS = 256;
constexpr int BLOCK_SIZE_TOPK = 128;

// Warp size (constant across all NVIDIA GPUs)
constexpr int WARP_SIZE = 32;

// Maximum shared memory per block (48KB default, can request up to 164KB on A100)
constexpr int MAX_SHARED_MEMORY = 48 * 1024;

// Vocabulary tile size for shared memory
constexpr int VOCAB_TILE_SIZE = 1024;

// ============================================================================
// Utility Functions
// ============================================================================

// Fast log1p approximation for positive values
__device__ __forceinline__ float fast_log1p(float x) {
    // For x > 0: log(1 + x) using hardware log
    return __logf(1.0f + x);
}

// Fused ReLU + log1p activation
__device__ __forceinline__ float splade_activation(float x) {
    float relu_x = fmaxf(x, 0.0f);
    return fast_log1p(relu_x);
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level max reduction using shared memory
__device__ float block_reduce_max(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Warp-level reduction first
    val = warp_reduce_max(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all warp results
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared_mem[threadIdx.x] : -INFINITY;

    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// Block-level sum reduction using shared memory
__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared_mem[threadIdx.x] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// Kernel Declarations
// ============================================================================

/*
 * Fused SPLADE Aggregation Kernel
 *
 * Computes: output[b, v] = max_s(log1p(relu(logits[b, s, v])) * mask[b, s])
 *
 * This kernel fuses 4 operations:
 * 1. ReLU activation
 * 2. log1p transformation
 * 3. Attention mask application
 * 4. Max-pooling over sequence dimension
 *
 * Grid: (batch_size, ceil(vocab_size / BLOCK_SIZE))
 * Block: (BLOCK_SIZE_AGGREGATE,)
 */
__global__ void splade_aggregate_kernel(
    const float* __restrict__ logits,      // [batch, seq_len, vocab_size]
    const float* __restrict__ attention_mask, // [batch, seq_len]
    float* __restrict__ output,            // [batch, vocab_size]
    int batch_size,
    int seq_len,
    int vocab_size
);

/*
 * Fused SPLADE Aggregation Kernel with Half Precision
 *
 * Same as above but uses FP16 for inputs (2x memory bandwidth)
 */
__global__ void splade_aggregate_kernel_fp16(
    const __half* __restrict__ logits,
    const __half* __restrict__ attention_mask,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size
);

/*
 * Vectorized SPLADE Aggregation Kernel
 *
 * Uses float4 loads for 4x memory bandwidth improvement
 * Requires vocab_size to be divisible by 4
 */
__global__ void splade_aggregate_kernel_vectorized(
    const float4* __restrict__ logits,
    const float* __restrict__ attention_mask,
    float4* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size_div4
);

/*
 * FLOPS Regularization Kernel - Phase 1: Column Sums
 *
 * Computes: col_sums[v] = sum_b(|activations[b, v]|)
 *
 * Grid: (ceil(vocab_size / BLOCK_SIZE),)
 * Block: (BLOCK_SIZE_FLOPS,)
 */
__global__ void flops_reg_column_sum_kernel(
    const float* __restrict__ activations,  // [batch, vocab_size]
    float* __restrict__ col_sums,           // [vocab_size]
    int batch_size,
    int vocab_size
);

/*
 * FLOPS Regularization Kernel - Phase 2: Final Reduction
 *
 * Computes: loss = sum_v((col_sums[v] / batch_size)^2)
 *
 * Grid: (1,)
 * Block: (BLOCK_SIZE_FLOPS,)
 */
__global__ void flops_reg_final_kernel(
    const float* __restrict__ col_sums,
    float* __restrict__ loss,
    int vocab_size,
    float batch_size_inv
);

/*
 * Fused FLOPS Regularization Kernel (Single-Pass)
 *
 * Computes the full FLOPS loss in a single kernel for small batch sizes.
 * Uses shared memory to accumulate column sums.
 *
 * L_FLOPS = sum_v (mean_b |activations[b,v]|)^2
 */
__global__ void flops_reg_fused_kernel(
    const float* __restrict__ activations,
    float* __restrict__ loss,
    int batch_size,
    int vocab_size
);

/*
 * Top-K Extraction Kernel
 *
 * Extracts top-k indices and values from sparse vectors for interpretability.
 * Uses partial sorting with shared memory.
 *
 * Grid: (batch_size,)
 * Block: (BLOCK_SIZE_TOPK,)
 */
__global__ void topk_kernel(
    const float* __restrict__ vectors,    // [batch, vocab_size]
    float* __restrict__ values,           // [batch, k]
    int* __restrict__ indices,            // [batch, k]
    int vocab_size,
    int k
);

/*
 * Sparse Vector Statistics Kernel
 *
 * Computes sparsity statistics for a batch of vectors.
 * Output: [num_nonzero, sum_abs, max_abs] per vector
 */
__global__ void sparse_stats_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ stats,            // [batch, 3]
    int vocab_size,
    float threshold
);

/*
 * Combined Forward Pass Kernel
 *
 * Fuses SPLADE aggregation with immediate classification for minimal memory traffic.
 * Useful when only predictions (not sparse vectors) are needed.
 */
__global__ void splade_classify_fused_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ attention_mask,
    const float* __restrict__ classifier_weights,  // [num_labels, vocab_size]
    const float* __restrict__ classifier_bias,     // [num_labels]
    float* __restrict__ class_logits,              // [batch, num_labels]
    int batch_size,
    int seq_len,
    int vocab_size,
    int num_labels
);

// ============================================================================
// Launcher Functions (C++ Interface)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Launch splade_aggregate with automatic configuration
void launch_splade_aggregate(
    const float* logits,
    const float* attention_mask,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    cudaStream_t stream = 0
);

// Launch splade_aggregate with FP16 inputs
void launch_splade_aggregate_fp16(
    const void* logits,
    const void* attention_mask,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    cudaStream_t stream = 0
);

// Launch FLOPS regularization
void launch_flops_reg(
    const float* activations,
    float* loss,
    float* workspace,  // Temporary buffer for column sums
    int batch_size,
    int vocab_size,
    cudaStream_t stream = 0
);

// Launch top-k extraction
void launch_topk(
    const float* vectors,
    float* values,
    int* indices,
    int batch_size,
    int vocab_size,
    int k,
    cudaStream_t stream = 0
);

// Get required workspace size for FLOPS regularization
size_t get_flops_reg_workspace_size(int vocab_size);

#ifdef __cplusplus
}
#endif

// ============================================================================
// Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // SPLADE_KERNELS_CUH

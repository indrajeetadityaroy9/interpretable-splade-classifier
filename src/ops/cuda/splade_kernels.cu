/*
 * SPLADE CUDA Kernels - Implementation
 *
 * High-performance CUDA implementations optimized for:
 * - Memory coalescing and vectorized access
 * - Warp-level primitives for fast reductions
 * - Shared memory tiling for large vocabularies
 * - Minimal synchronization overhead
 */

#include "splade_kernels.cuh"
#include <cfloat>
#include <cstdio>

// ============================================================================
// SPLADE Aggregation Kernels
// ============================================================================

/*
 * Main SPLADE Aggregation Kernel
 *
 * Each block processes one (batch, vocab_block) pair.
 * Threads iterate over sequence positions and maintain running max.
 *
 * Memory access pattern:
 * - Coalesced reads across vocab dimension
 * - Each thread handles multiple vocab positions if needed
 */
__global__ void splade_aggregate_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ attention_mask,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    // 2D grid: (batch_size, vocab_blocks)
    int batch_idx = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int vocab_idx = vocab_start + threadIdx.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size) return;

    // Pointer arithmetic for this batch
    const float* batch_logits = logits + batch_idx * seq_len * vocab_size;
    const float* batch_mask = attention_mask + batch_idx * seq_len;

    // Initialize max to negative infinity
    float max_val = -FLT_MAX;

    // Iterate over sequence positions
    #pragma unroll 4
    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float mask_val = batch_mask[seq_idx];

        // Skip masked positions (attention_mask == 0)
        if (mask_val > 0.0f) {
            // Load logit value
            float logit = batch_logits[seq_idx * vocab_size + vocab_idx];

            // Fused ReLU + log1p
            float activated = splade_activation(logit);

            // Update running max
            max_val = fmaxf(max_val, activated);
        }
    }

    // Handle case where all positions were masked
    if (max_val == -FLT_MAX) {
        max_val = 0.0f;
    }

    // Write output
    output[batch_idx * vocab_size + vocab_idx] = max_val;
}

/*
 * Vectorized SPLADE Aggregation using float4
 *
 * Processes 4 vocabulary positions per thread for better memory bandwidth.
 * Requires vocab_size to be divisible by 4.
 */
__global__ void splade_aggregate_kernel_vectorized(
    const float4* __restrict__ logits,
    const float* __restrict__ attention_mask,
    float4* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size_div4
) {
    int batch_idx = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int vocab_idx = vocab_start + threadIdx.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size_div4) return;

    const float4* batch_logits = logits + batch_idx * seq_len * vocab_size_div4;
    const float* batch_mask = attention_mask + batch_idx * seq_len;

    // Initialize max values for 4 vocab positions
    float4 max_val = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

    #pragma unroll 4
    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float mask_val = batch_mask[seq_idx];

        if (mask_val > 0.0f) {
            // Vectorized load
            float4 logit = batch_logits[seq_idx * vocab_size_div4 + vocab_idx];

            // Fused activation on all 4 values
            float4 activated;
            activated.x = splade_activation(logit.x);
            activated.y = splade_activation(logit.y);
            activated.z = splade_activation(logit.z);
            activated.w = splade_activation(logit.w);

            // Update max values
            max_val.x = fmaxf(max_val.x, activated.x);
            max_val.y = fmaxf(max_val.y, activated.y);
            max_val.z = fmaxf(max_val.z, activated.z);
            max_val.w = fmaxf(max_val.w, activated.w);
        }
    }

    // Handle masked positions
    if (max_val.x == -FLT_MAX) max_val.x = 0.0f;
    if (max_val.y == -FLT_MAX) max_val.y = 0.0f;
    if (max_val.z == -FLT_MAX) max_val.z = 0.0f;
    if (max_val.w == -FLT_MAX) max_val.w = 0.0f;

    // Vectorized store
    output[batch_idx * vocab_size_div4 + vocab_idx] = max_val;
}

/*
 * FP16 SPLADE Aggregation
 *
 * Uses half precision for inputs, float for accumulation.
 * 2x memory bandwidth improvement over FP32.
 */
__global__ void splade_aggregate_kernel_fp16(
    const __half* __restrict__ logits,
    const __half* __restrict__ attention_mask,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    int batch_idx = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int vocab_idx = vocab_start + threadIdx.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size) return;

    const __half* batch_logits = logits + batch_idx * seq_len * vocab_size;
    const __half* batch_mask = attention_mask + batch_idx * seq_len;

    float max_val = -FLT_MAX;

    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float mask_val = __half2float(batch_mask[seq_idx]);

        if (mask_val > 0.0f) {
            float logit = __half2float(batch_logits[seq_idx * vocab_size + vocab_idx]);
            float activated = splade_activation(logit);
            max_val = fmaxf(max_val, activated);
        }
    }

    if (max_val == -FLT_MAX) max_val = 0.0f;
    output[batch_idx * vocab_size + vocab_idx] = max_val;
}

// ============================================================================
// FLOPS Regularization Kernels
// ============================================================================

/*
 * Phase 1: Compute column-wise absolute sums
 *
 * Each block handles a portion of the vocabulary.
 * Threads iterate over batch dimension and accumulate.
 */
__global__ void flops_reg_column_sum_kernel(
    const float* __restrict__ activations,
    float* __restrict__ col_sums,
    int batch_size,
    int vocab_size
) {
    int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vocab_idx >= vocab_size) return;

    float sum = 0.0f;

    // Iterate over batch dimension
    #pragma unroll 8
    for (int b = 0; b < batch_size; b++) {
        float val = activations[b * vocab_size + vocab_idx];
        sum += fabsf(val);
    }

    col_sums[vocab_idx] = sum;
}

/*
 * Phase 2: Compute final FLOPS loss
 *
 * Reduces column sums: loss = sum_v((col_sum[v] / batch_size)^2)
 */
__global__ void flops_reg_final_kernel(
    const float* __restrict__ col_sums,
    float* __restrict__ loss,
    int vocab_size,
    float batch_size_inv
) {
    __shared__ float shared_sum[32];  // For warp reduction

    float local_sum = 0.0f;

    // Grid-stride loop over vocabulary
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab_size; v += gridDim.x * blockDim.x) {
        float mean = col_sums[v] * batch_size_inv;
        local_sum += mean * mean;
    }

    // Block-level reduction
    float block_sum = block_reduce_sum(local_sum, shared_sum);

    // First thread writes block result
    if (threadIdx.x == 0) {
        atomicAdd(loss, block_sum);
    }
}

/*
 * Fused FLOPS Regularization (for small batch sizes)
 *
 * Computes everything in one kernel using shared memory.
 * More efficient when batch_size is small.
 */
__global__ void flops_reg_fused_kernel(
    const float* __restrict__ activations,
    float* __restrict__ loss,
    int batch_size,
    int vocab_size
) {
    __shared__ float shared_sum[32];

    float local_sum = 0.0f;
    float batch_inv = 1.0f / batch_size;

    // Each thread handles multiple vocab positions
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab_size; v += gridDim.x * blockDim.x) {
        // Compute column sum
        float col_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            col_sum += fabsf(activations[b * vocab_size + v]);
        }

        // Mean and square
        float mean = col_sum * batch_inv;
        local_sum += mean * mean;
    }

    // Reduce across block
    float block_sum = block_reduce_sum(local_sum, shared_sum);

    if (threadIdx.x == 0) {
        atomicAdd(loss, block_sum);
    }
}

// ============================================================================
// Top-K Extraction Kernel
// ============================================================================

/*
 * Bitonic sort helper for top-k
 */
__device__ void bitonic_compare_and_swap(
    float* values, int* indices,
    int i, int j, bool ascending
) {
    if ((values[i] > values[j]) == ascending) {
        // Swap values
        float tmp_v = values[i];
        values[i] = values[j];
        values[j] = tmp_v;
        // Swap indices
        int tmp_i = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_i;
    }
}

/*
 * Top-K Extraction Kernel
 *
 * Uses partial bitonic sort to extract top-k elements.
 * More efficient than full sort for small k.
 */
__global__ void topk_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ out_values,
    int* __restrict__ out_indices,
    int vocab_size,
    int k
) {
    // Shared memory for partial sorting
    extern __shared__ char shared_mem[];
    float* s_values = (float*)shared_mem;
    int* s_indices = (int*)(s_values + blockDim.x);

    int batch_idx = blockIdx.x;
    const float* vec = vectors + batch_idx * vocab_size;

    // Initialize with -infinity
    int tid = threadIdx.x;
    s_values[tid] = -FLT_MAX;
    s_indices[tid] = -1;

    __syncthreads();

    // Each thread scans a portion of the vocabulary
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = vec[v];

        // Check if this value should be in top-k
        // Using simple insertion for small k
        if (val > s_values[k - 1]) {
            // Find insertion position
            int pos = k - 1;
            while (pos > 0 && val > s_values[pos - 1]) {
                s_values[pos] = s_values[pos - 1];
                s_indices[pos] = s_indices[pos - 1];
                pos--;
            }
            s_values[pos] = val;
            s_indices[pos] = v;
        }
        __syncthreads();
    }

    // Write output
    if (tid < k) {
        out_values[batch_idx * k + tid] = s_values[tid];
        out_indices[batch_idx * k + tid] = s_indices[tid];
    }
}

// ============================================================================
// Sparse Statistics Kernel
// ============================================================================

__global__ void sparse_stats_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ stats,
    int vocab_size,
    float threshold
) {
    __shared__ float s_count[32];
    __shared__ float s_sum[32];
    __shared__ float s_max[32];

    int batch_idx = blockIdx.x;
    const float* vec = vectors + batch_idx * vocab_size;

    float local_count = 0.0f;
    float local_sum = 0.0f;
    float local_max = 0.0f;

    // Grid-stride loop
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = fabsf(vec[v]);
        if (val >= threshold) {
            local_count += 1.0f;
            local_sum += val;
            local_max = fmaxf(local_max, val);
        }
    }

    // Reduce count
    float block_count = block_reduce_sum(local_count, s_count);
    __syncthreads();

    // Reduce sum
    float block_sum = block_reduce_sum(local_sum, s_sum);
    __syncthreads();

    // Reduce max
    float block_max = block_reduce_max(local_max, s_max);

    // Write stats
    if (threadIdx.x == 0) {
        stats[batch_idx * 3 + 0] = block_count;
        stats[batch_idx * 3 + 1] = block_sum;
        stats[batch_idx * 3 + 2] = block_max;
    }
}

// ============================================================================
// Combined Forward Pass Kernel
// ============================================================================

/*
 * Fused SPLADE + Classification
 *
 * Computes sparse vectors and immediately applies classifier.
 * Avoids storing full vocab_size vectors when only predictions needed.
 */
__global__ void splade_classify_fused_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ attention_mask,
    const float* __restrict__ classifier_weights,
    const float* __restrict__ classifier_bias,
    float* __restrict__ class_logits,
    int batch_size,
    int seq_len,
    int vocab_size,
    int num_labels
) {
    // Each block handles one (batch, label) pair
    int batch_idx = blockIdx.x;
    int label_idx = blockIdx.y;

    if (batch_idx >= batch_size || label_idx >= num_labels) return;

    __shared__ float shared_sum[32];

    const float* batch_logits = logits + batch_idx * seq_len * vocab_size;
    const float* batch_mask = attention_mask + batch_idx * seq_len;
    const float* weights = classifier_weights + label_idx * vocab_size;

    float local_sum = 0.0f;

    // Each thread handles multiple vocab positions
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        // Compute max over sequence for this vocab position
        float max_val = -FLT_MAX;

        for (int s = 0; s < seq_len; s++) {
            if (batch_mask[s] > 0.0f) {
                float logit = batch_logits[s * vocab_size + v];
                float activated = splade_activation(logit);
                max_val = fmaxf(max_val, activated);
            }
        }

        if (max_val == -FLT_MAX) max_val = 0.0f;

        // Immediately multiply by classifier weight
        local_sum += max_val * weights[v];
    }

    // Reduce across threads
    float block_sum = block_reduce_sum(local_sum, shared_sum);

    // Add bias and write output
    if (threadIdx.x == 0) {
        class_logits[batch_idx * num_labels + label_idx] = block_sum + classifier_bias[label_idx];
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

void launch_splade_aggregate(
    const float* logits,
    const float* attention_mask,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    cudaStream_t stream
) {
    // Use vectorized kernel if vocab_size is divisible by 4
    if (vocab_size % 4 == 0) {
        int vocab_size_div4 = vocab_size / 4;
        int threads = BLOCK_SIZE_AGGREGATE;
        dim3 grid(batch_size, (vocab_size_div4 + threads - 1) / threads);

        splade_aggregate_kernel_vectorized<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(logits),
            attention_mask,
            reinterpret_cast<float4*>(output),
            batch_size, seq_len, vocab_size_div4
        );
    } else {
        int threads = BLOCK_SIZE_AGGREGATE;
        dim3 grid(batch_size, (vocab_size + threads - 1) / threads);

        splade_aggregate_kernel<<<grid, threads, 0, stream>>>(
            logits, attention_mask, output,
            batch_size, seq_len, vocab_size
        );
    }
}

void launch_splade_aggregate_fp16(
    const void* logits,
    const void* attention_mask,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    cudaStream_t stream
) {
    int threads = BLOCK_SIZE_AGGREGATE;
    dim3 grid(batch_size, (vocab_size + threads - 1) / threads);

    splade_aggregate_kernel_fp16<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(logits),
        reinterpret_cast<const __half*>(attention_mask),
        output,
        batch_size, seq_len, vocab_size
    );
}

void launch_flops_reg(
    const float* activations,
    float* loss,
    float* workspace,
    int batch_size,
    int vocab_size,
    cudaStream_t stream
) {
    // Clear loss
    cudaMemsetAsync(loss, 0, sizeof(float), stream);

    // For small batches, use fused kernel
    if (batch_size <= 64) {
        int threads = BLOCK_SIZE_FLOPS;
        int blocks = (vocab_size + threads - 1) / threads;
        blocks = min(blocks, 256);  // Limit blocks for atomicAdd efficiency

        flops_reg_fused_kernel<<<blocks, threads, 0, stream>>>(
            activations, loss, batch_size, vocab_size
        );
    } else {
        // Two-phase approach for larger batches
        int threads = BLOCK_SIZE_FLOPS;

        // Phase 1: Column sums
        int blocks1 = (vocab_size + threads - 1) / threads;
        flops_reg_column_sum_kernel<<<blocks1, threads, 0, stream>>>(
            activations, workspace, batch_size, vocab_size
        );

        // Phase 2: Final reduction
        int blocks2 = min((vocab_size + threads - 1) / threads, 256);
        flops_reg_final_kernel<<<blocks2, threads, 0, stream>>>(
            workspace, loss, vocab_size, 1.0f / batch_size
        );
    }
}

void launch_topk(
    const float* vectors,
    float* values,
    int* indices,
    int batch_size,
    int vocab_size,
    int k,
    cudaStream_t stream
) {
    // Shared memory for values and indices
    int threads = max(k, 32);
    threads = min(threads, BLOCK_SIZE_TOPK);
    size_t shared_size = threads * (sizeof(float) + sizeof(int));

    topk_kernel<<<batch_size, threads, shared_size, stream>>>(
        vectors, values, indices, vocab_size, k
    );
}

size_t get_flops_reg_workspace_size(int vocab_size) {
    return vocab_size * sizeof(float);
}

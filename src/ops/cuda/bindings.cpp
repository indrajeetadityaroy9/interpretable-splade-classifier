/*
 * PyTorch C++ Extension Bindings for SPLADE CUDA Kernels
 *
 * Provides Python-accessible functions that wrap CUDA kernels
 * with automatic tensor validation and dtype handling.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of launcher functions
extern "C" {
    void launch_splade_aggregate(
        const float* logits,
        const float* attention_mask,
        float* output,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream
    );

    void launch_splade_aggregate_fp16(
        const void* logits,
        const void* attention_mask,
        float* output,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream
    );

    void launch_flops_reg(
        const float* activations,
        float* loss,
        float* workspace,
        int batch_size,
        int vocab_size,
        cudaStream_t stream
    );

    void launch_topk(
        const float* vectors,
        float* values,
        int* indices,
        int batch_size,
        int vocab_size,
        int k,
        cudaStream_t stream
    );

    size_t get_flops_reg_workspace_size(int vocab_size);
}

// ============================================================================
// Tensor Validation Utilities
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// Python-Accessible Functions
// ============================================================================

/*
 * SPLADE Aggregation: Fused log1p(relu(x)) * mask -> max_pool
 *
 * Args:
 *     logits: [batch_size, seq_len, vocab_size] float32 or float16
 *     attention_mask: [batch_size, seq_len] float32 or float16
 *
 * Returns:
 *     output: [batch_size, vocab_size] float32
 */
torch::Tensor splade_aggregate_cuda(
    torch::Tensor logits,
    torch::Tensor attention_mask
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(attention_mask);

    TORCH_CHECK(logits.dim() == 3, "logits must be 3D [batch, seq, vocab]");
    TORCH_CHECK(attention_mask.dim() == 2, "attention_mask must be 2D [batch, seq]");

    int batch_size = logits.size(0);
    int seq_len = logits.size(1);
    int vocab_size = logits.size(2);

    TORCH_CHECK(attention_mask.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(attention_mask.size(1) == seq_len, "seq_len mismatch");

    // Allocate output tensor (always float32 for numerical stability)
    auto output = torch::empty({batch_size, vocab_size},
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(logits.device()));

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch based on input dtype
    if (logits.dtype() == torch::kFloat32) {
        // Ensure mask is also float32
        auto mask_f32 = attention_mask.to(torch::kFloat32);

        launch_splade_aggregate(
            logits.data_ptr<float>(),
            mask_f32.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, seq_len, vocab_size,
            stream
        );
    } else if (logits.dtype() == torch::kFloat16) {
        auto mask_f16 = attention_mask.to(torch::kFloat16);

        launch_splade_aggregate_fp16(
            logits.data_ptr<at::Half>(),
            mask_f16.data_ptr<at::Half>(),
            output.data_ptr<float>(),
            batch_size, seq_len, vocab_size,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Expected float32 or float16.");
    }

    return output;
}

/*
 * FLOPS Regularization Loss
 *
 * Computes: L_FLOPS = sum_j (mean_i |w_ij|)^2
 *
 * Args:
 *     activations: [batch_size, vocab_size] float32
 *
 * Returns:
 *     loss: scalar float32 tensor
 */
torch::Tensor flops_reg_cuda(torch::Tensor activations) {
    CHECK_INPUT(activations);
    TORCH_CHECK(activations.dim() == 2, "activations must be 2D [batch, vocab]");
    TORCH_CHECK(activations.dtype() == torch::kFloat32, "activations must be float32");

    int batch_size = activations.size(0);
    int vocab_size = activations.size(1);

    // Allocate output and workspace
    auto loss = torch::zeros({1}, activations.options());

    size_t workspace_size = get_flops_reg_workspace_size(vocab_size);
    auto workspace = torch::empty({static_cast<long>(workspace_size / sizeof(float))},
                                  activations.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_flops_reg(
        activations.data_ptr<float>(),
        loss.data_ptr<float>(),
        workspace.data_ptr<float>(),
        batch_size, vocab_size,
        stream
    );

    return loss.squeeze(0);
}

/*
 * Top-K Extraction
 *
 * Extracts top-k values and indices from sparse vectors.
 *
 * Args:
 *     vectors: [batch_size, vocab_size] float32
 *     k: number of top elements to extract
 *
 * Returns:
 *     tuple of (values [batch, k], indices [batch, k])
 */
std::vector<torch::Tensor> topk_cuda(
    torch::Tensor vectors,
    int k
) {
    CHECK_INPUT(vectors);
    TORCH_CHECK(vectors.dim() == 2, "vectors must be 2D [batch, vocab]");
    TORCH_CHECK(vectors.dtype() == torch::kFloat32, "vectors must be float32");

    int batch_size = vectors.size(0);
    int vocab_size = vectors.size(1);

    TORCH_CHECK(k > 0 && k <= vocab_size, "k must be in range (0, vocab_size]");

    auto values = torch::empty({batch_size, k}, vectors.options());
    auto indices = torch::empty({batch_size, k},
                                torch::TensorOptions()
                                    .dtype(torch::kInt32)
                                    .device(vectors.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_topk(
        vectors.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size, vocab_size, k,
        stream
    );

    return {values, indices};
}

/*
 * Sparse Statistics
 *
 * Computes sparsity statistics for a batch of vectors.
 *
 * Args:
 *     vectors: [batch_size, vocab_size] float32
 *     threshold: values with abs < threshold are considered zero
 *
 * Returns:
 *     stats: [batch_size, 3] containing [num_nonzero, sum_abs, max_abs]
 */
torch::Tensor sparse_stats_cuda(
    torch::Tensor vectors,
    float threshold
) {
    CHECK_INPUT(vectors);
    TORCH_CHECK(vectors.dim() == 2, "vectors must be 2D");

    int batch_size = vectors.size(0);
    int vocab_size = vectors.size(1);

    auto stats = torch::empty({batch_size, 3}, vectors.options());

    // This kernel is declared in the header but we'll call it via launch function
    // For now, compute on CPU as a fallback
    auto vec_cpu = vectors.cpu();
    auto stats_cpu = torch::empty({batch_size, 3});

    for (int b = 0; b < batch_size; b++) {
        float count = 0, sum = 0, max_val = 0;
        auto row = vec_cpu[b];
        for (int v = 0; v < vocab_size; v++) {
            float val = std::abs(row[v].item<float>());
            if (val >= threshold) {
                count += 1;
                sum += val;
                max_val = std::max(max_val, val);
            }
        }
        stats_cpu[b][0] = count;
        stats_cpu[b][1] = sum;
        stats_cpu[b][2] = max_val;
    }

    return stats_cpu.to(vectors.device());
}

/*
 * Check if CUDA kernels are available
 */
bool cuda_available() {
    return true;
}

/*
 * Get CUDA device properties
 */
std::string get_cuda_info() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::stringstream ss;
    ss << "Device: " << props.name << "\n";
    ss << "Compute Capability: " << props.major << "." << props.minor << "\n";
    ss << "Total Memory: " << props.totalGlobalMem / (1024*1024*1024) << " GB\n";
    ss << "SM Count: " << props.multiProcessorCount << "\n";
    ss << "Shared Memory per Block: " << props.sharedMemPerBlock / 1024 << " KB\n";
    ss << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";

    return ss.str();
}

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SPLADE CUDA Kernels - High-performance implementations";

    m.def("splade_aggregate", &splade_aggregate_cuda,
          "Fused SPLADE aggregation: log1p(relu(x)) * mask -> max_pool",
          py::arg("logits"), py::arg("attention_mask"));

    m.def("flops_reg", &flops_reg_cuda,
          "FLOPS regularization loss for sparsity enforcement",
          py::arg("activations"));

    m.def("topk", &topk_cuda,
          "Extract top-k values and indices from sparse vectors",
          py::arg("vectors"), py::arg("k"));

    m.def("sparse_stats", &sparse_stats_cuda,
          "Compute sparsity statistics [num_nonzero, sum_abs, max_abs]",
          py::arg("vectors"), py::arg("threshold") = 1e-6f);

    m.def("cuda_available", &cuda_available,
          "Check if CUDA kernels are available");

    m.def("get_cuda_info", &get_cuda_info,
          "Get CUDA device information");
}

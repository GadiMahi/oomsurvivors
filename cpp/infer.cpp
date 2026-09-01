#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

using namespace nvinfer1;

// Simple logger required by TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    }

int main() {
    std::cout << "Loading TensorRT Engine..." << std::endl;
    
    // 1. Read the serialized engine from disk
    std::ifstream file("nafnet_fp16.engine", std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read engine file!" << std::endl;
        return -1;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    // 2. Deserialize Engine and Create Context
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // 3. Define the Dynamic Input Dimensions (e.g., evaluating a 256x256 patch)
    int32_t batchSize = 1;
    int32_t channels = 1;
    int32_t height = 256;
    int32_t width = 256;
    context->setInputShape("input", Dims4{batchSize, channels, height, width});

    // 4. Allocate GPU Memory
    size_t inputSize = batchSize * channels * height * width * sizeof(float);
    // Remember output is scaled by 2x in both spatial dimensions
    size_t outputSize = batchSize * channels * (height * 2) * (width * 2) * sizeof(float); 

    void* d_input;
    void* d_output;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&d_output, outputSize));

    // Map tensor names to the allocated GPU memory
    context->setTensorAddress("input", d_input);
    context->setTensorAddress("output", d_output);

    // 5. Create CUDA Stream for asynchronous execution
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // NOTE: Insert your OpenCV/libtiff image loading here. 
    // Use cudaMemcpyAsync(d_input, host_image_ptr, inputSize, cudaMemcpyHostToDevice, stream);

    // 6. Measure Latency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    
    // Execute the forward pass
    context->enqueueV3(stream);
    
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "✅ Native Inference executed in: " << milliseconds << " ms" << std::endl;

    // NOTE: Extract your output here.
    // Use cudaMemcpyAsync(host_output_ptr, d_output, outputSize, cudaMemcpyDeviceToHost, stream);

    // 7. Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
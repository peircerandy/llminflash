#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <utility>
#include <omp.h>

constexpr size_t HIDDEN_SIZE = 4096;
constexpr size_t FFN_DIM = 16384;
constexpr size_t NUM_LAYERS = 32;
constexpr size_t BYTES_PER_VAL = 2;

constexpr size_t NEURON_BUNDLE_VALS = HIDDEN_SIZE * 2;
constexpr size_t NEURON_BUNDLE_SIZE = NEURON_BUNDLE_VALS * BYTES_PER_VAL;
constexpr int TOP_K = 1024; // Apple's ~6% sparsity

class FlashFFNEngine {
private:
    uint8_t *ffn_mapped;
    size_t ffn_size;
    float *predictor_mapped;
    size_t predictor_size;

public:
    FlashFFNEngine(const char* ffn_path) {
        int fd = open(ffn_path, O_RDONLY);
        if (fd == -1) { perror("FFN open failed"); exit(1); }
        struct stat st; fstat(fd, &st); ffn_size = st.st_size;
        ffn_mapped = (uint8_t*)mmap(nullptr, ffn_size, PROT_READ, MAP_SHARED, fd, 0);
        madvise(ffn_mapped, ffn_size, MADV_RANDOM);

        int p_fd = open("/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin", O_RDONLY);
        if (p_fd != -1) {
            struct stat p_st; fstat(p_fd, &p_st); predictor_size = p_st.st_size;
            predictor_mapped = (float*)mmap(nullptr, predictor_size, PROT_READ, MAP_SHARED, p_fd, 0);
            madvise(predictor_mapped, predictor_size, MADV_NORMAL);
        }
    }

    ~FlashFFNEngine() {
        munmap(ffn_mapped, ffn_size);
        if (predictor_mapped) munmap(predictor_mapped, predictor_size);
    }

    inline float h2f(uint16_t h) {
        uint16_t exp = (h >> 10) & 0x001F;
        uint16_t mant = h & 0x03FF;
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t res;
        if (exp == 0) res = sign; 
        else if (exp == 31) res = sign | 0x7F800000 | (mant << 13);
        else res = sign | ((exp + 112) << 23) | (mant << 13);
        float f; std::memcpy(&f, &res, 4);
        return f;
    }

    void execute_ffn(int layer_idx, float* norm_x, float* ffn_out, float* fc1_bias, int mode) {
        size_t layer_offset = (size_t)layer_idx * FFN_DIM * NEURON_BUNDLE_SIZE;
        uint8_t* layer_base_ptr = ffn_mapped + layer_offset;

        std::vector<int> active_indices;
        std::fill_n(ffn_out, HIDDEN_SIZE, 0.0f);

        // ==========================================
        // MODE 0: ML PREDICTOR
        // ==========================================
        if (mode == 0 && predictor_mapped) {
            size_t predictor_layer_offset = (size_t)layer_idx * ((128 * 4096) + (16384 * 128));
            float* down_weight = predictor_mapped + predictor_layer_offset;
            float* up_weight = down_weight + (128 * 4096);

            float hidden[128] = {0};
            for (int i = 0; i < 128; i++) {
                for (int j = 0; j < 4096; j++) hidden[i] += down_weight[i * 4096 + j] * norm_x[j];
            }

            std::vector<std::pair<float, int>> scores(FFN_DIM);
            for (size_t i = 0; i < FFN_DIM; i++) {
                float out = 0;
                for (int j = 0; j < 128; j++) out += up_weight[i * 128 + j] * hidden[j];
                scores[i] = {out, (int)i};
            }

            std::nth_element(scores.begin(), scores.begin() + TOP_K, scores.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

            for (int i = 0; i < TOP_K; i++) active_indices.push_back(scores[i].second);
            std::sort(active_indices.begin(), active_indices.end());
        }

        // ==========================================
        // C++ SSD STREAMING EXECUTION
        // ==========================================
        #pragma omp parallel
        {
            std::vector<float> local_out(HIDDEN_SIZE, 0.0f);

            // MODE 0 (Predictor) & MODE 1 (Oracle): Stream sparsely
            if (mode == 0 || mode == 1) {
                size_t loop_count = (mode == 0) ? active_indices.size() : FFN_DIM;
                
                #pragma omp for schedule(dynamic, 64)
                for(size_t i = 0; i < loop_count; ++i) {
                    size_t n = (mode == 0) ? active_indices[i] : i;
                    size_t neuron_offset = n * NEURON_BUNDLE_SIZE;
                    uint16_t* b = (uint16_t*)(layer_base_ptr + neuron_offset);

                    float act = fc1_bias[n]; 
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += norm_x[h] * h2f(b[h]);
                    
                    if(!(act > 0.0f)) continue; // Hardware Sparsity Check

                    uint16_t* b_fc2 = b + HIDDEN_SIZE;
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) local_out[h] += act * h2f(b_fc2[h]);
                }
            } 
            // ==========================================
            // MODE 2: NAIVE (The Bad Apple Baseline)
            // ==========================================
            else if (mode == 2) {
                #pragma omp for schedule(static)
                for(size_t n = 0; n < FFN_DIM; ++n) {
                    size_t neuron_offset = n * NEURON_BUNDLE_SIZE;
                    uint16_t* b = (uint16_t*)(layer_base_ptr + neuron_offset);

                    float act = fc1_bias[n]; 
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += norm_x[h] * h2f(b[h]);
                    
                    uint16_t* b_fc2 = b + HIDDEN_SIZE;
                    volatile float dummy = 0; // Prevent compiler from optimizing out the memory read!
                    
                    // We FORCE the engine to read the memory from the SSD regardless of activation!
                    // This creates the massive PCIe bottleneck that proves Apple's point.
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) {
                        float val = h2f(b_fc2[h]);
                        if (act > 0.0f) local_out[h] += act * val;
                        else dummy += val; 
                    }
                }
            }

            #pragma omp critical
            {
                for(size_t h = 0; h < HIDDEN_SIZE; ++h) ffn_out[h] += local_out[h];
            }
        }
    }
};

extern "C" {
    void* init_engine(const char* ffn_path) { return new FlashFFNEngine(ffn_path); }
    
    void execute_ffn_layer(void* ptr, int layer_idx, float* in, float* out, int n, float* fc1_bias, int mode) {
        FlashFFNEngine* e = (FlashFFNEngine*)ptr;
        for(int i=0; i<n; ++i) e->execute_ffn(layer_idx, in + i*HIDDEN_SIZE, out + i*HIDDEN_SIZE, fc1_bias, mode);
    }
    
    void destroy_engine(void* ptr) { delete (FlashFFNEngine*)ptr; }
}
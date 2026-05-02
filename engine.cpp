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
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

constexpr size_t HIDDEN_SIZE = 4096;
constexpr size_t FFN_DIM = 16384;
constexpr size_t NUM_LAYERS = 32;
constexpr size_t BYTES_PER_VAL = 2;

constexpr size_t NEURON_BUNDLE_VALS = HIDDEN_SIZE * 2;
constexpr size_t NEURON_BUNDLE_SIZE = NEURON_BUNDLE_VALS * BYTES_PER_VAL;
constexpr int TOP_K = 1024; // Apple's ~6% sparsity
constexpr int WINDOW_SIZE = 5; // Section 3.1: "we keep the active neurons of past k tokens (we use k = 5)"

/**
 * Figure 6: Memory management; First we replace elements to be deleted by last elements 
 * to maintain a consecutive occupation of memory.
 */
struct LayerCache {
    float* active_weights; 
    int num_resident = 0;
    int max_resident;
    
    // Maps neuron_idx -> index in active_weights [0, num_resident-1]
    std::unordered_map<int, int> neuron_to_slot;
    // Reverse map: index in active_weights -> neuron_idx
    std::vector<int> slot_to_neuron;

    // Sliding window of active neuron sets for the last K tokens
    std::deque<std::unordered_set<int>> window;

    LayerCache(int max_neurons) : max_resident(max_neurons) {
        active_weights = new float[max_resident * NEURON_BUNDLE_VALS];
        slot_to_neuron.resize(max_resident, -1);
    }
    ~LayerCache() { delete[] active_weights; }
};

class FlashFFNEngine {
private:
    uint8_t *ffn_mapped;
    size_t ffn_size;
    float *predictor_mapped;
    size_t predictor_size;
    std::vector<LayerCache*> caches;

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

        // Section 3.3: Pre-allocate all necessary memory
        for(size_t i=0; i<NUM_LAYERS; ++i) {
            // Capping at ~25% of FFN_DIM as suggested in paper
            caches.push_back(new LayerCache(FFN_DIM / 4));
        }
    }

    ~FlashFFNEngine() {
        for(auto c : caches) delete c;
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
        LayerCache* cache = caches[layer_idx];
        std::vector<int> current_active_indices;
        std::fill_n(ffn_out, HIDDEN_SIZE, 0.0f);

        // ==========================================
        // 1. SELECTIVE PREDICTION (Section 3.1)
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

            for (int i = 0; i < TOP_K; i++) current_active_indices.push_back(scores[i].second);
        } else if (mode == 1) {
            // Oracle mode will still use sparsity but we'll assume ALL neurons 
            // could be active for the sake of simplicity in the loop, 
            // letting the act > 0 check handle actual activation.
            // Or we could pass active indices from Python.
            for (int i=0; i<(int)FFN_DIM; ++i) current_active_indices.push_back(i);
        }

        // ==========================================
        // 2. SLIDING WINDOW & DRAM MGMT (Section 3.3)
        // ==========================================
        if (mode != 2) {
            std::unordered_set<int> current_set(current_active_indices.begin(), current_active_indices.end());

            // A. Evict old neurons if window full
            if (cache->window.size() >= (size_t)WINDOW_SIZE) {
                std::unordered_set<int> oldest = cache->window.front();
                cache->window.pop_front();
                for (int n : oldest) {
                    // Check if still active in current set or any other token in window
                    bool needed = current_set.count(n);
                    if (!needed) {
                        for (auto& win_set : cache->window) {
                            if (win_set.count(n)) { needed = true; break; }
                        }
                    }
                    if (!needed) evict_neuron(cache, n);
                }
            }
            cache->window.push_back(current_set);

            // B. Bring in new neurons from Flash
            for (int n : current_active_indices) {
                if (cache->neuron_to_slot.find(n) == cache->neuron_to_slot.end()) {
                    load_neuron_to_cache(layer_idx, cache, n);
                }
            }
        }

        // ==========================================
        // 3. EXECUTION (DRAM-Resident GEMM)
        // ==========================================
        #pragma omp parallel
        {
            std::vector<float> local_out(HIDDEN_SIZE, 0.0f);

            if (mode == 0 || mode == 1) {
                // We only iterate over the CURRENT active indices, even if more are in cache
                #pragma omp for schedule(dynamic, 64)
                for(size_t i = 0; i < current_active_indices.size(); ++i) {
                    int n = current_active_indices[i];
                    int slot = cache->neuron_to_slot[n];
                    float* b = cache->active_weights + (slot * NEURON_BUNDLE_VALS);

                    float act = fc1_bias[n]; 
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += norm_x[h] * b[h];
                    
                    if(!(act > 0.0f)) continue; 

                    float* b_fc2 = b + HIDDEN_SIZE;
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) local_out[h] += act * b_fc2[h];
                }
            } 
            else if (mode == 2) {
                // Naive mode: no cache, direct SSD read
                size_t layer_offset = (size_t)layer_idx * FFN_DIM * NEURON_BUNDLE_SIZE;
                uint8_t* layer_base_ptr = ffn_mapped + layer_offset;
                #pragma omp for schedule(static)
                for(size_t n = 0; n < FFN_DIM; ++n) {
                    size_t neuron_offset = n * NEURON_BUNDLE_SIZE;
                    uint16_t* b_h = (uint16_t*)(layer_base_ptr + neuron_offset);

                    float act = fc1_bias[n]; 
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += norm_x[h] * h2f(b_h[h]);
                    
                    uint16_t* b_fc2_h = b_h + HIDDEN_SIZE;
                    volatile float dummy = 0;
                    for(size_t h = 0; h < HIDDEN_SIZE; ++h) {
                        float val = h2f(b_fc2_h[h]);
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

private:
    void evict_neuron(LayerCache* cache, int n) {
        auto it = cache->neuron_to_slot.find(n);
        if (it == cache->neuron_to_slot.end()) return;
        int slot_to_del = it->second;
        int last_slot = cache->num_resident - 1;

        if (slot_to_del != last_slot) {
            // Replace with last resident to maintain contiguous block (Figure 6)
            int last_neuron = cache->slot_to_neuron[last_slot];
            std::memcpy(cache->active_weights + slot_to_del * NEURON_BUNDLE_VALS,
                        cache->active_weights + last_slot * NEURON_BUNDLE_VALS,
                        NEURON_BUNDLE_SIZE);
            cache->neuron_to_slot[last_neuron] = slot_to_del;
            cache->slot_to_neuron[slot_to_del] = last_neuron;
        }

        cache->neuron_to_slot.erase(n);
        cache->slot_to_neuron[last_slot] = -1;
        cache->num_resident--;
    }

    void load_neuron_to_cache(int layer_idx, LayerCache* cache, int n) {
        if (cache->num_resident >= cache->max_resident) return; 

        size_t offset = (size_t)layer_idx*FFN_DIM*NEURON_BUNDLE_SIZE + (size_t)n*NEURON_BUNDLE_SIZE;
        uint16_t* src_h = (uint16_t*)(ffn_mapped + offset);
        float* dst = cache->active_weights + (cache->num_resident * NEURON_BUNDLE_VALS);

        for(size_t h=0; h<HIDDEN_SIZE; ++h) {
            dst[h] = h2f(src_h[h]);
            dst[h+HIDDEN_SIZE] = h2f(src_h[h+HIDDEN_SIZE]);
        }

        cache->neuron_to_slot[n] = cache->num_resident;
        cache->slot_to_neuron[cache->num_resident] = n;
        cache->num_resident++;
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
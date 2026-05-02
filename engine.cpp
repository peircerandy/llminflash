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
constexpr size_t NEURON_BUNDLE_VALS_SSD = HIDDEN_SIZE * 2;
constexpr size_t NEURON_BUNDLE_SIZE_SSD = NEURON_BUNDLE_VALS_SSD * BYTES_PER_VAL;
constexpr size_t NEURON_BUNDLE_VALS_CACHE = HIDDEN_SIZE * 2; 

constexpr int TOP_K = 1024; 
constexpr int WINDOW_SIZE = 5; 

struct LayerCache {
    float* active_weights; 
    int num_resident = 0;
    int max_resident;
    std::unordered_map<int, int> neuron_to_slot;
    std::vector<int> slot_to_neuron;
    std::deque<std::unordered_set<int>> window;

    LayerCache(int max_neurons) : max_resident(max_neurons) {
        active_weights = new float[max_resident * NEURON_BUNDLE_VALS_CACHE];
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

    struct PredictorInfo { size_t offset; int rank; };
    std::vector<PredictorInfo> predictor_infos;

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

        size_t current_p_offset = 0;
        for(size_t i=0; i<NUM_LAYERS; ++i) {
            caches.push_back(new LayerCache(FFN_DIM / 16)); 
            int rank = (i < 28) ? 128 : 1024;
            predictor_infos.push_back({current_p_offset, rank});
            current_p_offset += (size_t)rank * HIDDEN_SIZE + (size_t)FFN_DIM * rank;
        }
    }

    ~FlashFFNEngine() {
        for(auto c : caches) delete c;
        munmap(ffn_mapped, ffn_size);
        if (predictor_mapped) munmap(predictor_mapped, predictor_size);
    }

    inline float h2f(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF) << 13;
        uint32_t res;
        if (exp == 0x1F) res = sign | 0x7F800000 | mant;
        else if (exp == 0) res = sign | (mant ? (0x38800000 | mant) : 0);
        else res = sign | ((exp + 112) << 23) | mant;
        float f; std::memcpy(&f, &res, 4);
        return f;
    }

    void execute_ffn(int layer_idx, float* in_batch, float* out_batch, int n_tokens, float* fc1_bias, int mode) {
        LayerCache* cache = caches[layer_idx];
        std::unordered_set<int> union_active_set;

        for (int t = 0; t < n_tokens; ++t) {
            float* norm_x = in_batch + t * HIDDEN_SIZE;
            if (mode == 0 && predictor_mapped) {
                auto& p_info = predictor_infos[layer_idx];
                float* down_w = predictor_mapped + p_info.offset;
                float* up_w = down_w + (p_info.rank * HIDDEN_SIZE);

                std::vector<float> hidden(p_info.rank, 0.0f);
                for (int i = 0; i < p_info.rank; i++) {
                    for (int j = 0; j < (int)HIDDEN_SIZE; j++) hidden[i] += down_w[i * HIDDEN_SIZE + j] * norm_x[j];
                }

                std::vector<std::pair<float, int>> scores(FFN_DIM);
                for (size_t i = 0; i < FFN_DIM; i++) {
                    float val = 0;
                    for (int j = 0; j < p_info.rank; j++) {
                        float h_act = (hidden[j] > 0.0f) ? hidden[j] : 0.0f;
                        val += up_w[i * p_info.rank + j] * h_act;
                    }
                    scores[i] = {1.0f / (1.0f + std::exp(-val)), (int)i};
                }

                std::nth_element(scores.begin(), scores.begin() + TOP_K, scores.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
                for (int i = 0; i < TOP_K; i++) union_active_set.insert(scores[i].second);
            } else if (mode == 1) {
                for (int i=0; i<(int)FFN_DIM; ++i) union_active_set.insert(i);
            }
        }

        if (mode != 2) {
            std::unordered_set<int> current_set(union_active_set.begin(), union_active_set.end());
            if (cache->window.size() >= (size_t)WINDOW_SIZE) {
                std::unordered_set<int> oldest = cache->window.front();
                cache->window.pop_front();
                for (int n : oldest) {
                    bool needed = current_set.count(n);
                    if (!needed) for (auto& w : cache->window) if (w.count(n)) { needed = true; break; }
                    if (!needed) evict_neuron(cache, n);
                }
            }
            cache->window.push_back(current_set);
            for (int n : union_active_set) {
                if (cache->neuron_to_slot.find(n) == cache->neuron_to_slot.end()) load_neuron_to_cache(layer_idx, cache, n);
            }
        }

        uint8_t* layer_base = ffn_mapped + (size_t)layer_idx * FFN_DIM * NEURON_BUNDLE_SIZE_SSD;
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1)
            for (int t = 0; t < n_tokens; ++t) {
                float* x = in_batch + t * HIDDEN_SIZE;
                float* out = out_batch + t * HIDDEN_SIZE;
                std::fill_n(out, HIDDEN_SIZE, 0.0f);
                if (mode == 0 || mode == 1) {
                    for (int n : union_active_set) {
                        auto it = cache->neuron_to_slot.find(n);
                        if (it != cache->neuron_to_slot.end()) {
                            float* b = cache->active_weights + (it->second * NEURON_BUNDLE_VALS_CACHE);
                            float act = fc1_bias[n]; 
                            for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += x[h] * b[h];
                            if(act > 0.0f) {
                                float* w2 = b + HIDDEN_SIZE;
                                for(size_t h = 0; h < HIDDEN_SIZE; ++h) out[h] += act * w2[h];
                            }
                        } else {
                            uint16_t* src_h = (uint16_t*)(layer_base + (size_t)n * NEURON_BUNDLE_SIZE_SSD);
                            float act = fc1_bias[n];
                            for(size_t h=0; h<HIDDEN_SIZE; ++h) act += x[h] * h2f(src_h[h]);
                            if(act > 0.0f) {
                                uint16_t* w2_h = src_h + HIDDEN_SIZE;
                                for(size_t h=0; h<HIDDEN_SIZE; ++h) out[h] += act * h2f(w2_h[h]);
                            }
                        }
                    }
                } 
                else if (mode == 2) {
                    for(size_t n = 0; n < FFN_DIM; ++n) {
                        size_t neuron_offset = n * NEURON_BUNDLE_SIZE_SSD;
                        uint16_t* b_h = (uint16_t*)(layer_base + neuron_offset);
                        float act = fc1_bias[n]; 
                        for(size_t h = 0; h < HIDDEN_SIZE; ++h) act += x[h] * h2f(b_h[h]);
                        uint16_t* b_fc2_h = b_h + HIDDEN_SIZE;
                        if (act > 0.0f) {
                            for(size_t h = 0; h < HIDDEN_SIZE; ++h) out[h] += act * h2f(b_fc2_h[h]);
                        }
                    }
                }
            }
        }
    }

private:
    void evict_neuron(LayerCache* cache, int n) {
        auto it = cache->neuron_to_slot.find(n);
        if (it == cache->neuron_to_slot.end()) return;
        int slot = it->second; int last = cache->num_resident - 1;
        if (slot != last) {
            int last_n = cache->slot_to_neuron[last];
            std::memcpy(cache->active_weights + slot * NEURON_BUNDLE_VALS_CACHE, cache->active_weights + last * NEURON_BUNDLE_VALS_CACHE, NEURON_BUNDLE_VALS_CACHE * sizeof(float));
            cache->neuron_to_slot[last_n] = slot; cache->slot_to_neuron[slot] = last_n;
        }
        cache->neuron_to_slot.erase(n); cache->num_resident--;
    }

    void load_neuron_to_cache(int l_idx, LayerCache* cache, int n) {
        if (cache->num_resident >= cache->max_resident) return; 
        uint16_t* src = (uint16_t*)(ffn_mapped + (size_t)l_idx*FFN_DIM*NEURON_BUNDLE_SIZE_SSD + (size_t)n*NEURON_BUNDLE_SIZE_SSD);
        float* dst = cache->active_weights + (cache->num_resident * NEURON_BUNDLE_VALS_CACHE);
        for(size_t h=0; h<HIDDEN_SIZE; ++h) { dst[h] = h2f(src[h]); dst[h+HIDDEN_SIZE] = h2f(src[h+HIDDEN_SIZE]); }
        cache->neuron_to_slot[n] = cache->num_resident; cache->slot_to_neuron[cache->num_resident] = n; cache->num_resident++;
    }
};

extern "C" {
    void* init_engine(const char* p) { return new FlashFFNEngine(p); }
    void execute_ffn_layer(void* ptr, int l, float* in, float* out, int n, float* b, int m) { ((FlashFFNEngine*)ptr)->execute_ffn(l, in, out, n, b, m); }
    void destroy_engine(void* ptr) { delete (FlashFFNEngine*)ptr; }
}

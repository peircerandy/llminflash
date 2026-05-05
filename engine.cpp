/**
 * engine.cpp: High-performance C++ core with fine-grained parallelism.
 * Now with ARM NEON SIMD support and dynamic model architecture compatibility (OPT & Llama 3).
 */

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
#include <set>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

constexpr size_t BYTES_PER_VAL = 2; 

struct LayerCache {
    float* active_weights;
    int num_resident = 0;
    int max_resident;
    std::unordered_map<int, int> neuron_to_slot;
    std::vector<int> slot_to_neuron;
    std::deque<std::unordered_set<int>> window;

    LayerCache(int max_neurons, size_t vals_per_cache_neuron) : max_resident(max_neurons) {
        active_weights = new float[max_resident * vals_per_cache_neuron];
        slot_to_neuron.resize(max_resident, -1);
    }
    ~LayerCache() { delete[] active_weights; }
};

class FlashFFNEngine {
private:
    uint8_t *ffn_mapped = nullptr;
    size_t ffn_size = 0;
    float *predictor_mapped = nullptr;
    size_t predictor_size = 0;
    std::vector<LayerCache*> caches;

    struct PredictorInfo { size_t offset; int rank; };
    std::vector<PredictorInfo> predictor_infos;

    int top_k = 1024;
    float threshold = 0.5f;
    int window_size = 5;
    
    // Architecture Config
    size_t hidden_size;
    size_t ffn_dim;
    size_t num_layers;
    bool is_llama3;
    size_t vals_per_ssd_neuron;
    size_t bytes_per_ssd_neuron;
    size_t vals_per_cache_neuron;

public:
    FlashFFNEngine(const char* ffn_path, const char* predictor_path, size_t hs, size_t fd, size_t nl, bool llama3) 
        : hidden_size(hs), ffn_dim(fd), num_layers(nl), is_llama3(llama3) {
            
        vals_per_ssd_neuron = is_llama3 ? hidden_size * 3 : hidden_size * 2;
        bytes_per_ssd_neuron = vals_per_ssd_neuron * BYTES_PER_VAL;
        vals_per_cache_neuron = (is_llama3 ? hidden_size * 3 : hidden_size * 2) + 1; // +1 for bias (if OPT)

        int fd_ffn = open(ffn_path, O_RDONLY);
        if (fd_ffn != -1) {
            struct stat st; fstat(fd_ffn, &st); ffn_size = st.st_size;
            ffn_mapped = (uint8_t*)mmap(nullptr, ffn_size, PROT_READ, MAP_SHARED, fd_ffn, 0);
            madvise(ffn_mapped, ffn_size, MADV_RANDOM);
            close(fd_ffn);
        }

        if (predictor_path && std::strlen(predictor_path) > 0) {
            int p_fd = open(predictor_path, O_RDONLY);
            if (p_fd != -1) {
                struct stat p_st; fstat(p_fd, &p_st); predictor_size = p_st.st_size;
                predictor_mapped = (float*)mmap(nullptr, predictor_size, PROT_READ, MAP_SHARED, p_fd, 0);
                madvise(predictor_mapped, predictor_size, MADV_NORMAL);
                close(p_fd);
            }
        }

        size_t current_p_offset = 0;
        for(size_t i=0; i<num_layers; ++i) {
            caches.push_back(new LayerCache(1024, vals_per_cache_neuron)); 
            int rank = (i < 28) ? 128 : 1024;
            predictor_infos.push_back({current_p_offset, rank});
            current_p_offset += (size_t)rank * hidden_size + (size_t)ffn_dim * rank;
        }
    }

    ~FlashFFNEngine() {
        for(auto c : caches) delete c;
        if (ffn_mapped) munmap(ffn_mapped, ffn_size);
        if (predictor_mapped) munmap(predictor_mapped, predictor_size);
    }

    void set_config(int k, float t, int w) { top_k = k; threshold = t; window_size = w; }

    void set_predictor_info(int layer_idx, int rank, size_t offset) {
        if (layer_idx < (int)predictor_infos.size()) {
            predictor_infos[layer_idx] = {offset, rank};
        }
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

    float dot_product(float* a, float* b, size_t len) {
        float sum = 0.0f;
        size_t j = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (; j + 3 < len; j += 4) {
            float32x4_t a_vec = vld1q_f32(&a[j]);
            float32x4_t b_vec = vld1q_f32(&b[j]);
            sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
        }
        sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
              vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
#endif
        for (; j < len; j++) sum += a[j] * b[j];
        return sum;
    }
    
    void add_scaled(float* out, float* w, float scale, size_t len) {
        size_t j = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        float32x4_t scale_vec = vdupq_n_f32(scale);
        for (; j + 3 < len; j += 4) {
            float32x4_t out_vec = vld1q_f32(&out[j]);
            float32x4_t w_vec = vld1q_f32(&w[j]);
            out_vec = vmlaq_f32(out_vec, w_vec, scale_vec);
            vst1q_f32(&out[j], out_vec);
        }
#endif
        for (; j < len; j++) out[j] += w[j] * scale;
    }

    void execute_ffn(int layer_idx, float* in_batch, float* out_batch, int n_tokens, float* fc1_bias, int mode) {
        LayerCache* cache = caches[layer_idx];
        std::set<int> union_active_set;

        // --- Step 1: PREDICTION (Highly Parallel) ---
        if (mode == 0 && predictor_mapped) {
            auto& p_info = predictor_infos[layer_idx];
            float* down_w = predictor_mapped + p_info.offset;
            float* up_w = down_w + (p_info.rank * hidden_size);

            for (int t = 0; t < n_tokens; ++t) {
                float* norm_x = in_batch + t * hidden_size;
                std::vector<float> hidden(p_info.rank, 0.0f);
                
                // Parallel Hidden Pass
                #pragma omp parallel for
                for (int i = 0; i < p_info.rank; i++) {
                    float sum = dot_product(down_w + i * hidden_size, norm_x, hidden_size);
                    hidden[i] = (sum > 0.0f) ? sum : 0.0f;
                }

                // Parallel Output Pass
                std::vector<std::pair<float, int>> scores(ffn_dim);
                #pragma omp parallel for
                for (size_t i = 0; i < ffn_dim; i++) {
                    float val = dot_product(up_w + i * p_info.rank, hidden.data(), p_info.rank);
                    scores[i] = {1.0f / (1.0f + std::exp(-val)), (int)i};
                }

                // Top-K selection
                std::nth_element(scores.begin(), scores.begin() + top_k, scores.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
                
                for (int i = 0; i < top_k; i++) union_active_set.insert(scores[i].second);
                for (int i = top_k; i < (int)ffn_dim; i++) {
                    if (scores[i].first > threshold) union_active_set.insert(scores[i].second);
                }
            }
        } else if (mode == 1 || (mode == 0 && !predictor_mapped)) {
            for (int i=0; i<(int)ffn_dim; ++i) union_active_set.insert(i);
        }

        // --- Step 2: CACHE & SEQUENTIAL I/O ---
        if (mode == 0) {
            std::unordered_set<int> current_set(union_active_set.begin(), union_active_set.end());
            if (cache->window.size() >= (size_t)window_size) {
                std::unordered_set<int> oldest = cache->window.front();
                cache->window.pop_front();
                for (int n : oldest) {
                    bool needed = current_set.count(n);
                    if (!needed) { for (auto& win_set : cache->window) if (win_set.count(n)) { needed = true; break; } }
                    if (!needed) evict_neuron(cache, n);
                }
            }
            cache->window.push_back(current_set);
            for (int n : union_active_set) {
                if (cache->neuron_to_slot.find(n) == cache->neuron_to_slot.end()) load_neuron_to_cache(layer_idx, cache, n, fc1_bias);
            }
        }

        // --- Step 3: COMPUTE ---
        uint8_t* layer_base = ffn_mapped + (size_t)layer_idx * ffn_dim * bytes_per_ssd_neuron;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int t = 0; t < n_tokens; ++t) {
            float* x = in_batch + t * hidden_size;
            float* out = out_batch + t * hidden_size;
            std::fill_n(out, hidden_size, 0.0f);

            if (mode == 0) {
                for (int n : union_active_set) {
                    auto it = cache->neuron_to_slot.find(n);
                    if (it != cache->neuron_to_slot.end()) {
                        float* b = cache->active_weights + (it->second * vals_per_cache_neuron);
                        if (is_llama3) {
                            float gate_act = dot_product(b, x, hidden_size);
                            float up_act = dot_product(b + hidden_size, x, hidden_size);
                            float silu_gate = gate_act / (1.0f + std::exp(-gate_act));
                            float act = silu_gate * up_act;
                            add_scaled(out, b + hidden_size * 2, act, hidden_size);
                        } else {
                            float act = b[hidden_size * 2]; // bias
                            act += dot_product(b, x, hidden_size);
                            if(act > 0.0f) add_scaled(out, b + hidden_size, act, hidden_size);
                        }
                    } else {
                        // Fallback (should be rare if cache size is sufficient)
                        compute_neuron_from_ssd(layer_base, n, x, out, fc1_bias ? fc1_bias[n] : 0.0f);
                    }
                }
            } else {
                for(size_t n = 0; n < ffn_dim; ++n) {
                    compute_neuron_from_ssd(layer_base, n, x, out, fc1_bias ? fc1_bias[n] : 0.0f);
                }
            }
        }
    }

private:
    void compute_neuron_from_ssd(uint8_t* layer_base, int n, float* x, float* out, float bias) {
        uint16_t* src_h = (uint16_t*)(layer_base + (size_t)n * bytes_per_ssd_neuron);
        std::vector<float> w1(hidden_size);
        for(size_t h=0; h<hidden_size; ++h) w1[h] = h2f(src_h[h]);
        
        if (is_llama3) {
            std::vector<float> w2(hidden_size);
            std::vector<float> w3(hidden_size);
            uint16_t* src_up = src_h + hidden_size;
            uint16_t* src_down = src_h + hidden_size * 2;
            for(size_t h=0; h<hidden_size; ++h) {
                w2[h] = h2f(src_up[h]);
                w3[h] = h2f(src_down[h]);
            }
            float gate_act = dot_product(w1.data(), x, hidden_size);
            float up_act = dot_product(w2.data(), x, hidden_size);
            float silu_gate = gate_act / (1.0f + std::exp(-gate_act));
            float act = silu_gate * up_act;
            add_scaled(out, w3.data(), act, hidden_size);
        } else {
            float act = bias + dot_product(w1.data(), x, hidden_size);
            if(act > 0.0f) {
                std::vector<float> w2(hidden_size);
                uint16_t* src_w2 = src_h + hidden_size;
                for(size_t h=0; h<hidden_size; ++h) w2[h] = h2f(src_w2[h]);
                add_scaled(out, w2.data(), act, hidden_size);
            }
        }
    }

    void evict_neuron(LayerCache* cache, int n) {
        auto it = cache->neuron_to_slot.find(n);
        if (it == cache->neuron_to_slot.end()) return;
        int slot = it->second; int last = cache->num_resident - 1;
        if (slot != last) {
            int last_n = cache->slot_to_neuron[last];
            std::memcpy(cache->active_weights + slot * vals_per_cache_neuron, 
                        cache->active_weights + last * vals_per_cache_neuron, 
                        vals_per_cache_neuron * sizeof(float));
            cache->neuron_to_slot[last_n] = slot; cache->slot_to_neuron[slot] = last_n;
        }
        cache->neuron_to_slot.erase(n); cache->num_resident--;
    }

    void load_neuron_to_cache(int l_idx, LayerCache* cache, int n, float* fc1_bias_layer) {
        if (cache->num_resident >= cache->max_resident) return; 
        uint16_t* src = (uint16_t*)(ffn_mapped + (size_t)l_idx*ffn_dim*bytes_per_ssd_neuron + (size_t)n*bytes_per_ssd_neuron);
        float* dst = cache->active_weights + (cache->num_resident * vals_per_cache_neuron);
        
        for(size_t h=0; h<hidden_size; ++h) dst[h] = h2f(src[h]); // gate_proj / fc1
        if (is_llama3) {
            for(size_t h=0; h<hidden_size; ++h) dst[h+hidden_size] = h2f(src[h+hidden_size]); // up_proj
            for(size_t h=0; h<hidden_size; ++h) dst[h+hidden_size*2] = h2f(src[h+hidden_size*2]); // down_proj
            dst[hidden_size*3] = 0.0f; // no bias
        } else {
            for(size_t h=0; h<hidden_size; ++h) dst[h+hidden_size] = h2f(src[h+hidden_size]); // fc2
            dst[hidden_size*2] = fc1_bias_layer ? fc1_bias_layer[n] : 0.0f;
        }
        cache->neuron_to_slot[n] = cache->num_resident; cache->slot_to_neuron[cache->num_resident] = n; cache->num_resident++;
    }
};

extern "C" {
    void* init_engine(const char* ffn, const char* pred, size_t hs, size_t fd, size_t nl, int llama3) { 
        return new FlashFFNEngine(ffn, pred, hs, fd, nl, llama3 != 0); 
    }
    void set_engine_config(void* ptr, int k, float t, int w) { ((FlashFFNEngine*)ptr)->set_config(k, t, w); }
    void set_predictor_layer_info(void* ptr, int l, int r, size_t o) {
        ((FlashFFNEngine*)ptr)->set_predictor_info(l, r, o);
    }
    void execute_ffn_layer(void* ptr, int l, float* in, float* out, int n, float* b, int m) { 
        ((FlashFFNEngine*)ptr)->execute_ffn(l, in, out, n, b, m); 
    }
    void destroy_engine(void* ptr) { delete (FlashFFNEngine*)ptr; }
}

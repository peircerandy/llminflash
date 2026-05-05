import re

with open("engine.cpp", "r") as f:
    content = f.read()

# 1. Remove hardcoded constexprs
content = re.sub(r'constexpr size_t HIDDEN_SIZE = 4096;\n', '', content)
content = re.sub(r'constexpr size_t FFN_DIM = 16384;\n', '', content)
content = re.sub(r'constexpr size_t NUM_LAYERS = 32;\n', '', content)

# 2. Add ARM NEON header
if "<arm_neon.h>" not in content:
    content = content.replace("#include <omp.h>", "#include <omp.h>\n#if defined(__ARM_NEON) || defined(__aarch64__)\n#include <arm_neon.h>\n#endif")

# 3. Add dynamic variables to FlashFFNEngine class
content = content.replace("class FlashFFNEngine {\nprivate:", "class FlashFFNEngine {\nprivate:\n    size_t HIDDEN_SIZE;\n    size_t FFN_DIM;\n    size_t NUM_LAYERS;\n    bool is_llama3;\n    size_t NEURON_BUNDLE_VALS_SSD;\n    size_t NEURON_BUNDLE_SIZE_SSD;\n    size_t NEURON_BUNDLE_VALS_CACHE;")

# 4. Fix constructor
init_replace = """    FlashFFNEngine(const char* ffn_path, const char* predictor_path, size_t hidden_size, size_t ffn_dim, size_t num_layers, bool is_llama3) {
        HIDDEN_SIZE = hidden_size;
        FFN_DIM = ffn_dim;
        NUM_LAYERS = num_layers;
        this->is_llama3 = is_llama3;
        NEURON_BUNDLE_VALS_SSD = is_llama3 ? HIDDEN_SIZE * 3 : HIDDEN_SIZE * 2;
        NEURON_BUNDLE_SIZE_SSD = NEURON_BUNDLE_VALS_SSD * BYTES_PER_VAL;
        NEURON_BUNDLE_VALS_CACHE = (is_llama3 ? HIDDEN_SIZE * 3 : HIDDEN_SIZE * 2) + 1;
"""
content = re.sub(r'    FlashFFNEngine\(const char\* ffn_path, const char\* predictor_path\) \{', init_replace, content)

# I will use a more robust rewrite script next.

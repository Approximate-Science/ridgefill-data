// CPU native split-state check. No engine modifications or instrumented build.
// Use streamed graph boundaries only to observe layer 16 before later native
// identity layers reuse (and rename) the same tensor in a resident graph.
#include "llama.h"
#include "llama-ext.h"
#include "llama-ridgefill.h"
#include "server-ridgefill-sweep.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

struct observed { int rows[64] = {}; int values = 0; };

static bool callback(ggml_tensor * t, bool ask, void * user) {
    int layer = -1;
    if (std::sscanf(ggml_get_name(t), "ridgefill_inp-%d", &layer) != 1 || layer < 16 || layer > 63) { return false; }
    if (ask) { return true; }
    auto & result = *(observed *) user;
    // Native GDN identity is the h_S gather itself, with no intervening map.
    std::vector<float> actual(ggml_nelements(t));
    ggml_backend_tensor_get(t, actual.data(), 0, actual.size()*sizeof(float));
    ggml_tensor * source = t;
    if (layer % 4 == 3) {
        if (t->op != GGML_OP_ADD || t->src[0]->op != GGML_OP_MUL_MAT || t->src[0]->src[0]->type != GGML_TYPE_F16) {
            throw std::runtime_error("attention input is not F16 identity matmul plus bias");
        }
        source = t->src[0]->src[1];
    }
    if (source->op != GGML_OP_GET_ROWS || source->src[0]->type != GGML_TYPE_F16) {
        throw std::runtime_error("fill input is not a direct F16 h_S gather");
    }
    std::vector<int32_t> idx(t->ne[1]);
    ggml_backend_tensor_get(source->src[1], idx.data(), 0, idx.size()*sizeof(int32_t));
    std::vector<ggml_fp16_t> stored(t->ne[0]);
    for (int64_t row = 0; row < t->ne[1]; ++row) {
        ggml_backend_tensor_get(source->src[0], stored.data(), idx[row]*source->src[0]->nb[1], stored.size()*sizeof(ggml_fp16_t));
        for (int64_t col = 0; col < t->ne[0]; ++col) {
            if (actual[row*t->ne[0]+col] != ggml_fp16_to_fp32(stored[col])) {
                throw std::runtime_error("filled layer input differs from h_S");
            }
            ++result.values;
        }
        ++result.rows[layer];
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "model projector document.json\n"); return 1; }
    try {
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) { throw std::runtime_error("model load failed"); }
        std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(llama_ridgefill_init_streamed(model.get(), argv[2]), llama_ridgefill_free);
        if (!rf || rf->split != 16 || rf->w[16] || rf->b[16]) { throw std::runtime_error("layer16 is not native identity"); }
        std::ifstream input(argv[3]); nlohmann::json doc; input >> doc;
        auto ids = doc["ids"].get<std::vector<llama_token>>(); ids.resize(512);
        observed result;
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = cp.n_ubatch = 128;
        cp.n_threads = cp.n_threads_batch = 6;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = callback; cp.cb_eval_user_data = &result;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx || llama_set_ridgefill(ctx.get(), rf.get()) != 0) { throw std::runtime_error("context attach failed"); }
        llama_ridgefill_set_alpha(ctx.get(), 0);
        llama_ridgefill_seq_set(ctx.get(), 0, 112);
        server_ridgefill_sweep_params params;
        params.n_embd = llama_model_n_embd(model.get()); params.n_tokens = ids.size(); params.n_prefix = 112;
        params.n_batch = 128; params.split = 16; params.type = GGML_TYPE_F16;
        params.qattn_share = 0; params.vtail = true; params.vtail_rows = 32; params.projector = rf.get();
        server_ridgefill_sweep sweep;
        auto status = server_ridgefill_sweep_result::pending;
        int steps = 0;
        while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
            status = server_ridgefill_sweep_step(sweep, ctx.get(), 0, ids.data(), {}, params);
        }
        if (status != server_ridgefill_sweep_result::complete) { throw std::runtime_error("incomplete sweep"); }
        for (int il = 16; il < 64; ++il) {
            if (result.rows[il] != 112) { throw std::runtime_error("incomplete filled-layer observation"); }
        }
        std::printf("PASS: inputs of all layers 16..63 equal saved h_S exactly: 112 filled rows per layer, %d F32 values. Layer16 is native GET_ROWS of F16 split buffer, no map; attention uses F16 identity matmul.\n", result.values);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 2;
    }
}

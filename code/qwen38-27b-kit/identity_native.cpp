// CPU-only identity operator check, linked to the unchanged E3 ggml libraries.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    constexpr int n = 32;
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32}) {
        auto * ctx = ggml_init({16*1024*1024, nullptr, false});
        auto * w = ggml_new_tensor_2d(ctx, type, n, n);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
        std::memset(w->data, 0, ggml_nbytes(w));
        for (int i = 0; i < n; ++i) {
            // Values exactly representable in F16, as in E3's --h-type f16.
            ((float *) x->data)[i] = (i+1)*0.03125f;
            if (type == GGML_TYPE_Q8_0) {
                auto * row = (unsigned char *) w->data + i*34;
                auto one = ggml_fp32_to_fp16(1.f);
                std::memcpy(row, &one, 2);
                row[2+i] = 1;
            } else if (type == GGML_TYPE_F16) {
                ((ggml_fp16_t *) w->data)[i*n+i] = ggml_fp32_to_fp16(1.f);
            } else {
                ((float *) w->data)[i*n+i] = 1.f;
            }
        }
        auto * y = ggml_mul_mat(ctx, w, x);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) { return 1; }
        float error = 0;
        for (int i = 0; i < n; ++i) {
            error = std::fmax(error, std::fabs(((float *) y->data)[i] - ((float *) x->data)[i]));
        }
        std::printf("%s identity: max_abs_error=%.9g, byte_equal=%s\n", ggml_type_name(type), error,
                    std::memcmp(x->data, y->data, n*sizeof(float)) == 0 ? "true" : "false");
        if (type != GGML_TYPE_Q8_0 && error != 0) { return 2; }
        ggml_free(ctx);
    }
}

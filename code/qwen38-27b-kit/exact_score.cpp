// Independent exact forward linked to upstream, for scorer-off equivalence only.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
int main(int argc,char **argv) {
    try {
        if (argc!=4) throw std::runtime_error("exact-score MODEL INPUT OUTPUT");
        std::ifstream input(argv[2]); json doc; input >> doc;
        auto ids=doc.at("ids").get<std::vector<llama_token>>();
        const int n=ids.size(), k=512, begin=n-k-1,end=n-1;
        if (begin<0) throw std::runtime_error("document too short");
        ggml_backend_load_all(); llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
        auto *model=llama_model_load_from_file(argv[1],mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();
        cp.n_ctx=n; cp.n_batch=2048; cp.n_ubatch=512; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=4;
        cp.type_k=cp.type_v=GGML_TYPE_F16; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto *ctx=llama_init_from_model(model,cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_memory_clear(llama_get_memory(ctx),true);
        std::vector<double> lp;
        for (int first=0;first<n;) {
            const int count=std::min(2048,n-first);
            auto batch=llama_batch_init(count,0,1); batch.n_tokens=count;
            int outputs=0,first_output=-1;
            for (int i=0;i<count;i++) {
                batch.token[i]=ids[first+i]; batch.pos[i]=first+i;
                batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
                batch.logits[i]=first+i>=begin && first+i<end;
                if (batch.logits[i]) { if (!outputs)first_output=first+i; outputs++; }
            }
            int rc=llama_decode(ctx,batch); llama_batch_free(batch);
            if (rc) throw std::runtime_error("decode failed");
            for (int i=0;i<outputs;i++) {
                const float *logits=llama_get_logits_ith(ctx,i-outputs);
                double maximum=-std::numeric_limits<double>::infinity();
                for (int v=0;v<vocab;v++) {
                    if (!std::isfinite(logits[v]))throw std::runtime_error("nonfinite logits");
                    maximum=std::max(maximum,double(logits[v]));
                }
                double sum=0;
                for(int v=0;v<vocab;v++)sum+=std::exp(double(logits[v])-maximum);
                lp.push_back(double(logits[ids[first_output+i+1]])-(maximum+std::log(sum)));
            }
            first+=count;
        }
        if (lp.size()!=512)throw std::runtime_error("wrong score count");
        double mean=0; for(double x:lp)mean-=x/512;
        std::ofstream out(argv[3]); out << json{{"nll",mean},{"logprobs",lp}} << '\n';
        if (!out)throw std::runtime_error("write failed");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n';return 1; }
}

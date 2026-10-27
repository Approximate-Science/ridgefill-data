"""Identical observational hook in both servers; no effect unless the audit env var is set."""
import sys
from pathlib import Path

ANCHOR='    const int n_logits = llama_get_sampled_logits_count_ith(ctx, idx);\n'
HOOK='''
    // cloud-kit CHECK-A/B: dump full-vocabulary pre-softmax float32 rows verbatim.
    // This is an observation only; arithmetic and sampling below are unchanged.
    if (const char * path = std::getenv("RIDGEFILL_AUDIT_LOGITS")) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        if (!logits || n_logits != n_vocab) {
            throw std::runtime_error("audit logits require dense unsampled logits");
        }
        for (int v = 0; sampled_ids && v < n_vocab; ++v) {
            if (sampled_ids[v] != v) { throw std::runtime_error("audit logits require vocabulary order"); }
        }
        std::ofstream dump(path, std::ios::binary | std::ios::app);
        dump.write(reinterpret_cast<const char *>(logits), size_t(n_logits)*sizeof(float));
        dump.flush();
        if (!dump) { throw std::runtime_error("cannot write audit logits"); }
    }
'''


def instrument_text(text):
    assert text.count(ANCHOR)==1,'Pinned server source does not match hook location'
    assert 'RIDGEFILL_AUDIT_LOGITS' not in text,'Source already instrumented'
    return '#include <cstdlib>\n#include <stdexcept>\n'+text.replace(ANCHOR,ANCHOR+HOOK)


def main():
    path=Path(sys.argv[1])/'tools/server/server-common.cpp'
    path.write_text(instrument_text(path.read_text()))


if __name__=='__main__':
    main()

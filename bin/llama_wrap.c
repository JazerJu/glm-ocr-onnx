/*
 * C wrapper for llama.cpp functions that take structs by value.
 *
 * ctypes on some Python builds cannot correctly pass structs >16 bytes by value
 * on x86_64 (System V ABI). This wrapper exposes simple C functions with only
 * primitive/pointer arguments, calling the real llama.cpp API internally.
 *
 * Build:
 *   gcc -shared -fPIC -o libllama_wrap.so llama_wrap.c \
 *       -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
 *       -L<build>/bin -lllama -lggml \
 *       -Wl,-rpath,<build>/bin
 */
#include "llama.h"


void * wrap_model_load(const char * path, int n_gpu_layers, int use_mmap) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap = use_mmap;
    return llama_model_load_from_file(path, params);
}


void * wrap_context_init(void * model,
                         uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
                         uint32_t n_seq_max, int flash_attn, int embeddings,
                         int n_threads, int n_threads_batch,
                         int type_k, int type_v) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_seq_max = n_seq_max;
    params.flash_attn_type = flash_attn ? 1 : 0;
    params.embeddings = embeddings;
    params.offload_kqv = 1;
    params.no_perf = 1;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads_batch;
    params.type_k = type_k;
    params.type_v = type_v;
    return llama_init_from_model(model, params);
}


int32_t wrap_decode(void * ctx, struct llama_batch * batch) {
    return llama_decode(ctx, *batch);
}


void wrap_batch_free(struct llama_batch * batch) {
    llama_batch_free(*batch);
}

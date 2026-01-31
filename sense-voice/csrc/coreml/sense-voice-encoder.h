#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sense_voice_coreml_context;

struct sense_voice_coreml_context * sense_voice_coreml_init(const char * path_model);
void sense_voice_coreml_free(struct sense_voice_coreml_context * ctx);

// Encodes the audio features using the CoreML model
// n_ctx: number of audio frames (time steps)
// n_mel: dimension of features (e.g. 80)
// mel: input features, flat array of size n_ctx * n_mel
// out: output tensor buffer
void sense_voice_coreml_encode(
        const struct sense_voice_coreml_context * ctx,
        int64_t n_ctx,
        int64_t n_mel,
        int64_t in_stride,
        const float * mel,
        float * out);

#ifdef __cplusplus
}
#endif

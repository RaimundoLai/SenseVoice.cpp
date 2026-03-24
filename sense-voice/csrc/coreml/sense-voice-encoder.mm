#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "sense-voice-encoder.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <iostream>

#if __cplusplus
extern "C" {
#endif

struct sense_voice_coreml_context {
    void * model; // MLModel *
};

struct sense_voice_coreml_context * sense_voice_coreml_init(const char * path_model) {
    @autoreleasepool {
        NSString * path = [NSString stringWithUTF8String:path_model];
        NSURL * url = [NSURL fileURLWithPath:path];
        
        NSError * error = nil;
        
        if ([path hasSuffix:@".mlpackage"]) {
            NSURL * compiledUrl = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                fprintf(stderr, "%s: error compiling model: %s\n", __func__, [[error localizedDescription] UTF8String]);
                return NULL;
            }
            url = compiledUrl;
        }

        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        // Use CPU + Neural Engine to avoid macOS 15 MPSGraph bytecode parsing bug.
        // MLComputeUnitsAll includes the GPU/MPS path which generates broken bytecode
        // for certain SenseVoice encoder operations on macOS 15 Sequoia, causing:
        //   "error: attempting to parse a byte at the end of the bytecode"
        // CPU + ANE still provides hardware acceleration via the Neural Engine.
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        
        fprintf(stderr, "%s: loading CoreML model (CPU + Neural Engine)... \n", __func__);
        
        // Re-initialize error for model loading
        error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (error) {
            fprintf(stderr, "%s: error loading model: %s. Attempting to compile and reload...\n", __func__, [[error localizedDescription] UTF8String]);
            
            // Fallback: Try to compile it anyway, it might be a .mlpackage named as .mlmodelc, or just need recompilation
            NSError * compileError = nil;
            NSURL * compiledUrl = [MLModel compileModelAtURL:url error:&compileError];
            
            if (compileError || !compiledUrl) {
                 fprintf(stderr, "%s: compilation failed: %s\n", __func__, [[compileError localizedDescription] UTF8String]);
                 return NULL;
            }
            
            fprintf(stderr, "%s: compilation successful, loading from %s\n", __func__, [[compiledUrl path] UTF8String]);
            error = nil;
            model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
            
            if (error) {
                fprintf(stderr, "%s: error loading compiled model: %s\n", __func__, [[error localizedDescription] UTF8String]);
                return NULL;
            }
        }

        struct sense_voice_coreml_context * ctx = new sense_voice_coreml_context;
        ctx->model = (__bridge_retained void *)model;
        return ctx;
    }
}

void sense_voice_coreml_free(struct sense_voice_coreml_context * ctx) {
    if (ctx) {
        if (ctx->model) {
            CFRelease(ctx->model);
        }
        delete ctx;
    }
}

void sense_voice_coreml_encode(
        const struct sense_voice_coreml_context * ctx,
        int64_t n_ctx,
        int64_t n_mel,
        int64_t in_stride,
        const float * mel,
        float * out) {
    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->model;

        NSError * error = nil;

        // Prepare Input
        // SenseVoice inputs:
        // speech: (1, n_time, n_dim)
        // speech_lengths: (1)

        // MLMultiArray shape definition: (1, Time, Dim)
        // Create MLMultiArray with explicit shape
        NSArray<NSNumber *> *shape = @[@1, @(n_ctx), @(n_mel)];
        
        // Initialize error
        error = nil;

        // Use standard init which allocates its own contiguous memory
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:&error];
        
        if (error) {
            fprintf(stderr, "%s: failed to create MLMultiArray: %s\n", __func__, [[error localizedDescription] UTF8String]);
            return;
        }

        // Copy data from ggml tensor to MLMultiArray
        float *modelData = (float *)inputArray.dataPointer;
        const float *srcData = mel;
        
        for (int i = 0; i < n_ctx; i++) {
            memcpy(modelData + i * n_mel, srcData + i * in_stride, n_mel * sizeof(float));
        }

        // Create speech_lengths input (assuming it's always 1 for now, or derived from n_ctx)
        // For SenseVoice, speech_lengths is typically (1) and contains the actual sequence length.
        // If n_ctx is the actual length, then speech_lengths should be an MLMultiArray of shape (1) with value n_ctx.
        // For simplicity, let's assume a fixed length or derive it.
        // Assuming speech_lengths is a single value representing n_ctx
        MLMultiArray *speechLengthsArray = [[MLMultiArray alloc] initWithShape:@[@1] dataType:MLMultiArrayDataTypeInt32 error:&error];
        if (error) {
            fprintf(stderr, "%s: failed to create speech_lengths MLMultiArray: %s\n", __func__, [[error localizedDescription] UTF8String]);
            return;
        }
        ((int32_t*)speechLengthsArray.dataPointer)[0] = (int32_t)n_ctx;


        // Assuming SenseVoiceSmall_EncoderInput is a generated class from the CoreML model
        // This part needs the actual generated class name and its initializer.
        // Placeholder:
        // SenseVoiceSmall_EncoderInput *input = [[SenseVoiceSmall_EncoderInput alloc] initWithSpeech:inputArray speech_lengths:speechLengthsArray];
        // Since we don't have the generated class, we'll use MLFeatureProvider directly.
        
        // Create a dictionary for inputs
        NSMutableDictionary<NSString *, id> *inputFeatures = [NSMutableDictionary dictionary];
        [inputFeatures setObject:[MLFeatureValue featureValueWithMultiArray:inputArray] forKey:@"speech"];
        [inputFeatures setObject:[MLFeatureValue featureValueWithMultiArray:speechLengthsArray] forKey:@"speech_lengths"];

        MLDictionaryFeatureProvider *inputProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputFeatures error:&error];
        if (error) {
            fprintf(stderr, "%s: failed to create input provider: %s\n", __func__, [[error localizedDescription] UTF8String]);
            return;
        }
        
        // Prediction
        id<MLFeatureProvider> outputProvider = [model predictionFromFeatures:inputProvider error:&error];
        if (error) {
            fprintf(stderr, "%s: error predicting: %s\n", __func__, [[error localizedDescription] UTF8String]);
            return;
        }
        
        // Get Output
        // "encoder_out"
        MLMultiArray * outputArray = [outputProvider featureValueForName:@"encoder_out"].multiArrayValue;
        if (!outputArray) {
            fprintf(stderr, "%s: error getting output array 'encoder_out'\n", __func__);
            return;
        }
        
        // Copy to output buffer
        // outputArray shape should be checked or assumed correct.
        // float * out should have enough space.
        
        // We assume the caller knows size or we trust CoreML output matches expected size.
        // Copying data:
        // If output is float32
        if (outputArray.dataType == MLMultiArrayDataTypeFloat32) {
             memcpy(out, outputArray.dataPointer, outputArray.count * sizeof(float));
        } else if (outputArray.dataType == MLMultiArrayDataTypeFloat16) {
             uint16_t *f16_data = (uint16_t *)outputArray.dataPointer;
             for (NSInteger i = 0; i < outputArray.count; i++) {
                 uint16_t h = f16_data[i];
                 uint32_t sign = (h >> 15) & 0x00000001;
                 uint32_t exp  = (h >> 10) & 0x0000001f;
                 uint32_t mant =  h        & 0x000003ff;

                 exp = exp + (127 - 15);
                 if (exp == 127 - 15) {
                     if (mant == 0) {
                         exp = 0;
                     } else {
                         exp++;
                         while ((mant & 0x00000400) == 0) {
                             mant <<= 1;
                             exp--;
                         }
                         mant &= 0x000003ff;
                     }
                 } else if (exp == 127 - 15 + 31) {
                     exp = 255;
                 }

                 uint32_t v = (sign << 31) | (exp << 23) | (mant << 13);
                 memcpy(&out[i], &v, sizeof(float));
             }
        } else {
             fprintf(stderr, "%s: unsupported output data type %ld\n", __func__, (long)outputArray.dataType);
        }
    }
}

#if __cplusplus
}
#endif

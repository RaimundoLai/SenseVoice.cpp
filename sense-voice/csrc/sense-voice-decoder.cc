//
// Created by lovemefan on 2024/7/25.
//

#include "sense-voice-decoder.h"
#include "wenet_ctc_decoder.h"
#include <set>
#include <cstring>

#define SENSEVOICE_DECODER_MAX_NODES 16

// ============================================================================
// Hot Words Biasing using Aho-Corasick Automaton
// ============================================================================
// The Aho-Corasick algorithm builds a trie of all hot word token sequences
// and uses failure links to efficiently match multiple patterns simultaneously.
// During CTC beam search, we track the automaton state as tokens are decoded
// and boost log probabilities of tokens that continue hot word prefixes.
// ============================================================================

#include <queue>
#include <unordered_map>

// Aho-Corasick Automaton Node
struct ACNode {
    std::unordered_map<int, int> children;  // token_id -> child node index
    int fail = 0;           // failure link (index into nodes vector)
    float output = 0.0f;    // accumulated score when reaching this node (word ends here)
    bool is_end = false;    // true if a hot word pattern ends at this node
    int depth = 0;          // depth in trie (0 = root, 1 = first token, etc.)
};

// Aho-Corasick Automaton for hot words token sequences
class HotWordsAC {
public:
    std::vector<ACNode> nodes;
    
    HotWordsAC() {
        nodes.emplace_back();  // root node at index 0
    }
    
    // Insert a hot word token sequence into the trie
    void insert(const std::vector<int>& pattern, float score) {
        if (pattern.empty()) return;
        
        int cur = 0;  // start at root
        for (int token_id : pattern) {
            auto it = nodes[cur].children.find(token_id);
            if (it == nodes[cur].children.end()) {
                int new_node = static_cast<int>(nodes.size());
                nodes.emplace_back();
                nodes[cur].children[token_id] = new_node;
                // Calculate depth: parent depth + 1
                nodes[new_node].depth = nodes[cur].depth + 1;
                cur = new_node;
            } else {
                cur = it->second;
            }
        }
        nodes[cur].is_end = true;
        nodes[cur].output += score;  // accumulate if multiple patterns end here
    }
    
    // Build failure links using BFS
    void build() {
        std::queue<int> q;
        
        // Initialize: children of root have failure link to root
        for (auto it = nodes[0].children.begin(); it != nodes[0].children.end(); ++it) {
            int child = it->second;
            nodes[child].fail = 0;
            q.push(child);
        }
        
        // BFS to build failure links
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            
            for (auto it = nodes[cur].children.begin(); it != nodes[cur].children.end(); ++it) {
                int token_id = it->first;
                int child = it->second;
                // Find failure link for child
                int f = nodes[cur].fail;
                while (f != 0 && nodes[f].children.find(token_id) == nodes[f].children.end()) {
                    f = nodes[f].fail;
                }
                
                auto fit = nodes[f].children.find(token_id);
                if (fit != nodes[f].children.end() && fit->second != child) {
                    nodes[child].fail = fit->second;
                } else {
                    nodes[child].fail = 0;
                }
                
                // Accumulate output from failure chain
                nodes[child].output += nodes[nodes[child].fail].output;
                
                q.push(child);
            }
        }
    }
    
    // Given current state, transition on token_id and return (new_state, bonus_score)
    std::pair<int, float> step(int state, int token_id) const {
        int cur = state;
        
        // Follow failure links until we find a transition or reach root
        while (cur != 0 && nodes[cur].children.find(token_id) == nodes[cur].children.end()) {
            cur = nodes[cur].fail;
        }
        
        auto it = nodes[cur].children.find(token_id);
        if (it != nodes[cur].children.end()) {
            int next = it->second;
            return {next, nodes[next].output};
        }
        
        return {0, 0.0f};  // stay at root, no bonus
    }
    
    // Get tokens that would give a bonus from this state
    std::set<int> get_bonus_tokens(int state) const {
        std::set<int> result;
        
        // Collect all possible transitions from current state (including via failure links)
        int cur = state;
        while (true) {
            for (auto it = nodes[cur].children.begin(); it != nodes[cur].children.end(); ++it) {
                result.insert(it->first);
            }
            if (cur == 0) break;
            cur = nodes[cur].fail;
        }
        
        return result;
    }
    
    // Get depth of a state (0 = root, 1 = first token, etc.)
    int get_depth(int state) const {
        if (state >= 0 && state < static_cast<int>(nodes.size())) {
            return nodes[state].depth;
        }
        return 0;
    }
    
    // Check if a state is the end of a hot word
    bool is_end(int state) const {
        if (state >= 0 && state < static_cast<int>(nodes.size())) {
            return nodes[state].is_end;
        }
        return false;
    }
    
    bool empty() const { return nodes.size() <= 1; }
};

// Adapter to make HotWordsAC compatible with wenet_ctc::HotWordsACInterface
class HotWordsACAdapter : public wenet_ctc::HotWordsACInterface {
public:
    explicit HotWordsACAdapter(const HotWordsAC* ac) : ac_(ac) {}
    
    std::pair<int, float> step(int state, int token_id) const override {
        return ac_->step(state, token_id);
    }
    
    bool empty() const override {
        return ac_->empty();
    }
    
    std::set<int> get_bonus_tokens(int state) const override {
        return ac_->get_bonus_tokens(state);
    }
    
    int get_depth(int state) const override {
        return ac_->get_depth(state);
    }
    
    bool is_end(int state) const override {
        return ac_->is_end(state);
    }
    
private:
    const HotWordsAC* ac_;
};

// Greedy tokenization: tokenize text using longest-match greedy approach
static std::vector<int> greedy_tokenize(
    const sense_voice_vocab& vocab,
    const std::string& text) {
    std::vector<int> result;
    if (text.empty()) return result;
    
    size_t pos = 0;
    while (pos < text.size()) {
        int best_id = -1;
        size_t best_len = 0;
        
        for (const auto& pair : vocab.token_to_id) {
            const std::string& token = pair.first;
            if (token.empty()) continue;
            
            if (pos + token.size() <= text.size() &&
                text.compare(pos, token.size(), token) == 0) {
                if (token.size() > best_len) {
                    best_len = token.size();
                    best_id = pair.second;
                }
            }
        }
        
        if (best_id >= 0) {
            result.push_back(best_id);
            pos += best_len;
        } else {
            unsigned char c = text[pos];
            if ((c & 0x80) == 0) pos += 1;
            else if ((c & 0xE0) == 0xC0) pos += 2;
            else if ((c & 0xF0) == 0xE0) pos += 3;
            else if ((c & 0xF8) == 0xF0) pos += 4;
            else pos += 1;
        }
    }
    
    return result;
}

// Parse hotword with optional score suffix
// Format: "word" or "word:5.0" where 5.0 is the individual score
// Returns (word_without_suffix, score)
static std::pair<std::string, float> parse_hotword_with_score(const char* hotword, float default_score) {
    std::string hw_str(hotword);
    float score = default_score;
    
    // Find the last ':' that could be a score separator
    size_t colon_pos = hw_str.rfind(':');
    if (colon_pos != std::string::npos && colon_pos > 0 && colon_pos < hw_str.size() - 1) {
        // Check if everything after ':' is a valid number
        std::string suffix = hw_str.substr(colon_pos + 1);
        bool is_number = true;
        bool has_dot = false;
        for (size_t i = 0; i < suffix.size(); ++i) {
            char c = suffix[i];
            if (c == '.') {
                if (has_dot) { is_number = false; break; }
                has_dot = true;
            } else if (c == '-' && i == 0) {
                // Allow negative at start
            } else if (!isdigit(c)) {
                is_number = false;
                break;
            }
        }
        
        if (is_number && !suffix.empty()) {
            score = static_cast<float>(atof(suffix.c_str()));
            hw_str = hw_str.substr(0, colon_pos);
        }
    }
    
    return std::make_pair(hw_str, score);
}

// Build Aho-Corasick automaton from hot words
static HotWordsAC build_hotwords_ac(
    const sense_voice_context& ctx,
    const sense_voice_full_params& params) {
    HotWordsAC ac;
    
    // Helper: Check if string looks like English (ASCII letters)
    auto is_ascii_word = [](const std::string& s) {
        for (unsigned char c : s) {
            if (c >= 0x80) return false;  // Non-ASCII = likely Chinese
        }
        return true;
    };
    
    // Add hot words (supports per-word score via "word:score" format)
    if (params.hotwords && params.n_hotwords > 0) {
        for (int i = 0; i < params.n_hotwords; ++i) {
            if (params.hotwords[i] && params.hotwords[i][0] != '\0') {
                // Parse hotword with optional score suffix
                auto parsed = parse_hotword_with_score(params.hotwords[i], params.hotwords_score);
                const std::string& word = parsed.first;
                float score = parsed.second;
                
                // Tokenize the original word
                auto tokens = greedy_tokenize(ctx.vocab, word);
                
                std::string debug_str = "Hotword '" + word + "' (score=" + std::to_string(score) + ") -> tokens: ";
                for (int id : tokens) {
                    auto it = ctx.vocab.id_to_token.find(id);
                    if (it != ctx.vocab.id_to_token.end()) {
                        debug_str += "[" + std::to_string(id) + ":" + it->second + "] ";
                    }
                }
                if (params.debug_mode) {
                    SENSE_VOICE_LOG_INFO("%s: %s\n", __func__, debug_str.c_str());
                }
                
                // Only skip very short single-token hotwords (< 3 chars) that cause interference
                // English words like "Apple", "Google" are often single tokens and should NOT be skipped
                bool is_short_single_token = (tokens.size() == 1 && word.length() < 3);
                
                if (!is_short_single_token && !tokens.empty()) {
                    ac.insert(tokens, score);
                    
                    // For English words: add space-prefixed versions
                    // This handles mid-sentence occurrences where tokens have leading space
                    if (is_ascii_word(word)) {
                        // Try 1: ASCII Space (0x20) - some tokenizers use this
                        std::string space_word_ascii = " " + word;
                        auto tokens_ascii = greedy_tokenize(ctx.vocab, space_word_ascii);
                        if (!tokens_ascii.empty() && tokens_ascii != tokens) {
                            ac.insert(tokens_ascii, score);
                            debug_str = "  + ASCII space variant: ";
                            for (int id : tokens_ascii) {
                                auto it = ctx.vocab.id_to_token.find(id);
                                if (it != ctx.vocab.id_to_token.end()) {
                                    debug_str += "[" + std::to_string(id) + ":" + it->second + "] ";
                                }
                            }
                            if (params.debug_mode) {
                                SENSE_VOICE_LOG_INFO("%s: %s\n", __func__, debug_str.c_str());
                            }
                        }
                        
                        // Try 2: SentencePiece Space (U+2581 = \xe2\x96\x81) - SenseVoice uses this
                        std::string space_word_sp = "\xe2\x96\x81" + word;
                        auto tokens_sp = greedy_tokenize(ctx.vocab, space_word_sp);
                        if (!tokens_sp.empty() && tokens_sp != tokens && tokens_sp != tokens_ascii) {
                            ac.insert(tokens_sp, score);
                            debug_str = "  + SentencePiece space variant: ";
                            for (int id : tokens_sp) {
                                auto it = ctx.vocab.id_to_token.find(id);
                                if (it != ctx.vocab.id_to_token.end()) {
                                    debug_str += "[" + std::to_string(id) + ":" + it->second + "] ";
                                }
                            }
                            if (params.debug_mode) {
                                SENSE_VOICE_LOG_INFO("%s: %s\n", __func__, debug_str.c_str());
                            }
                        }
                    }
                } else if (is_short_single_token) {
                    if (params.debug_mode) {
                        SENSE_VOICE_LOG_INFO("%s: Skipping short hotword '%s' (< 3 chars, causes interference)\n",
                            __func__, word.c_str());
                    }
                }
            }
        }
    }
    
    if (!ac.empty()) {
        ac.build();
        if (params.debug_mode) {
            SENSE_VOICE_LOG_INFO("%s: Built Aho-Corasick automaton with %zu nodes\n", 
                __func__, ac.nodes.size());
        }
    }
    
    return ac;
}



// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
                    ggml_mul_mat(ctx, x_0, y_0),
                    ggml_mul_mat(ctx, x_1, y_1));
}

// copy from whisper.cpp
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif

struct ggml_cgraph *sense_voice_build_graph_ctc_decoder(sense_voice_context &ctx,
                                                    sense_voice_state &state){
    const auto &model = ctx.model.model;

    struct ggml_init_params params = {
            /*.mem_size   =*/state.sched_decode.meta.size(),
            /*.mem_buffer =*/state.sched_decode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_DECODER_MAX_NODES, false);

    ggml_tensor *encoder_out = ggml_new_tensor_3d(ctx0, state.encoder_out->type,
                                                  state.encoder_out->ne[0], state.encoder_out->ne[1], 
                                                  state.encoder_out->ne[2]);
    ggml_set_name(encoder_out, "encoder_out");
    ggml_set_input(encoder_out);

    ggml_tensor *cur;
    {
        // Reshape encoder_out to merge batch and time dimensions
        cur = ggml_reshape_2d(ctx0, encoder_out, encoder_out->ne[0], encoder_out->ne[1] * encoder_out->ne[2]);
        cur = ggml_mul_mat(ctx0, model->ctc_out_linear_weight, cur);
        cur = ggml_add(ctx0, cur, model->ctc_out_linear_bias);
        // Reshape back to 3D
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], encoder_out->ne[1], encoder_out->ne[2]);
    }
    ggml_tensor * probs = ggml_soft_max(ctx0, cur);
    probs = ggml_reshape_2d(ctx0, probs, probs->ne[0], probs->ne[1] * probs->ne[2] * probs->ne[3]);
    ggml_tensor * argmax_logit = ggml_argmax(ctx0, probs);
    argmax_logit = ggml_reshape_3d(ctx0, argmax_logit, cur->ne[1], cur->ne[2], cur->ne[3]);
    ggml_set_output(probs);
    ggml_set_output(argmax_logit);
    ggml_build_forward_expand(gf, argmax_logit);
    ggml_free(ctx0);
    return gf;
}


bool sense_voice_decode_internal(sense_voice_context &ctx,
                                 sense_voice_state &state,
                                 const sense_voice_full_params &params,
                                 const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    // decoder
    {
        auto & sched = state.sched_decode.sched;


        ggml_cgraph *gf = sense_voice_build_graph_ctc_decoder(ctx, state);

//        sched->callback_eval = ctx.params.cb_eval;
//        sched->callback_eval_data = ctx.params.cb_eval_user_data;

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }


        // set the input
        {
            struct ggml_tensor *encoder_out = ggml_graph_get_tensor(gf, "encoder_out");
            ggml_backend_tensor_copy(state.encoder_out, encoder_out);
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
        {
            // Get output nodes
            ggml_tensor *argmax_logit = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
            
            // Debug: print argmax_logit dimensions
            SENSE_VOICE_LOG_DEBUG("%s: argmax_logit dims: [%lld, %lld, %lld, %lld]\n",
                __func__, argmax_logit->ne[0], argmax_logit->ne[1], argmax_logit->ne[2], argmax_logit->ne[3]);
            
            // Get probs tensor - it's the first output (before argmax)
            ggml_tensor *probs = nullptr;
            for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
                ggml_tensor *node = ggml_graph_node(gf, i);
                // Find the 2D probs tensor (reshaped) before argmax
                if (node != argmax_logit && node->ne[0] > 1000 && node->ne[1] > 1) {
                    probs = node;
                    break;
                }
            }
            
            if (!probs) {
                SENSE_VOICE_LOG_ERROR("%s: Could not find probs tensor in graph\n", __func__);
                return false;
            }
            
            SENSE_VOICE_LOG_DEBUG("%s: probs dims: [%lld, %lld]\n", 
                __func__, probs->ne[0], probs->ne[1]);
            
            // Beam search vs Greedy decoding
            if (params.strategy == SENSE_VOICE_SAMPLING_BEAM_SEARCH && params.beam_search.beam_size > 1) {
                // === Beam Search Path ===
                // probs tensor is 2D: [vocab_size, n_frames] (column-major GGML layout)
                const int vocab_size = probs->ne[0];  // 25055
                const int n_frames = probs->ne[1];    // 8256
                
                // Validate with model vocab size
                SENSE_VOICE_LOG_INFO("%s: CTC Beam Search: vocab_size=%d, n_frames=%d, beam_size=%d\n",
                    __func__, vocab_size, n_frames, params.beam_search.beam_size);
                
                // Read probs tensor to CPU
                std::vector<float> probs_data(vocab_size * n_frames);
                ggml_backend_tensor_get(probs, probs_data.data(), 0, sizeof(float) * probs_data.size());
                
                // Convert to log probabilities and organize by timestep
                // GGML uses column-major: data[vocab + frame * vocab_size]
                std::vector<std::vector<float>> log_probs(n_frames);
                for (int t = 0; t < n_frames; ++t) {
                    log_probs[t].resize(vocab_size);
                    for (int v = 0; v < vocab_size; ++v) {
                        // Column-major access: data[row + col * nrows]
                        float prob = probs_data[v + t * vocab_size];
                        log_probs[t][v] = std::log(std::max(prob, 1e-10f));
                    }
                }
                
                // Apply hot words biasing using Aho-Corasick (if configured)
                HotWordsAC hotwords_ac = build_hotwords_ac(ctx, params);
                
                // NOTE: Static biasing removed - it corrupts English BPE output
                // We now rely entirely on contextual tracking during beam search
                
                // Create adapter for contextual biasing in beam search
                HotWordsACAdapter ac_adapter(&hotwords_ac);
                
                // Run CTC beam search using WeNet decoder with contextual hot word biasing
                wenet_ctc::CtcPrefixBeamSearch decoder(0, params.beam_search.beam_size);
                
                // Enable contextual biasing: track AC state and give progressive bonus
                // when tokens follow the hot word sequence
                if (!hotwords_ac.empty() && params.hotwords_score != 0.0f) {
                    float contextual_bonus = params.hotwords_score * 0.5f;  // 50% bonus for being on path
                    decoder.SetContextualAC(&ac_adapter, contextual_bonus);
                    if (params.debug_mode) {
                        SENSE_VOICE_LOG_INFO("%s: Enabled contextual hot word biasing (bonus=%.2f)\n",
                            __func__, contextual_bonus);
                    }
                }
                
                for (int t = 0; t < n_frames; ++t) {
                    decoder.SearchFrame(log_probs[t]);
                    if (params.debug_mode && t % 1000 == 0) {
                        SENSE_VOICE_LOG_INFO("%s: Beam search progress: %d/%d frames (%.1f%%)\n",
                            __func__, t, n_frames, 100.0f * t / n_frames);
                    }
                }
                
                auto best_path = decoder.GetBestPath();
                float best_score = decoder.GetBestScore();
                
                SENSE_VOICE_LOG_INFO("%s: Beam search result: %zu tokens, score=%.4f\n",
                    __func__, best_path.size(), best_score);
                
                // Debug: show first 10 tokens
                if (params.debug_mode && best_path.size() > 0) {
                    std::string preview = "First tokens: ";
                    for (size_t i = 0; i < std::min(size_t(10), best_path.size()); ++i) {
                        preview += std::to_string(best_path[i]) + " ";
                    }
                    SENSE_VOICE_LOG_INFO("%s: %s\n", __func__, preview.c_str());
                }
                
                // Store result
                if (state.result_all.empty()) {
                    state.ids = best_path;
                } else {
                    // Batch processing - fallback to greedy
                    state.ids.resize(argmax_logit->ne[0]);
                    ggml_backend_tensor_get(argmax_logit, state.ids.data(), 0, sizeof(int) * argmax_logit->ne[0]);
                }
            } else {
                // === Greedy Decoding Path (Corrected) ===
                // 1. Get raw argmax tokens
                const int32_t n_logits = argmax_logit->ne[0] * argmax_logit->ne[1];
                std::vector<int> raw_tokens(n_logits);
                ggml_backend_tensor_get(argmax_logit, raw_tokens.data(), 0, sizeof(int) * n_logits);
                
                // CTC blank token ID (typically 0 in SenseVoice vocab)
                const int blank_id = 0; 

                // CTC greedy decode: merge repeats and remove blanks
                auto ctc_greedy_decode = [&](const std::vector<int>& raw, int start, int len) -> std::vector<int> {
                    std::vector<int> result;
                    int prev_token = -1;
                    for (int i = 0; i < len; ++i) {
                        int curr_token = raw[start + i];
                        // 1. Merge repeats
                        if (curr_token != prev_token) {
                            // 2. Remove blanks
                            if (curr_token != blank_id) {
                                result.push_back(curr_token);
                            }
                            prev_token = curr_token;
                        }
                    }
                    return result;
                };

                if(state.result_all.empty()) {
                    // Single sentence case
                    state.ids = ctc_greedy_decode(raw_tokens, 0, argmax_logit->ne[0]);
                }
                else {
                    // Batch processing (multiple segments)
                    for(int32_t i = 0; i < argmax_logit->ne[1]; i++)
                    {
                        int posL = i * argmax_logit->ne[0];
                        state.result_all[state.segmentIDs[i]].tokens = 
                            ctc_greedy_decode(raw_tokens, posL, argmax_logit->ne[0]);
                    }
                }
            }
        }

    }
//    ggml_tensor *logit = ggml_get_tensor(ctx)
    state.t_decode_us += ggml_time_us() - t_start_us;

    return true;
}
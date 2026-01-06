// Adapted from WeNet (https://github.com/wenet-e2e/wenet)
// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
// Licensed under the Apache License, Version 2.0

#ifndef WENET_CTC_DECODER_H
#define WENET_CTC_DECODER_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace wenet_ctc {

constexpr float kFloatMax = std::numeric_limits<float>::max();

// LogAdd for numerical stability
inline float LogAdd(float x, float y) {
    const float kMinLogDiffFloat = std::log(std::numeric_limits<float>::min());
    if (x <= kMinLogDiffFloat) return y;
    if (y <= kMinLogDiffFloat) return x;
    float vmin = std::min(x, y);
    float vmax = std::max(x, y);
    if (vmax > vmin + 50.0) return vmax;
    return vmax + std::log(1.0 + std::exp(vmin - vmax));
}

// TopK selection
inline void TopK(const std::vector<float>& data, int k,
                 std::vector<float>* topk_score,
                 std::vector<int>* topk_index) {
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        pairs.push_back({data[i], i});
    }
    
    int actual_k = std::min(k, static_cast<int>(pairs.size()));
    std::nth_element(pairs.begin(), pairs.begin() + actual_k, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    pairs.resize(actual_k);
    
    topk_score->clear();
    topk_index->clear();
    for (const auto& p : pairs) {
        topk_score->push_back(p.first);
        topk_index->push_back(p.second);
    }
}

// Forward declaration
struct ACNode;
class HotWordsAC;

struct PrefixScore {
    float s = -kFloatMax;       // blank ending score
    float ns = -kFloatMax;      // non-blank ending score
    float v_s = -kFloatMax;     // viterbi blank ending score
    float v_ns = -kFloatMax;    // viterbi non-blank ending score
    float cur_token_prob = -kFloatMax;
    std::vector<int> times_s;   // times of viterbi blank path
    std::vector<int> times_ns;  // times of viterbi non-blank path
    
    // Contextual hot word state tracking
    int ac_state = 0;           // Current state in AC automaton
    
    float score() const { return LogAdd(s, ns); }
    float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
    const std::vector<int>& times() const {
        return v_s > v_ns ? times_s : times_ns;
    }
};

struct PrefixHash {
    size_t operator()(const std::vector<int>& prefix) const {
        size_t hash_code = 0;
        for (int id : prefix) {
            hash_code = id + 31 * hash_code;
        }
        return hash_code;
    }
};

// AC Node structure (minimal, for interface)
struct ACNodeInterface {
    std::unordered_map<int, int> children;
    int fail = 0;
    float output = 0.0f;
    bool is_end = false;
};

// Hot Words AC Interface for contextual biasing
class HotWordsACInterface {
public:
    virtual ~HotWordsACInterface() = default;
    virtual std::pair<int, float> step(int state, int token_id) const = 0;
    virtual bool empty() const = 0;
    virtual std::set<int> get_bonus_tokens(int state) const = 0;
    virtual int get_depth(int state) const = 0;  // Get depth of state in trie
    virtual bool is_end(int state) const = 0;    // Check if state is hotword ending
};

class CtcPrefixBeamSearch {
public:
    CtcPrefixBeamSearch(int blank_id, int beam_size)
        : blank_id_(blank_id), beam_size_(beam_size) {
        Reset();
    }
    
    // Set AC automaton for contextual hot word biasing
    void SetContextualAC(const HotWordsACInterface* ac, float contextual_bonus) {
        contextual_ac_ = ac;
        contextual_bonus_ = contextual_bonus;
    }
    
    void Reset() {
        cur_hyps_.clear();
        abs_time_step_ = 0;
        
        PrefixScore prefix_score;
        prefix_score.s = 0.0;
        prefix_score.ns = -kFloatMax;
        prefix_score.v_s = 0.0;
        prefix_score.v_ns = 0.0;
        prefix_score.ac_state = 0;  // Start at root of AC
        
        std::vector<int> empty;
        cur_hyps_[empty] = prefix_score;
    }
    
    // Search one frame with contextual hot word biasing
    void SearchFrame(const std::vector<float>& logp_t) {
        std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
        
        // 1. First beam prune - select top-k candidates
        std::vector<float> topk_score;
        std::vector<int> topk_index;
        int first_beam_size = std::min(static_cast<int>(logp_t.size()), beam_size_ * 2);
        TopK(logp_t, first_beam_size, &topk_score, &topk_index);
        
        // 2. Token passing
        for (size_t i = 0; i < topk_index.size(); ++i) {
            int id = topk_index[i];
            float prob = topk_score[i];
            
            for (const auto& it : cur_hyps_) {
                const std::vector<int>& prefix = it.first;
                const PrefixScore& prefix_score = it.second;
                
                // Calculate contextual hot word bonus
                float contextual_bonus = 0.0f;
                int next_ac_state = prefix_score.ac_state;
                
                if (id != blank_id_ && contextual_ac_ && !contextual_ac_->empty()) {
                    auto ac_result = contextual_ac_->step(prefix_score.ac_state, id);
                    next_ac_state = ac_result.first;
                    float completion_bonus = ac_result.second;
                    
                    // Simple contextual biasing: only give bonus for continuation and completion
                    if (next_ac_state > 0) {
                        int next_depth = contextual_ac_->get_depth(next_ac_state);
                        
                        if (next_depth > 1) {
                            // Continuation inside a hotword - give full bonus
                            contextual_bonus = contextual_bonus_;
                        }
                        // Note: depth == 1 (new hotword start) gets 0.0 - let acoustic model decide
                        
                        // Add completion bonus when hotword is fully matched
                        if (completion_bonus > 0) {
                            contextual_bonus += completion_bonus;
                            // RESET AC state to 0 after hotword completion
                            // This prevents failure links from causing ghost repetitions
                            // (e.g., nonsense -> ense, check -> ck)
                            next_ac_state = 0;
                        }
                    }
                }

                if (id == blank_id_) {
                    // Standard blank handling - no special boosting
                    PrefixScore& next_score = next_hyps[prefix];
                    next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
                    next_score.v_s = prefix_score.viterbi_score() + prob;
                    next_score.times_s = prefix_score.times();
                    // Blank doesn't change AC state
                    if (next_score.ac_state == 0) {
                        next_score.ac_state = prefix_score.ac_state;
                    }
                } else if (!prefix.empty() && id == prefix.back()) {
                    // Case 1: *a + a => *a (merge)
                    PrefixScore& next_score1 = next_hyps[prefix];
                    next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
                    if (next_score1.v_ns < prefix_score.v_ns + prob) {
                        next_score1.v_ns = prefix_score.v_ns + prob;
                        if (next_score1.cur_token_prob < prob) {
                            next_score1.cur_token_prob = prob;
                            next_score1.times_ns = prefix_score.times_ns;
                            if (!next_score1.times_ns.empty()) {
                                next_score1.times_ns.back() = abs_time_step_;
                            }
                        }
                    }
                    // Merge doesn't change AC state
                    if (next_score1.ac_state == 0) {
                        next_score1.ac_state = prefix_score.ac_state;
                    }
                    
                    // Case 2: *a[blank] + a => *aa (extend after blank)
                    std::vector<int> new_prefix(prefix);
                    new_prefix.push_back(id);
                    PrefixScore& next_score2 = next_hyps[new_prefix];
                    float total_prob = prefix_score.s + prob + contextual_bonus;
                    next_score2.ns = LogAdd(next_score2.ns, total_prob);
                    if (next_score2.v_ns < total_prob) {
                        next_score2.v_ns = total_prob;
                        next_score2.cur_token_prob = prob;
                        next_score2.times_ns = prefix_score.times_s;
                        next_score2.times_ns.push_back(abs_time_step_);
                    }
                    // Update AC state
                    next_score2.ac_state = next_ac_state;
                } else {
                    // Case 3: *a + b => *ab (new character)
                    std::vector<int> new_prefix(prefix);
                    new_prefix.push_back(id);
                    PrefixScore& next_score = next_hyps[new_prefix];
                    float total_prob = prefix_score.score() + prob + contextual_bonus;
                    next_score.ns = LogAdd(next_score.ns, total_prob);
                    if (next_score.v_ns < total_prob) {
                        next_score.v_ns = total_prob;
                        next_score.cur_token_prob = prob;
                        next_score.times_ns = prefix_score.times();
                        next_score.times_ns.push_back(abs_time_step_);
                    }
                    // Update AC state
                    next_score.ac_state = next_ac_state;
                }
            }
        }
        
        // 3. Second beam prune - keep only top beam_size paths
        std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
        int actual_beam = std::min(static_cast<int>(arr.size()), beam_size_);
        std::nth_element(arr.begin(), arr.begin() + actual_beam, arr.end(),
                         [](const auto& a, const auto& b) { return a.second.score() > b.second.score(); });
        arr.resize(actual_beam);
        
        // 4. Update current hypotheses
        cur_hyps_.clear();
        for (const auto& item : arr) {
            cur_hyps_[item.first] = item.second;
        }
        
        ++abs_time_step_;
    }
    
    // Get best path
    std::vector<int> GetBestPath() const {
        if (cur_hyps_.empty()) return {};
        
        auto best = cur_hyps_.begin();
        for (auto it = cur_hyps_.begin(); it != cur_hyps_.end(); ++it) {
            if (it->second.score() > best->second.score()) {
                best = it;
            }
        }
        return best->first;
    }
    
    // Get best path timestamps
    std::vector<int> GetBestTimes() const {
        if (cur_hyps_.empty()) return {};
        
        auto best = cur_hyps_.begin();
        for (auto it = cur_hyps_.begin(); it != cur_hyps_.end(); ++it) {
            if (it->second.score() > best->second.score()) {
                best = it;
            }
        }
        return best->second.times();
    }
    
    float GetBestScore() const {
        if (cur_hyps_.empty()) return -kFloatMax;
        
        auto best = cur_hyps_.begin();
        for (auto it = cur_hyps_.begin(); it != cur_hyps_.end(); ++it) {
            if (it->second.score() > best->second.score()) {
                best = it;
            }
        }
        return best->second.score();
    }

private:
    int blank_id_;
    int beam_size_;
    int abs_time_step_ = 0;
    const HotWordsACInterface* contextual_ac_ = nullptr;
    float contextual_bonus_ = 0.0f;
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
};

} // namespace wenet_ctc

#endif // WENET_CTC_DECODER_H

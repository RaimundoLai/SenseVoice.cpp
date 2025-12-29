// ctc_decoder.h - Optimized CTC Beam Search Decoder for SenseVoice
#ifndef CTC_DECODER_H
#define CTC_DECODER_H
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
namespace ctc {
constexpr float NUM_FLT_INF = std::numeric_limits<float>::max();
constexpr float NUM_FLT_MIN = -std::numeric_limits<float>::max();
template <typename T>
inline T log_sum_exp(const T& x, const T& y) {
    if (x <= NUM_FLT_MIN) return y;
    if (y <= NUM_FLT_MIN) return x;
    T xmax = std::max(x, y);
    return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}
struct PrefixNode {
    int character = -1;
    int timestep = 0;
    float log_prob_b_prev = NUM_FLT_MIN;
    float log_prob_nb_prev = NUM_FLT_MIN;
    float log_prob_b_cur = NUM_FLT_MIN;
    float log_prob_nb_cur = NUM_FLT_MIN;
    float score = 0.0f;
    PrefixNode* parent = nullptr;
    std::map<int, PrefixNode*> children;
    ~PrefixNode() {
        for (auto& kv : children) { delete kv.second; }
    }
    PrefixNode* get_child(int char_id, int ts) {
        auto it = children.find(char_id);
        if (it != children.end()) return it->second;
        PrefixNode* child = new PrefixNode();
        child->character = char_id; child->timestep = ts; child->parent = this;
        children[char_id] = child;
        return child;
    }
    void get_path(std::vector<int>& tokens, std::vector<int>& ts) {
        tokens.clear(); ts.clear();
        PrefixNode* node = this;
        while (node != nullptr && node->character != -1) {
            tokens.insert(tokens.begin(), node->character);
            ts.insert(ts.begin(), node->timestep);
            node = node->parent;
        }
    }
};
struct CTCResult {
    std::vector<int> tokens;
    std::vector<int> timesteps;
    float score;
};
class CTCDecoder {
    int blank_id_, beam_size_, timestep_ = 0;
    int topk_; // Top-k pruning
    PrefixNode* root_;
    std::vector<PrefixNode*> prefixes_;
    void collect_prefixes(PrefixNode* node, std::vector<PrefixNode*>& out) {
        if (node->children.empty()) out.push_back(node);
        else for (auto& kv : node->children) collect_prefixes(kv.second, out);
    }
public:
    CTCDecoder(int blank_id = 0, int beam_size = 1, int topk = 40) 
        : blank_id_(blank_id), beam_size_(beam_size), topk_(topk), root_(new PrefixNode()) {
        root_->score = 0.0f; root_->log_prob_b_prev = 0.0f; prefixes_.push_back(root_);
    }
    ~CTCDecoder() { delete root_; }
    void reset() { delete root_; root_ = new PrefixNode(); root_->score = 0.0f; root_->log_prob_b_prev = 0.0f; prefixes_.clear(); prefixes_.push_back(root_); timestep_ = 0; }
    void next(const std::vector<float>& lp) {
        const int vs = lp.size();
        
        // Efficient top-k using min-heap (priority_queue)
        auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> pq(cmp);
        
        int k = std::min(topk_, vs);
        for (int i = 0; i < vs; ++i) {
            if (pq.size() < static_cast<size_t>(k)) {
                pq.push({lp[i], i});
            } else if (lp[i] > pq.top().first) {
                pq.pop();
                pq.push({lp[i], i});
            }
        }
        
        // Extract top-k candidates
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(k);
        while (!pq.empty()) {
            candidates.push_back(pq.top());
            pq.pop();
        }
        
        std::vector<PrefixNode*> np;
        for (PrefixNode* p : prefixes_) {
            // Process blank
            float lpb = lp[blank_id_];
            p->log_prob_b_cur = log_sum_exp(p->log_prob_b_cur, lpb + p->score);
            
            // Process top-k non-blank candidates
            for (const auto& cand : candidates) {
                int c = cand.second;
                if (c == blank_id_) continue;
                
                float lpc = lp[c];
                
                if (c == p->character) {
                    // Repeated character: update current prefix probability
                    p->log_prob_nb_cur = log_sum_exp(p->log_prob_nb_cur, lpc + p->log_prob_nb_prev);
                    
                    // Can also extend after blank (aa pattern)
                    if (p->log_prob_b_prev > NUM_FLT_MIN) {
                        PrefixNode* pn = p->get_child(c, timestep_);
                        float log_p = lpc + p->log_prob_b_prev;
                        pn->log_prob_nb_cur = log_sum_exp(pn->log_prob_nb_cur, log_p);
                    }
                } else {
                    // New character: extend prefix
                    PrefixNode* pn = p->get_child(c, timestep_);
                    float log_p = lpc + p->score;
                    pn->log_prob_nb_cur = log_sum_exp(pn->log_prob_nb_cur, log_p);
                }
            }
        }
        
        // Collect all prefixes
        np.clear(); 
        collect_prefixes(root_, np);
        
        // Update probabilities for next iteration
        for (PrefixNode* p : np) {
            p->log_prob_nb_prev = p->log_prob_nb_cur;
            p->log_prob_b_prev = p->log_prob_b_cur;
            p->score = log_sum_exp(p->log_prob_b_cur, p->log_prob_nb_cur);
            p->log_prob_b_cur = NUM_FLT_MIN;
            p->log_prob_nb_cur = NUM_FLT_MIN;
        }
        
        // Prune to beam size
        if (np.size() > static_cast<size_t>(beam_size_)) {
            std::nth_element(np.begin(), np.begin() + beam_size_, np.end(), 
                           [](const PrefixNode* a, const PrefixNode* b) { return a->score > b->score; });
            np.resize(beam_size_);
        }
        
        prefixes_ = np;
        ++timestep_;
    }
    CTCResult decode() {
        CTCResult r;
        if (prefixes_.empty()) { r.score = NUM_FLT_MIN; return r; }
        PrefixNode* best = prefixes_[0];
        for (size_t i = 1; i < prefixes_.size(); ++i) if (prefixes_[i]->score > best->score) best = prefixes_[i];
        best->get_path(r.tokens, r.timesteps); r.score = best->score;
        return r;
    }
};
}
#endif

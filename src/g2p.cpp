#include "g2p.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "misaki/g2p.hpp"
#include "phonemizer.hpp"
#include "utf8.hpp"

namespace kokoro {
namespace {

// All espeak-ng access (misaki's English OOV fallback and the non-English path
// below) ultimately drives the same non-thread-safe global espeak state, so
// G2P::phonemize is serialized as a whole. Phonemization is microseconds-to-low-
// ms and runs once per request, well off the inference hot path.
std::mutex g_mu;
std::string g_lexiconDir;
std::string g_espeakDataPath;
std::unique_ptr<misaki::G2P> g_us, g_gb;  // lazily constructed per dialect
bool g_warnedNoLex = false;

// Lazily build (and cache) the misaki English engine for the requested dialect.
// Returns nullptr if construction fails (e.g. missing data dir), in which case
// the caller falls back to pure espeak.
const misaki::G2P* english_g2p(bool british) {  // call under g_mu
  if (g_lexiconDir.empty()) {
    if (!g_warnedNoLex) {
      spdlog::warn("G2P: no lexicon dir; English uses espeak only (less accurate)");
      g_warnedNoLex = true;
    }
    return nullptr;
  }
  std::unique_ptr<misaki::G2P>& slot = british ? g_gb : g_us;
  if (!slot) {
    try {
      slot = std::make_unique<misaki::G2P>(british, g_lexiconDir, g_espeakDataPath);
      spdlog::info("G2P: loaded misaki {} engine from {}", british ? "GB" : "US",
                   g_lexiconDir);
    } catch (const std::exception& e) {
      if (!g_warnedNoLex) {
        spdlog::warn("G2P: misaki engine load failed ({}); English uses espeak only",
                     e.what());
        g_warnedNoLex = true;
      }
      return nullptr;
    }
  }
  return slot.get();
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
  return s;
}

// Punctuation that maps into the kokoro vocab (used by the non-English path to
// re-insert clause delimiters that espeak's phoneme stream drops).
std::string punct_phoneme(char32_t c) {
  switch (c) {
    case ';': case ':': case ',': case '.': case '!': case '?':
    case '(': case ')':
      return std::string(1, (char)c);
    case 0x2014: return "\xE2\x80\x94";  // — em dash
    case 0x2026: return "\xE2\x80\xA6";  // … ellipsis
    case '-': case 0x2013: return "\xE2\x80\x94";  // -/– -> —
    case '"': return "\"";
    case 0x201C: return "\xE2\x80\x9C";  // “
    case 0x201D: return "\xE2\x80\x9D";  // ”
    default: return "";
  }
}

// ---- English: delegate to the misaki C++ port ---------------------------
std::string phonemize_english(const std::string& text, bool british) {  // call under g_mu
  const misaki::G2P* g = english_g2p(british);
  if (g) return g->phonemize(text);
  // Degraded path: no lexicon -> per-"word" espeak via kokoro's Phonemizer.
  // (misaki construction failed; keep producing *something* rather than empty.)
  return Phonemizer::phonemize(text, british);
}

// ---- non-English (EspeakG2P) --------------------------------------------
struct E2M { const char* from; const char* to; };
// EspeakG2P.e2m (version != 2.0), sorted longest-first by porting order.
const E2M ESPEAK_G2P_E2M[] = {
    {"a^\xC9\xAA", "I"},          // aɪ
    {"a^\xCA\x8A", "W"},          // aʊ
    {"d^z", "\xCA\xA3"},          // dz -> ʣ
    {"d^\xCA\x92", "\xCA\xA4"},   // dʒ -> ʤ
    {"e^\xC9\xAA", "A"},          // eɪ
    {"o^\xCA\x8A", "O"},          // oʊ
    {"\xC9\x99^\xCA\x8A", "Q"},   // əʊ
    {"s^s", "S"},                 // ss -> S
    {"t^s", "\xCA\xA6"},          // ts -> ʦ
    {"t^\xCA\x83", "\xCA\xA7"},   // tʃ -> ʧ
    {"\xC9\x94^\xC9\xAA", "Y"},   // ɔɪ
};

// Split on sentence punctuation, keeping the delimiters, so espeak punctuation
// (which the phoneme stream drops) is re-inserted for prosody/pauses.
std::vector<std::pair<std::string, std::string>> split_clauses(const std::string& text) {
  // returns list of (segment, trailing_delim)
  std::vector<std::pair<std::string, std::string>> out;
  std::vector<char32_t> cps = utf8::decode(text);
  std::string seg;
  auto flush = [&](const std::string& delim) { out.push_back({seg, delim}); seg.clear(); };
  for (char32_t c : cps) {
    std::string pp = punct_phoneme(c);
    bool isDelim = (c == ';' || c == ':' || c == ',' || c == '.' || c == '!' ||
                    c == '?' || c == 0x2026);
    if (isDelim) flush(pp);
    else utf8::append(seg, c);
  }
  if (!seg.empty()) flush("");
  return out;
}

std::string phonemize_nonenglish(const std::string& text, const std::string& lang) {
  std::string out;
  for (auto& [seg, delim] : split_clauses(text)) {
    std::string s = seg;
    // strip whitespace-only segments
    bool allWs = true;
    for (char c : s) if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r')) allWs = false;
    if (!allWs) {
      // protect literal parentheses (EspeakG2P maps ()->«» before phonemizing)
      s = replace_all(s, "(", "\xC2\xAB");  // «
      s = replace_all(s, ")", "\xC2\xBB");  // »
      std::string ps = Phonemizer::espeakRawIPA(s, lang);
      for (auto& e : ESPEAK_G2P_E2M) ps = replace_all(ps, e.from, e.to);
      ps = replace_all(ps, "^", "");
      ps = replace_all(ps, "-", "");  // version != 2.0
      // strip espeak language-switch flags "(en)" etc. (real parens are «»)
      std::string cleaned;
      bool inFlag = false;
      for (std::size_t k = 0; k < ps.size(); ++k) {
        if (ps[k] == '(') { inFlag = true; continue; }
        if (ps[k] == ')') { inFlag = false; continue; }
        if (!inFlag) cleaned += ps[k];
      }
      ps = cleaned;
      ps = replace_all(ps, "\xC2\xAB", "(");
      ps = replace_all(ps, "\xC2\xBB", ")");
      // trim
      while (!ps.empty() && ps.back() == ' ') ps.pop_back();
      out += ps;
    }
    out += delim;
    out += ' ';
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

struct Route { bool english; bool british; const char* espeak; };
Route route(const std::string& voice, bool britishOverride) {
  char p = voice.empty() ? 'a' : voice[0];
  switch (p) {
    case 'a': return {true, britishOverride, nullptr};
    case 'b': return {true, true, nullptr};
    case 'e': return {false, false, "es"};
    case 'f': return {false, false, "fr"};
    case 'h': return {false, false, "hi"};
    case 'i': return {false, false, "it"};
    case 'p': return {false, false, "pt-br"};
    case 'j': case 'z':
      spdlog::warn("G2P: voice '{}' ({}) needs a CJK engine; using English",
                   voice, p == 'j' ? "Japanese" : "Chinese");
      return {true, britishOverride, nullptr};
    default:
      return {true, britishOverride, nullptr};
  }
}

} // namespace

void G2P::init(const std::string& lexiconDir, const std::string& espeakDataPath) {
  std::lock_guard<std::mutex> g(g_mu);
  g_lexiconDir = lexiconDir;
  g_espeakDataPath = espeakDataPath;
}

void G2P::terminate() {
  std::lock_guard<std::mutex> g(g_mu);
  g_us.reset();
  g_gb.reset();
}

std::string G2P::phonemize(const std::string& text, const std::string& voiceName,
                           bool britishOverride) {
  Route r = route(voiceName, britishOverride);
  std::lock_guard<std::mutex> g(g_mu);
  if (r.english) return phonemize_english(text, r.british);
  return phonemize_nonenglish(text, r.espeak);
}

} // namespace kokoro

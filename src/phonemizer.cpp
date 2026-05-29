#include "phonemizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include <spdlog/spdlog.h>

namespace kokoro {

namespace {

std::mutex g_espeak_mu; // espeak-ng API is not thread-safe.

// UTF-8 helpers (no extra dep). We only need:
//   - replace a UTF-8 substring globally
//   - decode UTF-8 to codepoints

bool decode_utf8(const std::string& s, std::vector<char32_t>& out) {
  out.clear();
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp;
    int extra;
    if (c < 0x80) { cp = c; extra = 0; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else return false;
    if (i + extra >= s.size()) return false;
    for (int k = 0; k < extra; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[++i]);
      if ((cc & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (cc & 0x3F);
    }
    out.push_back(cp);
    ++i;
  }
  return true;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  if (from.empty()) return;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

// Vendored from hexgrad/misaki:misaki/espeak.py EspeakFallback.E2M plus
// the british/american post-pass tweaks. Apply in order — these are
// string substitutions; longest-match-first is implicit since we list the
// composite sequences (with the espeak ^ tie-marker) before single chars.
//
// Note: espeak-ng IPA output emits the U+0361 combining-double-inverted-
// breve (͡) between tied symbols (e.g. d͡ʒ). misaki's table uses ASCII ^
// as the tie marker because that's what their wrapper normalises to.
// We normalize U+0361 → '^' before applying the table.

struct Rule { const char* from; const char* to; };

constexpr Rule E2M_BASE[] = {
    // Pre-fixups
    {"\xCA\x94\xCC\x8Cn\xCC\xA9", "\xCA\x94n"},  // ʔˌn̩ -> ʔn
    {"\xCA\x94n\xCC\xA9",         "\xCA\x94n"},  // ʔn̩  -> ʔn

    // Diphthongs (with ^ tie marker, inserted by espeak ipa output)
    {"a^\xC9\xAA", "I"},  // aɪ
    {"a^\xCA\x8A", "W"},  // aʊ
    {"d^\xCA\x92", "\xCA\xA4"},  // dʒ -> ʤ
    {"e^\xC9\xAA", "A"},  // eɪ
    {"e",           "A"},
    {"t^\xCA\x83", "\xCA\xA7"},  // tʃ -> ʧ
    {"\xC9\x94^\xC9\xAA", "Y"},  // ɔɪ
    {"\xC9\x99^l", "\xE1\xB5\x8Al"}, // əl -> ᵊl
    {"\xCA\xB2o", "jo"}, // ʲo
    {"\xCA\xB2\xC9\x99", "j\xC9\x99"}, // ʲə
    {"\xCA\xB2", ""},
    {"\xC9\x9A", "\xC9\x99\xC9\xB9"}, // ɚ -> əɹ
    {"r",  "\xC9\xB9"}, // r -> ɹ
    {"x",  "k"},
    {"\xC3\xA7", "k"},  // ç
    {"\xC9\x90", "\xC9\x99"}, // ɐ -> ə
    {"\xC9\xAC", "l"}, // ɬ
    {"\xCC\x83", ""},  // combining tilde U+0303
};

constexpr Rule E2M_BRITISH[] = {
    {"e^\xC9\x99", "\xC9\x9B\xCB\x90"},  // eə -> ɛː
    {"i\xC9\x99",  "\xC9\xAA\xC9\x99"},  // iə -> ɪə
    {"\xC9\x99^\xCA\x8A", "Q"},          // əʊ
};

constexpr Rule E2M_AMERICAN[] = {
    {"o^\xCA\x8A", "O"},                              // oʊ
    {"\xC9\x9C\xCB\x90\xC9\xB9", "\xC9\x9C\xC9\xB9"}, // ɜːɹ -> ɜɹ
    {"\xC9\x9C\xCB\x90", "\xC9\x9C\xC9\xB9"},          // ɜː  -> ɜɹ
    {"\xC9\xAA\xC9\x99", "i\xC9\x99"},                 // ɪə  -> iə
};

void apply_rules(std::string& s, const Rule* rules, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) replace_all(s, rules[i].from, rules[i].to);
}

void post_remap(std::string& s, bool british) {
  // Normalize U+0361 combining tie -> '^'
  replace_all(s, "\xCD\xA1", "^");
  apply_rules(s, E2M_BASE, sizeof(E2M_BASE) / sizeof(Rule));
  if (british) {
    apply_rules(s, E2M_BRITISH, sizeof(E2M_BRITISH) / sizeof(Rule));
  } else {
    apply_rules(s, E2M_AMERICAN, sizeof(E2M_AMERICAN) / sizeof(Rule));
  }
  // Cleanup
  replace_all(s, "^", "");
  replace_all(s, "\xCB\x90", ""); // ː length marker
  replace_all(s, "o", "\xC9\x94"); // o -> ɔ
}

bool g_inited = false;

} // namespace

void Phonemizer::init(const std::string& espeakDataPath) {
  std::lock_guard<std::mutex> g(g_espeak_mu);
  if (g_inited) return;
  int rc = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, /*buflen*/ 0,
                             espeakDataPath.c_str(), /*options*/ 0);
  if (rc < 0)
    throw std::runtime_error("espeak_Initialize failed (data path: " +
                             espeakDataPath + ")");
  g_inited = true;
  spdlog::info("eSpeak-ng initialised (data: {})", espeakDataPath);
}

void Phonemizer::terminate() {
  std::lock_guard<std::mutex> g(g_espeak_mu);
  if (!g_inited) return;
  espeak_Terminate();
  g_inited = false;
}

std::string Phonemizer::espeakRawIPA(const std::string& text,
                                     const std::string& espeakVoice) {
  std::lock_guard<std::mutex> g(g_espeak_mu);
  if (!g_inited)
    throw std::runtime_error("Phonemizer::init() must be called first");

  if (espeak_SetVoiceByName(espeakVoice.c_str()) != EE_OK)
    throw std::runtime_error("espeak_SetVoiceByName failed: " + espeakVoice);

  std::string out;
  out.reserve(text.size());

  const void* in_ptr = text.c_str();
  // textmode = UTF-8; phonememode 0x02 = IPA output (no ascii separator). IPA
  // mode emits the U+0361 combining double-inverted-breve between tied symbols.
  const int textmode = espeakCHARS_UTF8;
  const int phonememode = 0x02;

  while (in_ptr != nullptr) {
    const char* p = espeak_TextToPhonemes(&in_ptr, textmode, phonememode);
    if (p != nullptr) {
      out += p;
      // Space between espeak clauses keeps word boundaries sane.
      out += ' ';
    }
  }

  // Normalize the U+0361 tie -> ASCII '^' (matches the misaki phonemizer
  // wrapper's tie='^'), so downstream E2M tables can match composite symbols.
  replace_all(out, "\xCD\xA1", "^");

  while (!out.empty() && (out.back() == ' ' || out.back() == '\n' ||
                          out.back() == '\r' || out.back() == '\t'))
    out.pop_back();
  return out;
}

std::string Phonemizer::phonemize(const std::string& text, bool british) {
  std::string out = espeakRawIPA(text, british ? "en-gb" : "en-us");
  post_remap(out, british);
  while (!out.empty() && (out.back() == ' ' || out.back() == '\n' ||
                          out.back() == '\r' || out.back() == '\t'))
    out.pop_back();
  return out;
}

std::vector<int64_t> Phonemizer::to_ids(const std::string& phonemes,
                                       const Vocab& vocab,
                                       std::size_t* dropped) {
  std::vector<char32_t> cps;
  if (!decode_utf8(phonemes, cps))
    throw std::runtime_error("invalid UTF-8 in phonemes");
  std::vector<int64_t> ids;
  ids.reserve(cps.size());
  std::size_t miss = 0;
  for (char32_t cp : cps) {
    auto it = vocab.find(cp);
    if (it == vocab.end()) { ++miss; continue; }
    ids.push_back(static_cast<int64_t>(it->second));
  }
  if (dropped) *dropped = miss;
  return ids;
}

} // namespace kokoro

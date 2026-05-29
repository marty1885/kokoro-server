#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace kokoro {

// Vocab is char32_t (UTF-32 codepoint) -> int id. Single-codepoint keys only,
// matching Kokoro's config.json vocab.
using Vocab = std::map<char32_t, int>;

struct Phonemizer {
  // espeakDataPath: path to espeak-ng-data directory. Must call init() once
  // before phonemize().
  static void init(const std::string& espeakDataPath);
  static void terminate();

  // Low-level: run espeak-ng with IPA output for the given espeak voice
  // ("en-us", "es", "fr-fr", ...). Returns raw IPA with the U+0361 tie
  // normalized to ASCII '^' (what misaki's phonemizer wrapper emits) and
  // clause boundaries joined by spaces. No E2M remap. Throws on espeak failure.
  static std::string espeakRawIPA(const std::string& text,
                                  const std::string& espeakVoice);

  // text -> Kokoro-vocab phoneme string (UTF-8). Runs espeak-ng with IPA
  // output mode, then applies the vendored misaki EspeakFallback E2M remap
  // (american by default; pass british=true for GB voices). This is the
  // English OOV fallback (misaki feeds it one word at a time). Throws.
  static std::string phonemize(const std::string& text, bool british = false);

  // Convert a Kokoro-vocab phoneme string (UTF-8) to token ids using the
  // supplied vocab. Characters not in vocab are dropped (caller logs).
  // Returned ids do NOT include the [0, ..., 0] BOS/EOS padding — the
  // caller wraps that since it's an inference detail.
  static std::vector<int64_t> to_ids(const std::string& phonemes,
                                     const Vocab& vocab,
                                     std::size_t* dropped = nullptr);
};

} // namespace kokoro

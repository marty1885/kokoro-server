#pragma once

#include <string>

namespace kokoro {

// Top-level grapheme->phoneme front end. Routes by voice language:
//   a -> American English   b -> British English   (misaki dict + espeak OOV)
//   e -> es   f -> fr-fr   h -> hi   i -> it   p -> pt-br   (espeak + per-lang E2M)
//   j/z (Japanese/Chinese) are unsupported (need OpenJTalk / jieba) and fall
//   back to American English with a warning.
//
// Requires Phonemizer::init() (espeak) to have been called first.
class G2P {
 public:
  // lexiconDir: misaki data directory — must contain the us/gb gold+silver
  // JSONs, tagger/, and spacy_exceptions.json. If empty or the files are
  // missing, English degrades to pure espeak (with a warning). espeakDataPath
  // is the espeak-ng-data dir for the English OOV fallback (espeak is also
  // initialized separately via Phonemizer::init for the non-English path).
  static void init(const std::string& lexiconDir,
                   const std::string& espeakDataPath = "");
  static void terminate();

  // Phonemize text into the Kokoro vocab phoneme string for the given voice.
  // britishOverride forces GB English for 'a'/'b' voices (from the API flag).
  static std::string phonemize(const std::string& text,
                               const std::string& voiceName,
                               bool britishOverride = false);
};

} // namespace kokoro

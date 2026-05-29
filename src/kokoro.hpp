#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inferer.hpp"
#include "npy.hpp"
#include "phonemizer.hpp"

namespace kokoro {

struct EngineConfig {
  // Decoder window — must match the RKNN compile-time T_fix.
  int T_fix = 50;
  int CTX = 2;
  // Overlap-add crossfade width (samples) per window seam, split DEPOP_OVL/2
  // into the real context both neighbouring windows decoded. Must be
  // <= 2*CTX*samplesPerFrame and small enough to stay inside well-supported
  // context (CTX*samplesPerFrame each side).
  int DEPOP_OVL = 480;
  int sampleRate = 24000;
  int samplesPerFrame = 600;
  int harChannels = 22;

  // Max phoneme tokens per encoder pass. BERT's position embeddings cap the
  // sequence at context_length=512; with the BOS/EOS wrap that leaves 510.
  // Longer input is split into sentence-aligned chunks before the encoder.
  int maxTokens = 510;

  // Worker threads for the per-window har/decoder/iSTFT pipeline. Match the
  // RKNN context count (3 on RK3588). ONNX-only builds set this to 1.
  int decoderWorkers = 3;
};

struct SynthesisResult {
  double encoderSeconds = 0.0;
  double harSeconds = 0.0;
  double decoderSeconds = 0.0;
  double istftSeconds = 0.0;
  double audioSeconds = 0.0;
  double realTimeFactor = 0.0;
};

// One audio chunk in playback order. int16 PCM, sampleRate=24000, mono.
using ChunkCallback = std::function<void(const int16_t* data, std::size_t n)>;

class VoicePack {
 public:
  static VoicePack load(const std::string& path);
  // Slice the (510, 1, 256) pack by n_phonemes-1 (clamped). Returns 256-d.
  std::vector<float> slice(int n_phonemes) const;
  bool empty() const { return arr_.shape.empty(); }
 private:
  NpyArray arr_;
};

class Engine {
 public:
  Engine();
  ~Engine();

  // Load encoder/har/decoder + Kokoro vocab (config.json with "vocab" key)
  // and a voices_npy/ directory for lazy voice loading.
  void load(const std::string& vocabConfigJson,
            const std::string& encoderPath,
            const std::string& harGenPath,
            const std::string& decoderPath,
            const std::string& voicesDir,
            const std::string& accelerator,
            const EngineConfig& cfg);

  // Lazy-load voice by name (looked up at voicesDir/<name>.npy).
  std::shared_ptr<VoicePack> getVoice(const std::string& name);

  // List available voice names (filesystem scan, cheap).
  std::vector<std::string> listVoices() const;

  // Lookup the vocab. Used by callers that want to phonemize externally
  // and pass ids; the convenience wrappers below do this internally.
  const Vocab& vocab() const;

  const EngineConfig& config() const;

  // Synthesize from Kokoro-vocab token ids (no BOS/EOS — wrapped here).
  // chunk emit fires in playback order; multi-window decoding runs in
  // parallel under the hood.
  SynthesisResult synthesizeIds(const std::vector<int64_t>& ids,
                                const VoicePack& voice,
                                float speed,
                                ChunkCallback emit);

  // Phonemize via espeak + E2M remap, then synthesize.
  SynthesisResult synthesizeText(const std::string& text,
                                 const std::string& voiceName,
                                 float speed,
                                 bool british,
                                 ChunkCallback emit);

  // Synthesize from a Kokoro-vocab phoneme string (no remap).
  SynthesisResult synthesizePhonemes(const std::string& phonemes,
                                     const std::string& voiceName,
                                     float speed,
                                     ChunkCallback emit);

  struct Impl;
 private:
  // Synthesize one token chunk that already fits BERT's context window.
  // synthesizeIds() splits oversized input and fans out to this.
  SynthesisResult synthChunk(Impl& I, const std::vector<int64_t>& ids,
                             const VoicePack& voice, float speed,
                             const ChunkCallback& emit);

  std::unique_ptr<Impl> impl_;
};

} // namespace kokoro

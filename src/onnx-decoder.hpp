#pragma once

#include <memory>
#include <string>

#include "inferer.hpp"

namespace Ort { class Session; class Env; class SessionOptions; }

namespace kokoro {

class OnnxDecoder : public Decoder {
 public:
  OnnxDecoder(int T_fix);
  ~OnnxDecoder() override;

  int T_fix() const override { return T_fix_; }
  void load(const std::string& path) override {
    load(path, std::string());
  }
  // Accelerator: "", "cuda", "tensorrt".
  void load(const std::string& path, const std::string& accelerator);

  DecoderOutput infer(const float* asr, const float* F0, const float* N,
                      const float* s, const float* har, int har_T) override;

 private:
  int T_fix_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace kokoro

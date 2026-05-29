#pragma once

#include <vector>
#include <string>

namespace kokoro {

// Output of one decoder window. Shapes: (1, F, T_frames) row-major, F = 11.
// T_frames is fixed for a given build (T_fix * 120 + 1 for the upstream model).
struct DecoderOutput {
  std::vector<float> spec;
  std::vector<float> phase;
  int F = 0;        // freq bins
  int T_frames = 0; // time frames
};

struct Decoder {
  virtual ~Decoder() = default;

  // T_fix is the number of ASR frames the decoder expects (compiled-in for
  // RKNN; ONNX could be dynamic but we standardize). Inputs are always padded
  // to this size by the caller.
  virtual int T_fix() const = 0;

  // All buffers are row-major float32. har is (1, 22, har_T) where
  // har_T = 2 * T_fix * 300 / 5 + 1.
  virtual DecoderOutput infer(const float* asr,    // (1, 512, T_fix)
                              const float* F0,     // (1, 2*T_fix)
                              const float* N,      // (1, 2*T_fix)
                              const float* s,      // (1, 128)
                              const float* har,    // (1, 22, har_T)
                              int har_T) = 0;

  virtual void load(const std::string& path) = 0;
};

} // namespace kokoro

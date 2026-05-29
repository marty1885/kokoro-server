#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rknn_api.h>

#include "inferer.hpp"

namespace kokoro {

// One RKNN context — holds its own input/output attr cache. Not thread-safe
// on its own; dispatcher serializes per-instance with the implTracker pattern.
struct RknnInstance {
  rknn_context ctx = 0;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;

  RknnInstance() = default;
  explicit RknnInstance(rknn_context c);
  RknnInstance(RknnInstance&&) noexcept;
  RknnInstance& operator=(RknnInstance&&) noexcept;
  RknnInstance(const RknnInstance&) = delete;
  RknnInstance& operator=(const RknnInstance&) = delete;
  ~RknnInstance();

  DecoderOutput infer(const float* asr, const float* F0, const float* N,
                      const float* s, const float* har, int har_T, int T_fix);
};

// Multi-context RKNN dispatcher (3 instances on RK3588, one per NPU core).
class RknnDecoder : public Decoder {
 public:
  explicit RknnDecoder(int T_fix);
  ~RknnDecoder() override;

  int T_fix() const override { return T_fix_; }
  void load(const std::string& path) override;

  DecoderOutput infer(const float* asr, const float* F0, const float* N,
                      const float* s, const float* har, int har_T) override;

 private:
  int T_fix_;
  std::vector<RknnInstance> impls_;
  std::vector<int> impl_busy_; // 0=free, 1=busy
  std::mutex mu_;
  std::condition_variable cv_;
};

} // namespace kokoro

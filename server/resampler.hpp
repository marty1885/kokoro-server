#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <soxr.h>

namespace kokoro_server {

// Streaming int16 mono resampler. One soxr_t lives for the duration of one
// streaming session (per WebSocket conn or per request). Re-using preserves
// the resampler's tail state across chunks — paroli's per-call create/destroy
// pattern dropped that state and produced boundary clicks.
class Resampler {
 public:
  Resampler(double in_sr, double out_sr, int channels) : ratio_(out_sr / in_sr) {
    if (in_sr == out_sr) { passthrough_ = true; return; }
    soxr_io_spec_t io = soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I);
    soxr_quality_spec_t q = soxr_quality_spec(SOXR_MQ, 0);
    soxr_error_t e = nullptr;
    soxr_ = soxr_create(in_sr, out_sr, channels, &e, &io, &q, nullptr);
    if (e) throw std::runtime_error(std::string("soxr_create: ") + e);
  }
  Resampler(const Resampler&) = delete;
  Resampler& operator=(const Resampler&) = delete;
  ~Resampler() { if (soxr_) soxr_delete(soxr_); }

  // Process a chunk; returns whatever output samples are ready *now*. Call
  // flush() at end of stream to drain the resampler tail.
  std::vector<int16_t> process(const int16_t* in, size_t n) {
    if (passthrough_) return std::vector<int16_t>(in, in + n);
    // Output buffer sized for the ratio with slack.
    std::vector<int16_t> out(static_cast<size_t>(n * ratio_ + 64));
    size_t idone = 0, odone = 0;
    soxr_error_t e = soxr_process(soxr_, in, n, &idone, out.data(),
                                  out.size(), &odone);
    if (e) throw std::runtime_error(std::string("soxr_process: ") + e);
    out.resize(odone);
    return out;
  }

  // Drain the internal tail (call once at end-of-stream).
  std::vector<int16_t> flush() {
    if (passthrough_) return {};
    std::vector<int16_t> out(1024);
    std::vector<int16_t> collected;
    for (;;) {
      size_t odone = 0;
      soxr_error_t e = soxr_process(soxr_, nullptr, 0, nullptr,
                                    out.data(), out.size(), &odone);
      if (e) throw std::runtime_error(std::string("soxr_process flush: ") + e);
      if (odone == 0) break;
      collected.insert(collected.end(), out.begin(), out.begin() + odone);
    }
    return collected;
  }

 private:
  soxr_t soxr_ = nullptr;
  double ratio_ = 1.0;
  bool passthrough_ = false;
};

} // namespace kokoro_server

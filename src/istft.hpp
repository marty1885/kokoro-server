#pragma once

#include <vector>

namespace kokoro {

// iSTFT matching infer.py's CustomSTFT.inverse: periodic Hann window, real
// branch computed as an explicit DFT matrix multiply via OpenBLAS sgemm (n_fft
// is tiny here — 20 — so a real FFT lib has no win and would refuse the size).
class IStft {
public:
  IStft(int n_fft = 20, int hop = 5);

  // spec, phase: (B, F, T_frames) row-major, F = n_fft/2 + 1.
  // phase is the sin-argument value (not radians-only). Audio out is
  // (B, out_len) with out_len = (T_frames - 1) * hop, pad-trimmed.
  std::vector<float> apply(const float* spec, const float* phase, int B,
                           int F, int T_frames, int& out_len_per_batch) const;

  int n_fft() const { return n_fft_; }
  int hop()   const { return hop_;   }
  int F()     const { return F_;     }

private:
  int n_fft_;
  int hop_;
  int F_;
  std::vector<float> cos_w_; // (F, n_fft) row-major, includes window/n_fft scale
  std::vector<float> sin_w_;
  std::vector<float> win_sq_; // (n_fft) Hann^2, for NOLA overlap-add envelope
};

} // namespace kokoro

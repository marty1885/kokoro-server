#include "istft.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#include <cblas.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace kokoro {

IStft::IStft(int n_fft, int hop)
    : n_fft_(n_fft), hop_(hop), F_(n_fft / 2 + 1) {
  cos_w_.resize(static_cast<size_t>(F_) * n_fft_);
  sin_w_.resize(static_cast<size_t>(F_) * n_fft_);

  std::vector<float> win(n_fft_);
  win_sq_.resize(n_fft_);
  for (int n = 0; n < n_fft_; ++n) {
    win[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n / n_fft_);
    win_sq_[n] = win[n] * win[n];
  }

  const float inv_n = 1.0f / static_cast<float>(n_fft_);
  for (int k = 0; k < F_; ++k) {
    // Real iFFT of a one-sided spectrum: bins 1..F-2 stand in for their
    // conjugate twins, so they carry a factor of 2; DC (k=0) and Nyquist
    // (k=F-1, n_fft even) appear once. CustomSTFT omits this — that halving
    // of the mid/high bins is what made the output sound muffled.
    const float dbl = (k == 0 || k == F_ - 1) ? 1.0f : 2.0f;
    for (int n = 0; n < n_fft_; ++n) {
      float angle = 2.0f * static_cast<float>(M_PI) * n * k / n_fft_;
      float w = win[n] * inv_n * dbl;
      cos_w_[static_cast<size_t>(k) * n_fft_ + n] = std::cos(angle) * w;
      sin_w_[static_cast<size_t>(k) * n_fft_ + n] = std::sin(angle) * w;
    }
  }
}

std::vector<float> IStft::apply(const float* spec, const float* phase, int B,
                                int F, int T, int& out_len_per_batch) const {
  assert(F == F_);
  const int n_fft = n_fft_;
  const int hop = hop_;
  const int ratio = n_fft / hop;
  assert(ratio * hop == n_fft);

  const int out_total = (T - 1) * hop + n_fft;
  const int pad = n_fft / 2;
  out_len_per_batch = out_total - 2 * pad;

  std::vector<float> real(static_cast<size_t>(F) * T);
  std::vector<float> imag(static_cast<size_t>(F) * T);
  std::vector<float> fr_r(static_cast<size_t>(T) * n_fft);
  std::vector<float> fr_i(static_cast<size_t>(T) * n_fft);
  std::vector<float> frames(static_cast<size_t>(T) * n_fft);
  std::vector<float> tmp_out(out_total);
  std::vector<float> out(static_cast<size_t>(B) * out_len_per_batch);

  // Window^2 overlap-add envelope (NOLA normalization, as torch.istft does).
  // Batch-independent — depends only on T/n_fft/hop/window — so build once.
  std::vector<float> env(out_total, 0.0f);
  for (int t = 0; t < T; ++t) {
    float* e = &env[static_cast<size_t>(t) * hop];
    for (int n = 0; n < n_fft; ++n) e[n] += win_sq_[n];
  }
  for (int i = 0; i < out_total; ++i)
    if (env[i] < 1e-11f) env[i] = 1e-11f;

  for (int b = 0; b < B; ++b) {
    const float* spec_b = spec + static_cast<size_t>(b) * F * T;
    const float* phase_b = phase + static_cast<size_t>(b) * F * T;
    const size_t FT = static_cast<size_t>(F) * T;
    for (size_t i = 0; i < FT; ++i) {
      float p = phase_b[i];
      real[i] = spec_b[i] * std::cos(p);
      imag[i] = spec_b[i] * std::sin(p);
    }

    // fr_r[t, n] = sum_f real[f, t] * cos_w[f, n]   = real^T @ cos_w
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                /*M*/ T, /*N*/ n_fft, /*K*/ F,
                1.0f,
                real.data(), /*lda*/ T,
                cos_w_.data(), /*ldb*/ n_fft,
                0.0f,
                fr_r.data(), /*ldc*/ n_fft);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                T, n_fft, F,
                1.0f,
                imag.data(), T,
                sin_w_.data(), n_fft,
                0.0f,
                fr_i.data(), n_fft);

    const size_t Tn = static_cast<size_t>(T) * n_fft;
    for (size_t i = 0; i < Tn; ++i) frames[i] = fr_r[i] - fr_i[i];

    std::memset(tmp_out.data(), 0, out_total * sizeof(float));
    for (int k = 0; k < ratio; ++k) {
      const int off = k * hop;
      for (int t = 0; t < T; ++t) {
        const float* row = &frames[static_cast<size_t>(t) * n_fft + off];
        float* dst = &tmp_out[off + t * hop];
        for (int j = 0; j < hop; ++j) dst[j] += row[j];
      }
    }
    for (int i = 0; i < out_total; ++i) tmp_out[i] /= env[i];
    std::memcpy(&out[static_cast<size_t>(b) * out_len_per_batch],
                &tmp_out[pad], out_len_per_batch * sizeof(float));
  }
  return out;
}

} // namespace kokoro

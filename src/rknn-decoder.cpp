#include "rknn-decoder.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace kokoro {

namespace {

// Portable fp32 -> fp16 (round-to-nearest-even). Avoids needing __fp16.
inline uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  uint16_t h;
  if (exp >= 31) {
    h = static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0)); // inf/nan
  } else if (exp <= 0) {
    if (exp < -10) { h = static_cast<uint16_t>(sign); }
    else {
      mant |= 0x800000u;
      uint32_t shift = static_cast<uint32_t>(14 - exp);
      uint32_t hmant = mant >> shift;
      // round
      if ((mant >> (shift - 1)) & 1) hmant += 1;
      h = static_cast<uint16_t>(sign | hmant);
    }
  } else {
    uint32_t hmant = mant >> 13;
    if (mant & 0x1000u) {
      hmant += 1;
      if (hmant & 0x400u) { hmant = 0; exp += 1; }
    }
    if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00u);
    else h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | hmant);
  }
  return h;
}

inline float fp16_to_fp32_local(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) out = sign << 31;
    else {
      while (!(mant & 0x400u)) { mant <<= 1; --exp; }
      ++exp; mant &= 0x3FFu;
      out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | (0xFFu << 23) | (mant << 13);
  } else {
    out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
  }
  float r;
  std::memcpy(&r, &out, 4);
  return r;
}

std::vector<uint8_t> loadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open rknn model: " + path);
  f.seekg(0, std::ios::end);
  std::vector<uint8_t> buf(static_cast<size_t>(f.tellg()));
  f.seekg(0, std::ios::beg);
  f.read(reinterpret_cast<char*>(buf.data()), buf.size());
  return buf;
}

} // namespace

// ---- RknnInstance --------------------------------------------------------

RknnInstance::RknnInstance(rknn_context c) : ctx(c) {
  rknn_input_output_num io_num;
  auto ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
    throw std::runtime_error("rknn_query (io num) failed: " + std::to_string(ret));
  input_attrs.resize(io_num.n_input);
  output_attrs.resize(io_num.n_output);
  std::memset(input_attrs.data(), 0, sizeof(rknn_tensor_attr) * io_num.n_input);
  std::memset(output_attrs.data(), 0, sizeof(rknn_tensor_attr) * io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i],
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
      throw std::runtime_error("rknn_query input failed: " + std::to_string(ret));
  }
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i],
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
      throw std::runtime_error("rknn_query output failed: " + std::to_string(ret));
  }
}

RknnInstance::RknnInstance(RknnInstance&& o) noexcept
    : ctx(o.ctx), input_attrs(std::move(o.input_attrs)),
      output_attrs(std::move(o.output_attrs)) {
  o.ctx = 0;
}

RknnInstance& RknnInstance::operator=(RknnInstance&& o) noexcept {
  if (this != &o) {
    if (ctx) rknn_destroy(ctx);
    ctx = o.ctx; o.ctx = 0;
    input_attrs = std::move(o.input_attrs);
    output_attrs = std::move(o.output_attrs);
  }
  return *this;
}

RknnInstance::~RknnInstance() {
  if (ctx) rknn_destroy(ctx);
}

DecoderOutput RknnInstance::infer(const float* asr, const float* F0,
                                  const float* N, const float* s,
                                  const float* har, int har_T, int T_fix) {
  // Build fp16 buffers per input. Order is determined by graph; we map by
  // shape since Kokoro names aren't guaranteed inside RKNN.
  std::array<const float*, 5> srcs{asr, F0, N, s, har};
  std::array<size_t, 5> nelem{
      static_cast<size_t>(1) * 512 * T_fix,
      static_cast<size_t>(1) * 2 * T_fix,
      static_cast<size_t>(1) * 2 * T_fix,
      static_cast<size_t>(1) * 128,
      static_cast<size_t>(1) * 22 * static_cast<size_t>(har_T),
  };
  if (input_attrs.size() != 5)
    throw std::runtime_error("rknn decoder: expected 5 inputs, got " +
                             std::to_string(input_attrs.size()));

  std::array<std::vector<uint16_t>, 5> bufs;
  for (int i = 0; i < 5; ++i) {
    bufs[i].resize(nelem[i]);
    for (size_t k = 0; k < nelem[i]; ++k)
      bufs[i][k] = fp32_to_fp16(srcs[i][k]);
  }

  std::array<rknn_input, 5> inputs{};
  for (int i = 0; i < 5; ++i) {
    inputs[i].index = static_cast<uint32_t>(i);
    inputs[i].size = static_cast<uint32_t>(bufs[i].size() * sizeof(uint16_t));
    inputs[i].type = RKNN_TENSOR_FLOAT16;
    inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[i].buf = bufs[i].data();
    inputs[i].pass_through = 0;
  }

  auto ret = rknn_inputs_set(ctx, 5, inputs.data());
  if (ret != RKNN_SUCC)
    throw std::runtime_error("rknn_inputs_set: " + std::to_string(ret));

  ret = rknn_run(ctx, nullptr);
  if (ret != RKNN_SUCC)
    throw std::runtime_error("rknn_run: " + std::to_string(ret));

  std::vector<rknn_output> outputs(output_attrs.size());
  std::memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
  // want_float = 0 to get fp16 directly; we convert ourselves.
  for (auto& o : outputs) o.want_float = 0;
  ret = rknn_outputs_get(ctx, static_cast<uint32_t>(outputs.size()),
                         outputs.data(), nullptr);
  if (ret != RKNN_SUCC)
    throw std::runtime_error("rknn_outputs_get: " + std::to_string(ret));

  // Convert outputs. Expect 2: spec, phase. Same shape (1, 11, T_frames).
  if (outputs.size() != 2)
    throw std::runtime_error("rknn decoder: expected 2 outputs, got " +
                             std::to_string(outputs.size()));

  auto& a0 = output_attrs[outputs[0].index];
  auto& a1 = output_attrs[outputs[1].index];

  auto shape_T_frames = [](const rknn_tensor_attr& a) -> int {
    // attr.dims is row-major; T is the last dim, F the middle.
    return static_cast<int>(a.dims[a.n_dims - 1]);
  };
  auto shape_F = [](const rknn_tensor_attr& a) -> int {
    return static_cast<int>(a.dims[a.n_dims - 2]);
  };

  DecoderOutput out;
  out.T_frames = shape_T_frames(a0);
  out.F = shape_F(a0);
  if (out.T_frames != shape_T_frames(a1) || out.F != shape_F(a1))
    throw std::runtime_error("rknn decoder: spec/phase shape mismatch");

  size_t nn = static_cast<size_t>(out.F) * out.T_frames;
  out.spec.resize(nn);
  out.phase.resize(nn);

  // We don't know which index is spec vs phase without name introspection.
  // RKNN preserves output order from the ONNX graph; Kokoro's decoder emits
  // (spec, phase). Trust that order; users with reversed models can patch.
  const uint16_t* p0 = static_cast<const uint16_t*>(outputs[0].buf);
  const uint16_t* p1 = static_cast<const uint16_t*>(outputs[1].buf);
  for (size_t k = 0; k < nn; ++k) {
    out.spec[k]  = fp16_to_fp32_local(p0[k]);
    out.phase[k] = fp16_to_fp32_local(p1[k]);
  }
  rknn_outputs_release(ctx, static_cast<uint32_t>(outputs.size()), outputs.data());
  return out;
}

// ---- RknnDecoder dispatcher ---------------------------------------------

RknnDecoder::RknnDecoder(int T_fix) : T_fix_(T_fix) {}
RknnDecoder::~RknnDecoder() = default;

void RknnDecoder::load(const std::string& path) {
  auto buf = loadFile(path);

  rknn_context root;
  auto ret = rknn_init(&root, buf.data(), static_cast<uint32_t>(buf.size()),
                       0, nullptr);
  if (ret != RKNN_SUCC)
    throw std::runtime_error("rknn_init: " + std::to_string(ret));

  // 3 contexts (one per RK3588 NPU core). dup_context only works before
  // any inference happens, hence we dup all 3 up front.
  std::array<rknn_context, 3> ctxs{root, 0, 0};
  for (int i = 1; i < 3; ++i) {
    ret = rknn_dup_context(&root, &ctxs[i]);
    if (ret != RKNN_SUCC)
      throw std::runtime_error("rknn_dup_context: " + std::to_string(ret));
  }
  impls_.reserve(3);
  for (int i = 0; i < 3; ++i) impls_.emplace_back(ctxs[i]);
  impl_busy_.assign(3, 0);
  spdlog::info("RknnDecoder loaded ({}, 3 NPU contexts)", path);
}

DecoderOutput RknnDecoder::infer(const float* asr, const float* F0,
                                 const float* N, const float* s,
                                 const float* har, int har_T) {
  int idx = -1;
  {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{
      for (size_t i = 0; i < impl_busy_.size(); ++i)
        if (impl_busy_[i] == 0) { idx = static_cast<int>(i); return true; }
      return false;
    });
    impl_busy_[idx] = 1;
  }
  DecoderOutput out;
  try {
    out = impls_[idx].infer(asr, F0, N, s, har, har_T, T_fix_);
  } catch (...) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      impl_busy_[idx] = 0;
    }
    cv_.notify_all();
    throw;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    impl_busy_[idx] = 0;
  }
  cv_.notify_all();
  return out;
}

} // namespace kokoro

#include "onnx-decoder.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

namespace kokoro {

struct OnnxDecoder::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "kokoro-decoder"};
  Ort::SessionOptions opt{};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> outNames;
};

OnnxDecoder::OnnxDecoder(int T_fix) : T_fix_(T_fix),
                                      impl_(std::make_unique<Impl>()) {}
OnnxDecoder::~OnnxDecoder() = default;

void OnnxDecoder::load(const std::string& path, const std::string& accelerator) {
  auto& I = *impl_;
  I.env.DisableTelemetryEvents();
  I.opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef USE_ORT_CUDA
  if (accelerator == "cuda") {
    OrtCUDAProviderOptions cuda{};
    cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    I.opt.AppendExecutionProvider_CUDA(cuda);
  } else if (accelerator == "tensorrt") {
    OrtTensorRTProviderOptions trt{};
    I.opt.AppendExecutionProvider_TensorRT(trt);
  }
#else
  (void)accelerator;
#endif
  I.session = std::make_unique<Ort::Session>(I.env, path.c_str(), I.opt);
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < I.session->GetOutputCount(); ++i)
    I.outNames.emplace_back(I.session->GetOutputNameAllocated(i, alloc).get());
  spdlog::info("OnnxDecoder loaded ({})", path);
}

DecoderOutput OnnxDecoder::infer(const float* asr, const float* F0,
                                 const float* N, const float* s,
                                 const float* har, int har_T) {
  auto& I = *impl_;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  const int T = T_fix_;
  std::array<std::vector<int64_t>, 5> shapes = {{
      {1, 512, T}, {1, 2 * T}, {1, 2 * T}, {1, 128}, {1, 22, har_T}
  }};
  std::array<const float*, 5> data{asr, F0, N, s, har};
  std::array<size_t, 5> nelem{
      static_cast<size_t>(1) * 512 * T,
      static_cast<size_t>(1) * 2 * T,
      static_cast<size_t>(1) * 2 * T,
      static_cast<size_t>(1) * 128,
      static_cast<size_t>(1) * 22 * har_T,
  };

  std::vector<Ort::Value> inputs;
  for (int i = 0; i < 5; ++i) {
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(data[i]), nelem[i],
        shapes[i].data(), shapes[i].size()));
  }

  static const std::array<const char*, 5> inNames{"asr", "F0", "N", "s", "har"};
  std::vector<const char*> outN;
  for (auto& s : I.outNames) outN.push_back(s.c_str());

  auto outs = I.session->Run(Ort::RunOptions{nullptr}, inNames.data(),
                             inputs.data(), inputs.size(),
                             outN.data(), outN.size());

  DecoderOutput out;
  auto take = [&](const std::string& name, std::vector<float>& dst,
                  int& F_out, int& T_out) {
    for (size_t i = 0; i < I.outNames.size(); ++i) {
      if (I.outNames[i] == name) {
        auto& t = outs[i];
        auto sh = t.GetTensorTypeAndShapeInfo().GetShape();
        if (sh.size() != 3)
          throw std::runtime_error("decoder output rank != 3");
        F_out = static_cast<int>(sh[1]);
        T_out = static_cast<int>(sh[2]);
        size_t n = static_cast<size_t>(sh[0]) * sh[1] * sh[2];
        dst.assign(t.GetTensorMutableData<float>(),
                   t.GetTensorMutableData<float>() + n);
        return;
      }
    }
    throw std::runtime_error("decoder missing output: " + name);
  };
  int fs = 0, ts = 0, fp = 0, tp = 0;
  take("spec",  out.spec,  fs, ts);
  take("phase", out.phase, fp, tp);
  if (fs != fp || ts != tp)
    throw std::runtime_error("spec/phase shape mismatch");
  out.F = fs;
  out.T_frames = ts;
  return out;
}

} // namespace kokoro

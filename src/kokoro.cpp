#include "kokoro.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

#include "g2p.hpp"
#include "istft.hpp"
#include "onnx-decoder.hpp"

#ifdef USE_RKNN
#include "rknn-decoder.hpp"
#endif

namespace kokoro {

using nlohmann::json;
using clk = std::chrono::steady_clock;
using secs = std::chrono::duration<double>;

// ---- VoicePack ------------------------------------------------------------

VoicePack VoicePack::load(const std::string& path) {
  VoicePack v;
  v.arr_ = load_npy(path);
  return v;
}

std::vector<float> VoicePack::slice(int n_phonemes) const {
  // Expected shapes: (510, 1, 256) or (256,) flat. Index by n_phonemes-1.
  if (arr_.shape.size() == 3) {
    const int N = static_cast<int>(arr_.shape[0]);
    const int M = static_cast<int>(arr_.shape[1]);
    const int D = static_cast<int>(arr_.shape[2]);
    if (M != 1) throw std::runtime_error("voice pack middle dim != 1");
    int idx = std::clamp(n_phonemes - 1, 0, N - 1);
    std::vector<float> out(D);
    std::memcpy(out.data(), arr_.data.data() + idx * D, D * sizeof(float));
    return out;
  }
  if (arr_.shape.size() == 1) {
    return arr_.data;
  }
  throw std::runtime_error("voice pack: unsupported shape rank " +
                           std::to_string(arr_.shape.size()));
}

// ---- Engine::Impl ---------------------------------------------------------

struct Engine::Impl {
  EngineConfig cfg;
  std::string voicesDir;
  std::string accelerator;
  Vocab vocab;

  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::SessionOptions> encOpt;
  std::unique_ptr<Ort::Session> enc;
  std::vector<std::string> encOutNames;

  std::unique_ptr<Ort::SessionOptions> harOpt;
  std::unique_ptr<Ort::Session> har;

  std::unique_ptr<Decoder> decoder;
  std::unique_ptr<IStft> istft;

  std::mutex voiceMu;
  std::map<std::string, std::shared_ptr<VoicePack>> voiceCache;
};

// ---- Engine ---------------------------------------------------------------

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

const Vocab& Engine::vocab() const { return impl_->vocab; }
const EngineConfig& Engine::config() const { return impl_->cfg; }

static void loadVocab(const std::string& path, Vocab& out) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open vocab config: " + path);
  json j; f >> j;
  auto& vobj = j.at("vocab");
  for (auto it = vobj.begin(); it != vobj.end(); ++it) {
    const std::string& k = it.key();
    int id = it.value().get<int>();
    // Decode the first UTF-8 codepoint of k.
    char32_t cp = 0;
    unsigned char c0 = static_cast<unsigned char>(k[0]);
    int extra = 0;
    if (c0 < 0x80) { cp = c0; }
    else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F; extra = 1; }
    else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F; extra = 2; }
    else if ((c0 & 0xF8) == 0xF0) { cp = c0 & 0x07; extra = 3; }
    else throw std::runtime_error("invalid utf-8 in vocab key");
    for (int i = 0; i < extra; ++i) {
      cp = (cp << 6) | (static_cast<unsigned char>(k[1 + i]) & 0x3F);
    }
    out.emplace(cp, id);
  }
}

void Engine::load(const std::string& vocabConfigJson,
                  const std::string& encoderPath,
                  const std::string& harGenPath,
                  const std::string& decoderPath,
                  const std::string& voicesDir,
                  const std::string& accelerator,
                  const EngineConfig& cfg) {
  auto& I = *impl_;
  I.cfg = cfg;
  I.voicesDir = voicesDir;
  I.accelerator = accelerator;

  loadVocab(vocabConfigJson, I.vocab);
  spdlog::info("Vocab loaded ({} tokens)", I.vocab.size());

  I.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "kokoro");
  I.env->DisableTelemetryEvents();

  auto makeOpt = [&]() {
    auto opt = std::make_unique<Ort::SessionOptions>();
    opt->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    opt->DisableProfiling();
#ifdef USE_ORT_CUDA
    if (accelerator == "cuda") {
      OrtCUDAProviderOptions cuda{};
      cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      opt->AppendExecutionProvider_CUDA(cuda);
    } else if (accelerator == "tensorrt") {
      OrtTensorRTProviderOptions trt{};
      opt->AppendExecutionProvider_TensorRT(trt);
    }
#endif
    return opt;
  };

  I.encOpt = makeOpt();
  I.enc = std::make_unique<Ort::Session>(*I.env, encoderPath.c_str(), *I.encOpt);
  Ort::AllocatorWithDefaultOptions alloc;
  size_t nout = I.enc->GetOutputCount();
  I.encOutNames.reserve(nout);
  for (size_t i = 0; i < nout; ++i)
    I.encOutNames.emplace_back(I.enc->GetOutputNameAllocated(i, alloc).get());

  I.harOpt = makeOpt();
  I.har = std::make_unique<Ort::Session>(*I.env, harGenPath.c_str(), *I.harOpt);

  auto ext = std::filesystem::path(decoderPath).extension().string();
  if (ext == ".rknn") {
#ifdef USE_RKNN
    I.decoder = std::make_unique<RknnDecoder>(I.cfg.T_fix);
#else
    throw std::runtime_error("decoder is .rknn but USE_RKNN was not enabled");
#endif
  } else {
    auto d = std::make_unique<OnnxDecoder>(I.cfg.T_fix);
    d->load(decoderPath, accelerator);
    I.decoder = std::move(d);
  }
  if (ext == ".rknn") I.decoder->load(decoderPath);

  I.istft = std::make_unique<IStft>(20, 5);

  spdlog::info("Engine loaded (encoder={}, har_gen={}, decoder={})",
               encoderPath, harGenPath, decoderPath);
}

std::shared_ptr<VoicePack> Engine::getVoice(const std::string& name) {
  auto& I = *impl_;
  std::lock_guard<std::mutex> g(I.voiceMu);
  auto it = I.voiceCache.find(name);
  if (it != I.voiceCache.end()) return it->second;

  std::filesystem::path p(name);
  if (!std::filesystem::exists(p)) {
    p = std::filesystem::path(I.voicesDir) / (p.stem().string() + ".npy");
  }
  if (!std::filesystem::exists(p))
    throw std::runtime_error("voice not found: " + name);

  auto v = std::make_shared<VoicePack>(VoicePack::load(p.string()));
  I.voiceCache.emplace(name, v);
  spdlog::info("Loaded voice '{}' from {}", name, p.string());
  return v;
}

std::vector<std::string> Engine::listVoices() const {
  auto& I = *impl_;
  std::vector<std::string> out;
  if (I.voicesDir.empty()) return out;
  for (auto& e : std::filesystem::directory_iterator(I.voicesDir)) {
    if (e.path().extension() == ".npy")
      out.push_back(e.path().stem().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// ---- Encoder run ---------------------------------------------------------

struct EncoderOut {
  // All row-major float32 (except input_ids int64 in).
  std::vector<float> asr;     // (1, 512, T)
  std::vector<float> F0;      // (1, 2T)
  std::vector<float> N;       // (1, 2T)
  std::vector<float> s;       // (1, 128)
  int T = 0;
  int trail = 0;              // frames the trailing EOS (0) token owns
};

static EncoderOut runEncoder(Engine::Impl& I,
                             const std::vector<int64_t>& padded_ids,
                             const float* ref_s, int ref_dim,
                             float speed) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;

  std::vector<int64_t> ids_shape{1, static_cast<int64_t>(padded_ids.size())};
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      mem, const_cast<int64_t*>(padded_ids.data()), padded_ids.size(),
      ids_shape.data(), ids_shape.size()));

  std::vector<int64_t> ref_shape{1, ref_dim};
  inputs.push_back(Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(ref_s),
      static_cast<size_t>(ref_dim), ref_shape.data(), ref_shape.size()));

  std::vector<float> sp{speed};
  std::vector<int64_t> sp_shape{1};
  inputs.push_back(Ort::Value::CreateTensor<float>(
      mem, sp.data(), sp.size(), sp_shape.data(), sp_shape.size()));

  static const std::array<const char*, 3> inNames{"input_ids", "ref_s", "speed"};
  std::vector<const char*> outNames;
  outNames.reserve(I.encOutNames.size());
  for (auto& s : I.encOutNames) outNames.push_back(s.c_str());

  auto outs = I.enc->Run(Ort::RunOptions{nullptr}, inNames.data(), inputs.data(),
                         inputs.size(), outNames.data(), outNames.size());

  EncoderOut eo;
  auto take = [&](const std::string& name, std::vector<float>& dst) {
    for (size_t i = 0; i < outNames.size(); ++i) {
      if (I.encOutNames[i] == name) {
        auto& t = outs[i];
        auto shp = t.GetTensorTypeAndShapeInfo().GetShape();
        size_t n = 1; for (auto d : shp) n *= static_cast<size_t>(d);
        dst.assign(t.GetTensorMutableData<float>(),
                   t.GetTensorMutableData<float>() + n);
        return;
      }
    }
    throw std::runtime_error("encoder missing output: " + name);
  };
  take("asr", eo.asr);
  take("F0",  eo.F0);
  take("N",   eo.N);
  take("s",   eo.s);

  // pred_dur is int64 (1, num_tokens); its last element is the EOS token's
  // own frame count, which we drop from emission so EOS isn't voiced.
  for (size_t i = 0; i < outNames.size(); ++i) {
    if (I.encOutNames[i] == "pred_dur") {
      auto& t = outs[i];
      auto shp = t.GetTensorTypeAndShapeInfo().GetShape();
      size_t n = 1; for (auto d : shp) n *= static_cast<size_t>(d);
      if (n > 0) {
        const int64_t* pd = t.GetTensorMutableData<int64_t>();
        eo.trail = static_cast<int>(pd[n - 1]);
      }
      break;
    }
  }

  // T = asr.size() / 512 (since asr shape is (1, 512, T))
  if (eo.asr.size() % 512 != 0)
    throw std::runtime_error("asr size not divisible by 512");
  eo.T = static_cast<int>(eo.asr.size() / 512);
  return eo;
}

// ---- har_gen run ---------------------------------------------------------

static std::vector<float> runHar(Engine::Impl& I, const float* F0_chunk,
                                 int F0_len, int& har_T) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<int64_t> shp{1, F0_len};
  std::vector<Ort::Value> ins;
  ins.push_back(Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(F0_chunk), static_cast<size_t>(F0_len),
      shp.data(), shp.size()));
  static const std::array<const char*, 1> inN{"F0"};
  Ort::AllocatorWithDefaultOptions alloc;
  auto name0 = I.har->GetOutputNameAllocated(0, alloc);
  std::array<const char*, 1> outN{name0.get()};
  auto outs = I.har->Run(Ort::RunOptions{nullptr}, inN.data(), ins.data(),
                         ins.size(), outN.data(), outN.size());
  auto& t = outs[0];
  auto sh = t.GetTensorTypeAndShapeInfo().GetShape();
  if (sh.size() != 3 || sh[0] != 1)
    throw std::runtime_error("har output expected (1, C, T)");
  size_t total = 1; for (auto d : sh) total *= static_cast<size_t>(d);
  har_T = static_cast<int>(sh[2]);
  std::vector<float> out(total);
  std::memcpy(out.data(), t.GetTensorMutableData<float>(),
              total * sizeof(float));
  return out;
}

// ---- Per-window worker ---------------------------------------------------

namespace {

struct WindowJob {
  int idx = 0;
  int emit_start = 0;
  int emit_end = 0;
  int a_lo = 0;
  int a_hi = 0;
  int lpad = 0;
  int rpad = 0;
};

struct WindowResult {
  std::atomic<bool> ready{false};
  std::vector<float> audio_emit; // emit_samples (no padding context)
  double har_s = 0, dec_s = 0, ist_s = 0;
  std::mutex mu;
  std::condition_variable cv;
};

} // namespace

static void clipToInt16(const float* in, std::size_t n, std::vector<int16_t>& out) {
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    float v = in[i];
    if (v > 1.f) v = 1.f;
    if (v < -1.f) v = -1.f;
    out[i] = static_cast<int16_t>(v * 32767.f);
  }
}

namespace {

// Split a phoneme-id sequence into chunks that each fit BERT's context window
// (budget = context_length - BOS/EOS). We break on the punctuation tokens G2P
// already emitted into the stream, preferring the strongest pause available
// within budget: sentence enders (. ! ? …) > clause (; :) > comma / em dash >
// a plain word boundary (space). Only a single >budget run with no break at
// all is hard-cut. Leading/trailing space tokens are trimmed per chunk.
std::vector<std::vector<int64_t>> splitIdsForBert(const std::vector<int64_t>& ids,
                                                  const Vocab& vocab, int budget) {
  auto idOf = [&](char32_t c) -> int64_t {
    auto it = vocab.find(c);
    return it == vocab.end() ? -1 : static_cast<int64_t>(it->second);
  };
  // id -> break tier (higher = stronger preferred break point).
  std::map<int64_t, int> tier;
  auto setTier = [&](char32_t c, int t) { int64_t id = idOf(c); if (id >= 0) tier[id] = t; };
  setTier(U'.', 3); setTier(U'!', 3); setTier(U'?', 3); setTier(U'…', 3);
  setTier(U';', 2); setTier(U':', 2);
  setTier(U',', 1); setTier(U'—', 1);
  const int64_t spaceId = idOf(U' ');

  std::vector<std::vector<int64_t>> out;
  if (budget < 1) budget = 1;
  const std::size_t n = ids.size();
  std::size_t i = 0;
  while (i < n) {
    std::size_t end = std::min(i + static_cast<std::size_t>(budget), n);
    std::size_t cutEnd, nextStart;
    if (end == n) {
      cutEnd = nextStart = n;                 // tail fits — take the rest
    } else {
      int bestTier = -1;
      std::size_t bestJ = n;                  // index of the chosen break token
      for (std::size_t j = i; j < end; ++j) {
        auto it = tier.find(ids[j]);
        if (it == tier.end()) continue;
        if (it->second >= bestTier) { bestTier = it->second; bestJ = j; } // latest, highest
      }
      if (bestJ == n) { cutEnd = nextStart = end; }      // no break: hard cut
      else            { cutEnd = nextStart = bestJ + 1; } // keep the punctuation
    }
    // Trim leading/trailing space tokens from the chunk.
    std::size_t lo = i, hi = cutEnd;
    while (lo < hi && ids[lo]   == spaceId) ++lo;
    while (hi > lo && ids[hi-1] == spaceId) --hi;
    if (hi > lo) out.emplace_back(ids.begin() + lo, ids.begin() + hi);
    i = nextStart;
    while (i < n && ids[i] == spaceId) ++i;   // skip the gap before the next chunk
  }
  if (out.empty() && !ids.empty()) out.push_back(ids);  // all-space safety net
  return out;
}

}  // namespace

SynthesisResult Engine::synthesizeIds(const std::vector<int64_t>& ids,
                                      const VoicePack& voice,
                                      float speed,
                                      ChunkCallback emit) {
  auto& I = *impl_;
  if (ids.empty()) return SynthesisResult{};

  if (static_cast<int>(ids.size()) > I.cfg.maxTokens) {
    auto chunks = splitIdsForBert(ids, I.vocab, I.cfg.maxTokens);
    spdlog::info("input {} tokens > {} (BERT limit); split into {} chunks",
                 ids.size(), I.cfg.maxTokens, chunks.size());
    SynthesisResult total{};
    for (auto& c : chunks) {
      SynthesisResult r = synthChunk(I, c, voice, speed, emit);
      total.encoderSeconds += r.encoderSeconds;
      total.harSeconds     += r.harSeconds;
      total.decoderSeconds += r.decoderSeconds;
      total.istftSeconds   += r.istftSeconds;
      total.audioSeconds   += r.audioSeconds;
    }
    double infer_total = total.encoderSeconds + total.harSeconds +
                         total.decoderSeconds + total.istftSeconds;
    if (infer_total > 0) total.realTimeFactor = total.audioSeconds / infer_total;
    return total;
  }

  return synthChunk(I, ids, voice, speed, emit);
}

SynthesisResult Engine::synthChunk(Impl& I, const std::vector<int64_t>& ids,
                                   const VoicePack& voice, float speed,
                                   const ChunkCallback& emit) {
  SynthesisResult res{};
  if (ids.empty()) return res;

  // Wrap with BOS/EOS zeros, matching infer.py.
  std::vector<int64_t> padded;
  padded.reserve(ids.size() + 2);
  padded.push_back(0);
  for (auto v : ids) padded.push_back(v);
  padded.push_back(0);

  auto ref_s = voice.slice(static_cast<int>(ids.size()));

  // ---- Encoder ----
  auto t0 = clk::now();
  auto enc = runEncoder(I, padded, ref_s.data(),
                        static_cast<int>(ref_s.size()), speed);
  res.encoderSeconds = secs(clk::now() - t0).count();

  const int T = enc.T;                          // total frames incl. BOS/EOS
  const int T_emit = std::max(1, T - enc.trail); // don't voice EOS; keep as context
  const int T_fix = I.cfg.T_fix;
  const int CTX = I.cfg.CTX;
  const int STEP = T_fix - 2 * CTX;
  const int SAMP_PER_FR = I.cfg.samplesPerFrame;
  const int emit_offset = CTX * SAMP_PER_FR;

  if (STEP <= 0) throw std::runtime_error("T_fix - 2*CTX must be > 0");

  // Build per-window jobs.
  std::vector<WindowJob> jobs;
  for (int es = 0; es < T_emit; es += STEP) {
    WindowJob j;
    j.idx = static_cast<int>(jobs.size());
    j.emit_start = es;
    j.emit_end = std::min(es + STEP, T_emit);
    int win_start = es - CTX;
    int win_end = j.emit_end + CTX;
    j.a_lo = std::max(0, win_start);
    j.a_hi = std::min(T, win_end);
    j.lpad = j.a_lo - win_start;
    j.rpad = win_end - j.a_hi;
    j.rpad += T_fix - (j.a_hi - j.a_lo + j.lpad + j.rpad);
    jobs.push_back(j);
  }

  const int n_windows = static_cast<int>(jobs.size());
  std::vector<std::unique_ptr<WindowResult>> results(n_windows);
  for (auto& r : results) r = std::make_unique<WindowResult>();

  // Worker pool. Each worker grabs the next window by atomic counter, runs
  // har→decoder→iSTFT, stores into results[i], notifies the emit thread.
  std::atomic<int> nextIdx{0};
  const int nWorkers = std::max(1, I.cfg.decoderWorkers);

  auto workerLoop = [&]() {
    for (;;) {
      int i = nextIdx.fetch_add(1, std::memory_order_relaxed);
      if (i >= n_windows) return;
      const auto& j = jobs[i];

      // Slice ASR (512, T_fix) with padding.
      std::vector<float> asr_win(512 * T_fix, 0.f);
      for (int c = 0; c < 512; ++c) {
        const float* src = &enc.asr[static_cast<size_t>(c) * T + j.a_lo];
        float* dst = &asr_win[static_cast<size_t>(c) * T_fix + j.lpad];
        std::memcpy(dst, src, (j.a_hi - j.a_lo) * sizeof(float));
      }
      // F0, N are (1, 2T) — slice 2*lo .. 2*hi with 2*lpad/2*rpad padding.
      std::vector<float> F0_win(2 * T_fix, 0.f);
      std::vector<float> N_win (2 * T_fix, 0.f);
      std::memcpy(&F0_win[2 * j.lpad], &enc.F0[2 * j.a_lo],
                  2 * (j.a_hi - j.a_lo) * sizeof(float));
      std::memcpy(&N_win [2 * j.lpad], &enc.N [2 * j.a_lo],
                  2 * (j.a_hi - j.a_lo) * sizeof(float));

      // har_gen
      auto th0 = clk::now();
      int har_T = 0;
      auto har = runHar(I, F0_win.data(), 2 * T_fix, har_T);
      double har_s = secs(clk::now() - th0).count();

      // decoder
      auto td0 = clk::now();
      DecoderOutput dec = I.decoder->infer(asr_win.data(), F0_win.data(),
                                           N_win.data(), enc.s.data(),
                                           har.data(), har_T);
      double dec_s = secs(clk::now() - td0).count();

      // iSTFT
      auto ti0 = clk::now();
      int out_per = 0;
      auto audio = I.istft->apply(dec.spec.data(), dec.phase.data(), 1,
                                  dec.F, dec.T_frames, out_per);
      double ist_s = secs(clk::now() - ti0).count();

      // Extract the emit region, extended by half the overlap into the real
      // context on each side so neighbouring windows share a 2*HALF region
      // straddling the seam (symmetric overlap-add; see the emit loop). The
      // first window has no left neighbour, the last no right.
      const int HALF = I.cfg.DEPOP_OVL / 2;
      int emit_samples = (j.emit_end - j.emit_start) * SAMP_PER_FR;
      int left_ext  = (j.emit_start > 0)     ? HALF : 0;
      int right_ext = (j.emit_end   < T_emit) ? HALF : 0;
      int start = emit_offset - left_ext;
      int end   = emit_offset + emit_samples + right_ext;
      if (start < 0) start = 0;
      if (end > out_per) end = out_per;          // defensive; context fits HALF

      auto& r = *results[i];
      r.audio_emit.assign(audio.begin() + start, audio.begin() + end);
      r.har_s = har_s;
      r.dec_s = dec_s;
      r.ist_s = ist_s;
      {
        std::lock_guard<std::mutex> g(r.mu);
        r.ready.store(true, std::memory_order_release);
      }
      r.cv.notify_all();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(nWorkers);
  for (int w = 0; w < nWorkers; ++w) workers.emplace_back(workerLoop);

  // ---- Emit loop with symmetric overlap-add ----
  // Each window's audio_emit is extended OVL/2 into the real context on both
  // sides (worker above), so consecutive windows share an OVL-sample region
  // straddling the seam. We stream every window except its trailing OVL
  // samples, holding those in pending_tail to crossfade against the next
  // window's leading OVL — window i tapers out, window i+1 tapers in.
  const int OVL = I.cfg.DEPOP_OVL;
  std::vector<float> pending_tail; // last OVL samples held for the next seam
  std::vector<int16_t> i16;
  std::vector<float> out;
  std::size_t total_samples_emitted = 0;

  auto flush = [&](const float* p, std::size_t n) {
    clipToInt16(p, n, i16);
    emit(i16.data(), i16.size());
    total_samples_emitted += i16.size();
  };

  for (int i = 0; i < n_windows; ++i) {
    auto& r = *results[i];
    {
      std::unique_lock<std::mutex> lk(r.mu);
      r.cv.wait(lk, [&]{ return r.ready.load(std::memory_order_acquire); });
    }
    res.harSeconds     += r.har_s;
    res.decoderSeconds += r.dec_s;
    res.istftSeconds   += r.ist_s;

    auto& chunk = r.audio_emit;
    if (chunk.empty()) continue;
    const int csz = static_cast<int>(chunk.size());
    const bool is_last = (i == n_windows - 1);

    if (pending_tail.empty()) {
      // First chunk: no left neighbour to crossfade.
      if (is_last || csz <= OVL) {
        flush(chunk.data(), csz);          // single/short window — emit whole
      } else {
        flush(chunk.data(), csz - OVL);    // hold trailing OVL for the seam
        pending_tail.assign(chunk.end() - OVL, chunk.end());
      }
      continue;
    }

    // Crossfade pending_tail (OVL) against chunk's leading OVL (same region).
    const int cf = std::min(OVL, csz);
    out.assign(chunk.begin(), chunk.begin() + cf);
    for (int k = 0; k < cf; ++k) {
      float w = static_cast<float>(k) / static_cast<float>(OVL);
      out[k] = pending_tail[k] * (1.f - w) + chunk[k] * w;
    }

    if (is_last) {
      // Emit the blend plus the rest of the chunk; nothing left to hold.
      out.insert(out.end(), chunk.begin() + cf, chunk.end());
      flush(out.data(), out.size());
      pending_tail.clear();
    } else {
      // Emit blend + body, hold the trailing OVL for the next seam.
      int body = csz - OVL;                // >= OVL for full-STEP windows
      out.insert(out.end(), chunk.begin() + cf, chunk.begin() + body);
      flush(out.data(), out.size());
      pending_tail.assign(chunk.begin() + body, chunk.end());
    }
  }

  // Flush any tail left over (e.g. trailing windows that came back empty).
  if (!pending_tail.empty())
    flush(pending_tail.data(), pending_tail.size());

  for (auto& t : workers) t.join();

  res.audioSeconds = static_cast<double>(total_samples_emitted) /
                     static_cast<double>(I.cfg.sampleRate);
  double infer_total = res.encoderSeconds + res.harSeconds +
                       res.decoderSeconds + res.istftSeconds;
  if (infer_total > 0)
    res.realTimeFactor = res.audioSeconds / infer_total;

  // Frame-level performance: how long we spend synthesizing one ASR frame vs
  // how much audio that frame yields. ratio == RTF (>1 = faster than realtime).
  if (T > 0) {
    double synth_ms_per_frame = infer_total * 1000.0 / T;
    double audio_ms_per_frame = res.audioSeconds * 1000.0 / T;
    spdlog::debug("perf: {} frames | {:.2f} ms/frame to synth vs {:.2f} ms/frame "
                 "audio out | RTF {:.2f}x  (enc {:.0f} har {:.0f} dec {:.0f} "
                 "istft {:.0f} ms over {} window{})",
                 T, synth_ms_per_frame, audio_ms_per_frame, res.realTimeFactor,
                 res.encoderSeconds * 1000, res.harSeconds * 1000,
                 res.decoderSeconds * 1000, res.istftSeconds * 1000,
                 n_windows, n_windows == 1 ? "" : "s");
  }
  return res;
}

SynthesisResult Engine::synthesizeText(const std::string& text,
                                       const std::string& voiceName,
                                       float speed, bool british,
                                       ChunkCallback emit) {
  std::string ph = G2P::phonemize(text, voiceName, british);
  spdlog::debug("phonemes ({}): {}", voiceName, ph);
  return synthesizePhonemes(ph, voiceName, speed, std::move(emit));
}

SynthesisResult Engine::synthesizePhonemes(const std::string& phonemes,
                                           const std::string& voiceName,
                                           float speed,
                                           ChunkCallback emit) {
  auto& I = *impl_;
  std::size_t dropped = 0;
  auto ids = Phonemizer::to_ids(phonemes, I.vocab, &dropped);
  if (dropped > 0)
    spdlog::warn("{} phoneme char(s) dropped (not in vocab)", dropped);
  if (ids.empty())
    throw std::runtime_error("no valid phonemes after vocab filter");
  auto voice = getVoice(voiceName);
  return synthesizeIds(ids, *voice, speed, std::move(emit));
}

} // namespace kokoro

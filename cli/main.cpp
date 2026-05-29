// kokoro-cli — minimal CLI for the C++ pipeline. Reads --text/--phonemes,
// writes a WAV file. Matches infer.py's UX so the two can be sanity-checked
// against each other.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "g2p.hpp"
#include "kokoro.hpp"
#include "phonemizer.hpp"

namespace {

void writeWav(const std::string& path, const std::vector<int16_t>& pcm, int sr) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  auto put32 = [&](uint32_t v){ f.write(reinterpret_cast<const char*>(&v), 4); };
  auto put16 = [&](uint16_t v){ f.write(reinterpret_cast<const char*>(&v), 2); };
  uint32_t pcm_bytes = static_cast<uint32_t>(pcm.size() * 2);
  f.write("RIFF", 4); put32(36 + pcm_bytes);
  f.write("WAVE", 4); f.write("fmt ", 4); put32(16); put16(1); put16(1);
  put32(sr); put32(sr * 2); put16(2); put16(16);
  f.write("data", 4); put32(pcm_bytes);
  f.write(reinterpret_cast<const char*>(pcm.data()), pcm_bytes);
}

void usage() {
  std::cerr <<
    "usage: kokoro-cli [--text STR | --phonemes STR] [opts]\n"
    "  --voice NAME         (default af_heart)\n"
    "  --speed FLOAT        (default 1.0)\n"
    "  --british            use en-gb voice and remap\n"
    "  --out FILE           output wav (default out.wav)\n"
    "  --encoder FILE       onnx/kokoro_encoder.onnx\n"
    "  --har-gen FILE       onnx/har_generator.onnx\n"
    "  --decoder FILE       onnx/kokoro_decoder.onnx or .rknn\n"
    "  --vocab FILE         Kokoro-82M/config.json\n"
    "  --voices-dir DIR     voices_npy\n"
    "  --espeak-data DIR    espeak-ng-data (else next to executable)\n"
    "  --lexicon-dir DIR    misaki us/gb JSONs (else next to executable)\n"
    "  --accelerator STR    cuda | tensorrt | (empty)\n"
    "  --debug\n";
}

} // namespace

int main(int argc, char** argv) {
  spdlog::set_default_logger(spdlog::stderr_color_st("kokoro"));

  std::string text, phonemes;
  std::string voice = "af_heart";
  std::string out   = "out.wav";
  std::string encoderPath = "onnx/kokoro_encoder.onnx";
  std::string harGenPath  = "onnx/har_generator.onnx";
  std::string decoderPath = "onnx/kokoro_decoder.rknn";
  std::string vocabPath   = "Kokoro-82M/config.json";
  std::string voicesDir   = "voices_npy";
  std::string accelerator = "";
  std::string espeakData  = "";
  std::string lexiconDir  = "";
  float speed = 1.0f;
  bool british = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](){
      if (i + 1 >= argc) { usage(); std::exit(1); }
      return std::string(argv[++i]);
    };
    if      (a == "--text")        text = need();
    else if (a == "--phonemes")    phonemes = need();
    else if (a == "--voice")       voice = need();
    else if (a == "--speed")       speed = std::stof(need());
    else if (a == "--british")     british = true;
    else if (a == "--out")         out = need();
    else if (a == "--encoder")     encoderPath = need();
    else if (a == "--har-gen")     harGenPath  = need();
    else if (a == "--decoder")     decoderPath = need();
    else if (a == "--vocab")       vocabPath = need();
    else if (a == "--voices-dir")  voicesDir = need();
    else if (a == "--accelerator") accelerator = need();
    else if (a == "--espeak-data") espeakData = need();
    else if (a == "--lexicon-dir") lexiconDir = need();
    else if (a == "--debug")       spdlog::set_level(spdlog::level::debug);
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { std::cerr << "unknown arg: " << a << "\n"; usage(); return 1; }
  }

  if (text.empty() && phonemes.empty()) {
    phonemes = "h\xC9\x99l\xCB\x88O w\xCB\x88\xC9\x9C\xC9\xB9ld"; // həlˈO wˈɜɹld
    spdlog::info("(no --text/--phonemes; using demo string)");
  }

  if (espeakData.empty()) {
    auto exe = std::filesystem::canonical("/proc/self/exe");
    espeakData = std::filesystem::absolute(exe.parent_path() / "espeak-ng-data").string();
  }
  kokoro::Phonemizer::init(espeakData);

  if (lexiconDir.empty())
    lexiconDir = (std::filesystem::current_path() / "misaki-data").string();
  kokoro::G2P::init(lexiconDir, espeakData);

  kokoro::EngineConfig cfg;
  cfg.decoderWorkers = (std::filesystem::path(decoderPath).extension() == ".rknn") ? 3 : 1;

  kokoro::Engine engine;
  engine.load(vocabPath, encoderPath, harGenPath, decoderPath, voicesDir,
              accelerator, cfg);

  std::vector<int16_t> pcm;
  auto emit = [&](const int16_t* d, std::size_t n) {
    pcm.insert(pcm.end(), d, d + n);
  };

  auto t0 = std::chrono::steady_clock::now();
  kokoro::SynthesisResult r;
  if (!text.empty()) r = engine.synthesizeText(text, voice, speed, british, emit);
  else               r = engine.synthesizePhonemes(phonemes, voice, speed, emit);
  auto t1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count();

  writeWav(out, pcm, engine.config().sampleRate);
  spdlog::info("wrote {} ({} samples, {:.2f}s audio)", out, pcm.size(),
               r.audioSeconds);
  spdlog::info("encoder={:.1f}ms har={:.1f}ms dec={:.1f}ms istft={:.1f}ms  wall={:.2f}s  RTF={:.2f}x",
               r.encoderSeconds * 1000, r.harSeconds * 1000,
               r.decoderSeconds * 1000, r.istftSeconds * 1000,
               wall, r.realTimeFactor);

  kokoro::G2P::terminate();
  kokoro::Phonemizer::terminate();
  return 0;
}

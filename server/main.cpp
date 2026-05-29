#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <drogon/drogon.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "g2p.hpp"
#include "kokoro.hpp"
#include "phonemizer.hpp"

using namespace drogon;

namespace kokoro_server {

struct RunConfig {
  std::filesystem::path encoderPath;   // required
  std::filesystem::path harGenPath;    // required
  std::filesystem::path decoderPath;   // required (.onnx or .rknn)
  std::filesystem::path vocabPath;     // required (Kokoro config.json)
  std::filesystem::path voicesDir;     // required
  std::optional<std::filesystem::path> espeakDataPath;
  std::optional<std::filesystem::path> lexiconDir;
  std::optional<std::filesystem::path> webRoot;
  std::string accelerator = "";
  std::string ip = "127.0.0.1";
  uint16_t port = 8848;
  std::string authToken = "";
  bool disableWebUI = false;
  std::string defaultVoice = "af_heart";
};

} // namespace kokoro_server

// Globals accessed by api.cpp.
kokoro::Engine g_engine;
std::string g_authToken;
std::string g_defaultVoice;

namespace {

void printUsage(const char* prog) {
  std::cerr <<
    "usage: " << prog << " [options]\n\n"
    "required:\n"
    "  --encoder FILE        encoder ONNX\n"
    "  --har-gen FILE        har generator ONNX\n"
    "  --decoder FILE        decoder .onnx or .rknn\n"
    "  --vocab FILE          Kokoro config.json (vocab is read from it)\n"
    "  --voices-dir DIR      directory of voice .npy files\n"
    "\noptional:\n"
    "  --espeak-data DIR     espeak-ng-data directory (else next to executable)\n"
    "  --lexicon-dir DIR     misaki us/gb JSONs (else ./misaki-data)\n"
    "  --web-root DIR        web UI directory (else auto-detect; see below)\n"
    "  --accelerator STR     ONNX accelerator: cuda, tensorrt (default none)\n"
    "  --default-voice NAME  default voice name (default af_heart)\n"
    "  --ip ADDR             bind address (default 127.0.0.1)\n"
    "  --port N              bind port (default 8848)\n"
    "  --auth [TOKEN]        require bearer token (random if not given)\n"
    "  --disable-web-ui      disable demo web UI\n"
    "  --debug               enable debug logging\n"
    "  -q, --quiet           silence logs\n"
    "  -h, --help            show this help\n"
    "\nWeb-root auto-detect (relative to CWD, first hit wins):\n"
    "  ./server/web-content, ./web-content\n";
}

void parseArgs(int argc, char** argv, kokoro_server::RunConfig& rc) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) { printUsage(argv[0]); std::exit(1); }
      return std::string(argv[++i]);
    };
    if      (a == "--encoder")        rc.encoderPath = need(a);
    else if (a == "--har-gen")        rc.harGenPath  = need(a);
    else if (a == "--decoder")        rc.decoderPath = need(a);
    else if (a == "--vocab")          rc.vocabPath   = need(a);
    else if (a == "--voices-dir")     rc.voicesDir   = need(a);
    else if (a == "--espeak-data")    rc.espeakDataPath = std::filesystem::path(need(a));
    else if (a == "--lexicon-dir")    rc.lexiconDir = std::filesystem::path(need(a));
    else if (a == "--web-root")       rc.webRoot = std::filesystem::path(need(a));
    else if (a == "--accelerator")    rc.accelerator = need(a);
    else if (a == "--default-voice")  rc.defaultVoice = need(a);
    else if (a == "--ip")             rc.ip   = need(a);
    else if (a == "--port")           rc.port = static_cast<uint16_t>(std::stoul(need(a)));
    else if (a == "--auth") {
      if (i + 1 < argc && argv[i + 1][0] != '-') rc.authToken = argv[++i];
      else rc.authToken = drogon::utils::secureRandomString(32);
    }
    else if (a == "--disable-web-ui") rc.disableWebUI = true;
    else if (a == "--debug")          spdlog::set_level(spdlog::level::debug);
    else if (a == "-q" || a == "--quiet") spdlog::set_level(spdlog::level::off);
    else if (a == "-h" || a == "--help") { printUsage(argv[0]); std::exit(0); }
    else { std::cerr << "unknown arg: " << a << "\n"; printUsage(argv[0]); std::exit(1); }
  }
}

} // namespace

int main(int argc, char** argv) {
  spdlog::set_default_logger(spdlog::stderr_color_st("kokoro"));

  kokoro_server::RunConfig rc;
  parseArgs(argc, argv, rc);

  auto requireArg = [&](const std::filesystem::path& p, const char* flag) {
    if (p.empty()) {
      std::cerr << "error: " << flag << " is required\n\n";
      printUsage(argv[0]);
      std::exit(1);
    }
  };
  requireArg(rc.encoderPath, "--encoder");
  requireArg(rc.harGenPath,  "--har-gen");
  requireArg(rc.decoderPath, "--decoder");
  requireArg(rc.vocabPath,   "--vocab");
  requireArg(rc.voicesDir,   "--voices-dir");

  // espeak data path: explicit, or next to the executable.
  std::string espeakData;
  if (rc.espeakDataPath) {
    espeakData = std::filesystem::absolute(*rc.espeakDataPath).string();
  } else {
    auto exe = std::filesystem::canonical("/proc/self/exe");
    espeakData = std::filesystem::absolute(exe.parent_path() / "espeak-ng-data").string();
  }
  kokoro::Phonemizer::init(espeakData);

  // misaki lexicon dir: explicit, or "misaki-data" under the current directory.
  std::string lexiconDir =
      rc.lexiconDir ? std::filesystem::absolute(*rc.lexiconDir).string()
                    : (std::filesystem::current_path() / "misaki-data").string();
  kokoro::G2P::init(lexiconDir, espeakData);

  kokoro::EngineConfig cfg;
  // RKNN: keep 3 workers. ONNX: 1 (no multi-context benefit).
  if (rc.decoderPath.extension() == ".rknn") cfg.decoderWorkers = 3;
  else cfg.decoderWorkers = 1;

  auto t0 = std::chrono::steady_clock::now();
  g_engine.load(rc.vocabPath.string(), rc.encoderPath.string(),
                rc.harGenPath.string(), rc.decoderPath.string(),
                rc.voicesDir.string(), rc.accelerator, cfg);
  auto t1 = std::chrono::steady_clock::now();
  spdlog::info("Engine loaded in {:.2f}s",
               std::chrono::duration<double>(t1 - t0).count());

  if (const char* env = std::getenv("KOKORO_TOKEN")) {
    g_authToken = env;
    spdlog::info("Auth token from KOKORO_TOKEN env");
  } else if (!rc.authToken.empty()) {
    g_authToken = rc.authToken;
    spdlog::info("Auth token: {}", rc.authToken);
  }
  g_defaultVoice = rc.defaultVoice;

  if (!rc.disableWebUI) {
    std::filesystem::path webDir;
    if (rc.webRoot) {
      webDir = *rc.webRoot;
      if (!std::filesystem::exists(webDir)) {
        spdlog::error("--web-root {} does not exist", webDir.string());
        return 1;
      }
    } else {
      auto cwd = std::filesystem::current_path();
      for (const auto& candidate : {
             cwd / "server" / "web-content",
             cwd / "web-content",
           }) {
        if (std::filesystem::exists(candidate)) { webDir = candidate; break; }
      }
    }
    if (!webDir.empty()) {
      app().setDocumentRoot(std::filesystem::absolute(webDir).string());
      spdlog::info("Web UI at http://{}:{}/  (root: {})", rc.ip, rc.port,
                   webDir.string());
    } else {
      spdlog::warn("Web UI requested but web-content/ not found "
                   "(pass --web-root DIR or --disable-web-ui)");
    }
  }

  app().addListener(rc.ip, rc.port).setThreadNum(3).run();

  kokoro::G2P::terminate();
  kokoro::Phonemizer::terminate();
  return 0;
}

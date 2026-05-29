// HTTP + WebSocket API. Mirrors paroli-server's surface, adapted for Kokoro:
//   POST /api/v1/synthesise   one-shot text/phonemes -> opus|pcm|wav
//   GET  /api/v1/voices       list available voice names
//   WS   /api/v1/stream       streaming text/phonemes -> binary frames
//   POST /v1/audio/speech     OpenAI-compatible TTS
//
// Multi-window decoder parallelism lives inside kokoro::Engine; the server
// just routes requests onto its thread pool and forwards emitted chunks.

#include <atomic>
#include <bit>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <drogon/HttpController.h>
#include <drogon/WebSocketController.h>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <trantor/net/EventLoopThreadPool.h>

#include "OggOpusEncoder.hpp"
#include "kokoro.hpp"
#include "resampler.hpp"

using namespace drogon;

extern kokoro::Engine g_engine;
extern std::string    g_authToken;
extern std::string    g_defaultVoice;

// Dedicated pool for synth work. Each synth call spawns N internal decoder
// worker threads (see EngineConfig.decoderWorkers), so this pool only needs
// to be large enough to handle queued requests without blocking IO loops.
static trantor::EventLoopThreadPool g_synthPool(2, "kokoro-synth");
static std::once_flag g_synthPoolOnce;
static void ensureSynthPool() {
  std::call_once(g_synthPoolOnce, []{ g_synthPool.start(); });
}

namespace {

struct ApiParams {
  // Exactly one of these is set:
  std::string text;       // run through espeak + E2M
  std::string phonemes;   // Kokoro vocab directly
  std::string voice;
  float speed = 1.0f;
  bool british = false;
  std::string audio_format = "opus"; // opus | pcm | wav | raw
};

ApiParams parseParams(std::string_view body) {
  ApiParams p;
  auto j = nlohmann::json::parse(body);
  if (j.contains("text") && !j["text"].is_null())
    p.text = j["text"].get<std::string>();
  if (j.contains("phonemes") && !j["phonemes"].is_null())
    p.phonemes = j["phonemes"].get<std::string>();
  if (p.text.empty() && p.phonemes.empty())
    throw std::runtime_error("must provide 'text' or 'phonemes'");
  if (!p.text.empty() && !p.phonemes.empty())
    throw std::runtime_error("provide only one of 'text' or 'phonemes'");

  // Voice resolution: explicit "voice" (Kokoro-native), or paroli-compat
  // "speaker" (name) / "speaker_id" (index into sorted listVoices()).
  if (j.contains("voice") && j["voice"].is_string()) {
    p.voice = j["voice"].get<std::string>();
  } else if (j.contains("speaker") && j["speaker"].is_string()) {
    p.voice = j["speaker"].get<std::string>();
  } else if (j.contains("speaker_id") && j["speaker_id"].is_number_integer()) {
    auto names = g_engine.listVoices();
    auto idx = j["speaker_id"].get<int64_t>();
    if (idx < 0 || static_cast<std::size_t>(idx) >= names.size())
      throw std::runtime_error("speaker_id out of range");
    p.voice = names[static_cast<std::size_t>(idx)];
  } else {
    p.voice = g_defaultVoice;
  }
  if (j.contains("speed") && j["speed"].is_number())
    p.speed = j["speed"].get<float>();
  if (j.contains("british") && j["british"].is_boolean())
    p.british = j["british"].get<bool>();
  if (j.contains("audio_format") && j["audio_format"].is_string())
    p.audio_format = j["audio_format"].get<std::string>();
  return p;
}

HttpResponsePtr badRequest(const std::string& msg) {
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k400BadRequest);
  r->setContentTypeCode(CT_TEXT_PLAIN);
  r->setBody(msg);
  return r;
}

bool checkAuth(const HttpRequestPtr& req) {
  if (g_authToken.empty()) return true;
  auto h = req->getHeader("Authorization");
  return h == "Bearer " + g_authToken;
}

bool checkAuthWs(const HttpRequestPtr& req) {
  if (g_authToken.empty()) return true;
  return req->getHeader("Authorization") == "Bearer " + g_authToken;
}

// Synth helper — runs entirely on whatever thread calls it, since the
// per-window parallelism is inside the engine.
kokoro::SynthesisResult doSynth(const ApiParams& p, kokoro::ChunkCallback emit) {
  if (!p.phonemes.empty())
    return g_engine.synthesizePhonemes(p.phonemes, p.voice, p.speed, std::move(emit));
  return g_engine.synthesizeText(p.text, p.voice, p.speed, p.british, std::move(emit));
}

void appendWavHeader(std::vector<uint8_t>& buf, int sr, int channels,
                     uint32_t pcm_bytes) {
  // 16-bit mono WAV header.
  auto put32 = [&](uint32_t v) {
    buf.push_back(v & 0xFF); buf.push_back((v >> 8) & 0xFF);
    buf.push_back((v >> 16) & 0xFF); buf.push_back((v >> 24) & 0xFF);
  };
  auto put16 = [&](uint16_t v) {
    buf.push_back(v & 0xFF); buf.push_back((v >> 8) & 0xFF);
  };
  buf.insert(buf.end(), {'R','I','F','F'}); put32(36 + pcm_bytes);
  buf.insert(buf.end(), {'W','A','V','E','f','m','t',' '});
  put32(16); put16(1); put16(channels); put32(sr);
  put32(sr * channels * 2); put16(channels * 2); put16(16);
  buf.insert(buf.end(), {'d','a','t','a'}); put32(pcm_bytes);
}

} // namespace

// ---- REST controllers ----------------------------------------------------

namespace api {

class v1 : public HttpController<v1> {
 public:
  METHOD_LIST_BEGIN
  METHOD_ADD(v1::synthesise, "/synthesise", {Post, Options});
  METHOD_ADD(v1::voices,     "/voices",     Get);
  METHOD_ADD(v1::speakers,   "/speakers",   Get);
  METHOD_LIST_END

  void synthesise(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& cb);
  void voices(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& cb);
  void speakers(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& cb);
};

void v1::synthesise(const HttpRequestPtr& req,
                    std::function<void(const HttpResponsePtr&)>&& cb) {
  if (req->method() == Options) {
    auto r = HttpResponse::newHttpResponse();
    r->addHeader("Access-Control-Allow-Origin", "*");
    r->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    r->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    cb(r); return;
  }
  if (req->getContentType() != CT_APPLICATION_JSON) {
    cb(badRequest("Content-Type must be application/json")); return;
  }
  if (!checkAuth(req)) { cb(badRequest("invalid Authorization")); return; }

  ApiParams p;
  try { p = parseParams(req->getBody()); }
  catch (const std::exception& e) { cb(badRequest(e.what())); return; }

  ensureSynthPool();
  g_synthPool.getNextLoop()->queueInLoop([p = std::move(p),
                                          cb = std::move(cb)]() mutable {
    std::vector<int16_t> audio;
    try {
      doSynth(p, [&](const int16_t* d, std::size_t n) {
        audio.insert(audio.end(), d, d + n);
      });
    } catch (const std::exception& e) {
      cb(badRequest(std::string("synthesis failed: ") + e.what())); return;
    }

    const int sr = g_engine.config().sampleRate;
    auto r = HttpResponse::newHttpResponse();
  if (p.audio_format == "opus" || p.audio_format.empty()) {
    auto opus = encodeOgg(std::vector<short>(audio.begin(), audio.end()), sr, 1);
    r->addHeader("Content-Type", "audio/ogg; codecs=opus");
    r->setBody(std::string(reinterpret_cast<const char*>(opus.data()),
                           opus.size()));
  } else if (p.audio_format == "wav") {
    std::vector<uint8_t> buf;
    appendWavHeader(buf, sr, 1, static_cast<uint32_t>(audio.size() * 2));
    auto* pp = reinterpret_cast<const uint8_t*>(audio.data());
    buf.insert(buf.end(), pp, pp + audio.size() * 2);
    r->addHeader("Content-Type", "audio/wav");
    r->setBody(std::string(reinterpret_cast<const char*>(buf.data()),
                           buf.size()));
  } else { // pcm | raw
    r->addHeader("Content-Type", "audio/raw");
    r->setBody(std::string(reinterpret_cast<const char*>(audio.data()),
                           audio.size() * sizeof(int16_t)));
    }
    cb(r);
  });
}

void v1::voices(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& cb) {
  auto names = g_engine.listVoices();
  nlohmann::json j = names;
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k200OK);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  r->setBody(j.dump());
  cb(r);
}

// Paroli-compatible /speakers. Kokoro voices have no native integer id, so
// we synthesize one from the alphabetical position. Clients posting
// {"speaker_id": N} hit the same voice every restart because listVoices()
// sorts.
void v1::speakers(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& cb) {
  auto names = g_engine.listVoices();
  nlohmann::json j = nlohmann::json::object();
  for (std::size_t i = 0; i < names.size(); ++i)
    j[names[i]] = static_cast<int64_t>(i);
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k200OK);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  r->setBody(j.dump());
  cb(r);
}

class v1ws : public WebSocketController<v1ws> {
 public:
  void handleNewConnection(const HttpRequestPtr& req,
                           const WebSocketConnectionPtr& ws) override {
    if (!checkAuthWs(req)) { ws->forceClose(); return; }
  }
  void handleNewMessage(const WebSocketConnectionPtr& ws, std::string&& msg,
                        const WebSocketMessageType& type) override;
  void handleConnectionClosed(const WebSocketConnectionPtr&) override {}

  WS_PATH_LIST_BEGIN
  WS_PATH_ADD("/api/v1/stream", Get);
  WS_PATH_LIST_END
};

void v1ws::handleNewMessage(const WebSocketConnectionPtr& ws, std::string&& msg,
                            const WebSocketMessageType& type) {
  if (type != WebSocketMessageType::Text) return;
  std::string message = std::move(msg);
  auto wsConn = ws;

  ensureSynthPool();
  g_synthPool.getNextLoop()->queueInLoop([wsConn, message]() mutable {
    ApiParams p;
    try { p = parseParams(message); }
    catch (const std::exception& e) {
      nlohmann::json j; j["status"] = "failed"; j["message"] = e.what();
      wsConn->send(j.dump()); return;
    }

    const int sr = g_engine.config().sampleRate;
    const bool send_opus = (p.audio_format == "opus" || p.audio_format.empty());
    StreamingOggOpusEncoder enc(sr, 1);

    try {
      doSynth(p, [&](const int16_t* d, std::size_t n) {
        if (send_opus) {
          std::vector<short> pcm(d, d + n);
          auto opus = enc.encode(pcm);
          if (!opus.empty())
            wsConn->send(reinterpret_cast<const char*>(opus.data()),
                         opus.size(), WebSocketMessageType::Binary);
        } else {
          if constexpr (std::endian::native == std::endian::big) {
            std::vector<int16_t> tmp(d, d + n);
            for (auto& s : tmp) s = static_cast<int16_t>((s >> 8) | (s << 8));
            wsConn->send(reinterpret_cast<const char*>(tmp.data()),
                         tmp.size() * sizeof(int16_t),
                         WebSocketMessageType::Binary);
          } else {
            wsConn->send(reinterpret_cast<const char*>(d),
                         n * sizeof(int16_t), WebSocketMessageType::Binary);
          }
        }
      });
    } catch (const std::exception& e) {
      nlohmann::json j; j["status"] = "failed"; j["message"] = e.what();
      wsConn->send(j.dump()); return;
    }

    if (send_opus) {
      auto tail = enc.finish();
      if (!tail.empty())
        wsConn->send(reinterpret_cast<const char*>(tail.data()), tail.size(),
                     WebSocketMessageType::Binary);
    }
    wsConn->send(R"({"status":"ok","message":"finished"})");
  });
}

} // namespace api

// ---- OpenAI-compatible TTS endpoint --------------------------------------

namespace v1 {

class audio : public HttpController<audio> {
 public:
  METHOD_LIST_BEGIN
  METHOD_ADD(audio::speech, "/speech", {Post, Options});
  METHOD_LIST_END

  void speech(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& cb);
};

void audio::speech(const HttpRequestPtr& req,
                   std::function<void(const HttpResponsePtr&)>&& cb) {
  if (req->method() == Options) {
    auto r = HttpResponse::newHttpResponse();
    r->addHeader("Access-Control-Allow-Origin", "*");
    r->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    cb(r); return;
  }
  if (req->getContentType() != CT_APPLICATION_JSON) {
    cb(badRequest("Content-Type must be application/json")); return;
  }
  if (!checkAuth(req)) { cb(badRequest("invalid Authorization")); return; }

  nlohmann::json j;
  try { j = nlohmann::json::parse(req->getBody()); }
  catch (...) { cb(badRequest("invalid JSON")); return; }

  if (!j.contains("input") || !j["input"].is_string()) {
    cb(badRequest("missing 'input'")); return;
  }
  ApiParams p;
  p.text = j["input"].get<std::string>();
  p.voice = j.value("voice", g_defaultVoice);
  if (j.contains("speed") && j["speed"].is_number())
    p.speed = j["speed"].get<float>();
  // OpenAI's response_format: opus|aac|mp3|flac|pcm|wav — we support
  // opus/pcm/wav directly; anything else degrades to opus.
  std::string fmt = j.value("response_format", "opus");
  if (fmt == "pcm" || fmt == "wav") p.audio_format = fmt;
  else p.audio_format = "opus";

  ensureSynthPool();
  g_synthPool.getNextLoop()->queueInLoop([p = std::move(p),
                                          cb = std::move(cb)]() mutable {
    std::vector<int16_t> audio;
    try {
      doSynth(p, [&](const int16_t* d, std::size_t n) {
        audio.insert(audio.end(), d, d + n);
      });
    } catch (const std::exception& e) {
      cb(badRequest(std::string("synthesis failed: ") + e.what())); return;
    }

    const int sr = g_engine.config().sampleRate;
    auto r = HttpResponse::newHttpResponse();
    if (p.audio_format == "wav") {
      std::vector<uint8_t> buf;
      appendWavHeader(buf, sr, 1, static_cast<uint32_t>(audio.size() * 2));
      auto* pp = reinterpret_cast<const uint8_t*>(audio.data());
      buf.insert(buf.end(), pp, pp + audio.size() * 2);
      r->addHeader("Content-Type", "audio/wav");
      r->setBody(std::string(reinterpret_cast<const char*>(buf.data()),
                             buf.size()));
    } else if (p.audio_format == "pcm") {
      r->addHeader("Content-Type", "audio/raw");
      r->setBody(std::string(reinterpret_cast<const char*>(audio.data()),
                             audio.size() * sizeof(int16_t)));
    } else {
      auto opus = encodeOgg(std::vector<short>(audio.begin(), audio.end()), sr, 1);
      r->addHeader("Content-Type", "audio/ogg; codecs=opus");
      r->setBody(std::string(reinterpret_cast<const char*>(opus.data()),
                             opus.size()));
    }
    cb(r);
  });
}

} // namespace v1

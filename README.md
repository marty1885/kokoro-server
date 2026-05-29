# kokoro-infer

Fast, lightweight end-to-end inference engine and server for Kokoro-82M TTS, optimized for CPU, CUDA, and Rockchip RK3588 NPU.

It provides both a Python prototyping pipeline (infer.py) and a production-ready C++ implementation (libkokoro, kokoro-cli, kokoro-server) with an OpenAI-compatible API.

## Features

- RK3588 NPU Acceleration: Run the Kokoro decoder on the Rockchip RK3588 NPU at ~3.5× RTF (via custom graph rewrites: fixed-stat normalization, Horner minimax Sin approximation, operator folding, and deconv-to-conv transform).
- Native C++ Library (libkokoro): Zero Python dependency, using ONNX Runtime (CPU/CUDA/TensorRT) or RKNN for inference.
- English G2P Frontend: Integration with misaki-cpp (English G2P frontend) and espeak-ng.
- OpenAI-Compatible Server: High-performance HTTP/WebSocket server using Drogon, offering:
  - POST /v1/audio/speech (OpenAI-compatible TTS)
  - POST /api/v1/synthesise (one-shot text/phonemes -> opus, pcm, wav, raw)
  - WS /api/v1/stream (streaming real-time audio)
  - GET /api/v1/voices (list available voices)
- Built-in Web UI: A lightweight demo web interface served directly by kokoro-server.

## Project Structure

- build.py: Host-side compilation and graph surgery (PyTorch -> ONNX -> RKNN)
- infer.py: Board-side Python reference runner (ORT + rknnlite)
- src/: Native C++ library (libkokoro)
- cli/: C++ CLI utility (kokoro-cli)
- server/: C++ Web/API Server (kokoro-server)
- misaki-cpp/: English G2P submodule
- NOTES.md: Detailed design & NPU optimization notes

## Python Quick Start (Prototyping / Board)

### 1. Compile RKNN Model (Host)
Prepare Kokoro weights and kokoro-src, then run:
```bash
python3 build.py
```
Outputs: onnx/kokoro_encoder.onnx, onnx/har_generator.onnx, onnx/kokoro_decoder.rknn, and voices_npy/.

### 2. Run Inference (Board)
Copy outputs and run:
```bash
# English text (needs misaki: pip install "misaki[en]")
python3 infer.py --text "Hello world." --voice af_heart --out hello.wav

# Or raw IPA phonemes directly
python3 infer.py --phonemes "həlˈO wˈɜɹld" --voice af_heart --out hello.wav
```

## C++ Quick Start

### 1. Requirements
Ensure the following dependencies are installed:
- ONNX Runtime C++ API (or RKNN runtime librknnrt)
- espeak-ng
- OpenBLAS (for iSTFT sgemm)
- fmt, spdlog, nlohmann_json
- For Server: Drogon, Opus, soxr

### 2. Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_RKNN=ON \             # Enable RK3588 NPU acceleration
         -DBUILD_SERVER=ON \
         -DBUILD_CLI=ON
make -j$(nproc)
```

### 3. Run C++ CLI
```bash
./kokoro-cli --text "Hello from C++." --voice af_heart --out out.wav
```

### 4. Run C++ Server
```bash
./kokoro-server \
  --encoder onnx/kokoro_encoder.onnx \
  --har-gen onnx/har_generator.onnx \
  --decoder onnx/kokoro_decoder.rknn \
  --vocab Kokoro-82M/config.json \
  --voices-dir voices_npy \
  --port 8848
```
Visit http://localhost:8848/ in your browser for the Web UI.

## Optimizations & Performance
For full technical details about fixed-stat normalization, Horner minimax polynomial approximation for Sin, Pow-to-Mul substitution, Snake1d Conv folding, and custom ConvTranspose padding rewrites, see the ON_RKNN_HACKING.md file.

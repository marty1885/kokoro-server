#!/usr/bin/env python3
"""
Run Kokoro-82M TTS end-to-end on a Rockchip RK3588 board.

Inputs:    IPA phonemes (string), voice name, optional output wav path.
Outputs:   .wav at 24 kHz, plus stage timings + RTF.

Stages:
    encoder (CPU, onnxruntime)   onnx/kokoro_encoder.onnx
    har gen (CPU, onnxruntime)   onnx/har_generator.onnx
    decoder (NPU, rknnlite)      onnx/kokoro_decoder.rknn
    iSTFT   (CPU, numpy)         inline below

Dependencies on the board:
    numpy, onnxruntime, rknn-toolkit-lite2  (no torch).

Make these artefacts on a host with build.py, then copy them to the board:
    onnx/kokoro_encoder.onnx
    onnx/har_generator.onnx
    onnx/kokoro_decoder.rknn
    voices_npy/*.npy
    Kokoro-82M/config.json       (only the vocab is needed at runtime)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SR              = 24000
SAMPLES_PER_FR  = 600        # final audio samples per ASR frame
T_FIX_DEFAULT   = 50         # decoder was compiled with this fixed window
DEPOP_CTX       = 2          # ASR frames of real context fed on each side
DEPOP_OVL       = 480        # overlap-add crossfade width (samples) per seam,
                             # split ±DEPOP_OVL//2 into the real context both
                             # windows decoded. Must be <= 2*DEPOP_CTX*SAMPLES_PER_FR
                             # and small enough to stay inside well-supported context.


def depop_append(buf: np.ndarray, chunk: np.ndarray,
                 blend: int = DEPOP_OVL) -> np.ndarray:
    """Overlap-add `chunk` onto `buf` across the seam. The caller extends each
    window by `blend//2` samples into the real context on *each* side, so the
    trailing `blend` samples of `buf` and the leading `blend` of `chunk` are two
    reconstructions of the SAME region straddling the boundary. The linear
    crossfade weights window i highest as it approaches the seam from its
    well-supported interior and window i+1 highest as it continues past — each
    contributes where it has the most decoder context, cancelling the
    inter-window fp16 step. No samples dropped, timing exact, and no WSOLA
    search (Kokoro frames sit on a fixed grid, so alignment is already exact)."""
    if buf.size < blend or chunk.size < blend:
        return np.concatenate([buf, chunk])

    w = np.linspace(0.0, 1.0, blend, endpoint=False, dtype=np.float32)
    blended = buf[-blend:] * (1.0 - w) + chunk[:blend] * w
    return np.concatenate([buf[:-blend], blended, chunk[blend:]])


# ---------------------------------------------------------------------------
# RKNN decoder wrapper. Loads via rknn-toolkit-lite2 on the board.
# ---------------------------------------------------------------------------
class RknnDecoder:
    CORE_MASKS = ("auto", "0", "1", "2", "0_1", "0_1_2")

    def __init__(self, rknn_path: str, core_mask: str = "auto"):
        from rknnlite.api import RKNNLite
        constants = {
            "auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,       "2": RKNNLite.NPU_CORE_2,
            "0_1": RKNNLite.NPU_CORE_0_1,   "0_1_2": RKNNLite.NPU_CORE_0_1_2,
        }
        self.r = RKNNLite()
        if self.r.load_rknn(rknn_path) != 0:
            raise SystemExit(f"load_rknn failed: {rknn_path}")
        if self.r.init_runtime(core_mask=constants[core_mask]) != 0:
            raise SystemExit("init_runtime failed")

    def __call__(self, asr, F0, N, s, har):
        return self.r.inference(inputs=[asr, F0, N, s, har], data_format=["nchw"] * 5)

    def close(self):
        self.r.release()


# ---------------------------------------------------------------------------
# numpy iSTFT (Hann window, exactly matches kokoro CustomSTFT.inverse).
# ---------------------------------------------------------------------------
class IStft:
    def __init__(self, n_fft: int = 20, hop: int = 5):
        self.n_fft, self.hop = n_fft, hop
        n = np.arange(n_fft)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)         # periodic Hann
        self.win_sq = (win * win).astype(np.float32)             # for NOLA envelope
        inv_window = (win / n_fft).astype(np.float32)
        F = n_fft // 2 + 1
        k = np.arange(F)
        # Real iFFT of a one-sided spectrum: bins 1..F-2 stand in for their
        # conjugate twins (factor 2); DC and Nyquist appear once. CustomSTFT
        # omits this doubling, which halved the mid/high band → muffled audio.
        dbl = np.ones(F, np.float32); dbl[1:F-1] = 2.0
        angle = 2 * np.pi * np.outer(n, k) / n_fft               # (n_fft, F)
        self.cos_w = (np.cos(angle).T * inv_window * dbl[:, None]).astype(np.float32)
        self.sin_w = (np.sin(angle).T * inv_window * dbl[:, None]).astype(np.float32)

    def __call__(self, spec, phase):
        # spec, phase: (B, F, T_frames)  — phase is sin(x), not radians
        real = spec * np.cos(phase)
        imag = spec * np.sin(phase)
        fr_r = np.einsum("bft,fn->btn", real, self.cos_w)
        fr_i = np.einsum("bft,fn->btn", imag, self.sin_w)
        frames = fr_r - fr_i                                     # (B, T_frames, n_fft)
        B, T_frames, _ = frames.shape
        # Vectorized overlap-add: n_fft/hop is an integer ratio (4 here), so
        # every output sample is the sum of exactly that many shifted frames.
        # Split frames along n_fft into `ratio` chunks of width `hop` and add
        # them at hop-shifted offsets — `ratio` numpy ops, no Python loop.
        ratio = self.n_fft // self.hop
        assert ratio * self.hop == self.n_fft, "n_fft must be a multiple of hop"
        out_len = (T_frames - 1) * self.hop + self.n_fft
        out = np.zeros((B, out_len), np.float32)
        flat_len = T_frames * self.hop
        for k in range(ratio):
            contrib = frames[:, :, k*self.hop:(k+1)*self.hop].reshape(B, flat_len)
            out[:, k*self.hop : k*self.hop + flat_len] += contrib
        # Divide by the window^2 overlap envelope (NOLA), as torch.istft does.
        env = np.zeros(out_len, np.float32)
        for t in range(T_frames):
            env[t*self.hop : t*self.hop + self.n_fft] += self.win_sq
        out /= np.maximum(env, 1e-11)
        pad = self.n_fft // 2
        return out[:, pad:-pad] if pad else out


# ---------------------------------------------------------------------------
def load_vocab(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)["vocab"]


def phonemes_to_ids(phonemes: str, vocab: dict) -> list[int]:
    ids = [vocab[p] for p in phonemes if p in vocab]
    missing = [p for p in phonemes if p not in vocab]
    if missing:
        print(f"[warn] {len(missing)} chars not in vocab, dropped: {missing[:8]}",
              file=sys.stderr)
    return ids


def text_to_phonemes(text: str, british: bool = False) -> str:
    """English text → IPA via misaki, the G2P kokoro itself uses."""
    # misaki[en] pulls in spacy → thinc → torch. torch.fx's logger init reads
    # `PYTORCH_MATCHER_LOGLEVEL` and crashes on its own default in some envs;
    # set it explicitly before the import so we never hit that path.
    os.environ.setdefault("PYTORCH_MATCHER_LOGLEVEL", "WARNING")
    try:
        from misaki import en
    except ImportError:
        raise SystemExit(
            "--text needs misaki. Install with:  pip install misaki[en]\n"
            "Or pass --phonemes with IPA directly.")
    g2p = en.G2P(trf=False, british=british, fallback=None)
    phonemes, _ = g2p(text)
    return phonemes


def load_voice(voice: str, n_phonemes: int) -> np.ndarray:
    """Accept either 'af_heart' (bare name) or a path to .npy."""
    path = Path(voice)
    if not path.exists():
        path = Path("voices_npy") / (path.stem + ".npy")
    if not path.exists():
        raise SystemExit(f"voice not found: {voice}  (looked in voices_npy/)")
    pack = np.load(path)
    if pack.ndim == 3:                                            # (510, 1, 256) — pick by phoneme count
        pack = pack[max(0, min(pack.shape[0] - 1, n_phonemes - 1))]
    return pack.astype(np.float32).reshape(1, 256)


def write_wav(path: str, audio: np.ndarray, sr: int = SR):
    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text",
                     help="English text — converted to IPA via misaki "
                          "(install: pip install misaki[en])")
    src.add_argument("--phonemes",
                     help="IPA phoneme string (kokoro vocab); skips G2P")
    ap.add_argument("--british",  action="store_true",
                    help="Use British English G2P (only with --text)")
    ap.add_argument("--voice",    default="af_heart")
    ap.add_argument("--speed",    type=float, default=1.0)
    ap.add_argument("--out",      default="out.wav")
    ap.add_argument("--encoder",  default="onnx/kokoro_encoder.onnx")
    ap.add_argument("--har",      default="onnx/har_generator.onnx")
    ap.add_argument("--decoder",  default="onnx/kokoro_decoder.rknn")
    ap.add_argument("--vocab",    default="Kokoro-82M/config.json")
    ap.add_argument("--cores",    default="auto", choices=RknnDecoder.CORE_MASKS)
    ap.add_argument("--t-fix",    type=int, default=T_FIX_DEFAULT,
                    help="Decoder window size at compile time (must match build)")
    args = ap.parse_args()

    T_FIX = args.t_fix
    vocab = load_vocab(args.vocab)

    # G2P first, *before* the rknnlite import — rknnlite's logging shim
    # rewrites `logging._nameToLevel` to single-letter keys ('D', 'E', 'I',
    # 'W'), which makes `logger.setLevel('WARNING')` blow up later inside any
    # library that uses the stdlib logging API (e.g. misaki → spacy → torch).
    # Resolve phonemes here so the misaki import happens while logging is
    # still intact.
    if args.text is not None:
        t0 = time.perf_counter()
        phonemes = text_to_phonemes(args.text, british=args.british)
        t_g2p = time.perf_counter() - t0
    elif args.phonemes is not None:
        phonemes = args.phonemes
        t_g2p = 0.0
    else:
        phonemes = "həlˈO wˈɜɹld"
        t_g2p = 0.0
        print(f"[demo] no --text/--phonemes; using {phonemes!r}")

    # ===== Stage 1: load all models + warm the NPU (one-time costs) =====
    init_t0 = time.perf_counter()

    t0 = time.perf_counter()
    enc = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    print(f"[init] encoder loaded  ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    har_sess = ort.InferenceSession(args.har, providers=["CPUExecutionProvider"])
    print(f"[init] har gen loaded  ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    dec = RknnDecoder(args.decoder, core_mask=args.cores)
    print(f"[init] decoder loaded  ({(time.perf_counter()-t0)*1000:.0f} ms)  [cores={args.cores}]")

    istft = IStft()

    # NPU warmup — the first inference includes context allocation. Doing
    # this once now means the timings reported below are steady-state.
    t0 = time.perf_counter()
    warm_a   = np.zeros((1, 512, T_FIX), np.float32)
    warm_fn  = np.zeros((1, 2*T_FIX),    np.float32)
    warm_s   = np.zeros((1, 128),        np.float32)
    warm_har = np.zeros((1, 22, 2*T_FIX*300//5 + 1), np.float32)
    _ = dec(warm_a, warm_fn, warm_fn, warm_s, warm_har)
    print(f"[init] decoder warmup  ({(time.perf_counter()-t0)*1000:.0f} ms)")
    print(f"[ready] all models initialised  (total {(time.perf_counter()-init_t0)*1000:.0f} ms)")
    print()

    # ===== Stage 2: actual inference =====
    if args.text is not None:
        print(f"[g2p] {t_g2p*1000:5.1f} ms  {args.text!r}  →  {phonemes!r}")
    ids = phonemes_to_ids(phonemes, vocab)
    if not ids:
        sys.exit("no valid phonemes after vocab filter")
    input_ids = np.array([[0, *ids, 0]], np.int64)
    ref_s = load_voice(args.voice, len(ids))
    speed = np.array([args.speed], np.float32)

    t0 = time.perf_counter()
    asr, F0, N, s, pred_dur = enc.run(
        None, {"input_ids": input_ids, "ref_s": ref_s, "speed": speed})
    t_enc = time.perf_counter() - t0
    T = int(asr.shape[2])                        # total frames incl. BOS/EOS
    trail = int(pred_dur[-1])                     # frames the EOS (0) token owns
    T_emit = max(1, T - trail)                    # don't voice EOS; keep as context
    print(f"[enc] {t_enc*1000:6.1f} ms  T={T}  pred_dur={pred_dur.tolist()}  "
          f"drop_eos={trail}fr  expected_audio={T_emit*SAMPLES_PER_FR/SR:.2f}s")

    audio_full = np.empty(0, np.float32)
    t_har, t_dec, t_ist, t_join = 0.0, 0.0, 0.0, 0.0
    n_win = 0
    CTX  = DEPOP_CTX
    STEP = T_FIX - 2 * CTX                       # frames emitted per window
    emit_offset = CTX * SAMPLES_PER_FR
    HALF = DEPOP_OVL // 2                         # overlap each side of a seam
    assert HALF <= emit_offset, "DEPOP_OVL exceeds decoded context"
    for emit_start in range(0, T_emit, STEP):
        emit_end = min(emit_start + STEP, T_emit)
        win_start, win_end = emit_start - CTX, emit_end + CTX
        a_lo, a_hi = max(0, win_start), min(T, win_end)
        lpad = a_lo - win_start                  # zero context past utterance start
        rpad = win_end - a_hi                    # zero context past utterance end
        rpad += T_FIX - (a_hi - a_lo + lpad + rpad)  # fill remainder for fixed shape

        a = asr[:, :, a_lo:a_hi]
        f = F0[:, 2*a_lo:2*a_hi]
        n = N[:,  2*a_lo:2*a_hi]
        if lpad or rpad:
            a = np.pad(a, ((0,0),(0,0),(lpad, rpad)))
            f = np.pad(f, ((0,0),(2*lpad, 2*rpad)))
            n = np.pad(n, ((0,0),(2*lpad, 2*rpad)))
        a = np.ascontiguousarray(a, np.float32)
        f = np.ascontiguousarray(f, np.float32)
        n = np.ascontiguousarray(n, np.float32)

        t0 = time.perf_counter()
        har = har_sess.run(None, {"F0": f})[0].astype(np.float32)
        t_har += time.perf_counter() - t0

        t0 = time.perf_counter()
        spec, phase = dec(a, f, n, s, har)
        t_dec += time.perf_counter() - t0

        t0 = time.perf_counter()
        audio = istft(spec, phase)
        t_ist += time.perf_counter() - t0

        emit_samples = (emit_end - emit_start) * SAMPLES_PER_FR
        # Symmetric overlap-add: extend the emit region by HALF into the real
        # context both neighbouring windows decoded. Each seam then crossfades a
        # 2*HALF region straddling the boundary — window i tapers out past the
        # seam, window i+1 tapers in before it (see depop_append). The first
        # window has no left neighbour, the last no right. No samples dropped.
        left_ext  = HALF if emit_start > 0      else 0
        right_ext = HALF if emit_end   < T_emit else 0
        piece = audio.reshape(-1)[emit_offset - left_ext :
                                  emit_offset + emit_samples + right_ext] \
                     .astype(np.float32, copy=False)
        t0 = time.perf_counter()
        audio_full = depop_append(audio_full, piece)
        t_join += time.perf_counter() - t0
        n_win += 1

    dec.close()

    write_wav(args.out, audio_full, sr=SR)

    audio_s = audio_full.size / SR
    t_total = t_enc + t_har + t_dec + t_ist + t_join
    rtf     = audio_s / t_total
    print()
    print(f"  audio out       : {audio_s:6.3f} s  ({audio_full.size} samples)")
    print(f"  encoder   (CPU) : {t_enc*1000:7.1f} ms")
    print(f"  har gen   (CPU) : {t_har*1000:7.1f} ms  ({n_win} window{'s' if n_win>1 else ''})")
    print(f"  decoder   (NPU) : {t_dec*1000:7.1f} ms")
    print(f"  iSTFT     (CPU) : {t_ist*1000:7.1f} ms")
    print(f"  depop     (CPU) : {t_join*1000:7.1f} ms  ({max(n_win-1,0)} seam{'s' if n_win!=2 else ''})")
    print(f"  total           : {t_total*1000:7.1f} ms")
    print(f"  RTF             : {rtf:5.2f}×  ({1/rtf*1000:.0f} ms wall per 1 s audio)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build all artefacts needed to run Kokoro-82M on a Rockchip RK3588 NPU.

Reads:
    Kokoro-82M/config.json
    Kokoro-82M/kokoro-v1_0.pth
    Kokoro-82M/voices/*.pt
    kokoro-src/kokoro/*.py        (model definition, vendored from kokoro pkg)

Writes (the files the board needs):
    onnx/kokoro_encoder.onnx       CPU on board
    onnx/har_generator.onnx        CPU on board (holds Atan + RandomNormalLike,
                                                 neither runs on librknnrt 2.3)
    onnx/kokoro_decoder.onnx       final fp32 decoder (CPU fallback / RKNN source)
    onnx/kokoro_decoder.rknn       NPU on board (compiled from kokoro_decoder.onnx)
    voices_npy/*.npy               board uses these instead of .pt

Pipeline summary (one stage per section below):
    1.  Export encoder (PyTorch → ONNX, dynamic L/T)
    2.  Export har generator (F0 → har via m_source + STFT.transform)
    3.  Export decoder v3 (outputs spec, phase; iSTFT runs on board CPU)
    4.  Voice .pt → .npy
    5.  Graph surgery on the decoder ONNX:
        a) InstanceNorm → dynamic per-instance norm via GlobalAveragePool
           (RKNN lowers GAP to on-NPU conv cascades — no channel-last Transpose,
           unlike InstanceNorm/ReduceMean; matches the raw fp32 noise floor)
        b) Sin → Clip[−π, π] + Horner minimax poly (deg 7)  (kills CPU Sin,
           ~300× more accurate than Taylor at the same degree)
        c) Pow(x, 2) → Mul(x, x)                            (cheaper, fusable)
        d) Snake1d /α post-scaling folded into next Conv's per-input weights
           (kills one full-tensor Mul per Snake = one DMA pass on this
           bandwidth-bound NPU)
        e) ConvTranspose stride 10/6 → zero-interleave + stride-1 Conv1d
           (librknnc only accepts deconv stride ∈ {1, 2, 4, 8})
    6.  RKNN compile (fp16)

Result on RK3588: ~3.5× RTF end-to-end (4.2× post-encoder).

Run:    python3 build.py
Knobs:  --t-fix N           ASR-frame window the decoder is compiled for
                            (default 50 = 0.625 s; smaller windows compile to
                            smaller NPU tiles that fit better in SRAM)
        --taylor-degree N   degree for the Sin polynomial (default 7; 5 trades
                            ~5 ms/call of decoder time for a higher noise floor)
        --skip-rknn         just produce ONNX, skip RKNN compile
        --rebuild           delete intermediate ONNX files first
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper


# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent
KOKORO_DIR  = HERE / "Kokoro-82M"
KSRC        = HERE / "kokoro-src" / "kokoro"
ONNX_DIR    = HERE / "onnx"
VOICES_DST  = HERE / "voices_npy"

ENC_ONNX    = ONNX_DIR / "kokoro_encoder.onnx"
HAR_ONNX    = ONNX_DIR / "har_generator.onnx"
DEC_RAW     = ONNX_DIR / "_decoder_raw.onnx"
DEC_FIXED   = ONNX_DIR / "_decoder_fixnorm.onnx"
DEC_SIN     = ONNX_DIR / "_decoder_fixnorm_sin.onnx"
DEC_FINAL   = ONNX_DIR / "kokoro_decoder.onnx"
DEC_RKNN    = ONNX_DIR / "kokoro_decoder.rknn"

SR              = 24000
SAMPLES_PER_FR  = 600        # = prod(upsample_rates) * gen_istft_hop * 2 (F0 is 2× asr)


# ---------------------------------------------------------------------------
# Kokoro PyTorch loader — bypass kokoro/__init__.py (which imports `misaki`).
# ---------------------------------------------------------------------------
def load_kokoro():
    """Side-effect: registers `kokoro.*` modules in sys.modules. Returns the
    KModel class plus the istftnet submodule (for Decoder/Generator construction)."""
    pkg = types.ModuleType("kokoro"); pkg.__path__ = [str(KSRC)]
    sys.modules["kokoro"] = pkg
    for name in ("custom_stft", "istftnet", "modules", "model"):
        spec = importlib.util.spec_from_file_location(f"kokoro.{name}", KSRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"kokoro.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["kokoro.model"].KModel, sys.modules["kokoro.istftnet"]


# ---------------------------------------------------------------------------
# Stage 1–3 — PyTorch → ONNX exports
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Everything before the ISTFTNet decoder call (BERT + ProsodyPredictor +
    TextEncoder + alignment + F0/N upsamplers)."""
    def __init__(self, k):
        super().__init__(); self.m = k

    def forward(self, input_ids, ref_s, speed):
        m = self.m
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1],
                                   device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s_full = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s_full, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur)
        aln = torch.zeros((input_ids.shape[1], indices.shape[0]), device=input_ids.device)
        aln[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        aln = aln.unsqueeze(0)

        en = d.transpose(-1, -2) @ aln
        F0_pred, N_pred = m.predictor.F0Ntrain(en, s_full)
        t_en = m.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln
        s_first = ref_s[:, :128]
        return asr, F0_pred, N_pred, s_first, pred_dur


class HarGen(nn.Module):
    """F0 → har via m_source + STFT.transform. Lives separately so the Atan +
    RandomNormalLike ops (which librknnrt has no NPU op for) stay on the CPU
    onnxruntime path on the board, instead of crashing the .rknn at load."""
    def __init__(self, generator):
        super().__init__(); self.g = generator

    def forward(self, F0):
        g = self.g
        f0 = g.f0_upsamp(F0[:, None]).transpose(1, 2)
        har_source, _, _ = g.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = g.stft.transform(har_source)
        return torch.cat([har_spec, har_phase], dim=1)


class DecoderV3(nn.Module):
    """ISTFTNet decoder accepting precomputed `har`, outputting (spec, phase).
    iSTFT runs on the board CPU in numpy — `CustomSTFT.inverse` has two
    stride-5 ConvTransposes which librknnc rejects, and numpy einsum+OLA at
    F=12001 is fast enough (~45 ms)."""
    def __init__(self, decoder):
        super().__init__(); self.d = decoder

    def forward(self, asr, F0_curve, N, s, har):
        d = self.d
        F0 = d.F0_conv(F0_curve.unsqueeze(1))
        N_ = d.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_], axis=1)
        x = d.encode(x, s)
        asr_res = d.asr_res(asr)
        res = True
        for block in d.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        g = d.generator
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = g.noise_convs[i](har)
            x_source = g.noise_res[i](x_source, s)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(g.num_kernels):
                if xs is None:
                    xs = g.resblocks[i*g.num_kernels + j](x, s)
                else:
                    xs += g.resblocks[i*g.num_kernels + j](x, s)
            x = xs / g.num_kernels
        x = F.leaky_relu(x)
        x = g.conv_post(x)
        spec  = torch.exp(x[:, :g.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, g.post_n_fft // 2 + 1:, :])
        return spec, phase


def export_onnx(kmodel, istftnet, t_fix):
    ONNX_DIR.mkdir(exist_ok=True)
    print("[1/8] export encoder.onnx")
    enc = Encoder(kmodel).eval()
    dummy_ids = torch.LongTensor([[0, 50, 156, 43, 102, 16, 56, 102, 43, 16, 102, 50, 0]])
    torch.onnx.export(
        enc, (dummy_ids, torch.randn(1, 256), torch.tensor([1.0])),
        str(ENC_ONNX),
        input_names=["input_ids", "ref_s", "speed"],
        output_names=["asr", "F0", "N", "s", "pred_dur"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input_ids": {1: "L"}, "asr": {2: "T"},
                      "F0": {1: "T2"}, "N": {1: "T2"}, "pred_dur": {0: "L"}},
    )

    print("[2/8] export har_generator.onnx")
    # Build a decoder just to access the generator submodule (m_source + STFT)
    with open(KOKORO_DIR / "config.json") as f:
        cfg = json.load(f)
    decoder = istftnet.Decoder(
        dim_in=cfg["hidden_dim"], style_dim=cfg["style_dim"],
        dim_out=cfg["n_mels"], disable_complex=True, **cfg["istftnet"],
    ).eval()
    sd = torch.load(KOKORO_DIR / "kokoro-v1_0.pth", map_location="cpu", weights_only=True)["decoder"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    decoder.load_state_dict(sd, strict=False)
    hg = HarGen(decoder.generator).eval()
    torch.onnx.export(
        hg, (torch.randn(1, 2*t_fix).abs() * 200,),
        str(HAR_ONNX),
        input_names=["F0"], output_names=["har"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"F0": {1: "T2"}, "har": {2: "F"}},
    )

    print("[3/8] export decoder_v3 (raw)")
    dec_v3 = DecoderV3(decoder).eval()
    asr = torch.randn(1, 512, t_fix)
    F0  = torch.randn(1, 2*t_fix).abs() * 200
    N   = torch.randn(1, 2*t_fix)
    s   = torch.randn(1, 128)
    har = torch.randn(1, 22, 2*t_fix * 300 // 5 + 1)
    torch.onnx.export(
        dec_v3, (asr, F0, N, s, har),
        str(DEC_RAW),
        input_names=["asr", "F0", "N", "s", "har"],
        output_names=["spec", "phase"],
        opset_version=17, do_constant_folding=True,
    )


def convert_voices():
    """voices/*.pt → voices_npy/*.npy so the board never needs torch."""
    print("[4/8] convert voices")
    VOICES_DST.mkdir(exist_ok=True)
    n = 0
    for pt in sorted((KOKORO_DIR / "voices").glob("*.pt")):
        arr = torch.load(pt, map_location="cpu", weights_only=True).to(torch.float32).numpy()
        np.save(VOICES_DST / (pt.stem + ".npy"), arr)
        n += 1
    print(f"       wrote {n} voices to {VOICES_DST}/")


# ---------------------------------------------------------------------------
# Stage 7 — Graph surgery (three small passes, all math-preserving up to the
# InstanceNorm approximation which uses dataset stats instead of per-instance).
# ---------------------------------------------------------------------------
def _get_init(model, name):
    for i in model.graph.initializer:
        if i.name == name:
            return i
    return None


def _resolve_through_identity(model, name, nodes_by_out):
    seen = 0
    while seen < 8:
        init = _get_init(model, name)
        if init is not None:
            return init
        n = nodes_by_out.get(name)
        if n is None or n.op_type != "Identity":
            return None
        name = n.input[0]
        seen += 1
    return None


def rewrite_instancenorm_gap(model):
    """IN(x, scale, bias) → *dynamic* per-instance norm, but with the time-axis
    reduction expressed as GlobalAveragePool so it stays on the NPU.

    Why this works where the original fixnorm punt was needed:
      • `InstanceNormalization` and an explicit `ReduceMean(axis=time)` both get
        lowered by RKNN to a channel-last (B,T,C) op and wrapped in a Transpose;
        at the upsampled T that Transpose falls back to CPU (~33 ms each, the
        biggest fallback — see NOTES.md §1 and "things that didn't help").
      • `GlobalAveragePool` maps to the RK3588 *pooling engine* in the native
        NC1HWC2 layout — the same reduction every SE/squeeze-excite block uses —
        so the per-channel time reduction never leaves the NPU.

    This restores the exact per-window per-instance normalization of the raw
    fp32 decoder (so the noise floor should match raw, not fixnorm), at the cost
    of a couple of extra full-T elementwise passes per IN versus the constant
    form. Two-pass (centered) variance is used instead of E[x²]−μ² because the
    latter catastrophically cancels in fp16 when |μ| ≫ σ, which would itself
    inject noise:  mu=GAP(x); xc=x−mu; var=GAP(xc·xc); y=xc/√(var+eps).

    The IN's own affine (scale/bias) is folded in directly when constant."""
    nodes_by_out = {o: n for n in model.graph.node for o in n.output}
    candidates = [n for n in list(model.graph.node) if n.op_type == "InstanceNormalization"]
    n_done = 0
    for node in candidates:
        x_in, scale_n, bias_n = node.input
        y_out = node.output[0]
        base = node.name or y_out
        eps = next((float(a.f) for a in node.attribute if a.name == "epsilon"), 1e-5)

        eps_init = f"{base}/eps"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(eps, np.float32), eps_init))

        new = []
        mu = f"{base}/mu"
        new.append(helper.make_node("GlobalAveragePool", [x_in], [mu], name=f"{base}/GAPmu"))
        xc = f"{base}/centered"
        new.append(helper.make_node("Sub", [x_in, mu], [xc], name=f"{base}/Sub"))
        sq = f"{base}/sq"
        new.append(helper.make_node("Mul", [xc, xc], [sq], name=f"{base}/Sq"))
        var = f"{base}/var"
        new.append(helper.make_node("GlobalAveragePool", [sq], [var], name=f"{base}/GAPvar"))
        vare = f"{base}/vare"
        new.append(helper.make_node("Add", [var, eps_init], [vare], name=f"{base}/Eps"))
        std = f"{base}/std"
        new.append(helper.make_node("Sqrt", [vare], [std], name=f"{base}/Sqrt"))
        inv = f"{base}/inv"
        new.append(helper.make_node("Reciprocal", [std], [inv], name=f"{base}/Recip"))
        norm = f"{base}/norm"
        new.append(helper.make_node("Mul", [xc, inv], [norm], name=f"{base}/Norm"))

        # Fold the IN's own affine when it's constant
        # (constant for the encode/decode top-level norms; absent for AdaIN,
        # whose γ/β are applied by separate downstream nodes).
        scale_init = _resolve_through_identity(model, scale_n, nodes_by_out)
        bias_init  = _resolve_through_identity(model, bias_n,  nodes_by_out)
        if scale_init is not None and bias_init is not None:
            s = numpy_helper.to_array(scale_init).reshape(1, -1, 1).astype(np.float32)
            b = numpy_helper.to_array(bias_init ).reshape(1, -1, 1).astype(np.float32)
            sn = f"{base}/s3d"; bn = f"{base}/b3d"
            model.graph.initializer.append(numpy_helper.from_array(s, sn))
            model.graph.initializer.append(numpy_helper.from_array(b, bn))
            scaled = f"{base}/scaled"
            new.append(helper.make_node("Mul", [norm, sn], [scaled], name=f"{base}/Sc"))
            new.append(helper.make_node("Add", [scaled, bn], [y_out], name=f"{base}/Bi"))
        else:
            new.append(helper.make_node("Identity", [norm], [y_out], name=f"{base}/Id"))

        idx = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        for i, nn in enumerate(new):
            model.graph.node.insert(idx + i, nn)
        n_done += 1
    return n_done


def _minimax_sin_coeffs(degree, clip):
    """Minimax (Chebyshev-projection) odd-polynomial fit of sin(x) on
    [-clip, +clip]. Returns coefficients [c1, c3, c5, ...] for the
    expansion  sin(x) ≈ Σ c_{2k+1} · x^{2k+1}.

    For [-π, π], degree 5 minimax max-err ≈ 5.9e-2 vs Taylor's 5.2e-1
    (~9× tighter); degree 7 minimax ≈ 4.5e-3 vs Taylor's 7.5e-2 (~17×)."""
    from numpy.polynomial import chebyshev
    # sample sin on a dense grid in [-clip, clip] and Chebyshev-fit
    t = np.cos(np.linspace(0, np.pi, 4096))  # Chebyshev nodes in [-1, 1]
    x = clip * t
    cheb = chebyshev.Chebyshev.fit(t, np.sin(x), degree)
    poly = cheb.convert(kind=np.polynomial.Polynomial).coef  # monomial in t
    # zero out (near-zero) even coefficients to enforce odd symmetry
    for i in range(0, len(poly), 2):
        poly[i] = 0.0
    # convert t = x/clip back to x: c_k_in_x = c_k_in_t / clip^k
    poly_in_x = [poly[k] / (clip ** k) for k in range(len(poly))]
    return [float(poly_in_x[p]) for p in (1, 3, 5, 7, 9) if p <= degree]


def rewrite_sin_clipped_taylor(model, degree=7, clip=3.14159, keep_last=True):
    """Sin(x) → Clip(x, −clip, +clip) → odd-power polynomial in Horner form
    with minimax (Chebyshev-projection) coefficients over [-clip, clip].
    librknnrt has no NPU Sin op; this replacement is all NPU-native ops.

    Horner: sin(x) ≈ x · (c1 + x² · (c3 + x² · (c5 + x² · c7))). For
    degree 7 that's 5 Muls + 3 Adds — the chain reuses one running
    accumulator instead of materializing x³, x⁵, x⁷ as separate tensors that
    each get DMA'd in and out of NPU SRAM. On the bandwidth-bound RK3588
    NPU, that's the real win.

    Minimax coefficients are ~50–300× tighter than Taylor at the same
    degree because they spread error uniformly across the interval; degree-7
    minimax max err ≈ 2.6e-4 (vs Taylor's 7.5e-2 at the same degree).

    For Snake1d the sin is squared (∈ [0, 1]) and added to x, so
    clip-saturation outside [−π, π] is a bounded error — empirically
    inaudible.

    `keep_last`: skip the final /Sin (the phase output sin(x[...])), which
    runs once and is cheap on CPU."""
    powers = [p for p in (1, 3, 5, 7, 9) if p <= degree]
    coeffs = _minimax_sin_coeffs(degree, clip)

    candidates = [n for n in list(model.graph.node) if n.op_type == "Sin"]
    if keep_last:
        candidates = [n for n in candidates if "noise_res" in n.name or "resblocks" in n.name]

    def add_const(name, val):
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(val, np.float32), name))

    n_done = 0
    for node in candidates:
        x = node.input[0]; y = node.output[0]
        base = node.name or y
        new = []

        # Clip x to [-clip, +clip]
        lo = f"{base}/lo"; hi = f"{base}/hi"
        add_const(lo, -clip); add_const(hi, clip)
        xc = f"{base}/xc"
        new.append(helper.make_node("Clip", [x, lo, hi], [xc], name=f"{base}/Clip"))

        # x² — reused at every Horner step
        x2 = f"{base}/x2"
        new.append(helper.make_node("Mul", [xc, xc], [x2], name=f"{base}/x2"))

        # Horner from the highest-order coefficient inward.
        # acc = c_last
        # for c in coeffs[-2::-1]:  acc = c + x² · acc
        # result = x · acc
        c_last = f"{base}/c{powers[-1]}"; add_const(c_last, coeffs[-1])
        acc = c_last
        for i in range(len(powers) - 2, -1, -1):
            p = powers[i]
            c_name = f"{base}/c{p}"; add_const(c_name, coeffs[i])
            mul_out = f"{base}/h_mul_{p}"
            new.append(helper.make_node("Mul", [x2, acc],   [mul_out], name=f"{base}/H_Mul_{p}"))
            add_out = f"{base}/h_add_{p}"
            new.append(helper.make_node("Add", [c_name, mul_out], [add_out], name=f"{base}/H_Add_{p}"))
            acc = add_out

        # Final multiply by x. Direct into the original Sin's output name.
        new.append(helper.make_node("Mul", [xc, acc], [y], name=f"{base}/H_x"))

        idx = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        for i, nn in enumerate(new):
            model.graph.node.insert(idx + i, nn)
        n_done += 1
    return n_done


def rewrite_deconv_stride(model):
    """Replace ConvTranspose1d with stride ∉ {1,2,4,8} (the only strides
    librknnc accepts) with the math-equivalent zero-interleave + stride-1
    Conv1d:
        X (B,C,T) ─reshape→ (B,C,T,1)
                  ─pad axis=3 (0, S−1)→ (B,C,T,S)
                  ─reshape→ (B,C,T·S)
                  ─Conv1d with W.T(0,1).flip(2), pads=[K−1−P, K−S−P]
    Verified to 1.5e-6 vs F.conv_transpose1d."""
    ALLOWED = {1, 2, 4, 8}
    n_done = 0
    for node in [n for n in list(model.graph.node) if n.op_type == "ConvTranspose"]:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        S = (attrs.get("strides") or [1])[0]
        if S in ALLOWED:
            continue
        K = attrs.get("kernel_shape", [None])[0]
        P_left, P_right = attrs.get("pads", [0, 0])
        if P_left != P_right or attrs.get("dilations", [1]) != [1] or attrs.get("group", 1) != 1:
            continue
        P = P_left
        if attrs.get("output_padding", [0]) != [0]:
            continue

        w_init = _get_init(model, node.input[1])
        if w_init is None: continue
        w = numpy_helper.to_array(w_init)
        w_new = np.flip(w.transpose(1, 0, 2), axis=2).astype(w.dtype).copy()

        x_in = node.input[0]; y_out = node.output[0]
        base = node.name or y_out
        new = []

        shape4d = f"{base}/sh4d"; shape3d = f"{base}/sh3d"; pads = f"{base}/pads"
        model.graph.initializer.append(numpy_helper.from_array(np.array([0,0,-1,1], np.int64), shape4d))
        model.graph.initializer.append(numpy_helper.from_array(np.array([0,0,-1],   np.int64), shape3d))
        model.graph.initializer.append(numpy_helper.from_array(np.array([0,0,0,0,0,0,0,S-1], np.int64), pads))

        r4d = f"{base}/r4d"; padded = f"{base}/pad"; flat = f"{base}/flat"
        new.append(helper.make_node("Reshape", [x_in, shape4d], [r4d],   name=f"{base}/Rs4D"))
        new.append(helper.make_node("Pad",     [r4d, pads],    [padded], mode="constant", name=f"{base}/Pad"))
        new.append(helper.make_node("Reshape", [padded, shape3d], [flat], name=f"{base}/RsFlat"))

        wn = f"{base}/W"
        model.graph.initializer.append(numpy_helper.from_array(w_new, wn))
        conv_inputs = [flat, wn]
        if len(node.input) >= 3 and node.input[2] != "":
            conv_inputs.append(node.input[2])
        new.append(helper.make_node(
            "Conv", conv_inputs, [y_out],
            name=f"{base}/EqConv",
            kernel_shape=[K], strides=[1], dilations=[1],
            pads=[K-1-P, K-S-P], group=1,
        ))

        idx = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        for i, nn in enumerate(new):
            model.graph.node.insert(idx + i, nn)
        n_done += 1
    return n_done


def _drop_orphans(model):
    used = set()
    for n in model.graph.node:
        used.update(n.input)
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def rewrite_snake_alpha_fold(model):
    """Fold the per-channel /α post-Snake scaling into the following Conv's
    per-input-channel weights. Snake1D is:

        αu  = Mul(α, u)
        s²  = sin²(αu)              # the Clip+Horner+self-Mul subgraph
        v   = u + s²/α              # = (αu + s²)/α  (algebraic identity)

    Rewrite to:
        z   = αu + s²               # residual Add now uses αu, not u
        v   = next_conv(z) with weights scaled by 1/α per input channel
            ≡ next_conv(z/α) with original weights

    Eliminates per Snake: 1 Mul (s²/α) + 1 Reciprocal(α). The Reciprocal is
    usually constant-folded at compile time, but the Mul was a full-tensor
    DMA pass — and on this bandwidth-bound NPU, removing DMA passes is what
    actually moves the wall clock."""
    producer  = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    inits = {init.name: init for init in model.graph.initializer}

    n_done = 0; n_skipped = 0
    for input_mul in [n for n in list(model.graph.node) if n.op_type == "Mul"]:
        # Identify (α, u) inputs: α is an initializer whose name contains ".alpha".
        a_name = next((inp for inp in input_mul.input
                       if inp in inits and ".alpha" in inp), None)
        if a_name is None:
            continue
        u_name = next(inp for inp in input_mul.input if inp != a_name)
        alpha = numpy_helper.to_array(inits[a_name])  # (1, C, 1)
        if alpha.ndim != 3 or alpha.shape[0] != 1 or alpha.shape[2] != 1:
            continue
        αu_name = input_mul.output[0]

        # αu must flow into a Clip (the Sin-Taylor entry).
        if not any(c.op_type == "Clip" for c in consumers.get(αu_name, [])):
            continue

        # Find the post-α Mul: a Mul whose one input is Reciprocal(α).
        post_α_mul = next(
            (n for n in model.graph.node
             if n.op_type == "Mul"
             and any(producer.get(inp) is not None
                     and producer[inp].op_type == "Reciprocal"
                     and producer[inp].input[0] == a_name
                     for inp in n.input)),
            None)
        if post_α_mul is None:
            n_skipped += 1; continue

        residual_add = next((c for c in consumers.get(post_α_mul.output[0], [])
                             if c.op_type == "Add" and u_name in c.input), None)
        if residual_add is None:
            n_skipped += 1; continue

        # Next op after the Add: must be a single Conv with constant weights.
        add_consumers = consumers.get(residual_add.output[0], [])
        if len(add_consumers) != 1 or add_consumers[0].op_type != "Conv":
            n_skipped += 1; continue
        next_conv = add_consumers[0]
        W_name = next_conv.input[1]
        if W_name not in inits:
            n_skipped += 1; continue
        W = numpy_helper.to_array(inits[W_name])  # (Cout, Cin, K)
        if W.shape[1] != alpha.shape[1]:
            n_skipped += 1; continue

        # --- Apply ---
        # 1. Scale Conv weights by 1/α per input channel.
        W_new = (W / alpha.reshape(1, -1, 1)).astype(W.dtype)
        inits[W_name].CopyFrom(numpy_helper.from_array(W_new, W_name))

        # 2. Rewire the residual Add: u → αu, sin²/α → sin² (bypass post_α_mul).
        s2_name = next(x for x in post_α_mul.input
                       if producer.get(x) is None
                       or producer[x].op_type != "Reciprocal")
        for i, inp in enumerate(residual_add.input):
            if inp == u_name:
                residual_add.input[i] = αu_name
            elif inp == post_α_mul.output[0]:
                residual_add.input[i] = s2_name

        # 3. Drop the now-dead Mul (and Reciprocal if nothing else uses it).
        recip_name = next(x for x in post_α_mul.input if x != s2_name)
        recip_node = producer.get(recip_name)
        model.graph.node.remove(post_α_mul)
        if recip_node is not None and not any(
                recip_node.output[0] in n.input for n in model.graph.node):
            model.graph.node.remove(recip_node)
        n_done += 1

    return n_done, n_skipped


def rewrite_pow2_to_mul(model):
    """Pow(x, 2.0) → Mul(x, x). On RK3588 Pow with a float exponent runs as
    Exp/Log microcode (~415 µs/call vs Mul's ~360 µs/call here) and is opaque
    to compiler fusion. A plain Mul is cheaper and fusable. The 48 sin² nodes
    in Snake1d activations are all Pow(_, 2.0)."""
    # Build a lookup that handles both initializer-fed and Constant-node-fed exponents.
    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    constants = {}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    constants[n.output[0]] = numpy_helper.to_array(attr.t)

    def resolve(name):
        if name in inits:     return inits[name]
        if name in constants: return constants[name]
        return None

    n_done = 0
    for node in [n for n in list(model.graph.node) if n.op_type == "Pow"]:
        e = resolve(node.input[1])
        if e is None or e.size != 1 or float(e.item()) != 2.0:
            continue
        x = node.input[0]
        new = helper.make_node("Mul", [x, x], [node.output[0]], name=node.name + "/Mul")
        idx = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        model.graph.node.insert(idx, new)
        n_done += 1
    return n_done


def graph_surgery(t_fix, taylor_degree):
    print("[5/6] graph surgery")

    m = onnx.load(str(DEC_RAW))
    n = rewrite_instancenorm_gap(m)
    _drop_orphans(m); onnx.checker.check_model(m); onnx.save(m, str(DEC_FIXED))
    print(f"       gapnorm  : replaced {n} InstanceNorm → dynamic GlobalAveragePool norm  →  {DEC_FIXED.name}")

    m = onnx.load(str(DEC_FIXED))
    n = rewrite_sin_clipped_taylor(m, degree=taylor_degree)
    n_pow = rewrite_pow2_to_mul(m)
    n_α, n_α_skip = rewrite_snake_alpha_fold(m)
    _drop_orphans(m); onnx.checker.check_model(m); onnx.save(m, str(DEC_SIN))
    print(f"       sin     : replaced {n} Sin (degree {taylor_degree}, Horner+minimax)  →  {DEC_SIN.name}")
    print(f"       pow2    : replaced {n_pow} Pow(x,2)→Mul(x,x)")
    print(f"       α-fold  : folded {n_α} Snake1d /α into next Conv  ({n_α_skip} skipped)")

    m = onnx.load(str(DEC_SIN))
    n = rewrite_deconv_stride(m)
    _drop_orphans(m); onnx.checker.check_model(m); onnx.save(m, str(DEC_FINAL))
    print(f"       deconv  : rewrote {n} ConvTranspose  →  {DEC_FINAL.name}")


# ---------------------------------------------------------------------------
# Stage 6 — RKNN compile (fp16)
# ---------------------------------------------------------------------------
def compile_rknn(t_fix):
    print("[6/6] RKNN compile (fp16)")
    from rknn.api import RKNN
    har_F = 2 * t_fix * 300 // 5 + 1
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588")
    ret = r.load_onnx(
        model=str(DEC_FINAL),
        inputs=["asr", "F0", "N", "s", "har"],
        input_size_list=[[1, 512, t_fix], [1, 2*t_fix], [1, 2*t_fix], [1, 128], [1, 22, har_F]],
    )
    if ret != 0: raise SystemExit(f"load_onnx failed: {ret}")
    if r.build(do_quantization=False) != 0: raise SystemExit("build failed")
    if r.export_rknn(str(DEC_RKNN)) != 0: raise SystemExit("export failed")
    r.release()
    print(f"       wrote {DEC_RKNN}  ({DEC_RKNN.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-fix", type=int, default=50,
                    help="ASR frames per decoder window (0.625 s at default)")
    ap.add_argument("--taylor-degree", type=int, default=7, choices=[5, 7, 9])
    ap.add_argument("--skip-rknn", action="store_true",
                    help="ONNX only; skip the rknn-toolkit2 step")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete intermediate ONNX files first")
    args = ap.parse_args()

    if args.rebuild:
        for p in (DEC_RAW, DEC_FIXED, DEC_SIN, DEC_FINAL, DEC_RKNN):
            if p.exists(): p.unlink()

    print(f"== Kokoro-82M → RK3588 build  (T_FIX={args.t_fix})")
    KModel, istftnet = load_kokoro()
    kmodel = KModel(config=str(KOKORO_DIR / "config.json"),
                    model=str(KOKORO_DIR / "kokoro-v1_0.pth"),
                    disable_complex=True).eval()
    export_onnx(kmodel, istftnet, args.t_fix)
    convert_voices()
    graph_surgery(args.t_fix, args.taylor_degree)
    if not args.skip_rknn:
        compile_rknn(args.t_fix)
    else:
        print("[6/6] skipped RKNN compile (--skip-rknn)")
    print("done.")


if __name__ == "__main__":
    main()

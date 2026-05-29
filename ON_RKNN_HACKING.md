# Hacking Kokoro to run on RKNN

To compile and run the Kokoro decoder efficiently on the Rockchip RK3588 NPU, several graph surgery rewrites and optimizations are applied. With these modifications (including data-dependent normalization), the end-to-end pipeline runs at approximately 2x Real-Time Factor (RTF).

Below are the key graph rewrites and architectural adjustments used.

## 1. InstanceNorm & Data-Dependent Normalization Layouts
RKNN lowers InstanceNormalization as a LayerNorm-style operator that expects a channel-last (B, T, C) layout. However, the native NPU layout is tiled (NC1HWC2). 
To bridge this, RKNN inserts Transpose operations to flip between layouts around every InstanceNorm. At larger time axes (such as T = 12001 at the final upsampled rate), these transposes exceed the NPU transpose engine limits and fall back to CPU execution (~33 ms each). 
We use data-dependent normalization, which achieves approximately 2x RTF.

## 2. Sin to Minimax Polynomial in Horner Form
librknnrt has no native NPU Sin operator, causing every Snake1d activation (x + sin²(αx)/α) to fall back to the CPU and trigger additional transposes.
We replace Sin with an odd-power polynomial approximation in Horner form over the interval [-π, π]:
- x_c = Clip(x, -π, +π)
- x² = x_c * x_c
- sin(x_c) ≈ x_c * (c1 + x² * (c3 + x² * (c5 + x² * c7)))

This 7th-degree minimax polynomial approximation requires only 5 Multiplications and 3 Additions. Using Horner form reuses a single running accumulator instead of creating intermediate tensors for power terms, saving DMA passes through SRAM on the bandwidth-bound NPU.

## 3. Pow(x, 2.0) to Mul(x, x)
The 48 sin² nodes in the Snake1d activations originally used Pow(_, 2.0). On the RK3588, Pow with a float exponent runs as Exp/Log microcode (~415 µs/call vs Mul's ~360 µs/call) and cannot be fused. Replacing it with Mul(x, x) is computationally cheaper and enables compile-time operator fusion.

## 4. Folding Snake1d Divisor into Conv1d
Snake1d is defined as:
v = u + (1/α) * sin²(α * u)

This can be rewritten algebraically as:
v = (α * u + sin²(α * u)) / α

By scaling the input to Snake1d by α and absorbing the division by α into the next Convolution's weights (W_new = W / α), we completely eliminate the explicit Reciprocal and Division operators. Since the upsampled post-Snake tensor is large, avoiding memory reads of this tensor saves significant DMA bandwidth.

## 5. ConvTranspose Stride Reformulation
librknnc only supports deconvolution strides in {1, 2, 4, 8}, whereas Kokoro's upsampling blocks use strides of 10 and 6. To avoid slow CPU-based padding fallbacks, we reformulate the ConvTranspose operations into an interleaving pattern followed by a stride-1 Conv1d:
1. Reshape input X (B, C, T) to (B, C, T, 1)
2. Zero-pad the last axis by (0, Stride - 1) to (B, C, T, Stride)
3. Reshape back to (B, C, T * Stride)
4. Apply a standard Conv1d with weights rearranged as W.transpose(0, 1).flip(2)

This bypasses the CPU exConvTransposePad fallback entirely.

## 6. Decoder Windowing
The compiled RKNN decoder has a fixed input shape (T_FIX = 50 ASR frames per call, or ~0.625 seconds of audio). 
To process longer inputs without boundary pops:
- Adjacent windows are decoded with 2 frames of context on each side. The corrupted edge samples are discarded, emitting only the middle 46 frames.
- Overlapping regions are stitched using a 16-sample linear crossfade to smooth out numerical jitter.

## Diagnostics
To profile operator execution and check CPU/NPU target placement:
```bash
RKNN_LOG_LEVEL=5 python3 infer.py 2> run.log
grep -E "FLOAT16 +(CPU|NPU)" run.log
grep "Operator Time Consuming" run.log -A 30
```

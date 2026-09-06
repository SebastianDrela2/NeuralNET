using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// Superoptimized 4D tensor (Batch, Channels, Height, Width) with zero-alloc L1-cached memory pooling.
/// </summary>
public unsafe class CnnMatrix : IDisposable
{
    public const int Alignment = 16;
    private const int ByteAlignment = Alignment * sizeof(float);

    private static readonly ConcurrentBag<CnnMatrix> _pool = [];
    private static readonly int CommonAllocatedLength = 25690112;

    public float* Pointer;
    public int Batch;
    public int Channels;
    public int Height;
    public int Width;
    public bool ReadOnly;
    public int UnsafeSize;

    public int StrideW => 1;
    public int StrideH => Width;
    public int StrideC => Width * Height;
    public int StrideN => Width * Height * Channels;

    private bool _inUse = true;

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, bool readOnly = false)
    {
        if (!_pool.TryTake(out var item))
        {
            return new CnnMatrix(batch, channels, height, width, readOnly);
        }

        item.Resize(batch, channels, height, width);
        return item;
    }

    private CnnMatrix(int batch, int channels, int height, int width, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        ReadOnly = readOnly;
        UnsafeSize = batch * channels * height * width;

        if (UnsafeSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException($"Tensor size {UnsafeSize} exceeds pool buffer size {CommonAllocatedLength}.");
        }

        Pointer = (float*)NativeMemory.AlignedAlloc((nuint)(CommonAllocatedLength * sizeof(float)), (nuint)ByteAlignment);
        _inUse = true;
        Clear();
    }

    public void Resize(int batch, int channels, int height, int width)
    {
        int newSize = batch * channels * height * width;
        if (newSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException($"Tensor size {newSize} exceeds pool buffer size {CommonAllocatedLength}.");
        }

        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        UnsafeSize = newSize;
        _inUse = true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetIndex(int batch, int channel, int y, int x)
        => (batch * StrideN) + (channel * StrideC) + (y * StrideH) + x;

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref Pointer[GetIndex(batch, channel, y, x)];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPointer(int batch, int channel) => Pointer + (batch * StrideN) + (channel * StrideC);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int batch, int channel, int y) => Pointer + (batch * StrideN) + (channel * StrideC) + (y * StrideH);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Fill(float value)
    {
        new Span<float>(Pointer, UnsafeSize).Fill(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyFrom(CnnMatrix other)
    {
        NativeMemory.Copy(other.Pointer, Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public NeuralMatrix Im2Col(int kernelH, int kernelW, int stride, int padding)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;
        int totalPatches = Batch * outH * outW;

        var colMatrix = NeuralMatrix.GetOrCreate(totalPatches, patchSize);
        float* colPtr = colMatrix.Pointer;
        int colStride = colMatrix.ColumnsStride;

        bool needsPadding = padding > 0;
        using var padded = needsPadding ? GetOrCreate(Batch, Channels, paddedH, paddedW) : null;
        if (needsPadding) padded.Clear();

        float* baseSrcPtr = Pointer;
        float* basePaddedPtr = needsPadding ? padded!.Pointer : baseSrcPtr;
        int srcStrideH = Width;
        int targetPaddedW = needsPadding ? paddedW : Width;

        if (needsPadding)
        {
            nuint rowBytes = (nuint)Width * sizeof(float);
            Parallel.For(0, Batch, b =>
            {
                long batchSrcOffset = b * Channels * Height * Width;
                long batchPadOffset = b * Channels * paddedH * paddedW;

                for (int c = 0; c < Channels; c++)
                {
                    float* srcPtr = baseSrcPtr + batchSrcOffset + c * Height * Width;
                    float* dstPtr = basePaddedPtr + batchPadOffset + c * paddedH * paddedW;

                    for (int y = 0; y < Height; y++)
                    {
                        float* srcRow = srcPtr + y * srcStrideH;
                        float* dstRow = dstPtr + (y + padding) * targetPaddedW + padding;
                        NativeMemory.Copy(srcRow, dstRow, rowBytes);
                    }
                }
            });
        }

        int spatialPadStride = paddedH * targetPaddedW;

        Parallel.For(0, Batch, b =>
        {
            int batchPatchBase = b * outH * outW;
            long batchPadOffset = b * Channels * spatialPadStride;
            long batchColOffset = (long)batchPatchBase * colStride;
            float* batchColPtr = colPtr + batchColOffset;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = oh * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    float* dstRow = batchColPtr + (patchRowBase + ow) * colStride;
                    int colIdx = 0;

                    for (int c = 0; c < Channels; c++)
                    {
                        float* channelPaddedPtr = basePaddedPtr + batchPadOffset + c * spatialPadStride;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* srcRow = channelPaddedPtr + (startY + ky) * targetPaddedW + startX;

                            if (kernelW == 3)
                            {
                                dstRow[colIdx] = srcRow[0];
                                dstRow[colIdx + 1] = srcRow[1];
                                dstRow[colIdx + 2] = srcRow[2];
                                colIdx += 3;
                            }
                            else if (kernelW == 1)
                            {
                                dstRow[colIdx++] = srcRow[0];
                            }
                            else
                            {
                                nuint copyBytes = (nuint)kernelW * sizeof(float);
                                NativeMemory.Copy(srcRow, dstRow + colIdx, copyBytes);
                                colIdx += kernelW;
                            }
                        }
                    }
                }
            }
        });

        return colMatrix;
    }

    public void Col2Im(NeuralMatrix colGradients, int kernelH, int kernelW, int stride, int padding, float scale = 1.0f)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);
        paddedGrad.Clear();

        float* colPtr = colGradients.Pointer;
        int colStride = colGradients.ColumnsStride;
        float* gradPtr = paddedGrad.Pointer;
        int kernelSpatial = kernelH * kernelW;

        Parallel.For(0, Batch, b =>
        {
            long batchOffsetGrad = b * paddedGrad.StrideN;
            int batchPatchBase = b * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = (b * outH + oh) * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    float* colRow = colPtr + (patchRowBase + ow) * colStride;

                    for (int c = 0; c < Channels; c++)
                    {
                        long channelOffsetGrad = batchOffsetGrad + c * paddedGrad.StrideC;
                        int channelOffsetCol = c * kernelSpatial;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* dstGrad = gradPtr + channelOffsetGrad + (startY + ky) * paddedGrad.StrideH + startX;
                            float* srcCol = colRow + channelOffsetCol + ky * kernelW;

                            int kx = 0;
                            if (Vector512.IsHardwareAccelerated)
                            {
                                var vScale512 = Vector512.Create(scale);
                                int vecLimit = kernelW - (kernelW % 16);
                                for (; kx < vecLimit; kx += 16)
                                {
                                    var vDst = Vector512.Load(dstGrad + kx);
                                    var vSrc = Vector512.Load(srcCol + kx);
                                    vDst = Vector512.FusedMultiplyAdd(vSrc, vScale512, vDst);
                                    vDst.Store(dstGrad + kx);
                                }
                            }

                            for (; kx < kernelW; kx++)
                            {
                                dstGrad[kx] += srcCol[kx] * scale;
                            }
                        }
                    }
                }
            }
        });

        nuint rowBytes = (nuint)Width * sizeof(float);
        float* baseDstPtr = Pointer;
        float* basePaddedGradPtr = paddedGrad.Pointer;

        Parallel.For(0, Batch, b =>
        {
            long batchSrcOffset = b * paddedGrad.StrideN;
            long batchDstOffset = (long)b * Channels * Height * Width;

            for (int c = 0; c < Channels; c++)
            {
                float* srcChannel = basePaddedGradPtr + batchSrcOffset + c * paddedGrad.StrideC;
                float* dstChannel = baseDstPtr + batchDstOffset + c * Height * Width;

                for (int y = 0; y < Height; y++)
                {
                    float* srcPtr = srcChannel + (y + padding) * paddedW + padding;
                    float* dstPtr = dstChannel + y * Width;
                    NativeMemory.Copy(srcPtr, dstPtr, rowBytes);
                }
            }
        });
    }

    public void Dispose()
    {
        _inUse = false;
        _pool.Add(this);
    }

    public static void ClearPool()
    {
        while (_pool.TryTake(out var item))
        {
            if (item.Pointer != null)
            {
                NativeMemory.AlignedFree(item.Pointer);
                item.Pointer = null;
            }
        }
    }
}

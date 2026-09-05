using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Utils;

namespace NeutralNET.Test.Data;

public class LetterDataLoader : DataLoaderBase
{
    private static readonly string[] FontFamilies =
    [
        "Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma",
        "Consolas", "Courier New", "Comic Sans MS", "Impact", "Trebuchet MS",
        "Palatino Linotype", "Segoe UI", "Lucida Console", "Garamond", "Century Gothic"
    ];

    private static readonly FontStyle[] SupportedStyles =
    [
        FontStyle.Regular,
        FontStyle.Bold,
        FontStyle.Italic,
        FontStyle.Bold | FontStyle.Italic
    ];

    public override int ImageScale => 28;
    public override string DatasetName => "LetterData";
    public override int NumClasses => 26;

    protected override (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                        List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples)
    {
        var (trainImages, trainLabels) = LoadFlattenedDataSet(DataSetType.Train, batchSize, maxTrainSamples);
        var (testImages, testLabels) = LoadFlattenedDataSet(DataSetType.Test, batchSize, maxTestSamples);

        return (trainImages, trainLabels, testImages, testLabels);
    }

    protected override void AddToBatches(float[][] images, int[] labels, int batchSize,
                                         List<CnnMatrix> outImages, List<NeuralMatrix> outLabels)
    {
        int scale = ImageScale;
        int numSamples = images.Length;

        for (int start = 0; start < numSamples; start += batchSize)
        {
            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            if (currentBatchSize < batchSize) continue;

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, Channels, scale, scale, readOnly: true);
            var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, NumClasses);

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = start + i;
                float[] pixels = images[idx];

                for (int c = 0; c < Channels; c++)
                {
                    int offset = c * scale * scale;
                    for (int y = 0; y < scale; y++)
                    {
                        for (int x = 0; x < scale; x++)
                        {
                            imgMat[i, c, y, x] = pixels[offset + y * scale + x];
                        }
                    }
                }

                int label = labels[idx];
                lblMat.Set(i, label, 1.0f);
            }

            outImages.Add(imgMat);
            outLabels.Add(lblMat);
        }
    }

    private (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadFlattenedDataSet(
        DataSetType dataSetType, int batchSize, int maxSamples)
    {
        bool isTrain = dataSetType == DataSetType.Train;
        var rng = Random.Shared;

        // Step 1: Pre-render all font/style variations ONCE (Exact total GDI+ passes = 15 fonts * 4 styles = 60 passes)
        var fontCache = new List<PixelStructRGB[]>();

        foreach (var fontName in FontFamilies)
        {
            var stylesToRender = isTrain ? SupportedStyles : [FontStyle.Regular];
            foreach (var style in stylesToRender)
            {
                try
                {
                    // Render clean base glyphs without GDI transformations (we do transformations lightning-fast in RAM if needed)
                    var set = GraphicsUtils.GetLettersDataSetRGB(fontName, applyTransformation: isTrain, style: style);
                    fontCache.Add(set);
                }
                catch
                {
                    // Skip unsupported font/style combos safely
                }
            }
        }

        if (fontCache.Count == 0)
        {
            fontCache.Add(GraphicsUtils.GetLettersDataSetRGB("Arial", applyTransformation: isTrain, style: FontStyle.Regular));
        }

        // Step 2: Sample directly from memory array (Instantaneous in-memory operations)
        var allSamples = new List<PixelStructRGB>(maxSamples);

        while (allSamples.Count < maxSamples)
        {
            var randomSet = fontCache[rng.Next(fontCache.Count)];
            allSamples.AddRange(randomSet);
        }

        var selectedData = allSamples.Take(maxSamples).ToArray();
        rng.Shuffle(selectedData);

        var labelArray = selectedData.Select(x => x.Label).ToArray();
        var imageArray = selectedData.Select(x => x.Flat.ToArray()).ToArray();

        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();

        AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);

        Console.WriteLine($"{DatasetName}: Loaded {selectedData.Length} {dataSetType} samples from {fontCache.Count} font/style variants.");

        return (batchImages, batchLabels);
    }

    private enum DataSetType
    {
        Train,
        Test
    }
}

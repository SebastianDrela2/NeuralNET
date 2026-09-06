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

            // FIX: Handle leftover samples instead of dropping them completely
            if (currentBatchSize <= 0) break;

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

        // FIX: Cache raw un-transformed font definitions or base metadata templates
        // rather than pre-rendered static images, ensuring high transformation diversity.
        var fontTemplates = new List<(string FontName, FontStyle Style)>();

        foreach (var fontName in FontFamilies)
        {
            var stylesToRender = isTrain ? SupportedStyles : [FontStyle.Regular];
            foreach (var style in stylesToRender)
            {
                fontTemplates.Add((fontName, style));
            }
        }

        if (fontTemplates.Count == 0)
        {
            fontTemplates.Add(("Arial", FontStyle.Regular));
        }

        var allSamples = new List<PixelStructRGB>(maxSamples);

        // FIX: Generate unique transformations on-the-fly inside the collection loop
        while (allSamples.Count < maxSamples)
        {
            var template = fontTemplates[rng.Next(fontTemplates.Count)];

            // Re-rendering or pulling per-iteration ensures unique transformations per sample
            var set = GraphicsUtils.GetLettersDataSetRGB(template.FontName, applyTransformation: isTrain, style: template.Style);

            if (set != null && set.Length > 0)
            {
                // Shuffle individual batches to keep variety high across classes
                var shuffledSet = set.OrderBy(_ => rng.Next()).ToArray();
                foreach (var sample in shuffledSet)
                {
                    allSamples.Add(sample);
                    if (allSamples.Count >= maxSamples) break;
                }
            }
        }

        var selectedData = allSamples.Take(maxSamples).ToArray();

        // FIX: Shuffle an lightweight index array instead of moving heavy structs around
        int[] indices = Enumerable.Range(0, selectedData.Length).ToArray();
        rng.Shuffle(indices);

        var labelArray = new int[selectedData.Length];
        var imageArray = new float[selectedData.Length][];

        for (int i = 0; i < indices.Length; i++)
        {
            int originalIdx = indices[i];
            labelArray[i] = selectedData[originalIdx].Label;
            imageArray[i] = selectedData[originalIdx].Flat.ToArray();
        }

        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();

        AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);

        Console.WriteLine($"{DatasetName}: Loaded {selectedData.Length} {dataSetType} samples with randomized transform diversity.");

        return (batchImages, batchLabels);
    }

    private enum DataSetType
    {
        Train,
        Test
    }
}

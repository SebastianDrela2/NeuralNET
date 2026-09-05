using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Utils;

namespace NeutralNET.Test.Data;

public class LetterDataLoader : DataLoaderBase
{
    private static readonly string[] _fontNames =
        ["Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma"];

    public override int ImageScale => 16;
    public override string DatasetName => "LetterData";
    public override int NumClasses => 26; // 26 uppercase letters A-Z

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

            // Skip incomplete tail batches to keep strictly uniform batch sizes
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
                lblMat.Set(i, label, 1.0f); // One-hot encoding
            }

            outImages.Add(imgMat);
            outLabels.Add(lblMat);
        }
    }

    private (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadFlattenedDataSet(
        DataSetType dataSetType, int batchSize, int maxSamples)
    {
        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();

        int passes = dataSetType == DataSetType.Train ? 5 : 1;
        bool applyTransform = dataSetType == DataSetType.Train;

        var allSamples = new List<PixelStructRGB>();

        for (int i = 0; i < passes; i++)
        {
            foreach (var font in _fontNames)
            {
                var fontData = GraphicsUtils.GetLettersDataSetRGB(font, applyTransformation: applyTransform);
                allSamples.AddRange(fontData);
            }
        }

        var sampleArray = allSamples.ToArray();
        Random.Shared.Shuffle(sampleArray);

        var selectedData = sampleArray.Take(maxSamples).ToArray();

        var labelArray = selectedData.Select(x => x.Label).ToArray();
        var imageArray = selectedData.Select(x => x.Flat.ToArray()).ToArray();

        AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);

        Console.WriteLine($"{DatasetName}: Loaded {selectedData.Length} {dataSetType} samples in {batchImages.Count} batches");

        // Return the batched matrices directly
        return (batchImages, batchLabels);
    }

    private enum DataSetType
    {
        Train,
        Test
    }
}

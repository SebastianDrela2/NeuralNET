using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 32;

    static void Main() => RunCnnNetwork();
    public static void RunDuniel()
    {
        var loader = DataLoaderFactory.Create(DataSourceType.Letters);

        var dataSet = loader.LoadCompleteDataset(
           batchSize: BatchSize,
           maxTrainSamples: 3000,
           maxTestSamples: 300
        );

        char[] chars = [
            ..EnumerateChars('A').Take(loader.NumClasses)
        ];
        CnnDisplayWriter displayWriter = new(chars, dataSet.TrainImages.Count);
        displayWriter.Clear();

        Span<float> buff = stackalloc float[(1 + loader.NumClasses) * BatchSize];
        float accuracyVal = 0;

        while (true)
        {
            int offset = 0;
            for (int y = 0; y < BatchSize; ++y)
            {
                var expected = Random.Shared.Next(loader.NumClasses);
                buff[offset++] = expected;

                float acc = 1;

                var prob = buff[offset..(offset += loader.NumClasses)];
                for (int x = 0; x < loader.NumClasses; ++x)
                {
                    float v = Random.Shared.NextSingle() * acc;
                    prob[x] = v;
                    acc -= v;
                }
                Random.Shared.Shuffle(prob);
            }

            var rng = Random.Shared.NextSingle() * Random.Shared.NextSingle() * Random.Shared.NextSingle() * Random.Shared.NextSingle();
            accuracyVal += rng * (1 - accuracyVal);
            displayWriter.Accuracy = float.Round(loader.NumClasses * accuracyVal) / loader.NumClasses;
            displayWriter.TotalLoss = dataSet.TrainImages.Count * Random.Shared.NextSingle();
            displayWriter.Update(buff);
            Thread.Sleep(250);
        }
    }

    public static void RunCnnNetwork()
    {
        // 1. Data Loading
        var datasetKey = DataSourceType.Letters;
        var loader = DataLoaderFactory.Create(datasetKey);
        var config = TrainingConfig.CreateDefault(loader.NumClasses);

        config.DatasetKey = datasetKey;

        var dataSet = loader.LoadCompleteDataset(
            batchSize: config.BatchSize,
            maxTrainSamples: config.MaxTrainSamples,
            maxTestSamples: config.MaxTestSamples
        );

        // 3. Network Initialization
        using var network = new CnnBuilder<Architecture>()
            .WithCnnConfig(config.CnnArchitecture)
            .WithDenseConfig(config.DenseConfig)
            .WithInputSize(loader.ImageScale, loader.ImageScale, 3)
            .Build();

        var validator = new CnnValidator();
        var trainer = new CnnTrainer(network, validator, config);
        trainer.Train(dataSet, loader.NumClasses);

        Console.WriteLine("\n=== FINAL EVALUATION ===");
        var finalResult = validator.Validate(network, dataSet.TestImages, dataSet.TestLabels);
        validator.PrintResults(finalResult);

        CleanupDataset(dataSet);
    }

    private static void CleanupDataset(NeuralDataset dataSet)
    {
        foreach (var img in dataSet.TrainImages) img?.Dispose();
        foreach (var lbl in dataSet.TrainLabels) lbl?.Dispose();
        foreach (var img in dataSet.TestImages) img?.Dispose();
        foreach (var lbl in dataSet.TestLabels) lbl?.Dispose();
    }

    private static void TrainAndRenderTable(
        CnnNetwork<Architecture> network,
        CnnValidator validator,
        NeuralDataset dataSet,
        string datasetName,
        int numClasses)
    {
        var learningRate = 0.0003f;
        var earlyStopPatience = 300;
        var targetAccuracy = 0.98f;

        var testImages = dataSet.TestImages;
        var testLabels = dataSet.TestLabels;

        char[] chars = [.. EnumerateChars('A').Take(numClasses)];
        CnnDisplayWriter display = new(chars, dataSet.TrainImages.Count);

        display.Clear();
        Span<float> results = new float[(1 + numClasses) * BatchSize];
        while (true)
        {
            float totalLoss = 0f;
            for (int batchIdx = 0; batchIdx < dataSet.TrainImages.Count; batchIdx++)
            {
                totalLoss += network.TrainBatch(dataSet.TrainImages[batchIdx], dataSet.TrainLabels[batchIdx], learningRate);
            }

            display.TotalLoss = totalLoss;

            var result = validator.Validate(network, testImages, testLabels);
            display.Accuracy = result.Accuracy;

            int offset = 0;
            if (testImages.Count > 0)
            {
                var sampleBatch = testImages[0];
                var sampleLabels = testLabels[0];
                using var pred = network.Forward(sampleBatch);

                var numSamples = Math.Min(numClasses, sampleBatch.Batch);
                for (int i = 0; i < numSamples; i++)
                {
                    results[offset++] = GetActualLabelFromRow(sampleLabels, i, numClasses);

                    var probs = results[offset..(offset += numClasses)];
                    pred.GetRowSpan(i).CopyTo(probs);
                }
            }

            display.Update(results[..offset]);

            if (display.Accuracy >= targetAccuracy)
            {
                Console.WriteLine($"\n🎯 Target accuracy {targetAccuracy:P2} reached! Stopping early at epoch {display.Epoch}");
                break;
            }

            if (display.EpochsSinceBest >= earlyStopPatience && display.BestAccuracy > 0.4f)
            {
                Console.WriteLine($"\n⏹️ No improvement for {earlyStopPatience} epochs. Stopping early at epoch {display.Epoch}");
                Console.WriteLine($"Best accuracy: {display.BestAccuracy:P2}");
                break;
            }
        }
    }

    private static int GetActualLabelFromRow(NeuralMatrix labelMatrix, int row, int numClasses)
    {
        int maxIndex = -1;
        float maxValue = -1f;

        for (int i = 0; i < labelMatrix.UsedColumns; i++)
        {
            float val = labelMatrix.At(row, i);
            if (val > maxValue)
            {
                maxValue = val;
                maxIndex = i;
            }
        }

        if (maxIndex >= numClasses)
        {
            maxIndex = numClasses - 1;
        }

        return maxIndex;
    }

    public static IEnumerable<char> CharRange(char first, char last)
    {
        if (first > last) (first, last) = (last, first);
        return EnumerateChars(first).TakeWhile(c => c <= last);
    }

    public static IEnumerable<char> EnumerateChars(char start)
    {
        for (char c = start; ; ++c)
        {
            yield return c;
            if (c is char.MaxValue) break;
        }
    }
}

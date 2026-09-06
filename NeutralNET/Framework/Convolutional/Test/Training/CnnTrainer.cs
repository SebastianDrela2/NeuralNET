using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnTrainer
{
    private readonly CnnNetwork<Architecture> _network;
    private readonly CnnValidator _validator;
    private readonly TrainingConfig _config;

    public CnnTrainer(CnnNetwork<Architecture> network, CnnValidator validator, TrainingConfig config)
    {
        _network = network;
        _validator = validator;
        _config = config;
    }

    public void Train(NeuralDataset dataSet, int numClasses)
    {
        try
        {
            _network.LoadWeights(_config.DatasetKey, _config.CheckpointDir);
            Console.WriteLine($"[INFO] Successfully loaded existing weights for {_config.DatasetKey}.");
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"[INFO] No existing weights found for {_config.DatasetKey}. Starting fresh training.");
        }

        char[] chars = [.. EnumerateChars('A').Take(numClasses)];
        CnnDisplayWriter display = new(chars, dataSet.TrainImages.Count);

        display.Clear();
        Span<float> results = new float[(1 + numClasses) * _config.BatchSize];

        var indexes = Enumerable.Range(0, dataSet.TrainImages.Count).ToArray();

        while (true)
        {
            Random.Shared.Shuffle(indexes);
            float totalLoss = 0;
            for (int batchIdx = 0; batchIdx < dataSet.TrainImages.Count; batchIdx++)
            {
                var index = indexes[batchIdx];
                float loss = _network.TrainBatch(dataSet.TrainImages[index], dataSet.TrainLabels[index], _config.LearningRate);
                totalLoss += loss;
            }
            display.TotalLoss = totalLoss;

            var result = _validator.Validate(_network, dataSet.TestImages, dataSet.TestLabels);
            display.Accuracy = result.Accuracy;

            int offset = 0;
            if (dataSet.TestImages.Count > 0)
            {
                var sampleBatch = dataSet.TestImages[0];
                var sampleLabels = dataSet.TestLabels[0];
                using var pred = _network.Forward(sampleBatch);

                var numSamples = Math.Min(numClasses, sampleBatch.Batch);
                for (int i = 0; i < numSamples; i++)
                {
                    results[offset++] = GetActualLabelFromRow(sampleLabels, i, numClasses);

                    var probs = results[offset..(offset += numClasses)];
                    pred.GetRowSpan(i).CopyTo(probs);
                }
            }

            display.Update(results[..offset]);


            if (display.Accuracy == display.BestAccuracy)
            {
                _network.SaveWeights(_config.DatasetKey, _config.CheckpointDir);
                Console.WriteLine($"Saved Weights!");
            }

            if (display.Accuracy >= _config.TargetAccuracy)
            {
                Console.WriteLine($"\n🎯 Target accuracy {_config.TargetAccuracy:P2} reached! Stopping early at epoch {display.Epoch}");
                break;
            }

            if (display.EpochsSinceBest >= _config.EarlyStopPatience && display.BestAccuracy > 0.4f)
            {
                Console.WriteLine($"\n⏹️ No improvement for {_config.EarlyStopPatience} epochs. Stopping early at epoch {display.Epoch}");
                Console.WriteLine($"Best accuracy: {display.BestAccuracy:P2}");
                _network.LoadWeights(_config.DatasetKey, _config.CheckpointDir);
                break;
            }
        }
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
}

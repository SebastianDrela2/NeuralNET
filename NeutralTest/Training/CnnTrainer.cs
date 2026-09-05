using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
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
        var bestAccuracy = 0f;
        var epochsSinceBest = 0;

        try
        {
            _network.LoadWeights(_config.DatasetKey, _config.CheckpointDir);
            Console.WriteLine($"[INFO] Successfully loaded existing weights for {_config.DatasetKey}.");
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"[INFO] No existing weights found for {_config.DatasetKey}. Starting fresh training.");
        }

        Console.Write("\e[2J\e[3J\e[H");
        for (int epoch = 0; ; epoch++)
        {
            var totalLoss = 0f;

            for (int batchIdx = 0; batchIdx < dataSet.TrainImages.Count; batchIdx++)
            {
                float loss = _network.TrainBatch(dataSet.TrainImages[batchIdx], dataSet.TrainLabels[batchIdx], _config.LearningRate);
                totalLoss += loss;
            }

            var avgLoss = totalLoss / dataSet.TrainImages.Count;
            var result = _validator.Validate(_network, dataSet.TestImages, dataSet.TestLabels);
            var accuracy = result.Accuracy;

            // Render table using dedicated display class
            if (dataSet.TestImages.Count > 0)
            {
                var sampleBatch = dataSet.TestImages[0];
                var sampleLabels = dataSet.TestLabels[0];
                using var pred = _network.Forward(sampleBatch);

                CnnDisplay.RenderTable(
                    epoch, avgLoss, accuracy, bestAccuracy, epochsSinceBest,
                    numClasses, _config.DatasetKey, sampleBatch, sampleLabels, pred
                );
            }
            else
            {
                CnnDisplay.RenderTable(
                    epoch, avgLoss, accuracy, bestAccuracy, epochsSinceBest,
                    numClasses, _config.DatasetKey, null, null, null
                );
            }

            // Persistence on improvement
            if (accuracy > bestAccuracy)
            {
                bestAccuracy = accuracy;
                epochsSinceBest = 0;
                _network.SaveWeights(_config.DatasetKey, _config.CheckpointDir);
            }
            else
            {
                epochsSinceBest++;
            }

            if (epoch % 10 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            // Early stopping rules
            if (accuracy >= _config.TargetAccuracy)
            {
                Console.WriteLine($"\n🎯 Target accuracy {_config.TargetAccuracy:P2} reached! Saving final {_config.DatasetKey} weights...");
                _network.SaveWeights(_config.DatasetKey, _config.CheckpointDir);
                break;
            }

            if (epochsSinceBest >= _config.EarlyStopPatience && bestAccuracy > 0.4f)
            {
                Console.WriteLine($"\n⏹️ No accuracy improvement for {_config.EarlyStopPatience} epochs. Stopping early at epoch {epoch + 1}");
                Console.WriteLine($"Best accuracy for {_config.DatasetKey}: {bestAccuracy:P2}. Restoring best checkpoint...");
                _network.LoadWeights(_config.DatasetKey, _config.CheckpointDir);
                break;
            }
        }
    }
}

using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    static void Main() => RunCnnNetwork();

    public static void RunCnnNetwork()
    {
        // 1. Data Loading
        var datasetKey = DataSourceType.Letters;
        var loader = DataLoaderFactory.Create(datasetKey);
        var config = TrainingConfig.CreateDefault(loader.NumClasses);

        config.DatasetKey = datasetKey;
        config.BatchSize = 64;
        config.MaxTrainSamples = 60_000;
        config.MaxTestSamples = 3_000;

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
}

using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Test.Data;
using NeutralTest;

namespace NeutralNET.ImageEpochViewer;

internal static class Program
{
    [STAThread]
    static void Main()
    {
        ApplicationConfiguration.Initialize();

        var datasetKey = DataSourceType.Letters;
        var loader = DataLoaderFactory.Create(datasetKey);
        var config = TrainingConfig.CreateDefault(loader.NumClasses);
        config.DatasetKey = datasetKey;

        var network = new CnnBuilder()
            .WithCnnConfig(config.CnnArchitecture)
            .WithDenseConfig(config.DenseConfig)
            .WithInputSize(loader.ImageScale, loader.ImageScale, 3)
            .Build();

        try
        {
            network.LoadWeights(config.DatasetKey, config.CheckpointDir);
            Console.WriteLine($"[INFO] Successfully loaded existing weights for {config.DatasetKey}.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARNING] Could not load weights for {config.DatasetKey}: {ex.Message}. Running with untrained weights.");
        }

        var mainForm = new LetterWindow(network);
        Application.Run(mainForm);
    }
}

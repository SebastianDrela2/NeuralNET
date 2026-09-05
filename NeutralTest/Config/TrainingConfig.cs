using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class TrainingConfig
{
    // Dataset Configuration
    public DataSourceType DatasetKey { get; set; } = DataSourceType.Letters;
    public int MaxTrainSamples { get; set; } = 60_000;
    public int MaxTestSamples { get; set; } = 3_000;
    public int BatchSize { get; set; } = 64;

    // Hyperparameters
    public float LearningRate { get; set; } = 0.0003f;
    public float TargetAccuracy { get; set; } = 0.99f;
    public int EarlyStopPatience { get; set; } = 300;
    public string CheckpointDir { get; set; } = "./checkpoints";

    // Sub-configs
    public CnnArchitectureConfig CnnArchitecture { get; set; } = new();
    public NeuralNetworkConfig DenseConfig { get; set; } = new();

    /// <summary>
    /// Creates a default baseline configuration for training.
    /// </summary>
    public static TrainingConfig CreateDefault(int numClasses)
    {
        return new TrainingConfig
        {
            CnnArchitecture = new CnnArchitectureConfig
            {
                ConvLayers =
                [
                    new() {
                        KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                        Activation = ActivationType.ReLU, UseMaxPool = true, PoolSize = 2
                    },
                    new() {
                        KernelHeight = 3, KernelWidth = 3, Filters = 64, Stride = 1, Padding = 1,
                        Activation = ActivationType.ReLU, UseMaxPool = true, PoolSize = 2
                    }
                ],
                DenseArchitecture = [128, numClasses],
                DenseHiddenActivation = ActivationType.ReLU,
                OutputActivation = ActivationType.Softmax,
                OptimizerConfig = new CnnOptimizerConfig
                {
                    OptimizerType = CnnOptimizerType.Adam,
                    LearningRate = 0.0003f,
                    WeightDecay = 1e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.999f,
                    Epsilon = 1e-8f
                }
            },
            DenseConfig = new NeuralNetworkConfig
            {
                LearningRate = 0.0003f,
                WeightDecay = 1e-4f,
                BatchSize = 64,
                Epochs = 100,
                DropoutRate = 0.25f,
                WithShuffle = true,
                OptimizerType = OptimizerType.Adam,
                Model = null
            }
        };
    }
}

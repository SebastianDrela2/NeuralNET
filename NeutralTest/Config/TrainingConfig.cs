using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class TrainingConfig
{
    public DataSourceType DatasetKey { get; set; } = DataSourceType.Letters;

    // CRITICAL: Must be ~60,000 so the network gets ~2,300 images per letter instead of 76
    public int MaxTrainSamples { get; set; } = 60_000;
    public int MaxTestSamples { get; set; } = 10_000;

    public int BatchSize { get; set; } = 1024;

    public float LearningRate { get; set; } = 0.0005f;
    public float TargetAccuracy { get; set; } = 1f;
    public int EarlyStopPatience { get; set; } = 50;
    public string CheckpointDir { get; set; } = "./checkpoints";

    public CnnArchitectureConfig CnnArchitecture { get; set; } = new();
    public NeuralNetworkConfig DenseConfig { get; set; } = new();

    public static TrainingConfig CreateDefault(int numClasses = 26)
    {
        return new TrainingConfig
        {
            CnnArchitecture = new CnnArchitectureConfig
            {
                ConvLayers =
                [
                   // Layer 1: 32x32 -> 16x16 (32 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   },
                   // Layer 2: 16x16 -> 8x8 (64 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 64, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   }
                ],
                // Wide single hidden layer avoids information loss on 26 output classes
                DenseArchitecture = [256, numClasses],
                DenseHiddenActivation = ActivationType.LeakyReLU,
                OutputActivation = ActivationType.Softmax,
                OptimizerConfig = new CnnOptimizerConfig
                {
                    OptimizerType = CnnOptimizerType.Adam,
                    LearningRate = 0.0005f,
                    WeightDecay = 1e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.999f,
                    Epsilon = 1e-8f
                }
            },
            DenseConfig = new NeuralNetworkConfig
            {
                LearningRate = 0.0005f,
                WeightDecay = 1e-4f,
                BatchSize = 1024,
                Epochs = 100,
                DropoutRate = 0.1f,
                WithShuffle = true,
                OptimizerType = OptimizerType.Adam,
                Model = null
            }
        };
    }
}

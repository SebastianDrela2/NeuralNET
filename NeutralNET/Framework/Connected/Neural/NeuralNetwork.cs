using System;
using System.IO;
using NeutralNET.Framework.Connected;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Connected.Neural;

public class NeuralNetwork<TArch> where TArch : IArchitecture<TArch>
{
    private readonly NeuralNetworkConfig _config;
    private readonly NeuralFramework<TArch> _neuralFramework;

    public NeuralNetwork(NeuralNetworkConfig config)
    {
        _config = config;
        _neuralFramework = new NeuralFramework<TArch>(config);
    }

    public NeuralForward RunModel() => _neuralFramework.Run(_config.Model);
    public NeuralForward RunDynamicModel() => _neuralFramework.Run(_config.DynamicModel);
    public IEnumerable<NeuralMatrix> RunEpoch() => _neuralFramework.RunEpoch(_config.Model);
    public IEnumerable<NeuralMatrix> EnumerateEpochs() => _neuralFramework.EnumerateEpochs(_config.Model);
    public NeuralMatrix Forward() => _neuralFramework.Forward();

    public TArch Architecture => _neuralFramework.Architecture;

    #region Save and Load Methods

    /// <summary>
    /// Saves weights to a directory path using an enum key.
    /// </summary>
    public NeuralNetwork<TArch> SaveWeights<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        _neuralFramework.SaveWeights(key, directoryPath);
        return this;
    }

    /// <summary>
    /// Loads weights from a directory path using an enum key.
    /// </summary>
    public NeuralNetwork<TArch> LoadWeights<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        _neuralFramework.LoadWeights(key, directoryPath);
        return this;
    }

    #endregion
}

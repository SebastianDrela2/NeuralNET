﻿using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural;

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
}

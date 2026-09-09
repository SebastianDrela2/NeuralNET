using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class ConvHyperParameters(
    CnnMatrix Weights,
    NeuralMatrix FlattenedWeights,
    CnnMatrix Biases) : IDisposable
{
    public void Dispose()
    {
        Weights.Dispose();
        FlattenedWeights.Dispose();
        Biases.Dispose();
    }
}

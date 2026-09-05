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
    static void Main() => RunCnnNetwork();

    public static void RunCnnNetwork()
    {
        var loader = DataLoaderFactory.Create(DataSourceType.Letters);

        const int batchSize = 32;

        var dataSet = loader.LoadCompleteDataset(
           batchSize: batchSize,
           maxTrainSamples: 30000,
           maxTestSamples: 3000
        );

        var cnnConfig = new CnnArchitectureConfig
        {
            ConvLayers =
            [
                // === CONV 1: 16x16x3 → MaxPool → 8x8x32 ===
                new() {
                    KernelHeight = 3,
                    KernelWidth = 3,
                    Filters = 32,
                    Stride = 1,
                    Padding = 1,
                    Activation = ActivationType.ReLU,
                    UseMaxPool = true,
                    PoolSize = 2
                },
                
                // === CONV 2: 8x8x32 → MaxPool → 4x4x64 ===
                new() {
                    KernelHeight = 3,
                    KernelWidth = 3,
                    Filters = 64,
                    Stride = 1,
                    Padding = 1,
                    Activation = ActivationType.ReLU,
                    UseMaxPool = true,
                    PoolSize = 2
                },
            ],

            DenseArchitecture = [128, loader.NumClasses],
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
        };

        var denseConfig = new NeuralNetworkConfig
        {
            LearningRate = 0.0003f,
            WeightDecay = 1e-4f,
            BatchSize = batchSize,
            Epochs = 100,
            DropoutRate = 0.25f,
            WithShuffle = true,
            OptimizerType = OptimizerType.Adam,
            Model = null
        };

        using var network = new CnnBuilder<Architecture>()
            .WithCnnConfig(cnnConfig)
            .WithDenseConfig(denseConfig)
            .WithInputSize(loader.ImageScale, loader.ImageScale, 3)
            .Build();

        var validator = new CnnValidator();

        TrainAndRenderTable(network, validator, dataSet, loader.DatasetName, loader.NumClasses);

        Console.WriteLine("\n=== FINAL EVALUATION ===");
        var finalResult = validator.Validate(network, dataSet.TestImages, dataSet.TestLabels);
        validator.PrintResults(finalResult);

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
        var bestAccuracy = 0f;
        var accuracy = 0f;
        var epochsSinceBest = 0;

        var testImages = dataSet.TestImages;
        var testLabels = dataSet.TestLabels;

        string hBorderMid = new string('═', numClasses * 6 + 1);
        string headerLabels = string.Join("", Enumerable.Range(0, numClasses).Select(i => $"  {(char)('A' + i)}   "));

        Console.Write("\e[2J\e[3J\e[H");
        for (int epoch = 0; ; epoch++)
        {
            var totalLoss = 0f;

            // Train Epoch
            for (int batchIdx = 0; batchIdx < dataSet.TrainImages.Count; batchIdx++)
            {
                float loss = network.TrainBatch(dataSet.TrainImages[batchIdx], dataSet.TrainLabels[batchIdx], learningRate);
                totalLoss += loss;
            }

            var avgLoss = totalLoss / dataSet.TrainImages.Count;

            var result = validator.Validate(network, testImages, testLabels);
            accuracy = result.Accuracy;

            Console.Write("\e[H");
            Console.WriteLine($"╔══════════════╤{hBorderMid}╤══════════════╗\e[K");
            Console.WriteLine($"║  Epoch {epoch + 1,5} │ Loss: {avgLoss,9:F6}  │  Accuracy: {accuracy,7:P2} │  Best: {bestAccuracy,7:P2}  ║\e[K");
            Console.WriteLine($"╠═══════╤══════╧{hBorderMid}╧══════════════╣\e[K");
            Console.WriteLine($"║       │{headerLabels}│ Pred  Actual ║\e[K");
            Console.WriteLine($"╠═══════╪{hBorderMid}╪══════════════╣\e[K");

            if (testImages.Count > 0)
            {
                var sampleBatch = testImages[0];
                var sampleLabels = testLabels[0];
                var pred = network.Forward(sampleBatch);

                var numSamples = Math.Min(numClasses, sampleBatch.Batch);
                for (int i = 0; i < numSamples; i++)
                {
                    var probs = new float[numClasses];
                    for (var j = 0; j < numClasses; j++)
                    {
                        probs[j] = pred.At(i, j);
                    }

                    var predicted = ArgMax(probs);
                    var actual = GetActualLabelFromRow(sampleLabels, i, numClasses);

                    char predChar = (char)('A' + predicted);
                    char actualChar = actual >= 0 ? (char)('A' + actual) : '?';

                    Console.Write($"║ {i,2}    │");
                    for (int j = 0; j < numClasses; j++)
                    {
                        Console.Write($" {FmtPogression(probs[j], j == actual)}");
                    }

                    Console.WriteLine($" │  {(predicted == actual ? AsGreen(predChar.ToString()) : AsRed(predChar.ToString())),2}      {actualChar,2}   ║\e[K");
                }
                pred.Dispose();
            }

            Console.WriteLine($"╚═══════╧{hBorderMid}╧══════════════╝\e[K");
            Console.WriteLine();
            Console.WriteLine($"Best accuracy: {bestAccuracy:P2}  |  Epochs since best: {epochsSinceBest}\e[K");

            if (accuracy > bestAccuracy)
            {
                bestAccuracy = accuracy;
                epochsSinceBest = 0;
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

            if (accuracy >= targetAccuracy)
            {
                Console.WriteLine($"\n🎯 Target accuracy {targetAccuracy:P2} reached! Stopping early at epoch {epoch + 1}");
                break;
            }

            if (epochsSinceBest >= earlyStopPatience && bestAccuracy > 0.4f)
            {
                Console.WriteLine($"\n⏹️ No improvement for {earlyStopPatience} epochs. Stopping early at epoch {epoch + 1}");
                Console.WriteLine($"Best accuracy: {bestAccuracy:P2}");
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

    private static string FmtPogression(float x, bool hl)
    {
        const int ColWidth = 5;
        const char ChZero = ' ';
        const char ChMax = '\u2588';

        const string bg = "12;12;12";
        const string fg = "163;163;163";
        const string errBg = "227;61;48";
        const string errFg = "151;41;32";

        const string hlBg = "10;12;13";
        const string hlFg = "78;91;106";
        const string hlErrBg = "98;67;75";
        const string hlErrFg = "227;61;48";

        switch (x)
        {
            case 0: return $"\e[48;2;{(hl ? hlBg : bg)}m{new string(ChZero, ColWidth)}\e[49m";
            case 1: return $"\e[38;2;{(hl ? hlFg : fg)}m{new string(ChMax, ColWidth)}\e[39m";
            case <= 0: return $"\e[38;2;{(hl ? hlErrBg : errBg)}m{x,5:f2}\e[39m";
            case >= 1: return $"\e[38;2;{(hl ? hlErrFg : errFg)};48;2;{(hl ? hlErrBg : errBg)}m{x,5:f3}\e[39;49m";
        }
        Span<char> xs = stackalloc char[ColWidth];
        xs.Fill(ChZero);

        var scaled = x * ColWidth;
        var maxEnd = int.Clamp((int)scaled, 0, ColWidth);
        xs[..maxEnd].Fill(ChMax);

        if (maxEnd != ColWidth)
        {
            int frame = int.Clamp((int)((8 * (scaled - maxEnd)) + 0.5f), 0, 7);
            xs[maxEnd] = (char)(ChMax + (7 - frame));
        }

        return $"\e[38;2;{(hl ? hlFg : fg)}m{xs}\e[39m";
    }

    private static int ArgMax(float[] array)
    {
        int maxIdx = 0;
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static string AsGreen(string x) => $"\e[38;2;124;179;66m{x}\e[39m";
    private static string AsRed(string x) => $"\e[38;2;230;74;25m{x}\e[39m";
}

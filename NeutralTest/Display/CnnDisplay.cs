using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnDisplayWriter(char[] items, int dataSetSize)
{
    private const int LabelColSize = 5;
    private const int LabelPadL = (LabelColSize - 1) / 2;
    private const int LabelPadR = LabelColSize / 2;

    public char[] Items { get; } = items;
    private readonly string _txtBorderMid = new string('═', items.Length * LabelColSize);
    private readonly string _txtLabels = string.Join("", items.Select(c => $"{"",LabelPadL}{c}{"",LabelPadR}"));

    public int Epoch { get; set; }
    public float Accuracy
    {
        get;
        set
        {
            field = value;
            if (field <= BestAccuracy) return;
            EpochsSinceBest = 0;
            BestAccuracy = field;
        }
    }
    public float AvgLoss => TotalLoss / dataSetSize;
    public float TotalLoss { get; set; }
    public float BestAccuracy { get; set; }
    public int EpochsSinceBest { get; set; }

    public void Clear()
    {
        Console.Write("\e[2J\e[3J\e[H");
    }

    public void Update(ReadOnlySpan<float> xss)
    {
        Epoch += 1;

        const string Sep1 = "══════════════╤══════════════════╤════════════════════╤═════════════════";
        const string Sep2 = "══════════════╧══════════════════╧════════════════════╧═════════════════";

        Console.Write("\e[H");

        Console.WriteLine($"╔{Sep1}╗\e[K");
        Console.WriteLine($"║  Epoch {Epoch,5} │ Loss: {AvgLoss,9:F6}  │  Accuracy: {Accuracy,7:P2} │  Best: {BestAccuracy,7:P2}  ║\e[K");
        Console.WriteLine($"╠{Sep2}╝\e[K");
        Console.WriteLine($"║{_txtLabels}│\e[A\e[D╤\e[B\e[K");
        Console.WriteLine($"╠{_txtBorderMid}╡\e[K");

        while (xss is [var x, .. var xs])
        {
            var probs = xs[..Items.Length];
            xss = xs[Items.Length..];
            var predicted = ArgMax(probs);
            var actual = (int)x;

            char predChar = (char)('A' + predicted);
            char actualChar = actual >= 0 ? (char)('A' + actual) : '?';

            Console.Write($"║");
            for (int j = 0; j < Items.Length; j++)
            {
                Console.Write($"{FmtPogression(probs[j], j == actual)}");
            }

            bool isOk = predicted == actual;
            var mark = isOk ? AsGreen("✓") : AsRed("✗");
            var lhs = predChar.ToString();
            var rhs = actualChar.ToString();

            Console.WriteLine($"│ {lhs} {mark} {rhs}\e[K");

        }

        Console.WriteLine($"╚{_txtBorderMid}╛\e[K");
        Console.WriteLine();
        Console.WriteLine($"Best accuracy: {BestAccuracy:P2}  |  Epochs since best: {EpochsSinceBest}\e[K");

        EpochsSinceBest += 1;
    }

    private static string FmtPogression(float x, bool hl)
    {
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
            case 0: return $"\e[48;2;{(hl ? hlBg : bg)}m{new string(ChZero, LabelColSize)}\e[49m";
            case 1: return $"\e[38;2;{(hl ? hlFg : fg)}m{new string(ChMax, LabelColSize)}\e[39m";
            case <= 0: return $"\e[38;2;{(hl ? hlErrBg : errBg)}m{x,5:f2}\e[39m";
            case >= 1: return $"\e[38;2;{(hl ? hlErrFg : errFg)};48;2;{(hl ? hlErrBg : errBg)}m{x,5:f3}\e[39;49m";
        }
        Span<char> xs = stackalloc char[LabelColSize];
        xs.Fill(ChZero);

        var scaled = x * LabelColSize;
        var maxEnd = int.Clamp((int)scaled, 0, LabelColSize);
        xs[..maxEnd].Fill(ChMax);

        if (maxEnd != LabelColSize)
        {
            int frame = int.Clamp((int)((8 * (scaled - maxEnd)) + 0.5f), 0, 7);
            xs[maxEnd] = (char)(ChMax + (7 - frame));
        }

        if (!hl) return $"\e[48;2;35;35;35;38;2;{fg}m{xs}\e[39;49m";

        switch (x)
        {
            case < 0.1f: return $"\e[48;2;35;11;0;38;2;163;53;0m{xs}\e[39;49m";
            case > 0.9f: return $"\e[48;2;0;35;11;38;2;0;163;53m{xs}\e[39;49m";
            default: return $"\e[48;2;168;152;46;38;2;245;222;67m{xs}\e[39;49m";
        }
    }

    private static int ArgMax(ReadOnlySpan<float> array)
    {
        int maxIdx = 0;
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static string AsGreenOrRed(bool cnd, string x) => cnd ? AsGreen(x) : AsRed(x);
    private static string AsGreen(string x) => $"\e[38;2;124;179;66m{x}\e[39m";
    private static string AsRed(string x) => $"\e[38;2;230;74;25m{x}\e[39m";
}

public static class CnnDisplay
{
    public static void RenderTable(
        int epoch,
        float avgLoss,
        float accuracy,
        float bestAccuracy,
        int epochsSinceBest,
        int numClasses,
        DataSourceType datasetKey,
        CnnMatrix? sampleBatch,
        NeuralMatrix? sampleLabels,
        NeuralMatrix? predictions)
    {
        string hBorderMid = new string('═', numClasses * 6 + 1);
        string headerLabels = string.Join("", Enumerable.Range(0, numClasses).Select(i => $"  {(char)('A' + i)}   "));

        Console.Write("\e[H");
        Console.WriteLine($"╔══════════════╤{hBorderMid}╤══════════════╗\e[K");
        Console.WriteLine($"║  Epoch {epoch + 1,5} │ Loss: {avgLoss,9:F6}  │  Accuracy: {accuracy,7:P2} │  Best: {bestAccuracy,7:P2}  ║\e[K");
        Console.WriteLine($"╠═══════╤══════╧{hBorderMid}╧══════════════╣\e[K");
        Console.WriteLine($"║       │{headerLabels}│ Pred  Actual ║\e[K");
        Console.WriteLine($"╠═══════╪{hBorderMid}╪══════════════╣\e[K");

        if (sampleBatch != null && sampleLabels != null && predictions != null)
        {
            var numSamples = Math.Min(numClasses, sampleBatch.Batch);
            for (int i = 0; i < numSamples; i++)
            {
                var probs = new float[numClasses];
                for (var j = 0; j < numClasses; j++)
                {
                    probs[j] = predictions.At(i, j);
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
        }

        Console.WriteLine($"╚═══════╧{hBorderMid}╧══════════════╝\e[K");
        Console.WriteLine();
        Console.WriteLine($"Dataset: {datasetKey}  |  Best accuracy: {bestAccuracy:P2}  |  Epochs since best: {epochsSinceBest}\e[K");
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

        return maxIndex >= numClasses ? numClasses - 1 : maxIndex;
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

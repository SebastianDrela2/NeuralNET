using System.Diagnostics;

namespace NeutralNET.Framework.Convolutional;

public class SourceLocation(
    MatrixInfo info,
    [CallerLineNumber] int ln = 0,
    [CallerFilePath] string fp = "")
{
    public static SourceLocation Current(
        MatrixInfo info,
        [CallerLineNumber] int ln = 0,
        [CallerFilePath] string fp = "") => new SourceLocation(info, ln, fp);
    public StackTrace Trace {get; } = new StackTrace();

    public int ThreadID = Environment.CurrentManagedThreadId;
    public long TimeStamp = Stopwatch.GetTimestamp();

    public MatrixInfo Info = info;
    private int LineNumber { get; } = ln;
    private string FilePath { get; } = fp[53..];

    public string Debug => $"[{TimeStamp}|{ThreadID}] {FilePath}:{LineNumber}\n{Trace}";
    public override string ToString() => Debug;
}

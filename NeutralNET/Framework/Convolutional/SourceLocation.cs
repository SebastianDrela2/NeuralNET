using System.Diagnostics;

namespace NeutralNET.Framework.Convolutional;

public class SourceLocation([CallerLineNumber] int ln = 0, [CallerFilePath] string fp = "")
{
    public static SourceLocation Current([CallerLineNumber] int ln = 0, [CallerFilePath] string fp = "") => new SourceLocation(ln, fp);
    public StackTrace Trace {get; } = new StackTrace();

    public int ThreadID = Thread.CurrentThread.ManagedThreadId;
    public long TimeStamp = Stopwatch.GetTimestamp();
    private int LineNumber { get; } = ln;
    private string FilePath { get; } = fp[53..];

    public string Debug => $"[{TimeStamp}|{ThreadID}] {FilePath}:{LineNumber}\n{Trace}";
    public override string ToString() => Debug;
}

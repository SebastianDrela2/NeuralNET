global using System.Runtime.CompilerServices;
// System.Runtime.CompilerServices
//  .MethodImplOptions
//  .MethodImplAttribute

global using static GlobalScope;
global using SIMD = NeutralNET.SIMD_512;
using System.Diagnostics.CodeAnalysis;

public static partial class GlobalScope
{
    public const MethodImplOptions Inline = MethodImplOptions.AggressiveInlining;


    [return: NotNullIfNotNull(nameof(target))]
    public static ref T DisposeReplace<T>([NotNullIfNotNull(nameof(value))] ref T target, T value)
    where T : IDisposable?, allows ref struct
    {
        Exchange(ref target, value)?.Dispose();
        return ref target;
    }

    [return: NotNullIfNotNull(nameof(target))]
    public static T Exchange<T>([NotNullIfNotNull(nameof(value))] scoped ref T target, T value)
    where T : allows ref struct
    {
        var prev = target;
        target = value;
        return prev;
    }
}

public static partial class Extensions;
partial class Extensions
{
    extension(NotSupportedException)
    {
        public static void ThrowIfFalse(
            [DoesNotReturnIf(false)] bool condition,
            [CallerArgumentExpression(nameof(condition))] string? expr = null,
            [CallerFilePath] string? origin = null,
            [CallerLineNumber] int ln = -1)
        {
            throw new NotSupportedException((expr, Path.GetFileName(origin), ln) switch
            {
                (null, _, _) => null,
                (var e, null, _) => $"{e} was false",
                var (e, name, line) => $"{e} was false (in {name}:{line})",
            });
        }
    }

    extension<T>(List<T> xs)
        where T : IDisposable?
    {
        public void ClearAndDispose()
        {
            foreach (var x in xs) x?.Dispose();
            xs.Clear();
        }

    }
}

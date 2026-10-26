using System.Numerics;

namespace EvoToolbox.Constructs;

public readonly record struct UnorderedPair<T> where T : IBinaryInteger<T>
{
    public T Min { get; }
    public T Max { get; }

    public UnorderedPair(T a, T b)
    {
        Min = T.Min(a, b);
        Max = T.Max(a, b);
    }
}
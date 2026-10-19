namespace EvoToolbox.Objects;

public readonly struct Node
{
    public Node(byte index, ushort x, ushort y, ushort cost)
    {
        Index = index;
        Location = new Point(x, y);
        Cost = cost;
    }

    public readonly Point Location { get; }
    public ushort X => Location.X;
    public ushort Y => Location.Y;
    public readonly ushort Cost { get; }

    public readonly byte Index { get; }

    public ushort this[int index]
    {
        get =>
            index switch
            {
                0 => X,
                1 => Y,
                2 => Cost,
                _ => throw new ArgumentOutOfRangeException(),
            };
    }
}

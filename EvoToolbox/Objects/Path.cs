namespace EvoToolbox.Objects;

public class Path
{
    private readonly Node[] _masterArray;
    private List<byte> _nodeIdxs { get; }

    public Path(Node[] masterArray)
    {
        _masterArray = masterArray;
        _nodeIdxs = new List<byte>();
    }

    public void AppendNode(byte nodeIdx)
    {
        _nodeIdxs.Add(nodeIdx);
    }

    public void AddAfterNode(byte nodeIdx, byte afterIdx)
    {
        _nodeIdxs.Insert(afterIdx + 1, nodeIdx);
    }

    public void AddBeforeNode(byte nodeIdx, byte beforeIdx)
    {
        _nodeIdxs.Insert(beforeIdx, nodeIdx);
    }

    public Node this[byte index] => _masterArray[_nodeIdxs[index]];

    public Node OnLoopingIndex(int index)
    {
        int n = _nodeIdxs.Count;
        return _masterArray[_nodeIdxs[((index % n) + n) % n]];
    }

    public int Count => _nodeIdxs.Count;
}
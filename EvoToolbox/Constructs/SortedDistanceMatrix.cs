using EvoToolbox.Objects;

namespace EvoToolbox.Constructs;

public class SortedDistanceMatrix
{
    private readonly Node[] _masterArray;
    private readonly (uint, byte)[][] _sortedDistanceMatrix;

    public SortedDistanceMatrix(Node[] masterArray, bool penalizeSelf)
    {
        _masterArray = masterArray;
        _sortedDistanceMatrix = new (uint, byte)[_masterArray.Length][];
        for (byte i = 0; i < masterArray.Length; i++)
        {
            _sortedDistanceMatrix[i] = new(uint, byte)[_masterArray.Length];
            for (byte j = 0; j < _masterArray.Length; j++)
            {
                if (i == j && penalizeSelf)
                {
                    _sortedDistanceMatrix[i][j] = (uint.MaxValue, j);
                }
                else
                {
                    _sortedDistanceMatrix[i][j] = (
                        (uint)Math.Round(Math.Sqrt(Math.Pow(_masterArray[i].X - _masterArray[j].X, 2) +
                                                   Math.Pow(_masterArray[i].Y - _masterArray[j].Y, 2))), j);
                }
            }
            Array.Sort(_sortedDistanceMatrix[i], (a, b) => a.Item1.CompareTo(b.Item1));
        }
    }

    public int Length => _sortedDistanceMatrix.Length;
    
    public Node this[byte i, byte j] => _masterArray[_sortedDistanceMatrix[i][j].Item2];

    public (uint, byte)[] GetRawRow(byte row) => _sortedDistanceMatrix[row];

    public Node NthClosest(Node node, byte n) => _masterArray[_sortedDistanceMatrix[node.Index][n].Item2];
    
    public Node[] NClosest(Node node, byte n)
    {
        var nodeArray = new Node[n];
        var view = new ArraySegment<(uint, byte)>(_sortedDistanceMatrix[node.Index], 0, n);
        for (int i = 0; i < view.Count; i++)
        {
            nodeArray[i] = _masterArray[view[i].Item2];
        }

        return nodeArray;
    }
}
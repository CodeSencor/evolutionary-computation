using EvoToolbox.Objects;

namespace EvoToolbox.Utilities;

public class CostUtility
{
    private readonly Node[] _masterArray;
    private uint[,] _distanceMatrix;

    public CostUtility(Node[] masterArray)
    {
        _masterArray = masterArray;
        _distanceMatrix = new uint[masterArray.Length, masterArray.Length];
        for (byte i = 0; i < masterArray.Length; i++)
        {
            for (byte j = 0; j < masterArray.Length; j++)
            {
                _distanceMatrix[i, j] = (uint)Math.Round(Math.Sqrt(Math.Pow(_masterArray[i].X - _masterArray[j].X, 2) + Math.Pow(_masterArray[i].Y - _masterArray[j].Y, 2)));
            }
        }
    }

    public uint Delta(Node left, Node right, Node joins)
    {
        return _distanceMatrix[left.Index, joins.Index] + _distanceMatrix[right.Index, joins.Index] - _distanceMatrix[left.Index, right.Index] + joins.Cost;
    }
}
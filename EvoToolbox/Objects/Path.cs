using EvoToolbox.Constructs;

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

    public Node Successor(byte index)
    {
        return index == _nodeIdxs.Count - 1 ? _masterArray[_nodeIdxs[0]] : _masterArray[_nodeIdxs[index + 1]];
    }

    public Node Predecessor(byte index)
    {
        return index == 0 ? _masterArray[_nodeIdxs[^1]] : _masterArray[_nodeIdxs[index - 1]];
    }

    public int Count => _nodeIdxs.Count;

    public long TestNodeSwap(byte firstNodeIdx, byte secondNodeIdx, DistanceMatrix distanceMatrix)
    {
        Node first = this[firstNodeIdx],
            second = this[secondNodeIdx],
            firstLeft = Predecessor(firstNodeIdx),
            firstRight = Successor(firstNodeIdx),
            secondLeft = Predecessor(secondNodeIdx),
            secondRight = Successor(secondNodeIdx);

        if (second.Index == firstRight.Index)
        {
            return (long)(distanceMatrix.Distance(firstLeft, second) + distanceMatrix.Distance(first, secondRight)) - distanceMatrix.Distance(firstLeft, first) - distanceMatrix.Distance(second, secondRight);
        }

        if (first.Index == secondRight.Index)
        {
            return (long)(distanceMatrix.Distance(secondLeft, first) + distanceMatrix.Distance(second, firstRight)) - distanceMatrix.Distance(secondLeft, second) - distanceMatrix.Distance(first, firstRight);
        }
        
        long oldDistance = distanceMatrix.Distance(first, firstLeft) + distanceMatrix.Distance(first, firstRight)
                                                                    + distanceMatrix.Distance(second, secondLeft) +
                                                                    distanceMatrix.Distance(second, secondRight);
        long newDistance = distanceMatrix.Distance(firstLeft, second) + distanceMatrix.Distance(second, firstRight)
                                                                     + distanceMatrix.Distance(secondRight, first) +
                                                                     distanceMatrix.Distance(first, secondLeft);
        return newDistance - oldDistance;
    }

    public void NodeSwap(byte firstNodeIdx, byte secondNodeIdx)
    {
        (_nodeIdxs[firstNodeIdx], _nodeIdxs[secondNodeIdx]) = (_nodeIdxs[secondNodeIdx], _nodeIdxs[firstNodeIdx]);
    }

    public long TestEdgeSwap(byte firstNodeIdx, byte secondNodeIdx, DistanceMatrix distanceMatrix)
    {
        Node first = this[firstNodeIdx],
            second = this[secondNodeIdx],
            firstSuccessor = Successor(firstNodeIdx),
            secondSuccessor = Successor(secondNodeIdx);
        long oldDistance = distanceMatrix.Distance(first, firstSuccessor) +
                          distanceMatrix.Distance(second, secondSuccessor);
        long newDistance = distanceMatrix.Distance(first, second) +
                          distanceMatrix.Distance(firstSuccessor, secondSuccessor);

        return newDistance - oldDistance;
    }

    public void EdgeSwap(int firstNodeIdx, int secondNodeIdx)
    {
        if (firstNodeIdx > secondNodeIdx)
        {
            (firstNodeIdx, secondNodeIdx) = (secondNodeIdx, firstNodeIdx);
        }
        _nodeIdxs.Reverse(firstNodeIdx + 1, secondNodeIdx - firstNodeIdx);
    }

    public long TestNodeExchange(byte nodeIdx, Node newNode, DistanceMatrix distanceMatrix)
    {
        Node oldNode = this[nodeIdx], leftNeighbor = Predecessor(nodeIdx), rightNeighbor = Successor(nodeIdx);
        long oldCost = distanceMatrix.Distance(oldNode, leftNeighbor) +
                          distanceMatrix.Distance(oldNode, rightNeighbor) + oldNode.Cost;
        long newCost = distanceMatrix.Distance(newNode, leftNeighbor) +
                          distanceMatrix.Distance(newNode, rightNeighbor) + newNode.Cost;

        return newCost - oldCost;
    }

    public void NodeExchange(byte oldNodeMasterIdx, byte newNodeIdx)
    {
        var location = _nodeIdxs.IndexOf(oldNodeMasterIdx);
        _nodeIdxs.RemoveAt(location);
        _nodeIdxs.Insert(location, newNodeIdx);
    }
}
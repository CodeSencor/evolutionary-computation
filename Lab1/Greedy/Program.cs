using var reader = new StreamReader("TSPA.csv");

var content = reader.ReadToEnd().Replace('\n', ';').Replace("\r", "").TrimEnd(';').Split(';');
Node[] masterArray = new Node[content.Length / 3];
for (int i = 0; i < content.Length; )
{
    masterArray[i / 3] = new Node(
        ushort.Parse(content[i++]),
        ushort.Parse(content[i++]),
        ushort.Parse(content[i++])
    );
}

byte upTo = (byte)(masterArray.Length / 2 + (masterArray.Length % 2 == 0 ? 0 : 1));
var r = new Random(42);
HashSet<byte> usedNodes = new();

// 1. Random solutions
byte[,] randomPaths = new byte[masterArray.Length, upTo];

for (byte i = 0; i < masterArray.Length; i++)
{
    for (byte j = 0; j < upTo; j++)
    {
        byte idx;
        while (usedNodes.Contains(idx = (byte)r.Next(0, masterArray.Length)))
            ;
        randomPaths[i, j] = idx;
        usedNodes.Add(idx);
    }
    usedNodes.Clear();
}

// 2. Nearest neighbor end paths
var nNEPs = Enumerable.Range(0, masterArray.Length).Select(_ => new List<byte>(upTo)).ToArray();
for (byte i = 0; i < masterArray.Length; i++)
{
    nNEPs[i].Add(i);
    usedNodes.Add(i);
    for (byte j = 1; j < upTo; j++)
    {
        uint leastCost = uint.MaxValue;
        int idx = -1;
        for (byte h = 0; h < masterArray.Length; h++)
        {
            if (usedNodes.Contains(h))
            {
                continue;
            }
            uint estimateCost = Globals.TrueCost(masterArray[nNEPs[i][j - 1]], masterArray[h]);
            if (estimateCost < leastCost)
            {
                idx = h;
                leastCost = estimateCost;
            }
        }
        usedNodes.Add((byte)idx);
        nNEPs[i].Insert(j, (byte)idx);
    }
    usedNodes.Clear();
}

// 3. Nearest neighbor all paths
int[,] distanceMatrix = new int[masterArray.Length, masterArray.Length];
for (byte i = 0; i < masterArray.Length; i++)
{
    for (byte j = 0; j < masterArray.Length; j++)
    {
        distanceMatrix[i, j] = (int)Globals.TrueDistance(masterArray[i], masterArray[j]);
    }
}

var nNAPs = Enumerable.Range(0, masterArray.Length).Select(_ => new List<byte>(upTo)).ToArray();
for (byte i = 0; i < masterArray.Length; i++)
{
    nNAPs[i].Add(i);
    usedNodes.Add(i);
    while (nNAPs[i].Count < upTo)
    {
        int bestDelta = int.MaxValue,
            bestK = -1,
            bestPos = -1,
            count = nNAPs[i].Count;

        for (byte k = 0; k < masterArray.Length; k++)
        {
            if (usedNodes.Contains(k))
                continue;
            int delta = distanceMatrix[k, nNAPs[i][0]] + masterArray[k].Cost;
            if (delta < bestDelta)
            {
                bestDelta = delta;
                bestK = k;
                bestPos = 0;
            }
        }
        for (byte j = 0; j < count; j++)
        {
            byte a = nNAPs[i][j];
            byte? b = j != count - 1 ? nNAPs[i][j + 1] : null;
            for (byte k = 0; k < masterArray.Length; k++)
            {
                if (usedNodes.Contains(k))
                {
                    continue;
                }
                int delta = b is not null
                    ? distanceMatrix[a, k]
                        + distanceMatrix[k, (int)b]
                        - distanceMatrix[a, (int)b]
                        + masterArray[k].Cost
                    : distanceMatrix[a, k] + masterArray[k].Cost;
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestK = k;
                    bestPos = j + 1;
                }
            }
        }
        usedNodes.Add((byte)bestK);
        nNAPs[i].Insert(bestPos, (byte)bestK);
    }
    usedNodes.Clear();
}

// 4. Greedy cycle
var greedyCycles = Enumerable
    .Range(0, masterArray.Length)
    .Select(_ => new List<byte>(upTo))
    .ToArray();

for (byte i = 0; i < masterArray.Length; i++)
{
    usedNodes.Add(i);
    greedyCycles[i].Add(i);
    while (greedyCycles[i].Count < upTo)
    {
        int bestDelta = int.MaxValue,
            bestK = -1,
            bestPos = -1,
            count = greedyCycles[i].Count;
        for (byte j = 0; j < count; j++)
        {
            byte a = greedyCycles[i][j],
                b = greedyCycles[i][(j + 1) % greedyCycles[i].Count];
            for (byte k = 0; k < masterArray.Length; k++)
            {
                if (usedNodes.Contains(k))
                {
                    continue;
                }
                int delta =
                    distanceMatrix[a, k]
                    + distanceMatrix[k, b]
                    - distanceMatrix[a, b]
                    + masterArray[k].Cost;
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestK = k;
                    bestPos = j + 1;
                }
            }
        }
        usedNodes.Add((byte)bestK);
        greedyCycles[i].Insert(bestPos, (byte)bestK);
    }
    usedNodes.Clear();
}

// Write to file for visualizer
using var writer = new StreamWriter("random.csv");
for (byte i = 0; i < randomPaths.GetLength(0); i++)
{
    for (byte j = 0; j < randomPaths.GetLength(1); j++)
    {
        var node = masterArray[randomPaths[i, j]];
        writer.WriteLine($"{node.X};{node.Y};{node.Cost}");
    }
}
writer.Close();

using var writer2 = new StreamWriter("NNEPs.csv");
foreach (var nNEP in nNEPs)
{
    foreach (var idx in nNEP)
    {
        var node = masterArray[idx];
        writer2.WriteLine($"{node.X};{node.Y};{node.Cost}");
    }
}
writer2.Close();

using var writer3 = new StreamWriter("NNAPs.csv");
foreach (var nNAP in nNAPs)
{
    foreach (var idx in nNAP)
    {
        var node = masterArray[idx];
        writer3.WriteLine($"{node.X};{node.Y};{node.Cost}");
    }
}
writer3.Close();

using var writer4 = new StreamWriter("greedy_cycles.csv");
foreach (var greedyCycle in greedyCycles)
{
    foreach (var idx in greedyCycle)
    {
        var node = masterArray[idx];
        writer4.WriteLine($"{node.X};{node.Y};{node.Cost}");
    }
}
writer4.Close();

static class Globals
{
    public static uint DistanceEstimate(Node current, Node next) =>
        (uint)(Math.Pow(current.X - next.X, 2) + Math.Pow(current.Y - next.Y, 2));

    public static uint DistanceEstimate(Node current, TreeNode next) =>
        DistanceEstimate(current, next.Node);

    public static uint CostEstimate(Node current, Node next) =>
        DistanceEstimate(current, next) + next.Cost;

    public static uint CostEstimate(TreeNode current, TreeNode next)
    {
        if (next.Deleted || current.Deleted)
        {
            return uint.MaxValue;
        }
        else if (current.Equals(next))
        {
            return 0;
        }
        else
        {
            return CostEstimate(current.Node, next.Node);
        }
    }

    public static uint CostEstimate(Node current, TreeNode next)
    {
        if (next is null || next.Deleted)
        {
            return uint.MaxValue;
        }
        else if (current.Equals(next.Node))
        {
            return 0;
        }
        else
        {
            return CostEstimate(current, next.Node);
        }
    }

    public static uint TrueDistance(Node current, Node next) =>
        (uint)Math.Round(Math.Sqrt(DistanceEstimate(current, next)));

    public static uint TrueCost(Node current, Node next) => TrueDistance(current, next) + next.Cost;
}

readonly struct Node
{
    public Node(ushort x, ushort y, ushort cost)
    {
        Location = new Point(x, y);
        Cost = cost;
    }

    public Point Location { get; }
    public ushort X => Location.X;
    public ushort Y => Location.Y;
    public ushort Cost { get; }

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
};

struct Point
{
    public Point(ushort x, ushort y)
    {
        X = x;
        Y = y;
    }

    public ushort X;
    public ushort Y;
}

class TreeNode
{
    public TreeNode(Node node)
    {
        Node = node;
    }

    public Node Node { get; }

    public bool Deleted = false;

    public ushort this[int index]
    {
        get =>
            index switch
            {
                0 => Node.X,
                1 => Node.Y,
                2 => Node.Cost,
                _ => throw new ArgumentOutOfRangeException(),
            };
    }
}

class TwoDTree
{
    public TwoDTree(IEnumerable<Node> nodeArray, int depth = 0)
    {
        if (!nodeArray.Any())
        {
            Root = null;
            return;
        }
        int axis = depth % 2;
        List<Node> nodeList = nodeArray.OrderBy(node => node[axis]).ToList();
        var pivot = nodeList[nodeList.Count / 2];
        Root = new TreeNode(pivot);
        nodeList.RemoveAt(nodeList.Count / 2);
        var leftHalf = nodeList.Where(node => node[axis] <= pivot[axis]);
        var rightHalf = nodeList.Where(node => node[axis] > pivot[axis]);
        depth++;
        LeftTree = new TwoDTree(leftHalf, depth);
        RightTree = new TwoDTree(rightHalf, depth);
    }

    public TreeNode? Root { get; }

    public TwoDTree? LeftTree = null;
    public TwoDTree? RightTree = null;

    public TreeNode? NearestNeighbor(Node target, bool deleteFromTree, int depth = 0)
    {
        if (Root is null)
        {
            return null;
        }
        TwoDTree nextBranch;
        TwoDTree otherBranch;
        if (target[depth % 2] <= Root[depth % 2])
        {
            nextBranch = LeftTree;
            otherBranch = RightTree;
        }
        else
        {
            nextBranch = RightTree;
            otherBranch = LeftTree;
        }

        TreeNode temp = nextBranch.NearestNeighbor(target, false, depth + 1);
        TreeNode best = temp is not null
            ? Globals.CostEstimate(target, temp) < Globals.CostEstimate(target, Root)
                ? temp
                : Root
            : Root;

        int dist = target[depth % 2] - Root[depth % 2];
        if (Globals.CostEstimate(target, best) >= dist * dist)
        {
            temp = otherBranch.NearestNeighbor(target, false, depth + 1);
            best = temp is not null
                ? Globals.CostEstimate(target, temp) < Globals.CostEstimate(target, best)
                    ? temp
                    : best
                : best;
        }

        if (deleteFromTree)
        {
            best.Deleted = true;
        }
        return best;
    }

    public void ResetTree()
    {
        Root?.Deleted = false;
        LeftTree?.ResetTree();
        RightTree?.ResetTree();
    }
}

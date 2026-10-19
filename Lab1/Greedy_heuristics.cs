// Baseline heuristics (random, nearest neighbor end / any position, greedy cycle)
// run on every instance listed below.
// Result files: TSPA_random.csv, TSPA_NNEPs.csv, TSPA_NNAPs.csv, TSPA_greedy_cycles.csv
// and the same four with the TSPB_ prefix.
(string Instance, string Prefix)[] instances =
{
    ("TSPA", "TSPA_"),
    ("TSPB", "TSPB_"),
};

foreach (var (instance, prefix) in instances)
{
    var path = $"{instance}.csv";
    if (!File.Exists(path))
    {
        Console.WriteLine($"Skipping {instance}: {path} not found.");
        continue;
    }

    Console.WriteLine($"Running {instance}...");
    new Solver(path).Run(prefix);
}

class Solver
{
    private readonly Node[] masterArray;
    private readonly byte upTo;
    private readonly int[,] distanceMatrix;
    private readonly HashSet<byte> usedNodes = new();

    public Solver(string instancePath)
    {
        using var reader = new StreamReader(instancePath);

        var content = reader.ReadToEnd().Replace('\n', ';').Replace("\r", "").TrimEnd(';').Split(';');
        masterArray = new Node[content.Length / 3];
        for (int i = 0; i < content.Length; )
        {
            masterArray[i / 3] = new Node(
                ushort.Parse(content[i++]),
                ushort.Parse(content[i++]),
                ushort.Parse(content[i++])
            );
        }

        upTo = (byte)(masterArray.Length / 2 + (masterArray.Length % 2 == 0 ? 0 : 1));

        distanceMatrix = new int[masterArray.Length, masterArray.Length];
        for (int i = 0; i < masterArray.Length; i++)
        {
            for (int j = 0; j < masterArray.Length; j++)
            {
                distanceMatrix[i, j] = (int)Globals.TrueDistance(masterArray[i], masterArray[j]);
            }
        }
    }

    public void Run(string filePrefix)
    {
        var randomPaths = RandomSolutions();
        var nNEPs = NearestNeighborEnd();
        var nNAPs = NearestNeighborAnyPosition();
        var greedyCycles = GreedyCycle();

        WriteSolutions($"{filePrefix}random.csv", randomPaths);
        WriteSolutions($"{filePrefix}NNEPs.csv", nNEPs);
        WriteSolutions($"{filePrefix}NNAPs.csv", nNAPs);
        WriteSolutions($"{filePrefix}greedy_cycles.csv", greedyCycles);
    }

    // 1. Random solutions (same seed for every instance)
    private List<byte>[] RandomSolutions()
    {
        var r = new Random(42);
        var paths = Enumerable.Range(0, masterArray.Length).Select(_ => new List<byte>(upTo)).ToArray();

        for (int i = 0; i < masterArray.Length; i++)
        {
            for (int j = 0; j < upTo; j++)
            {
                byte idx;
                while (usedNodes.Contains(idx = (byte)r.Next(0, masterArray.Length)))
                    ;
                paths[i].Add(idx);
                usedNodes.Add(idx);
            }
            usedNodes.Clear();
        }
        return paths;
    }

    // 2. Nearest neighbor end paths
    private List<byte>[] NearestNeighborEnd()
    {
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
        return nNEPs;
    }

    // 3. Nearest neighbor all paths
    private List<byte>[] NearestNeighborAnyPosition()
    {
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
        return nNAPs;
    }

    // 4. Greedy cycle
    private List<byte>[] GreedyCycle()
    {
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
        return greedyCycles;
    }

    private void WriteSolutions(string fileName, IEnumerable<List<byte>> solutions)
    {
        using var w = new StreamWriter(fileName);
        foreach (var solution in solutions)
        {
            foreach (var idx in solution)
            {
                var node = masterArray[idx];
                w.WriteLine($"{node.X};{node.Y};{node.Cost}");
            }
        }
        Console.WriteLine($"Wrote {fileName}");
    }
}

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
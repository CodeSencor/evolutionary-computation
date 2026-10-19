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

    Console.WriteLine($"\n################ {instance} ################");
    new Solver(path).Run(prefix);
}

class Solver
{
    private readonly Node[] masterArray;
    private readonly byte upTo;
    private readonly int[,] distanceMatrix;
    private readonly HashSet<byte> usedNodes = new();

    private static readonly (double Regret, double Delta)[] WeightSets =
    {
        (1.0, 1.5)
    };

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
        // 1. Nearest neighbor all paths + 2-regret
        var nNAPRegrets = RegretHeuristic(false, 1.0, 0.0);

        // 2. Greedy cycle + 2-regret
        var greedyCycleRegrets = RegretHeuristic(true, 1.0, 0.0);

        // 3. Nearest neighbor all paths + weighted sum (all weight sets, best kept)
        Console.WriteLine("=== Weight sets: Nearest neighbor all ===");
        var (nNAPWr, nNAPWd, nNAPWeighted) = BestWeighted(false, "Nearest neighbor all");

        // 4. Greedy cycle + weighted sum (all weight sets, best kept)
        Console.WriteLine("\n=== Weight sets: Greedy cycle ===");
        var (greedyWr, greedyWd, greedyCycleWeighted) = BestWeighted(true, "Greedy cycle");

        // Results summary (min / avg / max of the objective over all starting nodes)
        Console.WriteLine("\n=== Results ===");
        PrintStats("Nearest neighbor all + 2-regret", nNAPRegrets);
        PrintStats("Greedy cycle + 2-regret", greedyCycleRegrets);
        PrintStats($"Nearest neighbor all + weighted (best {nNAPWr}:{nNAPWd})", nNAPWeighted);
        PrintStats($"Greedy cycle + weighted (best {greedyWr}:{greedyWd})", greedyCycleWeighted);

        // Write to files for the visualizer
        WriteSolutions($"{filePrefix}NNAP_regrets.csv", nNAPRegrets);
        WriteSolutions($"{filePrefix}greedy_cycle_regrets.csv", greedyCycleRegrets);
        WriteSolutions($"{filePrefix}NNAP_weighted.csv", nNAPWeighted);
        WriteSolutions($"{filePrefix}greedy_cycle_weighted.csv", greedyCycleWeighted);
    }

    private int Evaluate(List<byte> path)
    {
        int sum = 0;
        for (int i = 0; i < path.Count; i++)
        {
            sum += distanceMatrix[path[i], path[(i + 1) % path.Count]] + masterArray[path[i]].Cost;
        }
        return sum;
    }

    private List<byte>[] RegretHeuristic(bool cycle, double wRegret, double wDelta)
    {
        var solutions = Enumerable
            .Range(0, masterArray.Length)
            .Select(_ => new List<byte>(upTo))
            .ToArray();

        for (byte i = 0; i < masterArray.Length; i++)
        {
            var solution = solutions[i];
            solution.Add(i);
            usedNodes.Add(i);
            while (solution.Count < upTo)
            {
                int count = solution.Count,
                    chosenK = -1,
                    chosenPos = -1,
                    chosenDelta = int.MaxValue;
                double bestScore = double.NegativeInfinity;

                for (byte k = 0; k < masterArray.Length; k++)
                {
                    if (usedNodes.Contains(k))
                    {
                        continue;
                    }

                    int best1 = int.MaxValue,
                        best2 = int.MaxValue,
                        pos1 = -1;

                    void Consider(int delta, int pos)
                    {
                        if (delta < best1)
                        {
                            best2 = best1;
                            best1 = delta;
                            pos1 = pos;
                        }
                        else if (delta < best2)
                        {
                            best2 = delta;
                        }
                    }

                    if (!cycle)
                    {
                        Consider(distanceMatrix[k, solution[0]] + masterArray[k].Cost, 0);
                        for (int j = 0; j < count; j++)
                        {
                            byte a = solution[j];
                            byte? b = j != count - 1 ? solution[j + 1] : null;
                            int delta = b is not null
                                ? distanceMatrix[a, k]
                                    + distanceMatrix[k, (int)b]
                                    - distanceMatrix[a, (int)b]
                                    + masterArray[k].Cost
                                : distanceMatrix[a, k] + masterArray[k].Cost;
                            Consider(delta, j + 1);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < count; j++)
                        {
                            byte a = solution[j],
                                b = solution[(j + 1) % count];
                            Consider(
                                distanceMatrix[a, k]
                                    + distanceMatrix[k, b]
                                    - distanceMatrix[a, b]
                                    + masterArray[k].Cost,
                                j + 1
                            );
                        }
                    }

                    int regret = best2 == int.MaxValue ? 0 : best2 - best1;
                    double score = wRegret * regret - wDelta * best1;

                    if (score > bestScore || (score == bestScore && best1 < chosenDelta))
                    {
                        bestScore = score;
                        chosenK = k;
                        chosenPos = pos1;
                        chosenDelta = best1;
                    }
                }

                usedNodes.Add((byte)chosenK);
                solution.Insert(chosenPos, (byte)chosenK);
            }
            usedNodes.Clear();
        }
        return solutions;
    }

    private void PrintStats(string name, List<byte>[] solutions)
    {
        var values = solutions.Select(Evaluate).ToArray();
        int bestIdx = Array.IndexOf(values, values.Min());
        Console.WriteLine(
            $"{name,-50} min: {values.Min(),7}  avg: {values.Average(),10:F1}  max: {values.Max(),7}  (best start: {bestIdx})"
        );
    }

    private (double Regret, double Delta, List<byte>[] Solutions) BestWeighted(bool cycle, string name)
    {
        double bestRegret = 0,
            bestDelta = 0,
            bestAvg = double.MaxValue;
        int bestMin = int.MaxValue;
        List<byte>[] bestSolutions = null!;

        foreach (var (wr, wd) in WeightSets)
        {
            var solutions = RegretHeuristic(cycle, wr, wd);
            PrintStats($"{name} weighted ({wr}:{wd})", solutions);

            var values = solutions.Select(Evaluate).ToArray();
            int min = values.Min();
            double avg = values.Average();
            if (min < bestMin || (min == bestMin && avg < bestAvg))
            {
                bestMin = min;
                bestAvg = avg;
                bestRegret = wr;
                bestDelta = wd;
                bestSolutions = solutions;
            }
        }
        return (bestRegret, bestDelta, bestSolutions);
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

    public static uint TrueDistance(Node current, Node next) =>
        (uint)Math.Round(Math.Sqrt(DistanceEstimate(current, next)));
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
}

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
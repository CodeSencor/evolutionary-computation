#!/usr/bin/env dotnet
#:project ../EvoToolbox/EvoToolbox.csproj
// Multiple start local search (MSLS): 400 times (random start -> steepest local search), the best local optimum
// is the result of one MSLS run. MSLS is run 20 times per instance.
// The local search uses MoveListSearch (list of improving moves).
// ILS (IteratedLocalSearch): random start -> local search, then repeat (perturb, local search, accept when better)
// until the time limit, which is the average time of MSLS. ILS is run 20 times per instance.
using System.Diagnostics;
using EvoToolbox.Constructs;
using EvoToolbox.Objects;
using EvoToolbox.Utilities;
using Path = EvoToolbox.Objects.Path;
using Move = (int Delta, int Kind, int A, int B, int C, int D);

(string Instance, string Prefix)[] instances =
{
    ("TSPA", "tspa"),
    ("TSPB", "tspb"),
};

bool runMsls = false, runIls = false;
double? timeLimitMs = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i].ToLowerInvariant())
    {
        case "--msls":
            runMsls = true;
            break;
        case "--ils":
            runIls = true;
            break;
        case "--time" when i + 1 < args.Length
                           && double.TryParse(args[i + 1], System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture, out var ms):
            timeLimitMs = ms;
            i++;
            break;
        default:
            Console.WriteLine($"Unknown or incomplete argument: {args[i]}");
            Console.WriteLine("Usage: [--MSLS] [--ILS] [--time <ms>]   (no method flag = both methods)");
            return;
    }
}
if (!runMsls && !runIls)
{
    runMsls = runIls = true;
}

foreach (var (instance, prefix) in instances)
{
    var path = $"{instance}.csv";
    if (!File.Exists(path))
    {
        Console.WriteLine($"Skipping {instance}: {path} not found.");
        continue;
    }

    Console.WriteLine($"Running {instance}...");
    new Solver(path).Run(prefix, runMsls, runIls, timeLimitMs);
}

class Solver
{
    private const int MslsRuns = 20;
    private const int MslsStarts = 400;
    private static readonly bool UseMoveList = true;
    private const int NodeExchange = 0, EdgeExchange = 1;

    private readonly Node[] masterArray;
    private readonly DistanceMatrix distanceMatrix;
    private readonly byte upTo;
    private readonly Random random = new(42);

    private Path solution = null!;
    private readonly int[] position;
    private long objective;

    private readonly SortedSet<Move> moves = new();

    public Solver(string instancePath)
    {
        masterArray = DataUtility.ImportNodes(instancePath);
        distanceMatrix = new DistanceMatrix(masterArray);
        upTo = (byte)(masterArray.Length / 2 + (masterArray.Length % 2 == 0 ? 0 : 1));
        position = new int[masterArray.Length];
    }

    public void Run(string filePrefix, bool runMsls, bool runIls, double? timeLimitMs)
    {
        long mslsAverageTime = 0;

        if (runMsls || (runIls && timeLimitMs is null))
        {
            List<Path> paths = new(MslsRuns);
            List<long> times = new(MslsRuns);
            List<long> objectives = new(MslsRuns);
            for (int run = 0; run < MslsRuns; run++)
            {
                var watch = Stopwatch.StartNew();
                var (best, bestObjective) = Msls();
                watch.Stop();

                paths.Add(best);
                times.Add(watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond);
                objectives.Add(bestObjective);
            }
            mslsAverageTime = (long)times.Average();

            if (runMsls)
            {
                DataUtility.ExportPaths($"{filePrefix}S_MSLS.csv", (paths, times));
                Console.WriteLine($"  MSLS  objective {objectives.Average():F2} ({objectives.Min()} - {objectives.Max()}), " +
                                  $"time {times.Average() / 1000:F3} ms");
                PrintBest("MSLS", paths, objectives);
            }
            else
            {
                Console.WriteLine($"  MSLS run only to measure the time limit for ILS: {mslsAverageTime / 1000.0:F3} ms");
            }
        }

        if (!runIls) return;

        List<List<int>> perturbDistributions = new()
        {
            new() { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5 },
            new() { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10 },
            new() { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 20 },
            new() { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10 },
            new() { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20 },
            new() { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10 },
            new() { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 20 },
        };

        var csvRows = new List<string> { "Strategy,Distribution,AvgObjective,MinObjective,MaxObjective,AvgTimeMs,AvgLocalSearches" };

        foreach (var strategy in new[] { 0 })
        {
            string strategyName = strategy == 0 ? "random" : "randomSeparate";
            Console.WriteLine($"ILS perturbation strategy: {strategy} ({strategyName})");
            
            foreach (var perturbDistribution in perturbDistributions)
            {
                string distStr = string.Join(";", perturbDistribution);
                Console.WriteLine($"  ILS with perturbation distribution: {string.Join(", ", perturbDistribution)}");
                
                long timeLimit = timeLimitMs is double ms ? (long)(ms * 1000) : mslsAverageTime;
                List<Path> ilsPaths = new(MslsRuns);
                List<long> ilsTimes = new(MslsRuns);
                List<long> ilsObjectives = new(MslsRuns);
                List<int> ilsIterations = new(MslsRuns);
                
                for (int run = 0; run < MslsRuns; run++)
                {
                    var watch = Stopwatch.StartNew();
                    var (best, bestObjective, iterations) = IteratedLocalSearch(timeLimit, strategy, perturbDistribution);
                    watch.Stop();

                    ilsPaths.Add(best);
                    ilsTimes.Add(watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond);
                    ilsObjectives.Add(bestObjective);
                    ilsIterations.Add(iterations);
                }

                DataUtility.ExportPaths($"{filePrefix}S_ILS.csv", (ilsPaths, ilsTimes));
                
                double avgObj = ilsObjectives.Average();
                long minObj = ilsObjectives.Min();
                long maxObj = ilsObjectives.Max();
                double avgTimeMs = ilsTimes.Average() / 1000.0;
                double avgIts = ilsIterations.Average();

                Console.WriteLine($"  ILS   objective {avgObj:F2} ({minObj} - {maxObj}), " +
                                  $"time {avgTimeMs:F3} ms, " +
                                  $"local searches {avgIts:F1} ({ilsIterations.Min()} - {ilsIterations.Max()})");
                PrintBest("ILS", ilsPaths, ilsObjectives);

                csvRows.Add($"{strategyName},\"{distStr}\",{avgObj:F2},{minObj},{maxObj},{avgTimeMs:F3},{avgIts:F1}");
            }
        }

        string summaryFileName = $"{filePrefix}_experiment_summary.csv";
        File.WriteAllLines(summaryFileName, csvRows);
        Console.WriteLine($"  -> Saved experiment summary table to {summaryFileName}");
    }

    private static void PrintBest(string name, List<Path> paths, List<long> objectives)
    {
        int bestRun = objectives.IndexOf(objectives.Min());
        var bestPath = paths[bestRun];
        Console.WriteLine($"  {name} best solution ({objectives[bestRun]}): " +
                          string.Join(", ", Enumerable.Range(0, bestPath.Count).Select(i => bestPath[(byte)i].Index)));
    }

    private (Path Best, long Objective) Msls()
    {
        Path best = null!;
        long bestObjective = long.MaxValue;
        for (int i = 0; i < MslsStarts; i++)
        {
            Load(RandomStart());
            LocalSearch();

            if (objective != Objective())
                throw new Exception($"tracked objective {objective} != real objective {Objective()}");

            if (objective < bestObjective)
            {
                bestObjective = objective;
                best = solution;
            }
        }
        return (best, bestObjective);
    }

    private (Path Best, long Objective, int Iterations) IteratedLocalSearch(long timeLimitMicroseconds, int perturbStrategy, List<int> perturbDistribution)
    {
        var watch = Stopwatch.StartNew();

        Load(RandomStart());
        LocalSearch();
        int iterations = 1;

        byte[] current = Snapshot();
        long currentObjective = objective;

        int pt = 0;
        while (watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond < timeLimitMicroseconds)
        {
            Perturb(perturbDistribution[pt], perturbStrategy);
            pt = Math.Min(pt + 1, perturbDistribution.Count - 1);
            LocalSearch();
            iterations++;

            if (objective != Objective())
                throw new Exception($"tracked objective {objective} != real objective {Objective()}");

            if (objective < currentObjective)
            {
                current = Snapshot();
                currentObjective = objective;
                pt = 0;
            }
            else
            {
                Load(current);
            }
        }

        Load(current);
        return (solution, objective, iterations);
    }

    private void Perturb(int count, int strategy = 0)
    {
        switch (strategy)
        {
            case 1:
                PerturbRandomSeparate(count);
                break;
            default:
                PerturbRandom(count);
                break;
        }
    }

    private void PerturbRandom(int count)
    {
        var places = Enumerable.Range(0, upTo).ToArray();
        random.Shuffle(places);

        foreach (var place in places.Take(Math.Min(count, upTo)))
        {
            ReplaceNodeAt((byte)place);
        }
    }

    private void PerturbRandomSeparate(int count)
    {
        int actualCount = Math.Min(count, upTo);
        if (actualCount <= 0) return;

        var currentNodes = Snapshot().ToList();
        var unusedNodes = masterArray
            .Select(n => n.Index)
            .Where(idx => position[idx] < 0)
            .ToArray();
        random.Shuffle(unusedNodes);

        for (int i = 0; i < actualCount; i++)
        {
            int removeIdx = random.Next(currentNodes.Count);
            currentNodes.RemoveAt(removeIdx);
        }

        for (int i = 0; i < actualCount; i++)
        {
            int insertIdx = random.Next(currentNodes.Count + 1);
            currentNodes.Insert(insertIdx, unusedNodes[i]);
        }

        Load(currentNodes.ToArray());
    }

    private void ReplaceNodeAt(byte place)
    {
        int u;
        do
        {
            u = random.Next(masterArray.Length);
        } while (position[u] >= 0);

        Apply(Exchange(solution[place].Index, u));
    }

    private byte[] Snapshot() => Enumerable.Range(0, upTo).Select(i => solution[(byte)i].Index).ToArray();

    private byte[] RandomStart()
    {
        var nodes = masterArray.Select(n => n.Index).ToArray();
        random.Shuffle(nodes);
        return nodes[..upTo];
    }

    private void LocalSearch()
    {
        if (UseMoveList) MoveListSearch();
        else SteepestSearch();
    }

    private void SteepestSearch()
    {
        while (true)
        {
            Move best = (0, 0, 0, 0, 0, 0);
            EvaluateAll(false, m => { if (m.Delta < best.Delta) best = m; });
            if (best.Delta >= 0) return;
            Apply(best);
        }
    }

    private void MoveListSearch()
    {
        moves.Clear();
        EvaluateAll(true, AddImproving);
        bool justEvaluated = true;
        var stale = new List<Move>();
        while (true)
        {
            Move? found = null;
            stale.Clear();
            foreach (var m in moves)
            {
                var state = Check(m);
                if (state == MoveState.Remove) stale.Add(m);
                else if (state == MoveState.Apply) { found = m; break; }
            }
            foreach (var m in stale) moves.Remove(m);

            if (found is Move move)
            {
                moves.Remove(move);
                Apply(move);
                EvaluateNew(move);
                justEvaluated = false;
                continue;
            }

            if (justEvaluated) return;
            EvaluateAll(true, AddImproving);
            justEvaluated = true;
        }
    }

    private MoveState Check(Move m)
    {
        if (m.Kind == NodeExchange)
            return position[m.D] < 0 && Edge(m.A, m.B) != 0 && Edge(m.B, m.C) != 0 ? MoveState.Apply : MoveState.Remove;
        
        int first = Edge(m.A, m.B), second = Edge(m.C, m.D);
        if (first == 0 || second == 0) return MoveState.Remove;
        return first == second ? MoveState.Apply : MoveState.Keep;
    }

    private void EvaluateAll(bool withInverted, Action<Move> consider)
    {
        for (byte i = 0; i < upTo; i++)
            foreach (var u in masterArray)
                if (position[u.Index] < 0)
                    consider(Exchange(solution[i].Index, u.Index));

        for (byte i = 0; i < upTo - 2; i++)
        {
            for (byte j = (byte)(i + 2); j < upTo; j++)
            {
                if (i == 0 && j == upTo - 1) continue;
                Node a = solution[i], an = solution.Successor(i), b = solution[j], bn = solution.Successor(j);
                consider(TwoEdges(a.Index, an.Index, b.Index, bn.Index));
                if (withInverted) consider(TwoEdges(an.Index, a.Index, b.Index, bn.Index));
            }
        }
    }

    private void EvaluateNew(Move applied)
    {
        int[] changed = applied.Kind == NodeExchange
            ? [applied.A, applied.D, applied.C]
            : [applied.A, applied.B, applied.C, applied.D];

        foreach (var v in changed)
            foreach (var u in masterArray)
                if (position[u.Index] < 0)
                    AddImproving(Exchange(v, u.Index));

        if (applied.Kind == NodeExchange)
            for (byte i = 0; i < upTo; i++)
                AddImproving(Exchange(solution[i].Index, applied.B));

        foreach (var v in changed)
        {
            int vn = Next(v);
            for (byte i = 0; i < upTo; i++)
            {
                int x = solution[i].Index, xn = solution.Successor(i).Index;
                if (x == v || x == vn || xn == v || xn == vn) continue;
                AddImproving(TwoEdges(v, vn, x, xn));
                AddImproving(TwoEdges(vn, v, x, xn));
            }
        }
    }

    private void AddImproving(Move m)
    {
        if (m.Delta >= 0) return;
        if (m.Kind == EdgeExchange)
        {
            int min = Math.Min(Math.Min(m.A, m.B), Math.Min(m.C, m.D));
            m = min == m.A ? m
                : min == m.B ? (m.Delta, m.Kind, m.B, m.A, m.D, m.C)
                : min == m.C ? (m.Delta, m.Kind, m.C, m.D, m.A, m.B)
                : (m.Delta, m.Kind, m.D, m.C, m.B, m.A);
        }
        moves.Add(m);
    }

    private Move Exchange(int v, int u)
    {
        var at = (byte)position[v];
        int a = solution.Predecessor(at).Index, b = solution.Successor(at).Index;
        return ((int)solution.TestNodeExchange(at, masterArray[u], distanceMatrix), NodeExchange, Math.Min(a, b), v, Math.Max(a, b), u);
    }

    private Move TwoEdges(int a, int an, int b, int bn) =>
        (Distance(a, b) + Distance(an, bn) - Distance(a, an) - Distance(b, bn), EdgeExchange, a, an, b, bn);

    private void Apply(Move m)
    {
        objective += m.Delta;
        if (m.Kind == NodeExchange)
        {
            solution.NodeExchange((byte)m.B, (byte)m.D);
            position[m.D] = position[m.B];
            position[m.B] = -1;
            return;
        }

        int first = Edge(m.A, m.B) == 1 ? position[m.A] : position[m.B],
            second = Edge(m.A, m.B) == 1 ? position[m.C] : position[m.D];
        solution.EdgeSwap(first, second);
        for (int k = Math.Min(first, second) + 1; k <= Math.Max(first, second); k++)
            position[solution[(byte)k].Index] = k;
    }

    private int Edge(int a, int b) => position[a] < 0 || position[b] < 0 ? 0 : Next(a) == b ? 1 : Next(b) == a ? -1 : 0;
    private int Next(int v) => solution.Successor((byte)position[v]).Index;
    private int Distance(int a, int b) => (int)distanceMatrix.Distance(masterArray[a], masterArray[b]);

    private void Load(byte[] start)
    {
        solution = new Path(masterArray);
        Array.Fill(position, -1);
        foreach (var v in start)
        {
            position[v] = solution.Count;
            solution.AppendNode(v);
        }
        objective = Objective();
    }

    private long Objective()
    {
        long sum = 0;
        for (byte i = 0; i < upTo; i++)
            sum += distanceMatrix.Distance(solution[i], solution.Successor(i)) + solution[i].Cost;
        return sum;
    }
}

enum MoveState
{
    Remove,
    Apply,
    Keep
}
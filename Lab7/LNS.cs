#!/usr/bin/env dotnet
#:project ../EvoToolbox/EvoToolbox.csproj
// Large neighbourhood search (LNS): random start -> local search, then repeat (destroy, repair, optionally local
// search, accept when better) until the time limit = the average running time of MSLS from Lab6.
// Three destroy operators (each removes 30% of the nodes of the solution, used separately):
//   RN - random nodes:        30% of the positions, chosen uniformly at random,
//   SP - single subpath:      one random subpath of 30% of the nodes,
//   MP - multiple subpaths:   3-5 random disjoint subpaths with 30% of the nodes in total.
// Repair: greedy cycle + weighted 2-regret (score = regret - best insertion cost, weights 1:1, the best greedy
// heuristic of Lab2) inserting into the partial cycle until it has ceil(n/2) nodes again.
// Local search: steepest, list of improving moves (LM) from Lab5. Every destroy operator is run with and without
// the local search after repair (6 methods), 20 runs each.
using System.Diagnostics;
using EvoToolbox.Constructs;
using EvoToolbox.Objects;
using EvoToolbox.Utilities;
using Path = EvoToolbox.Objects.Path;
using Move = (int Delta, int Kind, int A, int B, int C, int D);

// MSLS time = average running time of one MSLS run in Lab6 (tspaS_MSLS.csv / tspbS_MSLS.csv)
(string Instance, string Prefix, double MslsTimeMs)[] instances =
{
    ("TSPA", "tspa", 2994.66),
    ("TSPB", "tspb", 2980.06),
};

double? timeLimitMs = null;
for (int i = 0; i < args.Length; i++)
{
    if (args[i].ToLowerInvariant() == "--time" && i + 1 < args.Length
        && double.TryParse(args[i + 1], System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var ms))
    {
        timeLimitMs = ms;
        i++;
        continue;
    }
    Console.WriteLine($"Unknown or incomplete argument: {args[i]}");
    Console.WriteLine("Usage: [--time <ms>]   (default: the average MSLS time of the instance from Lab6)");
    return;
}

foreach (var (instance, prefix, mslsTimeMs) in instances)
{
    var path = $"{instance}.csv";
    if (!File.Exists(path))
    {
        Console.WriteLine($"Skipping {instance}: {path} not found.");
        continue;
    }

    Console.WriteLine($"Running {instance}...");
    new Solver(path).Run(prefix, timeLimitMs ?? mslsTimeMs);
}

// A move is a tuple (Delta, Kind, A, B, C, D) of node indices, so the sorted list orders moves from the best.
//   Kind 0, node exchange:  remove edges (A, B), (B, C), put not selected node D in place of B.
//   Kind 1, edges exchange: remove edges A -> B and C -> D, add edges (A, C) and (B, D).
//                           Applicable when both removed edges have the saved relative direction.
class Solver
{
    private const int Runs = 20;
    private const double DestroyFraction = 0.3;
    private const double RegretWeight = 1.0, DeltaWeight = 1.0;
    private const int NodeExchange = 0, EdgeExchange = 1;
    private const int RandomNodes = 0, SingleSubpath = 1, MultipleSubpaths = 2;

    private readonly Node[] masterArray;
    private readonly DistanceMatrix distanceMatrix;
    private readonly byte upTo;
    private readonly Random random = new(42);

    // Current solution and position[v] = position of node v in it, or -1 when v is not selected
    private Path solution = null!;
    private readonly int[] position;
    private long objective;

    // LM - list of improving moves, ordered from the best to the worst
    private readonly SortedSet<Move> moves = new();

    public Solver(string instancePath)
    {
        masterArray = DataUtility.ImportNodes(instancePath);
        distanceMatrix = new DistanceMatrix(masterArray);
        upTo = (byte)(masterArray.Length / 2 + (masterArray.Length % 2 == 0 ? 0 : 1));
        position = new int[masterArray.Length];
    }

    public void Run(string filePrefix, double timeLimitMs)
    {
        Console.WriteLine($"  time limit {timeLimitMs:F2} ms");
        var timeLimit = (long)(timeLimitMs * 1000);
        var iterationRows = new List<string> { "Method,Iterations" };
        foreach (var (destroyName, destroy) in new[] { ("RN", RandomNodes), ("SP", SingleSubpath), ("MP", MultipleSubpaths) })
        {
            foreach (var useLocalSearch in new[] { true, false })
            {
                var name = $"LNS_{destroyName}_{(useLocalSearch ? "LS" : "noLS")}";
                List<Path> paths = new(Runs);
                List<long> times = new(Runs);
                List<long> objectives = new(Runs);
                List<int> iterations = new(Runs);
                for (int run = 0; run < Runs; run++)
                {
                    var watch = Stopwatch.StartNew();
                    var (best, bestObjective, its) = LargeNeighbourhoodSearch(timeLimit, destroy, useLocalSearch);
                    watch.Stop();

                    paths.Add(best);
                    times.Add(watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond);
                    objectives.Add(bestObjective);
                    iterations.Add(its);
                    iterationRows.Add($"{name},{its}");
                }

                DataUtility.ExportPaths($"{filePrefix}S_{name}.csv", (paths, times));
                Console.WriteLine($"  {name,-12}  objective {objectives.Average():F2} ({objectives.Min()} - {objectives.Max()}), " +
                                  $"time {times.Average() / 1000:F3} ms, " +
                                  $"iterations {iterations.Average():F1} ({iterations.Min()} - {iterations.Max()})");
            }
        }
        File.WriteAllLines($"{filePrefix}_LNS_iterations.csv", iterationRows);
    }

    // 1. LNS: random start + local search, then destroy -> repair (-> local search), accept only improvements
    private (Path Best, long Objective, int Iterations) LargeNeighbourhoodSearch(long timeLimitMicroseconds, int destroy,
        bool useLocalSearch)
    {
        var watch = Stopwatch.StartNew();

        Load(RandomStart());
        MoveListSearch();
        int iterations = 0;

        byte[] current = Snapshot();
        long currentObjective = objective;

        while (watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond < timeLimitMicroseconds)
        {
            Load(Repair(Destroy(current, destroy)));
            if (useLocalSearch)
            {
                MoveListSearch();
            }
            iterations++;

            // self-check: the objective tracked through the move deltas must match the recomputed one
            if (objective != Objective())
            {
                throw new Exception($"tracked objective {objective} != real objective {Objective()}");
            }

            if (objective < currentObjective)
            {
                current = Snapshot();
                currentObjective = objective;
            }
        }

        Load(current);
        return (solution, objective, iterations);
    }

    // 2. Destroy: remove 30% of the nodes, the remaining nodes keep their order (a partial cycle)
    private List<byte> Destroy(byte[] cycle, int destroy)
    {
        int remove = (int)Math.Round(DestroyFraction * cycle.Length);
        var removed = new bool[cycle.Length];
        if (destroy == RandomNodes)
        {
            var places = Enumerable.Range(0, cycle.Length).ToArray();
            random.Shuffle(places);
            foreach (var place in places[..remove])
            {
                removed[place] = true;
            }
        }
        else
        {
            // k removed subpaths alternate with k kept subpaths around the cycle; random lengths, random rotation
            int count = destroy == SingleSubpath ? 1 : random.Next(3, 6);
            int[] removedLengths = RandomParts(remove, count), keptLengths = RandomParts(cycle.Length - remove, count);
            int at = random.Next(cycle.Length);
            for (int s = 0; s < count; s++)
            {
                for (int l = 0; l < removedLengths[s]; l++)
                {
                    removed[(at + l) % cycle.Length] = true;
                }
                at += removedLengths[s] + keptLengths[s];
            }
        }

        var partial = cycle.Where((_, i) => !removed[i]).ToList();
        if (partial.Count != cycle.Length - remove)
        {
            throw new Exception($"destroy {destroy}: {cycle.Length - partial.Count} nodes removed, expected {remove}");
        }
        return partial;
    }

    // total split into count random positive parts (count - 1 distinct random cut points)
    private int[] RandomParts(int total, int count)
    {
        var cuts = Enumerable.Range(1, total - 1).ToArray();
        random.Shuffle(cuts);
        var bounds = cuts[..(count - 1)].Append(0).Append(total).Order().ToArray();
        return Enumerable.Range(0, count).Select(i => bounds[i + 1] - bounds[i]).ToArray();
    }

    // 3. Repair: greedy cycle + weighted 2-regret, every not selected node (also the removed ones) can be inserted
    private byte[] Repair(List<byte> partial)
    {
        var used = new bool[masterArray.Length];
        foreach (var v in partial)
        {
            used[v] = true;
        }

        while (partial.Count < upTo)
        {
            int count = partial.Count,
                chosenK = -1,
                chosenPos = -1,
                chosenDelta = int.MaxValue;
            double bestScore = double.NegativeInfinity;

            for (int k = 0; k < masterArray.Length; k++)
            {
                if (used[k])
                {
                    continue;
                }

                int best1 = int.MaxValue,
                    best2 = int.MaxValue,
                    pos1 = -1;
                for (int j = 0; j < count; j++)
                {
                    int a = partial[j], b = partial[(j + 1) % count];
                    int delta = Distance(a, k) + Distance(k, b) - Distance(a, b) + masterArray[k].Cost;
                    if (delta < best1)
                    {
                        best2 = best1;
                        best1 = delta;
                        pos1 = j + 1;
                    }
                    else if (delta < best2)
                    {
                        best2 = delta;
                    }
                }

                int regret = best2 == int.MaxValue ? 0 : best2 - best1;
                double score = RegretWeight * regret - DeltaWeight * best1;
                if (score > bestScore || (score == bestScore && best1 < chosenDelta))
                {
                    bestScore = score;
                    chosenK = k;
                    chosenPos = pos1;
                    chosenDelta = best1;
                }
            }

            used[chosenK] = true;
            partial.Insert(chosenPos, (byte)chosenK);
        }
        return partial.ToArray();
    }

    private byte[] Snapshot() => Enumerable.Range(0, upTo).Select(i => solution[(byte)i].Index).ToArray();

    // 4. Random starting solution
    private byte[] RandomStart()
    {
        var nodes = masterArray.Select(n => n.Index).ToArray();
        random.Shuffle(nodes);
        return nodes[..upTo];
    }

    // 5. Steepest local search with the list of improving moves (LM), as in Lab5
    private void MoveListSearch()
    {
        moves.Clear();
        EvaluateAll(AddImproving);
        bool justEvaluated = true;
        var stale = new List<Move>();
        while (true)
        {
            Move? found = null;
            stale.Clear();
            foreach (var m in moves)
            {
                var state = Check(m);
                if (state == MoveState.Remove)
                {
                    stale.Add(m);
                }
                else if (state == MoveState.Apply)
                {
                    found = m;
                    break;
                }
                // MoveState.Keep: edges exist but in a different relative direction, browse further
            }
            foreach (var m in stale)
            {
                moves.Remove(m);
            }

            if (found is Move move)
            {
                moves.Remove(move);
                Apply(move);
                EvaluateNew(move);
                justEvaluated = false;
                continue;
            }

            if (justEvaluated)
            {
                return;
            }
            // no applicable move in LM: evaluate the whole neighbourhood again (finds nothing in a local optimum)
            EvaluateAll(AddImproving);
            justEvaluated = true;
        }
    }

    private MoveState Check(Move m)
    {
        if (m.Kind == NodeExchange)
        {
            return position[m.D] < 0 && Edge(m.A, m.B) != 0 && Edge(m.B, m.C) != 0 ? MoveState.Apply : MoveState.Remove;
        }
        int first = Edge(m.A, m.B), second = Edge(m.C, m.D);
        if (first == 0 || second == 0)
        {
            return MoveState.Remove;
        }
        return first == second ? MoveState.Apply : MoveState.Keep;
    }

    // Evaluates every move of the current solution, also the edge exchanges with one edge relatively inverted
    private void EvaluateAll(Action<Move> consider)
    {
        for (byte i = 0; i < upTo; i++)
        {
            foreach (var u in masterArray)
            {
                if (position[u.Index] < 0)
                {
                    consider(Exchange(solution[i].Index, u.Index));
                }
            }
        }

        for (byte i = 0; i < upTo - 2; i++)
        {
            for (byte j = (byte)(i + 2); j < upTo; j++)
            {
                if (i == 0 && j == upTo - 1)
                {
                    continue; // the two edges would share node solution[0]
                }
                Node a = solution[i], an = solution.Successor(i), b = solution[j], bn = solution.Successor(j);
                consider(TwoEdges(a.Index, an.Index, b.Index, bn.Index));
                consider(TwoEdges(an.Index, a.Index, b.Index, bn.Index));
            }
        }
    }

    // After applying a move only the moves using a new edge or a node with new neighbours can be new
    private void EvaluateNew(Move applied)
    {
        int[] changed = applied.Kind == NodeExchange
            ? [applied.A, applied.D, applied.C]
            : [applied.A, applied.B, applied.C, applied.D];

        foreach (var v in changed)
        {
            foreach (var u in masterArray)
            {
                if (position[u.Index] < 0)
                {
                    AddImproving(Exchange(v, u.Index));
                }
            }
        }
        if (applied.Kind == NodeExchange)
        {
            // the removed node can now replace any selected node
            for (byte i = 0; i < upTo; i++)
            {
                AddImproving(Exchange(solution[i].Index, applied.B));
            }
        }

        // every new edge is v -> Next(v) for one of the changed nodes v (both directions relative to the other edge)
        foreach (var v in changed)
        {
            int vn = Next(v);
            for (byte i = 0; i < upTo; i++)
            {
                int x = solution[i].Index, xn = solution.Successor(i).Index;
                if (x == v || x == vn || xn == v || xn == vn)
                {
                    continue;
                }
                AddImproving(TwoEdges(v, vn, x, xn));
                AddImproving(TwoEdges(vn, v, x, xn));
            }
        }
    }

    private void AddImproving(Move m)
    {
        if (m.Delta >= 0)
        {
            return;
        }
        if (m.Kind == EdgeExchange)
        {
            // the same move has four notations (one per first node), keep the one starting with the smallest node
            // so that LM holds no duplicates
            int min = Math.Min(Math.Min(m.A, m.B), Math.Min(m.C, m.D));
            m = min == m.A ? m
                : min == m.B ? (m.Delta, m.Kind, m.B, m.A, m.D, m.C)
                : min == m.C ? (m.Delta, m.Kind, m.C, m.D, m.A, m.B)
                : (m.Delta, m.Kind, m.D, m.C, m.B, m.A);
        }
        moves.Add(m);
    }

    // Selected node v replaced by not selected node u
    private Move Exchange(int v, int u)
    {
        var at = (byte)position[v];
        int a = solution.Predecessor(at).Index, b = solution.Successor(at).Index;
        return ((int)solution.TestNodeExchange(at, masterArray[u], distanceMatrix), NodeExchange, Math.Min(a, b), v,
            Math.Max(a, b), u);
    }

    // Not through Path.TestEdgeSwap: the relatively inverted moves are not moves of the current path
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

        // A -> B and C -> D: reverse B ... C, both reversed (B -> A and D -> C): reverse A ... D
        int first = Edge(m.A, m.B) == 1 ? position[m.A] : position[m.B],
            second = Edge(m.A, m.B) == 1 ? position[m.C] : position[m.D];
        solution.EdgeSwap(first, second);
        for (int k = Math.Min(first, second) + 1; k <= Math.Max(first, second); k++)
        {
            position[solution[(byte)k].Index] = k;
        }
    }

    // 1 = edge a -> b in the path direction, -1 = edge b -> a, 0 = no such edge
    private int Edge(int a, int b) =>
        position[a] < 0 || position[b] < 0 ? 0 : Next(a) == b ? 1 : Next(b) == a ? -1 : 0;

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
        {
            sum += distanceMatrix.Distance(solution[i], solution.Successor(i)) + solution[i].Cost;
        }
        return sum;
    }
}

enum MoveState
{
    Remove,
    Apply,
    Keep
}

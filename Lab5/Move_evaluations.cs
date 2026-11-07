#!/usr/bin/env dotnet
#:project ../EvoToolbox/EvoToolbox.csproj
// Steepest local search: two-edges exchange (intra-route) + exchange of a selected and a not selected node
// (inter-route), random starting solutions. Two variants are run on the same 200 starting solutions:
//   base - the whole neighbourhood is evaluated in every iteration (baseline, as in Lab3),
//   LM   - list of improving moves: evaluations from previous iterations are reused, after a move only the
//          moves touching the changed edges / nodes are evaluated.
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

// A move is a tuple (Delta, Kind, A, B, C, D) of node indices, so the sorted list orders moves from the best.
//   Kind 0, node exchange:  remove edges (A, B), (B, C), put not selected node D in place of B.
//   Kind 1, edges exchange: remove edges A -> B and C -> D, add edges (A, C) and (B, D).
//                           Applicable when both removed edges have the saved relative direction.
class Solver
{
    private const int Runs = 200;
    private const int NodeExchange = 0, EdgeExchange = 1;

    private readonly Node[] masterArray;
    private readonly DistanceMatrix distanceMatrix;
    private readonly byte upTo;

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

    public void Run(string filePrefix)
    {
        var starts = RandomSolutions();
        foreach (var (name, useList) in new[] { ("base", false), ("LM", true) })
        {
            List<Path> paths = new(Runs);
            List<long> times = new(Runs);
            List<long> objectives = new(Runs);
            foreach (var start in starts)
            {
                Load(start);
                var watch = Stopwatch.StartNew();
                if (useList)
                {
                    MoveListSearch();
                }
                else
                {
                    SteepestSearch();
                }
                watch.Stop();

                // self-check: the objective tracked through the move deltas must match the recomputed one
                if (objective != Objective())
                {
                    throw new Exception($"{name}: tracked objective {objective} != real objective {Objective()}");
                }
                paths.Add(solution);
                times.Add(watch.Elapsed.Ticks / TimeSpan.TicksPerMicrosecond);
                objectives.Add(objective);
            }

            DataUtility.ExportPaths($"{filePrefix}S_ES_R_{name}.csv", (paths, times));
            Console.WriteLine($"  {name,-4}  objective {objectives.Average():F2} ({objectives.Min()} - {objectives.Max()}), " +
                              $"time {times.Average() / 1000:F3} ms");
        }
    }

    // 1. Random starting solutions (same seed for every instance, the same starts for both methods)
    private List<byte[]> RandomSolutions()
    {
        var r = new Random(42);
        var solutions = new List<byte[]>(Runs);
        for (int i = 0; i < Runs; i++)
        {
            var nodes = masterArray.Select(n => n.Index).ToArray();
            r.Shuffle(nodes);
            solutions.Add(nodes[..upTo]);
        }
        return solutions;
    }

    // 2. Baseline steepest local search: evaluate the whole neighbourhood, apply the best move
    private void SteepestSearch()
    {
        while (true)
        {
            Move best = (0, 0, 0, 0, 0, 0);
            EvaluateAll(false, m =>
            {
                if (m.Delta < best.Delta)
                {
                    best = m;
                }
            });
            if (best.Delta >= 0)
            {
                return;
            }
            Apply(best);
        }
    }

    // 3. Steepest local search with the list of improving moves (LM)
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
            EvaluateAll(true, AddImproving);
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

    // Evaluates every move of the current solution;
    // withInverted: also the edge exchanges with one edge relatively inverted (not applicable now, maybe later)
    private void EvaluateAll(bool withInverted, Action<Move> consider)
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
                if (withInverted)
                {
                    consider(TwoEdges(an.Index, a.Index, b.Index, bn.Index));
                }
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

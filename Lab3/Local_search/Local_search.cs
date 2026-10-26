using System.Diagnostics;
using System.Diagnostics.Tracing;
using EvoToolbox.Constructs;
using EvoToolbox.Utilities;
using EvoToolbox.Objects;
using Path = EvoToolbox.Objects.Path;

var r = new Random(42);

var tspaArray = DataUtility.ImportNodes("TSPA.csv");
var tspaDistanceMatrix = new DistanceMatrix(tspaArray);
var tspbArray = DataUtility.ImportNodes("TSPB.csv");
var tspbDistanceMatrix = new DistanceMatrix(tspbArray);
byte upTo = (byte)(tspaArray.Length / 2 + (tspaArray.Length % 2 == 0 ? 0 : 1));

var tspaG_NS_R = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspaG_NS_GR = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspaG_ES_R = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspaG_ES_GR = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspaS_NS_R = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspaS_NS_GR = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspaS_ES_R = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspaS_ES_GR = Hypersolver(tspaArray, tspaDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);

var tspbG_NS_R = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspbG_NS_GR = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspbG_ES_R = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspbG_ES_GR = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.GREEDY, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspbS_NS_R = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspbS_NS_GR = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.NODE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);
var tspbS_ES_R = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.RANDOM, r);
var tspbS_ES_GR = Hypersolver(tspbArray, tspbDistanceMatrix, upTo, SearchMode.STEEP, NeighborhoodType.EDGE_SWAP,
    InitialSolutionsType.GREEDY_REGRET, r);

DataUtility.ExportPaths($"./{nameof(tspaG_NS_R)}.csv", tspaG_NS_R);
DataUtility.ExportPaths($"./{nameof(tspaG_NS_GR)}.csv", tspaG_NS_GR);
DataUtility.ExportPaths($"./{nameof(tspaG_ES_R)}.csv", tspaG_ES_R);
DataUtility.ExportPaths($"./{nameof(tspaG_ES_GR)}.csv", tspaG_ES_GR);
DataUtility.ExportPaths($"./{nameof(tspaS_NS_R)}.csv", tspaS_NS_R);
DataUtility.ExportPaths($"./{nameof(tspaS_NS_GR)}.csv", tspaS_NS_GR);
DataUtility.ExportPaths($"./{nameof(tspaS_ES_R)}.csv", tspaS_ES_R);
DataUtility.ExportPaths($"./{nameof(tspaS_ES_GR)}.csv", tspaS_ES_GR);

DataUtility.ExportPaths($"./{nameof(tspbG_NS_R)}.csv", tspbG_NS_R);
DataUtility.ExportPaths($"./{nameof(tspbG_NS_GR)}.csv", tspbG_NS_GR);
DataUtility.ExportPaths($"./{nameof(tspbG_ES_R)}.csv", tspbG_ES_R);
DataUtility.ExportPaths($"./{nameof(tspbG_ES_GR)}.csv", tspbG_ES_GR);
DataUtility.ExportPaths($"./{nameof(tspbS_NS_R)}.csv", tspbS_NS_R);
DataUtility.ExportPaths($"./{nameof(tspbS_NS_GR)}.csv", tspbS_NS_GR);
DataUtility.ExportPaths($"./{nameof(tspbS_ES_R)}.csv", tspbS_ES_R);
DataUtility.ExportPaths($"./{nameof(tspbS_ES_GR)}.csv", tspbS_ES_GR);



(Path, HashSet<int>) RandomSolution(Node[] masterArray, Random random, byte upTo)
{
    var solution = new Path(masterArray);
    var usedNodes = new HashSet<int>();
    for (int j = 0; j < upTo; j++)
    {
        byte idx;
        while (usedNodes.Contains(idx = (byte)random.Next(0, masterArray.Length)))
            ;
        solution.AppendNode(idx);
        usedNodes.Add(idx);
    }

    return (solution, usedNodes);
}

(Path, HashSet<int>) GreedyRegret(byte seedIdx, Node[] masterArray, DistanceMatrix distanceMatrix, double wRegret,
    double wDelta, byte upTo)
{
    var solution = new Path(masterArray);
    var usedNodes = new HashSet<int>();
    solution.AppendNode(seedIdx);
    usedNodes.Add(seedIdx);
    while (solution.Count < upTo)
    {
        int count = solution.Count,
            chosenK = -1,
            chosenPos = -1;
        uint chosenDelta = uint.MaxValue;
        double bestScore = double.NegativeInfinity;

        for (byte k = 0; k < masterArray.Length; k++)
        {
            if (usedNodes.Contains(k))
            {
                continue;
            }

            uint best1 = uint.MaxValue,
                best2 = uint.MaxValue;
            int pos1 = -1;

            void Consider(uint delta, int pos)
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

            for (byte j = 0; j < count; j++)
            {
                Node a = solution[j],
                    b = solution.OnLoopingIndex(j + 1);
                Consider(
                    distanceMatrix.Delta(a, b, masterArray[k]),
                    j + 1
                );
            }

            uint regret = best2 == uint.MaxValue ? 0 : best2 - best1;
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
        solution.AddBeforeNode((byte)chosenK, (byte)chosenPos);
    }

    return (solution, usedNodes);
}

void GreedySolver((Path solution, HashSet<int> usedNodes) p, Node[] masterArray, DistanceMatrix distanceMatrix, List<(byte, byte)> possibleExchanges, List<(byte, byte)> possibleSwaps, int[] usableNodes, Random random, ref bool improvement, NeighborhoodType neighborhoodType)
{
    improvement = false;
    int exchanges = 0, swaps = 0;
    while (exchanges < possibleExchanges.Count || swaps < possibleSwaps.Count)
    {
        var coinToss = exchanges < possibleExchanges.Count && swaps < possibleSwaps.Count ?
            random.Next(2)
            : exchanges < possibleExchanges.Count ? 1 : 0;
        if (coinToss == 0)
        {
            var swapPair = possibleSwaps[swaps++];
            var trial = neighborhoodType switch
            {
                NeighborhoodType.NODE_SWAP => p.solution.TestNodeSwap(swapPair.Item1, swapPair.Item2, distanceMatrix),
                NeighborhoodType.EDGE_SWAP => p.solution.TestEdgeSwap(swapPair.Item1, swapPair.Item2, distanceMatrix),
                _ => throw new Exception("Invalid SolveMode.")
            };
            if (trial < 0)
            { 
                switch(neighborhoodType)
                {
                    case NeighborhoodType.NODE_SWAP:
                        p.solution.NodeSwap(swapPair.Item1, swapPair.Item2);
                        break;
                    case NeighborhoodType.EDGE_SWAP:
                        p.solution.EdgeSwap(swapPair.Item1, swapPair.Item2);
                        break;
                    default:
                        throw new Exception("Invalid SolveMode.");
                };
                improvement = true;
                break;
            }
        }
        else
        {
            var exchangePair = possibleExchanges[exchanges++];
            var newNode = masterArray[usableNodes[exchangePair.Item2]];
            var trial = p.solution.TestNodeExchange(exchangePair.Item1, newNode,
                distanceMatrix);
            if (trial < 0)
            {
                p.usedNodes.Remove(p.solution[exchangePair.Item1].Index);
                p.usedNodes.Add(newNode.Index);
                p.solution.NodeExchange(p.solution[exchangePair.Item1].Index, newNode.Index);
                
                improvement = true;
                break;
            }
        }
    }
}

void SteepSolver((Path solution, HashSet<int> usedNodes) p, Node[] masterArray, DistanceMatrix distanceMatrix, List<(byte, byte)> possibleExchanges, List<(byte, byte)> possibleSwaps, int[] usableNodes, ref bool improvement, NeighborhoodType neighborhoodType)
{
    improvement = false;
    long bestInterPerformance = long.MaxValue, bestIntraPerformance = long.MaxValue;
    (byte, byte) bestInterPair = (byte.MaxValue, byte.MaxValue), bestIntraPair = (byte.MaxValue, byte.MaxValue);
    
    for (var i = 0; i < possibleExchanges.Count; i++)
    {
        var exchangePair = possibleExchanges[i];
        var newNode = masterArray[usableNodes[exchangePair.Item2]];
        var trial = p.solution.TestNodeExchange(exchangePair.Item1, newNode,
            distanceMatrix);
        if (trial < bestInterPerformance)
        {
            bestInterPerformance = trial;
            bestInterPair = exchangePair;
        }
    }
    
    for (var i = 0; i < possibleSwaps.Count; i++)
    {
        var swapPair = possibleSwaps[i];
        var trial = neighborhoodType switch
        {
            NeighborhoodType.NODE_SWAP => p.solution.TestNodeSwap(swapPair.Item1, swapPair.Item2, distanceMatrix),
            NeighborhoodType.EDGE_SWAP => p.solution.TestEdgeSwap(swapPair.Item1, swapPair.Item2, distanceMatrix),
            _ => throw new ArgumentOutOfRangeException(nameof(neighborhoodType), neighborhoodType, null)
        };
        if (trial < bestIntraPerformance)
        {
            bestIntraPerformance = trial;
            bestIntraPair = swapPair;
        }
    }

    if (bestInterPerformance >= 0 && bestIntraPerformance >= 0)
    {
        improvement = false;
        return;
    }

    improvement = true;
    
    if (bestInterPerformance < bestIntraPerformance)
    {
        var newNode = masterArray[usableNodes[bestInterPair.Item2]];
        p.usedNodes.Remove(p.solution[bestInterPair.Item1].Index);
        p.usedNodes.Add(newNode.Index);
        p.solution.NodeExchange(p.solution[bestInterPair.Item1].Index, newNode.Index);
        return;
    }

    if (bestIntraPerformance <= bestInterPerformance)
    {
        switch (neighborhoodType)
        {
            case NeighborhoodType.NODE_SWAP:
                p.solution.NodeSwap(bestIntraPair.Item1, bestIntraPair.Item2);
                break;
            case NeighborhoodType.EDGE_SWAP:
                p.solution.EdgeSwap(bestIntraPair.Item1, bestIntraPair.Item2);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(neighborhoodType), neighborhoodType, null);
        }
    }
}

(List<Path>, List<long>) Hypersolver(Node[] masterArray, DistanceMatrix distanceMatrix, byte upTo,
    SearchMode searchMode, NeighborhoodType neighborhoodType, InitialSolutionsType initialSolutionsType, Random random)
{
    Console.WriteLine($"Hypersolving... SearchMode: {searchMode}, NeighborhoodType: {neighborhoodType}, InitialSolutionsType: {initialSolutionsType}");
    List<(Path, HashSet<int>)> solutions = new (masterArray.Length);
    List<long> times = new(masterArray.Length);
    var referenceHashset = Enumerable.Range(0, masterArray.Length).ToHashSet();
    List<(byte, byte)> possibleNodeSwaps = new();
    for (int i = 0; i < upTo - 1; i++)
    {
        for (int j = i + 1; j < upTo; j++)
        {
            possibleNodeSwaps.Add(((byte)i, (byte)j));
        }
    }
    List<(byte, byte)> possibleEdgeSwaps = new();
    for (int i = 0; i < upTo - 2; i++)
    {
        for (int j = i + 2; j < upTo; j++)
        {
            if (i == 0 && j == upTo - 1)
            {
                continue;
            }
            possibleEdgeSwaps.Add(((byte)i, (byte)j));
        }
    }
    List<(byte, byte)> possibleExchanges = new();
    for (int i = 0; i < upTo; i++)
    {
        for (int j = 0; j < masterArray.Length - upTo; j++)
        {
            possibleExchanges.Add(((byte)i, (byte)j));
        }
    }
    
    for (var i = 0; i < masterArray.Length; i++)
    {
        var watch = Stopwatch.StartNew();
        solutions.Add(initialSolutionsType switch
        {
            InitialSolutionsType.RANDOM => RandomSolution(masterArray, random, upTo),
            InitialSolutionsType.GREEDY_REGRET => GreedyRegret((byte)i, masterArray, distanceMatrix, 1.0, 1.5, upTo),
            _ => throw new ArgumentOutOfRangeException(nameof(initialSolutionsType), initialSolutionsType, null)
        });
        watch.Stop();
        times.Add(watch.ElapsedMilliseconds);
    }

    for (var i = 0; i < solutions.Count; i++)
    {
        Console.WriteLine($"Currently processing solution {i}");
        var solution = solutions[i];
        bool improvement = true;
        var watch = Stopwatch.StartNew();
        watch.Start();
        while (improvement)
        {
            var usableNodes = (from idx in referenceHashset.Except(solution.Item2) select idx).ToArray();
            possibleExchanges = [..possibleExchanges.Shuffle()];
            
            switch (searchMode, neighborhoodType)
            {
                case (SearchMode.GREEDY, NeighborhoodType.NODE_SWAP):
                    possibleNodeSwaps = [.. possibleNodeSwaps.Shuffle()];
                    GreedySolver(solution, masterArray, distanceMatrix, possibleExchanges, possibleNodeSwaps, usableNodes, random, ref improvement, neighborhoodType);
                    break;
                case (SearchMode.GREEDY, NeighborhoodType.EDGE_SWAP):
                    possibleEdgeSwaps = [.. possibleEdgeSwaps.Shuffle()];
                    GreedySolver(solution, masterArray, distanceMatrix, possibleExchanges, possibleEdgeSwaps, usableNodes, random, ref improvement, neighborhoodType);
                    break;
                case (SearchMode.STEEP, NeighborhoodType.NODE_SWAP):
                    possibleNodeSwaps = [.. possibleNodeSwaps.Shuffle()];
                    SteepSolver(solution, masterArray, distanceMatrix, possibleExchanges, possibleNodeSwaps, usableNodes, ref improvement, neighborhoodType);
                    break;
                case (SearchMode.STEEP, NeighborhoodType.EDGE_SWAP):
                    possibleEdgeSwaps = [.. possibleEdgeSwaps.Shuffle()];
                    SteepSolver(solution, masterArray, distanceMatrix, possibleExchanges, possibleEdgeSwaps, usableNodes, ref improvement, neighborhoodType);
                    break;
            }
        }
        watch.Stop();
        times[i] += watch.ElapsedMilliseconds;
    }

    return ([.. solutions.Select(x => x.Item1)], times);
}

public enum SearchMode
{
    GREEDY,
    STEEP
}

public enum NeighborhoodType
{
    NODE_SWAP,
    EDGE_SWAP
}

public enum InitialSolutionsType
{
    RANDOM,
    GREEDY_REGRET
}
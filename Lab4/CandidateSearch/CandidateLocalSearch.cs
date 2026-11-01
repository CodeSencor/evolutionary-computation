// =====================================================================================
//  CandidateLocalSearch.cs  -  samodzielny, modularny plik do zadania "candidate moves".
//
//  Zawiera WSZYSTKO:
//    1. Instance            - dane instancji (wezly, macierz odleglosci, upTo)
//    2. Geometry            - odleglosc, funkcja celu
//    3. CandidateBuilder    - listy K najblizszych sasiadow (dlugosc krawedzi + koszt wierzcholka)
//    4. StartSolutions      - losowe rozwiazania startowe
//    5. SteepestSearch      - steepest LS (wariant BASELINE = pelne sasiedztwo,
//                                           wariant CANDIDATES = tylko ruchy kandydackie)
//    6. Stats / Reporter    - min/avg/max, tabele (konsola + plik .md), eksport wszystkich uruchomien (czas + trasa)
//    7. Experiment          - uruchomienie calosci: Experiment.RunAll("TSPA.csv", "TSPB.csv");
//    8. Cli                 - punkt wejscia (Main) z parsowaniem argumentow
//
//  Plik jest czescia projektu Candidate_search.csproj (Lab4), ktory referencjonuje
//  ..\..\EvoToolbox\EvoToolbox.csproj. Jedyny punkt wejscia to Cli.Main.
//
//  Uruchomienie (z folderu Lab4/Candidate_search):
//      dotnet run -c Release
//      dotnet run -c Release -- TSPA.csv TSPB.csv --move edge --k 5,10,20 --seed 42 --out wyniki
// =====================================================================================

using System.Diagnostics;
using System.Globalization;
using System.Text;
using EvoToolbox.Constructs;
using EvoToolbox.Objects;
using EvoToolbox.Utilities;
using Path = EvoToolbox.Objects.Path;

namespace CandidateLs;

public enum MoveType
{
    NodeSwap,
    EdgeSwap
}

public enum Variant
{
    Baseline,   // steepest, pelne sasiedztwo
    Candidates  // steepest, tylko ruchy wprowadzajace krawedz kandydacka
}

// -------------------------------------------------------------------------------------
// 1. Instancja
// -------------------------------------------------------------------------------------
public sealed class Instance
{
    public string Name { get; }
    public Node[] Nodes { get; }
    public DistanceMatrix Dm { get; }
    public int UpTo { get; }

    public Instance(string name, string csvPath)
    {
        Name = name;
        Nodes = DataUtility.ImportNodes(csvPath);
        Dm = new DistanceMatrix(Nodes);
        UpTo = Nodes.Length / 2 + (Nodes.Length % 2 == 0 ? 0 : 1);
    }
}

// -------------------------------------------------------------------------------------
// 2. Geometria i funkcja celu
// -------------------------------------------------------------------------------------
public static class Geometry
{
    // Jedyne miejsce zalezne od pol Node. Dostosuj, jesli trzeba.
    public static long Dist(Node a, Node b)
    {
        double dx = a.X - b.X, dy = a.Y - b.Y;
        return (long)Math.Round(Math.Sqrt(dx * dx + dy * dy));
    }

    public static long Cost(Node a) => a.Cost;

    /// <summary>Dlugosc cyklu + suma kosztow wybranych wierzcholkow.</summary>
    public static long Objective(Path path, Node[] nodes)
    {
        int n = path.Count;
        long total = 0;
        for (int i = 0; i < n; i++)
        {
            var a = nodes[path[(byte)i].Index];
            var b = nodes[path[(byte)((i + 1) % n)].Index];
            total += Dist(a, b) + Cost(a);
        }

        return total;
    }
}

// -------------------------------------------------------------------------------------
// 3. Kandydaci
// -------------------------------------------------------------------------------------
public static class CandidateBuilder
{
    /// <summary>cand[v] = K wierzcholkow u minimalizujacych Dist(v,u) + Cost(u).</summary>
    public static int[][] Build(Node[] nodes, int k)
    {
        var result = new int[nodes.Length][];
        for (int v = 0; v < nodes.Length; v++)
        {
            result[v] = Enumerable.Range(0, nodes.Length)
                .Where(u => u != v)
                .OrderBy(u => Geometry.Dist(nodes[v], nodes[u]) + Geometry.Cost(nodes[u]))
                .Take(k)
                .ToArray();
        }

        return result;
    }
}

// -------------------------------------------------------------------------------------
// 4. Rozwiazania startowe
// -------------------------------------------------------------------------------------
public static class StartSolutions
{
    public static (Path, HashSet<int>) RandomSolution(Node[] nodes, int upTo, Random rng)
    {
        var path = new Path(nodes);
        var used = new HashSet<int>();
        while (used.Count < upTo)
        {
            int idx = rng.Next(0, nodes.Length);
            if (used.Add(idx))
                path.AppendNode((byte)idx);
        }

        return (path, used);
    }
}

// -------------------------------------------------------------------------------------
// 5. Steepest local search (baseline + kandydaci)
// -------------------------------------------------------------------------------------
public sealed class SteepestSearch
{
    private readonly Instance _inst;
    private readonly MoveType _move;
    private readonly int[][]? _cand; // null => pelne sasiedztwo (baseline)
    private readonly int[] _pos;

    // stan biezacego kroku
    private Path _s = null!;
    private HashSet<int> _used = null!;
    private int _n;
    private long _best;
    private int _kind, _x, _y; // kind: -1 brak, 0 intra (swap), 1 inter (exchange)

    public SteepestSearch(Instance inst, MoveType move, int[][]? candidates)
    {
        _inst = inst;
        _move = move;
        _cand = candidates;
        _pos = new int[inst.Nodes.Length];
    }

    public void Run(Path solution, HashSet<int> used)
    {
        _s = solution;
        _used = used;
        while (Step())
        {
        }
    }

    private bool Step()
    {
        _n = _s.Count;
        Array.Fill(_pos, -1);
        for (int i = 0; i < _n; i++)
            _pos[_s[(byte)i].Index] = i;

        _best = 0; // tylko ruchy z delta < 0
        _kind = -1;

        if (_cand == null) ScanFullNeighborhood();
        else ScanCandidateMoves();

        if (_kind < 0) return false;
        Apply();
        return true;
    }

    // ---- ewaluacja pojedynczych ruchow (z normalizacja indeksow) -----------------------

    private int Mod(int v) => ((v % _n) + _n) % _n;

    private void TryEdgeSwap(int x, int y)
    {
        x = Mod(x);
        y = Mod(y);
        int lo = Math.Min(x, y), hi = Math.Max(x, y);
        if (hi - lo < 2 || (lo == 0 && hi == _n - 1)) return;
        long d = _s.TestEdgeSwap((byte)lo, (byte)hi, _inst.Dm);
        if (d < _best)
        {
            _best = d; _kind = 0; _x = lo; _y = hi;
        }
    }

    private void TryNodeSwap(int x, int y)
    {
        x = Mod(x);
        y = Mod(y);
        if (x == y) return;
        int lo = Math.Min(x, y), hi = Math.Max(x, y);
        long d = _s.TestNodeSwap((byte)lo, (byte)hi, _inst.Dm);
        if (d < _best)
        {
            _best = d; _kind = 0; _x = lo; _y = hi;
        }
    }

    private void TryIntra(int x, int y)
    {
        if (_move == MoveType.EdgeSwap) TryEdgeSwap(x, y);
        else TryNodeSwap(x, y);
    }

    private void TryExchange(int position, int newNodeIdx)
    {
        position = Mod(position);
        long d = _s.TestNodeExchange((byte)position, _inst.Nodes[newNodeIdx], _inst.Dm);
        if (d < _best)
        {
            _best = d; _kind = 1; _x = position; _y = newNodeIdx;
        }
    }

    // ---- baseline: wszystkie ruchy -----------------------------------------------------

    private void ScanFullNeighborhood()
    {
        for (int i = 0; i < _n - 1; i++)
            for (int j = i + 1; j < _n; j++)
                TryIntra(i, j); // niepoprawne pary (np. sasiednie krawedzie) odrzuca TryEdgeSwap

        var unselected = new List<int>();
        for (int u = 0; u < _pos.Length; u++)
            if (_pos[u] < 0) unselected.Add(u);

        for (int i = 0; i < _n; i++)
            foreach (int u in unselected)
                TryExchange(i, u);
    }

    // ---- kandydaci: iterujemy TYLKO po ruchach wprowadzajacych krawedz kandydacka -----

    private void ScanCandidateMoves()
    {
        for (int i = 0; i < _n; i++)
        {
            int a = _s[(byte)i].Index;
            foreach (int c in _cand![a])
            {
                int j = _pos[c];
                if (j >= 0)
                {
                    // INTRA - dwa ruchy wprowadzajace krawedz (a,c)
                    if (_move == MoveType.EdgeSwap)
                    {
                        TryEdgeSwap(i, j);         // dodaje (i,j) i (i+1,j+1)
                        TryEdgeSwap(i - 1, j - 1); // dodaje (i-1,j-1) i (i,j)
                    }
                    else
                    {
                        TryNodeSwap(j, i + 1);     // c laduje na i+1, obok a
                        TryNodeSwap(j, i - 1);     // c laduje na i-1, obok a
                    }
                }
                else
                {
                    // INTER - c zastepuje nastepnika albo poprzednika a => krawedz (a,c) w rozwiazaniu
                    TryExchange(i + 1, c);
                    TryExchange(i - 1, c);
                }
            }
        }
    }

    // ---- wykonanie najlepszego ruchu ---------------------------------------------------

    private void Apply()
    {
        if (_kind == 0)
        {
            if (_move == MoveType.EdgeSwap) _s.EdgeSwap((byte)_x, (byte)_y);
            else _s.NodeSwap((byte)_x, (byte)_y);
        }
        else
        {
            int oldIdx = _s[(byte)_x].Index;
            _used.Remove(oldIdx);
            _used.Add(_y);
            _s.NodeExchange((byte)oldIdx, (byte)_y);
        }
    }
}

// -------------------------------------------------------------------------------------
// 6. Statystyki i raport
// -------------------------------------------------------------------------------------
public sealed record RunResult(
    string Instance, string Label, int Runs,
    long ObjMin, double ObjAvg, long ObjMax,
    double TimeMin, double TimeAvg, double TimeMax,
    Path BestPath);

public static class Reporter
{
    public static string Fmt(double v) => v.ToString("0.##", CultureInfo.InvariantCulture);

    public static string ToMarkdown(IEnumerable<RunResult> results)
    {
        var sb = new StringBuilder();
        foreach (var group in results.GroupBy(r => r.Instance))
        {
            sb.AppendLine($"### Instancja {group.Key}");
            sb.AppendLine();
            sb.AppendLine("| Metoda | Funkcja celu: avg (min - max) | Czas [ms]: avg (min - max) |");
            sb.AppendLine("|---|---|---|");
            foreach (var r in group)
            {
                sb.AppendLine(
                    $"| {r.Label} | {Fmt(r.ObjAvg)} ({r.ObjMin} - {r.ObjMax}) | " +
                    $"{Fmt(r.TimeAvg)} ({Fmt(r.TimeMin)} - {Fmt(r.TimeMax)}) |");
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }

    public static void Save(string markdown, string file) => File.WriteAllText(file, markdown);

    /// <summary>
    /// Nazwa pliku z wynikami, np. tspaS_ES_R_base.csv / tspbS_NS_R_C10.csv
    /// (czytana przez lab4_report.py: S = steepest, ES/NS = sasiedztwo, R = start losowy,
    /// base = bez kandydatow, C&lt;K&gt; = K kandydatow).
    /// </summary>
    public static string ResultKey(string instanceName, MoveType move, Variant variant, int k) =>
        $"{instanceName.ToLowerInvariant()}S_{(move == MoveType.EdgeSwap ? "ES" : "NS")}_R_" +
        (variant == Variant.Baseline ? "base" : $"C{k}");

    /// <summary>
    /// Format jak w Lab3 (DataUtility.ExportPaths z czasami): dla kazdego uruchomienia linia z czasem [ms]
    /// (ulamek dziesietny z kropka), potem N linii "x;y;cost". Czas jest ulamkowy, bo kandydackie
    /// uruchomienia trwaja czesto ponizej 1 ms.
    /// </summary>
    public static void ExportRuns(string file, IReadOnlyList<Path> paths, IReadOnlyList<double> times)
    {
        using var writer = new StreamWriter(file);
        for (int i = 0; i < paths.Count; i++)
        {
            writer.WriteLine(times[i].ToString("0.000", CultureInfo.InvariantCulture));
            for (int j = 0; j < paths[i].Count; j++)
            {
                var node = paths[i][(byte)j];
                writer.WriteLine($"{node.X};{node.Y};{node.Cost}");
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// 7. Eksperyment
// -------------------------------------------------------------------------------------
public static class Experiment
{
    /// <summary>
    /// Uruchamia baseline oraz warianty z kandydatami dla kazdej wartosci K.
    /// Liczba uruchomien kazdej metody = liczba wezlow instancji (200).
    /// </summary>
    public static List<RunResult> RunAll(string csvA, string csvB,
        int[]? candidateCounts = null, MoveType move = MoveType.EdgeSwap,
        int seed = 42, string outputDir = ".") =>
        RunAll(csvA, csvB, candidateCounts, new[] { move }, seed, outputDir);

    /// <summary>To samo dla kilku sasiedztw naraz (np. EdgeSwap i NodeSwap): kazde ma wlasny baseline.</summary>
    public static List<RunResult> RunAll(string csvA, string csvB,
        int[]? candidateCounts, IReadOnlyList<MoveType> moves,
        int seed = 42, string outputDir = ".")
    {
        candidateCounts ??= new[] { 5, 10, 15, 20 };
        var all = new List<RunResult>();

        foreach (var (name, path) in new[] { ("TSPA", csvA), ("TSPB", csvB) })
        {
            var inst = new Instance(name, path);

            foreach (var move in moves)
            {
                all.Add(RunMethod(inst, move, null, Variant.Baseline, 0, seed, outputDir));
                foreach (int k in candidateCounts)
                {
                    var cand = CandidateBuilder.Build(inst.Nodes, k); // budowa list poza pomiarem czasu LS
                    all.Add(RunMethod(inst, move, cand, Variant.Candidates, k, seed, outputDir));
                }
            }
        }

        var md = Reporter.ToMarkdown(all);
        Console.WriteLine(md);
        Reporter.Save(md, System.IO.Path.Combine(outputDir, "results_candidate_ls.md"));
        return all;
    }

    private static RunResult RunMethod(Instance inst, MoveType move, int[][]? cand,
        Variant variant, int k, int seed, string outputDir)
    {
        string label = variant == Variant.Baseline
            ? $"Steepest {move}, random start (baseline)"
            : $"Steepest {move}, random start, candidates K={k}";
        Console.WriteLine($"[{inst.Name}] {label}");

        var rng = new Random(seed); // ten sam seed => te same rozwiazania startowe dla wszystkich metod
        var search = new SteepestSearch(inst, move, cand);
        int runs = inst.Nodes.Length;

        var objectives = new List<long>(runs);
        var times = new List<double>(runs);
        var paths = new List<Path>(runs);
        Path? bestPath = null;
        long bestObj = long.MaxValue;

        for (int i = 0; i < runs; i++)
        {
            var watch = Stopwatch.StartNew();
            var (path, used) = StartSolutions.RandomSolution(inst.Nodes, inst.UpTo, rng);
            search.Run(path, used);
            watch.Stop();

            long obj = Geometry.Objective(path, inst.Nodes);
            objectives.Add(obj);
            paths.Add(path);
            times.Add(watch.Elapsed.TotalMilliseconds);
            if (obj < bestObj)
            {
                bestObj = obj;
                bestPath = path;
            }
        }

        // wszystkie uruchomienia (czas + trasa) - to czyta lab4_report.py
        Reporter.ExportRuns(
            System.IO.Path.Combine(outputDir, Reporter.ResultKey(inst.Name, move, variant, k) + ".csv"),
            paths, times);

        return new RunResult(inst.Name, label, runs,
            objectives.Min(), objectives.Average(), objectives.Max(),
            times.Min(), times.Average(), times.Max(), bestPath!);
    }
}

// -------------------------------------------------------------------------------------
// 8. Uruchamianie z konsoli
//
//   dotnet run -c Release -- [TSPA.csv TSPB.csv] [--move edge|node|both] [--k 5,10,20] [--seed 42] [--out ./wyniki]
//
//   Przyklady (w folderze Lab4/CandidateSearch):
//     dotnet run -c Release
//     dotnet run -c Release -- --move both
//     dotnet run -c Release -- --move node --k 10 --out wyniki
//     dotnet run -c Release -- ../../Lab2/TSPA.csv ../../Lab2/TSPB.csv
// -------------------------------------------------------------------------------------
public static class Cli
{
    /// <summary>Parsuje argumenty i uruchamia eksperyment. Zwraca false przy --help / bledzie.</summary>
    public static bool TryRun(string[] args)
    {
        if (args.Any(a => a is "-h" or "--help" or "/?"))
            return false;

        var files = new List<string>();
        MoveType[] moves = { MoveType.EdgeSwap };
        int[]? ks = null;
        int seed = 42;
        string outDir = ".";

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--move":
                    string m = args[++i];
                    moves = m.StartsWith("b", StringComparison.OrdinalIgnoreCase)
                        ? new[] { MoveType.EdgeSwap, MoveType.NodeSwap }          // both
                        : m.StartsWith("n", StringComparison.OrdinalIgnoreCase)
                            ? new[] { MoveType.NodeSwap }                          // node
                            : new[] { MoveType.EdgeSwap };                         // edge
                    break;
                case "--k":
                    ks = args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse).ToArray();
                    break;
                case "--seed":
                    seed = int.Parse(args[++i]);
                    break;
                case "--out":
                    outDir = args[++i];
                    break;
                default:
                    files.Add(args[i]);
                    break;
            }
        }

        string csvA = files.Count > 0 ? files[0] : "TSPA.csv";
        string csvB = files.Count > 1 ? files[1] : "TSPB.csv";

        Directory.CreateDirectory(outDir);
        Experiment.RunAll(csvA, csvB, ks, moves, seed, outDir);
        return true;
    }

    /// <summary>Jedyny punkt wejscia projektu Candidate_search.</summary>
    public static int Main(string[] args)
    {
        try
        {
            if (TryRun(args)) return 0;
        }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException
                                      or FormatException or IndexOutOfRangeException)
        {
            Console.Error.WriteLine($"Blad: {e.Message}");
        }

        Console.WriteLine("Uzycie: dotnet run -c Release -- [TSPA.csv TSPB.csv] " +
                          "[--move edge|node|both] [--k 5,10,20] [--seed 42] [--out ./wyniki]");
        return 1;
    }
}
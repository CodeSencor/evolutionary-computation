using EvoToolbox.Objects;
using Path = EvoToolbox.Objects.Path;

namespace EvoToolbox.Utilities;

public static class DataUtility
{
    public static Node[] ImportNodes(string filePath)
    {
        Node[] masterArray;
        using (var reader = new StreamReader(filePath))
        {
            var content = reader.ReadToEnd().Replace('\n', ';').Replace("\r", "").TrimEnd(';').Split(';');
            masterArray = new Node[content.Length / 3];
            for (int i = 0; i < content.Length;)
            {
                masterArray[i / 3] = new Node(
                    (byte)(i/3),
                    ushort.Parse(content[i++]),
                    ushort.Parse(content[i++]),
                    ushort.Parse(content[i++])
                );
            }
        }
        return masterArray;
    }

    public static void ExportPaths(string filePath, List<Path> paths)
    {
        using (var writer = new StreamWriter(filePath))
        {
            foreach (var path in paths)
            {
                for (byte i = 0; i < path.Count; i++)
                {
                    var node = path[i];
                    writer.WriteLine($"{node.X};{node.Y};{node.Cost}");
                }
            }
        }
    }
    
    public static void ExportPaths(string filePath, (List<Path> paths, List<long> times) p)
    {
        using (var writer = new StreamWriter(filePath))
        {
            for (var i = 0; i < p.paths.Count; i++)
            {
                writer.WriteLine(p.times[i]);
                var path = p.paths[i];
                for (byte j = 0; j < path.Count; j++)
                {
                    var node = path[j];
                    writer.WriteLine($"{node.X};{node.Y};{node.Cost}");
                }
            }
        }
    }
}

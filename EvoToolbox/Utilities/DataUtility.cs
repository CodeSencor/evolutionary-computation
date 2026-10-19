using EvoToolbox.Objects;
using Path = EvoToolbox.Objects.Path;

namespace EvoToolbox.Utilities;

public static class DataUtility
{
    static Node[] ImportNodes(string filePath)
    {
        Node[] masterArray;
        using (var reader = new StreamReader("TSPA.csv"))
        {
            var content = reader.ReadToEnd().Replace('\n', ';').Replace("\r", "").TrimEnd(';').Split(';');
            masterArray = new Node[content.Length / 3];
            for (int i = 0; i < content.Length;)
            {
                masterArray[i / 3] = new Node(
                    (byte)i,
                    ushort.Parse(content[i++]),
                    ushort.Parse(content[i++]),
                    ushort.Parse(content[i++])
                );
            }
        }
        return masterArray;
    }

    static void ExportPath(string filePath, Path path)
    {
        using (var writer = new StreamWriter(filePath))
        {
            for (byte i = 0; i < path.Count; i++)
            {
                var node = path[i];
                writer.WriteLine($"{node.X};{node.Y};{node.Cost}");
            }
        }
    }
}

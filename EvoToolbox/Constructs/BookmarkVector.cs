using EvoToolbox.Objects;

namespace EvoToolbox.Constructs;

public class BookmarkVector
{
    private readonly byte[] _bookmarkVector;

    public BookmarkVector(SortedDistanceMatrix sortedDistanceMatrix, uint maxDistance)
    {
        _bookmarkVector = new byte[sortedDistanceMatrix.Length];
        for (byte i = 0; i < _bookmarkVector.Length; i++)
        {
            _bookmarkVector[i] = BinarySearch(sortedDistanceMatrix.GetRawRow(i), maxDistance);
        }
    }

    private byte BinarySearch((uint, byte)[] array, uint lessThan)
    {
        int low = 0, high = array.Length;
        while (high != low)
        {
            int split = (high - low) / 2 + low;
            var pivot = array[split];
            if (pivot.Item1 >= lessThan)
            {
                high = split;
            }
            else
            {
                low = split;
            }
        }
        return (byte)low;
    }
}
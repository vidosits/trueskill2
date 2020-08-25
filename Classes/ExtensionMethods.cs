using System;
using System.Collections.Generic;
using System.Linq;

namespace GGScore.Classes
{
    public static class ExtensionMethods
    {
        public static IEnumerable<(int index, T item)> Enumerate<T>(this IEnumerable<T> self)       
            => self.Select((item, index) => (index, item));

        public static IEnumerable<T[]> Batch<T>(
            this IEnumerable<T> source, int size)
        {
            T[] bucket = null;
            var count = 0;

            foreach (var item in source)
            {
                if (bucket == null)
                    bucket = new T[size];

                bucket[count++] = item;

                if (count != size)                
                    continue;

                yield return bucket;

                bucket = null;
                count = 0;
            }

            // Return the last bucket with all remaining elements
            if (bucket != null && count > 0)
            {
                Array.Resize(ref bucket, count);
                yield return bucket;
            }
        }
    }
}
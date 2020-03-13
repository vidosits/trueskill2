using System.Collections.Generic;
using System.Linq;

namespace ts.core
{
    public static class ExtensionMethods
    {
        public static IEnumerable<(int index, T item)> Enumerate<T>(this IEnumerable<T> self)       
            => self.Select((item, index) => (index, item)); 
    }
}
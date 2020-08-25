using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace GGScore.Classes
{
    public static class Utils
    {
        public static IEnumerable<T> ReadMatchesFromFile<T>(string fileName)
        {
            using var r = new StreamReader(fileName);
            Console.Write("Reading matches from file...");
            var matches = JsonConvert.DeserializeObject<List<T>>(r.ReadToEnd());
            // return matches.OrderBy(x => x.Date);
            return matches;
        }
    }
}
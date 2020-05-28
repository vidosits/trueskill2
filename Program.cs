using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace ts.core
{
    internal static class Program
    {
        private static void Main()
        {
            // Trueskill2.Test();
            // Trueskill2.Run();
            // OnlineLearning.Run();
            
            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/abios_dota_matches_with_stats.json");
            Console.WriteLine("OK.");
        }

        private static IEnumerable<Match> ReadMatchesFromFile(string fileName)
        {
            using (var r = new StreamReader(fileName))
            {
                Console.Write("Reading matches from file...");
                return JsonConvert.DeserializeObject<List<Match>>(r.ReadToEnd());
            }
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Newtonsoft.Json;
using ts.core.Classes;
using ts.core.TrueSkill2;

namespace ts.core
{
    internal static class Program
    {
        private static void Main()
        {
            // TwoPersonTrueskill.RunExample(new []{0}, new []{1});
            // Trueskill2.Test();
            Trueskill2.Run();
            // OnlineLearning.Run();
            // TrueskillThroughTime.RunExample();
            // var skill = TwoTeamTrueskill.Run("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_dota_matches_with_stats.json");

            // var skill = Reimplementation<LeaguePlayerStat>.Run("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_lol_matches_with_stats_and_converted_champion_ids.json");
            // var players = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_lol_player_names.json"));
            //
            //
            // var orderings = new Func<KeyValuePair<int, Gaussian>, double>[]
            // {
            //     s => s.Value.GetMean() - (Reimplementation<LeaguePlayerStat>.SkillMean / Reimplementation<LeaguePlayerStat>.SkillDeviation) * Math.Sqrt(s.Value.GetVariance()),
            //     s => s.Value.GetMean(),
            //     s => s.Value.GetMean() - 3 * Math.Sqrt(s.Value.GetVariance())
            // };
            //
            //
            // foreach (var ordering in orderings)
            // {
            //     Console.WriteLine();
            //     var i = 1;
            //     foreach (var (key, value) in skill.OrderByDescending(ordering).Take(15))
            //     {
            //         Console.WriteLine($"{i}. {players[key]} ({key}) : {value.GetMean():F0}, {Math.Sqrt(value.GetVariance()):F}");
            //         i++;
            //     }
            // }
        }
    }
}

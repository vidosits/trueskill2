using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Newtonsoft.Json;

// ReSharper disable InconsistentNaming
// ReSharper disable ClassNeverInstantiated.Global

namespace ts.core
{
    public class Player
    {
        public int? assists;
        public int? camps_stacked;
        public int? deaths;
        public int? dn_t;
        public int? gold_spent;
        public int? hero_damage;
        public int? hero_healing;
        public int? kills;
        public int? lh_t;
        public int? obs_placed;
        public int? observer_kills;
        public int? sen_placed;
        public int? sentry_kills;
        public long? steam_id;
        public double? stuns;
        public double? teamfight_participation;
        public int? total_gold;
        public int? total_xp;
        public int? tower_damage;
    }

    public class Match
    {
        public DateTime date;
        public List<Player> dire;
        public int duration;
        public long match_id;
        public List<Player> radiant;
        public bool radiant_win;
    }

    internal static class Program
    {
        private static void RunTwoTeamTrueSkill(List<Match> matches)
        {
            // set up player skill priors
            var playerSkillPriors = new Dictionary<long, Gaussian>();

            // define batch size
            const int batchSize = 10000;

            var matchesToRank = new long[batchSize][][];
            var matchWinners = new int[batchSize];
            var matchLosers = new int[batchSize];
            var playerPriors = new Gaussian[batchSize][][];

            foreach (var (matchIndex, match) in matches.Enumerate())
            {
                foreach (var player in match.radiant.Union(match.dire))
                {
                    if (!playerSkillPriors.ContainsKey(player.steam_id.Value))
                    {
                        playerSkillPriors[player.steam_id.Value] = Gaussian.FromMeanAndVariance(TwoTeamTrueskill.SkillMean, Math.Pow(TwoTeamTrueskill.SkillDeviation, 2));
                    }
                }

                matchesToRank[matchIndex % batchSize] = new long[2][] {match.radiant.Select(player => player.steam_id.Value).ToArray(), match.dire.Select(player => player.steam_id.Value).ToArray()};

                matchWinners[matchIndex % batchSize] = match.radiant_win ? 0 : 1;
                matchLosers[matchIndex % batchSize] = match.radiant_win ? 1 : 0;

                playerPriors[matchIndex % batchSize] = new Gaussian[][]
                {
                    match.radiant.Select(player => playerSkillPriors[player.steam_id.Value]).ToArray(),
                    match.dire.Select(player => playerSkillPriors[player.steam_id.Value]).ToArray()
                };

                if (matchIndex % batchSize == batchSize - 1)
                {
                    var updatedPriors = TwoTeamTrueskill.Run(playerPriors, matchWinners, matchLosers).ToArray();

                    foreach (var (processedMatchIndex, processedMatch) in updatedPriors.Enumerate())
                    {
                        foreach (var (teamIndex, teamSkills) in processedMatch.Enumerate())
                        {
                            foreach (var (playerIndex, playerSkill) in teamSkills.Enumerate())
                            {
                                playerSkillPriors[matchesToRank[processedMatchIndex][teamIndex][playerIndex]] = Gaussian.FromMeanAndVariance(playerSkill.GetMean(),
                                    playerSkill.GetVariance() + Math.Pow(TwoTeamTrueskill.SkillDynamicsFactor, 2));
                            }
                        }
                    }
                }
            }
        }
        
        private static IEnumerable<Match> ReadMatchesFromFile(string fileName)
        {
            using (var r = new StreamReader(fileName))
            {
                var matchesByDate = JsonConvert.DeserializeObject<Dictionary<string, List<Match>>>(r.ReadToEnd());
                // set up matches as a chronological list
                Console.Write("Setting up chronological match order...");
                var matches = new List<Match>();
            
                foreach (var matchesOnDate in matchesByDate)
                {
                    foreach (var match in matchesOnDate.Value)
                    {
                        match.date = Convert.ToDateTime(matchesOnDate.Key);
            
                        if (match.radiant.Union(match.dire).All(player => player.steam_id != null))
                        {
                            matches.Add(match);
                        }
                    }
                }

                Console.WriteLine("OK.");

                return matches;
            }
        }

        private static void Main()
        {
            var matches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/sorted_dota2_ts2.json");
            
            // define variable keeping track of player skills
            var playerRating = new Dictionary<long, Gaussian>();
            
            // enumerate through all matches
            foreach (var (matchIndex, match) in matches.Enumerate())
            {
                // make sure each player has a prior
                foreach (var player in match.radiant.Union(match.dire))
                {
                    if (!playerRating.ContainsKey(player.steam_id.Value))
                    {
                        playerRating[player.steam_id.Value] = Gaussian.FromMeanAndVariance(Trueskill2.SkillMean, Math.Pow(Trueskill2.SkillDeviation, 2));
                    }
                }
            
                // define the players in a match
                var matchPlayers = new long[2][] {match.radiant.Select(player => player.steam_id.Value).ToArray(), match.dire.Select(player => player.steam_id.Value).ToArray()};
                
                // use the current skill level of players as the prior for their skill
                var matchPriors = new Gaussian[][]
                {
                    match.radiant.Select(player => playerRating[player.steam_id.Value]).ToArray(),
                    match.dire.Select(player => playerRating[player.steam_id.Value]).ToArray()
                };
            
                
                // do the online update for player priors with TS2
                var updatedPriors = Trueskill2.RunOnline(matchPriors, match.radiant_win ? 0 : 1, match.radiant_win ? 1 : 0).ToArray();
                
                // update the player skills
                foreach (var (teamIndex, teamSkills) in updatedPriors.Enumerate())
                {
                    foreach (var (playerIndex, playerSkill) in teamSkills.Enumerate())
                    {
                        playerRating[matchPlayers[teamIndex][playerIndex]] = playerSkill;
                    }
                }
                
            }
        }
    }
}
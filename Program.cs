using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Learners;
using Microsoft.ML.Probabilistic.Models;
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
                Console.Write("Reading matches from file...");
                var matchesByDate = JsonConvert.DeserializeObject<Dictionary<string, List<Match>>>(r.ReadToEnd());
                Console.WriteLine("OK.");

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

        private static void Rating()
        {
            var matches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/sorted_dota2_ts2.json");

            const int batchSize = 32;

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<long, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<long, DateTime>();

            foreach (var batch in matches.Batch(batchSize))
            {
                var steamIdToIndex = new Dictionary<long, int>();
                var indexToSteamId = new Dictionary<int, long>();

                // priors for all the players appearing at least once 
                var priors = new List<Gaussian>();

                // holds the indexes of each player in each game w.r.t. the priors array
                var playerIndex = new int[batchSize][][];

                // holds the amount of time that has passed for the m-th player in n-th game since the last time that player has played, index order of this array is [n][m] / [game][player]  
                var batchPlayerTimeLapse = new List<List<double>>();

                // holds the amount of games the ith player (w.r.t. the priors array) has played in this batch of games
                var gameCountsPerPlayer = new List<int>();

                // holds the mapping between the batch index of a game and the j-th player's skill in that game
                var gameMapping = new List<int[]>();

                // holds the length of each match in this batch
                var matchLength = new double[batchSize];

                // holds the kill counts for the k-th player in the j-th team in the i-th game, index order of this array is [i][j][k] / [game][team][player]
                var killCounts = new double[batchSize][][];

                // holds the value for whether the kill count for k-th player in the j-th team in the i-th game is missing or not
                var killCountMissing = new bool[batchSize][][];

                // holds the death counts for the k-th player in the j-th team in the i-th game, index order of this array is [i][j][k] / [game][team][player]
                var deathCounts = new double[batchSize][][];

                // holds the value for whether the death count for k-th player in the j-th team in the i-th game is missing or not
                var deathCountMissing = new bool[batchSize][][];

                // holds the index of the winning team for each match in the batch
                var winnerIndex = new int[batchSize];

                // holds the index of the losing team for each match in the batch
                var loserIndex = new int[batchSize];

                foreach (var (matchIndex, match) in batch.Enumerate())
                {
                    var teams = new[] {match.radiant, match.dire};
                    foreach (var player in match.radiant.Union(match.dire))
                    {
                        // if this is the first time that we've ever seen this player, initialize his/her global skill with the prior
                        if (!playerSkill.ContainsKey(player.steam_id.Value))
                        {
                            playerSkill[player.steam_id.Value] = Gaussian.FromMeanAndVariance(TwoTeamTrueskill.SkillMean, Math.Pow(TwoTeamTrueskill.SkillDeviation, 2));
                            globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                        }

                        int pIndex;

                        // check if this player has not already appeared in this batch 
                        if (!steamIdToIndex.ContainsKey(player.steam_id.Value))
                        {
                            // get the index that the player will receive
                            steamIdToIndex[player.steam_id.Value] = priors.Count;
                            indexToSteamId[priors.Count] = player.steam_id.Value;
                            pIndex = priors.Count;

                            // init player prior from global skill tracker
                            priors.Add(playerSkill[player.steam_id.Value]);

                            // init the number of games played in current batch
                            gameCountsPerPlayer.Add(1);

                            // set the time elapsed since the last time this player has played
                            batchPlayerTimeLapse.Add(new List<double>() {(match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days});

                            // set up the array that will hold the mapping between the i-th game and the players' games
                            gameMapping.Add(new int[batchSize]);
                        }
                        else
                        {
                            // index of the player among all the unique players that have appear at least once in this batchset pIndex
                            pIndex = steamIdToIndex[player.steam_id.Value];

                            // increase the number of games played in the current batch by the current player
                            gameCountsPerPlayer[pIndex] += 1;

                            // update batchPlayerTimeLapse, so that we can tell how much time has passed (in days) since the last time this player has played  
                            batchPlayerTimeLapse[pIndex].Add((match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days);
                        }

                        // set up the mapping between the game index and the player's games 
                        gameMapping[pIndex][matchIndex] = gameCountsPerPlayer[pIndex] - 1;

                        // update the date of the last played match for the player
                        globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                    }

                    // set playerIndex
                    playerIndex[matchIndex] = teams.Select(t => t.Select(p => steamIdToIndex[p.steam_id.Value]).ToArray()).ToArray();

                    // set matchLength
                    matchLength[matchIndex] = match.duration;

                    // set winnerIndex
                    winnerIndex[matchIndex] = match.radiant_win ? 0 : 1;

                    // set killCounts and killCountMissing
                    killCounts[matchIndex] = teams.Select(t => t.Select(p => p.kills != null ? (double) p.kills.Value : 0.0).ToArray()).ToArray();
                    killCountMissing[matchIndex] = teams.Select(t => t.Select(p => p.kills == null).ToArray()).ToArray();

                    // set deathCounts and deathCountMissing
                    deathCounts[matchIndex] = teams.Select(t => t.Select(p => p.deaths != null ? (double) p.deaths.Value : 0.0).ToArray()).ToArray();
                    deathCountMissing[matchIndex] = teams.Select(t => t.Select(p => p.deaths == null).ToArray()).ToArray();

                    // set loserIndex
                    loserIndex[matchIndex] = match.radiant_win ? 1 : 0;
                }

                // process this batch with TS2
                var inferredSkills = Trueskill2.Run(priors.ToArray(), playerIndex, batchPlayerTimeLapse.Select(Enumerable.ToArray).ToArray(), gameCountsPerPlayer.ToArray(),
                    gameMapping.ToArray(), matchLength, killCounts, killCountMissing, deathCounts, deathCountMissing, winnerIndex, loserIndex);

                // update the priors for the players in this batch
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate())
                {
                    playerSkill[indexToSteamId[i]] = skillOverTime.Last();
                }
            }
        }

        private static void Main()
        {
            Rating();
        }
    }
}
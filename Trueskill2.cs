using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Newtonsoft.Json;

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

    public static class Trueskill2
    {
        public static IEnumerable<Match> ReadMatchesFromFile(string fileName)
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
                foreach (var match in matchesOnDate.Value)
                {
                    match.date = Convert.ToDateTime(matchesOnDate.Key);

                    if (match.radiant.Union(match.dire).All(player => player.steam_id != null)) matches.Add(match);
                }

                Console.WriteLine("OK.");

                return matches;
            }
        }

        public static void Run()
        {
            #region Parameters

            var skillPrior = Gaussian.FromMeanAndVariance(1500, 800 * 800); // N ~ (μ, σ)
            var skillClassWidthPriorValue = Gamma.Uniform(); // β
            var skillDynamicsFactorPriorValue = Gamma.Uniform(); // γ
            var skillSharpnessDecreaseFactorPriorValue = Gamma.Uniform(); // τ

            var killWeightPlayerTeamPerformancePriorValue = Beta.Uniform();
            var killWeightPlayerOpponentPerformancePriorValue = Beta.Uniform();
            var killCountVariancePriorValue = Beta.Uniform();

            var deathWeightPlayerTeamPerformancePriorValue = Beta.Uniform();
            var deathWeightPlayerOpponentPerformancePriorValue = Beta.Uniform();
            var deathCountVariancePriorValue = Beta.Uniform();


            var skillClassWidthPrior = Variable.New<Gamma>().Named("skillClassWidthPrior");
            var skillDynamicsFactorPrior = Variable.New<Gamma>().Named("skillDynamicsFactorPrior");
            var skillSharpnessDecreaseFactorPrior = Variable.New<Gamma>().Named("skillSharpnessDecreaseFactorPrior");

            var killWeightPlayerTeamPerformancePrior = Variable.New<Beta>().Named("killWeightPlayerTeamPerformancePrior");
            var killWeightPlayerOpponentPerformancePrior = Variable.New<Beta>().Named("killWeightPlayerOpponentPerformancePrior");
            var killCountVariancePrior = Variable.New<Beta>().Named("killCountVariancePrior");

            var deathWeightPlayerTeamPerformancePrior = Variable.New<Beta>().Named("deathWeightPlayerTeamPerformancePrior");
            var deathWeightPlayerOpponentPerformancePrior = Variable.New<Beta>().Named("deathWeightPlayerOpponentPerformancePrior");
            var deathCountVariancePrior = Variable.New<Beta>().Named("deathCountVariancePrior");


            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth");
            var skillDynamicsFactor = Variable<double>.Random(skillDynamicsFactorPrior).Named("skillDynamicsFactor");
            var skillSharpnessDecreaseFactor = Variable<double>.Random(skillSharpnessDecreaseFactorPrior).Named("skillSharpnessDecreaseFactor");

            var killWeightPlayerTeamPerformance = Variable<double>.Random(killWeightPlayerTeamPerformancePrior).Named("killWeightPlayerTeamPerformance");
            var killWeightPlayerOpponentPerformance = Variable<double>.Random(killWeightPlayerOpponentPerformancePrior).Named("killWeightPlayerOpponentPerformance");
            var killCountVariance = Variable<double>.Random(killCountVariancePrior).Named("killCountVariance");

            var deathWeightPlayerTeamPerformance = Variable<double>.Random(deathWeightPlayerTeamPerformancePrior).Named("deathWeightPlayerTeamPerformance");
            var deathWeightPlayerOpponentPerformance = Variable<double>.Random(deathWeightPlayerOpponentPerformancePrior).Named("deathWeightPlayerOpponentPerformance");
            var deathCountVariance = Variable<double>.Random(deathCountVariancePrior).Named("deathCountVariance");

            #endregion

            #region Constants

            var zero = Variable.Observed(0.0).Named("zero");
            var batchLength = Variable.New<int>();
            var numOfPlayers = Variable.New<int>();

            #endregion

            #region Ranges

            // Ranges to loop through the data: [matches][teams][players]
            var allPlayers = new Range(numOfPlayers).Named("allPlayers");
            var allMatches = new Range(batchLength).Named("allMatches");
            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch");
            var teamSize = Variable.Observed(5.0).Named("teamSize");

            #endregion

            #region "Matches & Outcomes"

            // Array to hold the player lookup table. With this array player's details can be found in the skills array (to be defined later)
            var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("matches");

            // Array to hold the index of the winning and losing team in each match
            var winners = Variable.Array<int>(allMatches).Named("winners");
            var losers = Variable.Array<int>(allMatches).Named("losers");

            #endregion

            #region Skill setup

            // This array is used to hold the value of the total number of matches played by the specific players in this current batch.
            // These numbers can be used to create the 2D jagged array holding the skills of the player over time.
            var numberOfMatchesPlayedPerPlayer = Variable.Array<int>(allPlayers).Named("numberOfMatchesPlayedPerPlayer");

            // This range is used to access the 2d jagged array (because players play different amount of matches in the batch) holding the player skills
            var matchCounts = new Range(numberOfMatchesPlayedPerPlayer[allPlayers]).Named("matchCounts");

            // Array to hold the prior of the skill for each of the players
            var skillPriors = Variable.Array<Gaussian>(allPlayers).Named("skillPriors");

            // Jagged array holding the skills for all players through all their matches, the first column is the prior
            var skills = Variable.Array(Variable.Array<double>(matchCounts), allPlayers).Named("skills");

            // Array to hold the time elapsed between matches of each player used for calculating the decay of skills
            var playerTimeLapse = Variable.Array(Variable.Array<double>(matchCounts), allPlayers).Named("playerTimeElapsed");

            // Array used to hold the match length information, used for calculating skill updates
            var matchLengths = Variable.Array<double>(allMatches).Named("matchLengths");

            #endregion

            #region Stats

            // Initialize arrays holding player stat(s) (e.g.: kills, deaths, etc.) information and whether they are available
            var killCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("killCount");
            var isKillCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("killCountMissing");

            var deathCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("deathCount");
            var isDeathCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("deathCountMissing");

            #endregion

            #region Mapping

            // This array is used to hold the mapping between the index (m) of the m-th match in the batch
            // and the index of the same match for the n-th player in the skills array
            // This is needed because we keep track of player skills in a jagged array like the following:
            //
            // "skills":
            //
            //          m a t c h e s (m)
            //        -------------------
            //      p | 0 1 2 3 4
            //      l | 0 1
            //      a | 0 1 2 3
            //      y | 0 1 2 3 4 5
            //      e | 0 1
            //      r | 0
            //      s | 0 1 2 3
            //     (n)| 0 1 2 3
            // 
            // Because we're iterating through a rectangular table of "player ⨯ matches" we need to translate
            // the indices/elements of the rectangular table to the elements of the jagged "skills" array.
            var playerMatchMapping = Variable.Array(Variable.Array<int>(allMatches), allPlayers).Named("playerMatchMapping");

            #endregion

            // Initialize skills variable array, the outer index is the matchIndex, the innerIndex is the player index
            using (var playerBlock = Variable.ForEach(allPlayers))
            {
                using (var matchBlock = Variable.ForEach(matchCounts))
                {
                    using (Variable.If(matchBlock.Index == 0))
                    {
                        skills[allPlayers][matchCounts] = Variable<double>.Random(skillPriors[allPlayers]).Named($"{playerBlock.Index}. player prior");
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[allPlayers][matchCounts] =
                            Variable.GaussianFromMeanAndPrecision(skills[allPlayers][matchBlock.Index - 1], skillSharpnessDecreaseFactor * playerTimeLapse[allPlayers][matchCounts] + skillDynamicsFactor)
                                .Named($"{playerBlock.Index}. player skill in {matchBlock.Index}. match");
                    }
                }
            }

            using (Variable.ForEach(allMatches))
            {
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch).Named("playerPerformance");
                var teamPerformance = Variable.Array<double>(nTeamsPerMatch).Named("teamPerformance");

                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matches[allMatches][nTeamsPerMatch][nPlayersPerTeam].Named("playerIndex");
                        var matchIndex = playerMatchMapping[playerIndex][allMatches];
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = Variable.GaussianFromMeanAndVariance(skills[playerIndex][matchIndex], skillClassWidth).Named("playerPerformanceInNthMatchInIthTeam");
                    }

                    teamPerformance[nTeamsPerMatch] = Variable.Sum(playerPerformance[nTeamsPerMatch]);
                }

                Variable.ConstrainTrue(teamPerformance[winners[allMatches]] > teamPerformance[losers[allMatches]]);

                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var opponentTeamIndex = Variable.New<int>();
                        using (Variable.Case(team.Index, 0))
                        {
                            opponentTeamIndex.ObservedValue = 1;
                        }

                        using (Variable.Case(team.Index, 1))
                        {
                            opponentTeamIndex.ObservedValue = 0;
                        }

                        using (Variable.IfNot(isKillCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            killCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    killWeightPlayerTeamPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    killWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], killCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isDeathCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            deathCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    deathWeightPlayerTeamPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    deathWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], deathCountVariance * matchLengths[allMatches]));
                        }
                    }
                }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10, ModelName = "TrueSkill2",
                Compiler = {IncludeDebugInformation = true, GenerateInMemory = false, WriteSourceFiles = true, ShowWarnings = true, UseParallelForLoops = true},
                OptimiseForVariables = new List<IVariable>
                {
                    skills,
                    skillClassWidth,
                    skillDynamicsFactor,
                    skillSharpnessDecreaseFactor,

                    killWeightPlayerTeamPerformance,
                    killWeightPlayerOpponentPerformance,
                    killCountVariance,

                    deathWeightPlayerTeamPerformance,
                    deathWeightPlayerOpponentPerformance,
                    deathCountVariance
                }
            };

            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/sorted_dota2_ts2.json");

            const int batchSize = 32;

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<long, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<long, DateTime>();

            foreach (var batch in rawMatches.Batch(batchSize))
            {
                var steamIdToIndex = new Dictionary<long, int>();
                var indexToSteamId = new Dictionary<int, long>();

                // priors for all the players appearing at least once 
                var batchPriors = new List<Gaussian>();

                // holds the indices of each player in each match w.r.t. the priors array
                var batchMatches = new int[batchSize][][];

                // holds the amount of time that has passed for the m-th player in n-th match since the last time that player has played, index order of this array is [n][m] / [match][player]  
                var batchPlayerTimeLapse = new List<List<double>>();

                // holds the amount of matches the ith player (w.r.t. the priors array) has played in this batch of matches
                var batchPlayerMatchCount = new List<int>();

                // holds the mapping between the batch index of a match and the j-th player's skill in that match
                var batchPlayerMatchMapping = new List<int[]>();

                // holds the length of each match in this batch
                var batchMatchLengths = new double[batchSize];

                // holds the kill counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchKillCounts = new double[batchSize][][];

                // holds the value for whether the kill count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsKillCountMissing = new bool[batchSize][][];

                // holds the death counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchDeathCounts = new double[batchSize][][];

                // holds the value for whether the death count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsDeathCountMissing = new bool[batchSize][][];

                // holds the index of the winning team for each match in the batch
                var batchWinners = new int[batchSize];

                // holds the index of the losing team for each match in the batch
                var batchLosers = new int[batchSize];

                foreach (var (matchIndex, match) in batch.Enumerate())
                {
                    var teams = new[] {match.radiant, match.dire};
                    foreach (var player in match.radiant.Union(match.dire))
                    {
                        // if this is the first time that we've ever seen this player, initialize his/her global skill with the prior
                        if (!playerSkill.ContainsKey(player.steam_id.Value))
                        {
                            playerSkill[player.steam_id.Value] = skillPrior;
                            globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                        }

                        int pIndex;

                        // check if this player has not already appeared in this batch 
                        if (!steamIdToIndex.ContainsKey(player.steam_id.Value))
                        {
                            // set the index that the player will receive
                            steamIdToIndex[player.steam_id.Value] = batchPriors.Count;
                            indexToSteamId[batchPriors.Count] = player.steam_id.Value;
                            pIndex = batchPriors.Count;

                            // init player prior from global skill tracker
                            batchPriors.Add(playerSkill[player.steam_id.Value]);

                            // init the number of matches played in current batch
                            batchPlayerMatchCount.Add(1);

                            // set the time elapsed since the last time this player has played
                            batchPlayerTimeLapse.Add(new List<double> {(match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days});

                            // set up the array that will hold the mapping between the i-th match and the players' matches
                            batchPlayerMatchMapping.Add(new int[batchSize]);
                        }
                        else
                        {
                            // get the index of this player
                            pIndex = steamIdToIndex[player.steam_id.Value];

                            // increase the number of matches played in the current batch by the current player
                            batchPlayerMatchCount[pIndex] += 1;

                            // update batchPlayerTimeLapse, so that we can tell how much time has passed (in days) since the last time this player has played  
                            batchPlayerTimeLapse[pIndex].Add((match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days);
                        }

                        // set up the mapping between the match index and the player's matches 
                        batchPlayerMatchMapping[pIndex][matchIndex] = batchPlayerMatchCount[pIndex] - 1;

                        // update the date of the last played match for the player
                        globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                    }

                    // set playerIndex
                    batchMatches[matchIndex] = teams.Select(t => t.Select(p => steamIdToIndex[p.steam_id.Value]).ToArray()).ToArray();

                    // set matchLength
                    batchMatchLengths[matchIndex] = match.duration;

                    // set winnerIndex
                    batchWinners[matchIndex] = match.radiant_win ? 0 : 1;

                    // set killCounts and killCountMissing
                    batchKillCounts[matchIndex] = teams.Select(t => t.Select(p => p.kills != null ? (double) p.kills.Value : 0.0).ToArray()).ToArray();
                    batchIsKillCountMissing[matchIndex] = teams.Select(t => t.Select(p => p.kills == null).ToArray()).ToArray();

                    // set deathCounts and deathCountMissing
                    batchDeathCounts[matchIndex] = teams.Select(t => t.Select(p => p.deaths != null ? (double) p.deaths.Value : 0.0).ToArray()).ToArray();
                    batchIsDeathCountMissing[matchIndex] = teams.Select(t => t.Select(p => p.deaths == null).ToArray()).ToArray();

                    // set loserIndex
                    batchLosers[matchIndex] = match.radiant_win ? 1 : 0;
                }

                // process this batch with TS2

                #region Parameters

                skillClassWidthPrior.ObservedValue = skillClassWidthPriorValue;
                skillDynamicsFactorPrior.ObservedValue = skillDynamicsFactorPriorValue;
                skillSharpnessDecreaseFactorPrior.ObservedValue = skillSharpnessDecreaseFactorPriorValue;

                killWeightPlayerTeamPerformancePrior.ObservedValue = killWeightPlayerTeamPerformancePriorValue;
                killWeightPlayerOpponentPerformancePrior.ObservedValue = killWeightPlayerOpponentPerformancePriorValue;
                killCountVariancePrior.ObservedValue = killCountVariancePriorValue;

                deathWeightPlayerTeamPerformancePrior.ObservedValue = deathWeightPlayerTeamPerformancePriorValue;
                deathWeightPlayerOpponentPerformancePrior.ObservedValue = deathWeightPlayerOpponentPerformancePriorValue;
                deathCountVariancePrior.ObservedValue = deathCountVariancePriorValue;

                #endregion

                #region Constants

                batchLength.ObservedValue = batchSize;
                numOfPlayers.ObservedValue = batchPriors.Count;

                #endregion

                #region Matches & Outcomes

                matches.ObservedValue = batchMatches;
                winners.ObservedValue = batchWinners;
                losers.ObservedValue = batchLosers;

                #endregion

                #region Skill setup

                numberOfMatchesPlayedPerPlayer.ObservedValue = batchPlayerMatchCount.ToArray();
                skillPriors.ObservedValue = batchPriors.ToArray();
                playerTimeLapse.ObservedValue = batchPlayerTimeLapse.Select(Enumerable.ToArray).ToArray();
                matchLengths.ObservedValue = batchMatchLengths;

                #endregion

                #region Stats

                killCounts.ObservedValue = batchKillCounts;
                isKillCountMissing.ObservedValue = batchIsKillCountMissing;
                deathCounts.ObservedValue = batchDeathCounts;
                isDeathCountMissing.ObservedValue = batchIsDeathCountMissing;

                #endregion

                #region Mapping

                playerMatchMapping.ObservedValue = batchPlayerMatchMapping.ToArray();

                #endregion

                var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);

                // update the priors for the players in this batch
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate()) playerSkill[indexToSteamId[i]] = skillOverTime.Last();

                // update the parameters
                skillClassWidthPriorValue = inferenceEngine.Infer<Gamma>(skillClassWidthPrior);
                skillDynamicsFactorPriorValue = inferenceEngine.Infer<Gamma>(skillDynamicsFactorPrior);
                skillSharpnessDecreaseFactorPriorValue = inferenceEngine.Infer<Gamma>(skillSharpnessDecreaseFactorPrior);

                killWeightPlayerTeamPerformancePriorValue = inferenceEngine.Infer<Beta>(killWeightPlayerTeamPerformancePrior);
                killWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Beta>(killWeightPlayerOpponentPerformancePrior);
                killCountVariancePriorValue = inferenceEngine.Infer<Beta>(killCountVariancePrior);

                deathWeightPlayerTeamPerformancePriorValue = inferenceEngine.Infer<Beta>(deathWeightPlayerTeamPerformancePrior);
                deathWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Beta>(deathWeightPlayerOpponentPerformancePrior);
                deathCountVariancePriorValue = inferenceEngine.Infer<Beta>(deathCountVariancePrior);
            }
        }
    }
}
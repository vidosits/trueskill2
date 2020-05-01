using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Newtonsoft.Json;

// ReSharper disable PossibleInvalidOperationException

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

            // parameter prior values
            var skillPrior = Gaussian.FromMeanAndVariance(1500, 500 * 500); // N ~ (μ, σ)

            var skillClassWidthPriorValue = Gamma.FromMeanAndVariance(250, 100 * 100); // β
            var skillDynamicsPriorValue = Gamma.FromMeanAndVariance(400, 200 * 200); // γ
            var skillSharpnessDecreasePriorValue = Gamma.FromMeanAndVariance(1, 10); // τ

            var killWeightPlayerPerformancePriorValue = Gaussian.FromMeanAndVariance(1, 100);
            var killWeightPlayerOpponentPerformancePriorValue = Gaussian.FromMeanAndVariance(-1, 100);
            var killCountVariancePriorValue = Gamma.FromMeanAndVariance(1, 10);

            var deathWeightPlayerPerformancePriorValue = Gaussian.FromMeanAndVariance(-1, 100);
            var deathWeightPlayerOpponentPerformancePriorValue = Gaussian.FromMeanAndVariance(1, 100);
            var deathCountVariancePriorValue = Gamma.FromMeanAndVariance(1, 10);


            // parameter prior variables
            var skillClassWidthPrior = Variable.New<Gamma>().Named("skillClassWidthPrior");
            var skillDynamicsPrior = Variable.New<Gamma>().Named("skillDynamicsPrior");
            var skillSharpnessDecreasePrior = Variable.New<Gamma>().Named("skillSharpnessDecreasePrior");

            var killWeightPlayerPerformancePrior = Variable.New<Gaussian>().Named("killWeightPlayerPerformancePrior");
            var killWeightPlayerOpponentPerformancePrior = Variable.New<Gaussian>().Named("killWeightPlayerOpponentPerformancePrior");
            var killCountVariancePrior = Variable.New<Gamma>().Named("killCountVariancePrior");

            var deathWeightPlayerPerformancePrior = Variable.New<Gaussian>().Named("deathWeightPlayerPerformancePrior");
            var deathWeightPlayerOpponentPerformancePrior = Variable.New<Gaussian>().Named("deathWeightPlayerOpponentPerformancePrior");
            var deathCountVariancePrior = Variable.New<Gamma>().Named("deathCountVariancePrior");

            // parameter variables
            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth");
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamics = Variable<double>.Random(skillDynamicsPrior).Named("skillDynamics");
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecrease = Variable<double>.Random(skillSharpnessDecreasePrior).Named("skillSharpnessDecrease");
            skillSharpnessDecrease.AddAttribute(new PointEstimate());


            var killWeightPlayerPerformance = Variable<double>.Random(killWeightPlayerPerformancePrior).Named("killWeightPlayerPerformance");
            killWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var killWeightPlayerOpponentPerformance = Variable<double>.Random(killWeightPlayerOpponentPerformancePrior).Named("killWeightPlayerOpponentPerformance");
            killWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var killCountVariance = Variable<double>.Random(killCountVariancePrior).Named("killCountVariance");
            killCountVariance.AddAttribute(new PointEstimate());


            var deathWeightPlayerPerformance = Variable<double>.Random(deathWeightPlayerPerformancePrior).Named("deathWeightPlayerPerformance");
            deathWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var deathWeightPlayerOpponentPerformance = Variable<double>.Random(deathWeightPlayerOpponentPerformancePrior).Named("deathWeightPlayerOpponentPerformance");
            deathWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var deathCountVariance = Variable<double>.Random(deathCountVariancePrior).Named("deathCountVariance");
            deathCountVariance.AddAttribute(new PointEstimate());

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
            var killCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("killCounts");
            var isKillCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isKillCountMissing");

            var deathCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("deathCounts");
            var isDeathCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isDeathCountMissing");

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

            // Initialize skills variable array
            using (var playerBlock = Variable.ForEach(allPlayers))
            {
                using (var matchBlock = Variable.ForEach(matchCounts))
                {
                    using (Variable.If(matchBlock.Index == 0))
                    {
                        skills[allPlayers][matchBlock.Index] = Variable<double>.Random(skillPriors[allPlayers]).Named($"{playerBlock.Index}. player prior");
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[allPlayers][matchBlock.Index] =
                            Variable.GaussianFromMeanAndVariance(
                                Variable.GaussianFromMeanAndVariance(
                                    skills[allPlayers][matchBlock.Index - 1], skillSharpnessDecrease * playerTimeLapse[allPlayers][matchCounts]),
                                skillDynamics);
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
                                    killWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    killWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], killCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isDeathCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            deathCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    deathWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    deathWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], deathCountVariance * matchLengths[allMatches]));
                        }
                    }
                }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 250, ModelName = "TrueSkill2",
                Compiler =
                {
                    IncludeDebugInformation = true,
                    GenerateInMemory = false,
                    WriteSourceFiles = true,
                    ShowWarnings = false,
                    GeneratedSourceFolder = "/mnt/win/Andris/Work/WIN/trueskill/ts.core/generated_source/",
                    UseParallelForLoops = true
                },
                OptimiseForVariables = new List<IVariable>
                {
                    skills,
                    skillClassWidth,
                    skillDynamics,
                    skillSharpnessDecrease,

                    killWeightPlayerPerformance,
                    killWeightPlayerOpponentPerformance,
                    killCountVariance,

                    deathWeightPlayerPerformance,
                    deathWeightPlayerOpponentPerformance,
                    deathCountVariance
                }
            };
            inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));

            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/sorted_dota2_ts2.json");
            // var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/small.json");

            var batchSize = 32873;

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<long, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<long, DateTime>();

            foreach (var batch in rawMatches.Batch(batchSize))
            {
                // batchSize = batch.Count();

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
                skillDynamicsPrior.ObservedValue = skillDynamicsPriorValue;
                skillSharpnessDecreasePrior.ObservedValue = skillSharpnessDecreasePriorValue;

                killWeightPlayerPerformancePrior.ObservedValue = killWeightPlayerPerformancePriorValue;
                killWeightPlayerOpponentPerformancePrior.ObservedValue = killWeightPlayerOpponentPerformancePriorValue;
                killCountVariancePrior.ObservedValue = killCountVariancePriorValue;

                deathWeightPlayerPerformancePrior.ObservedValue = deathWeightPlayerPerformancePriorValue;
                deathWeightPlayerOpponentPerformancePrior.ObservedValue = deathWeightPlayerOpponentPerformancePriorValue;
                deathCountVariancePrior.ObservedValue = deathCountVariancePriorValue;

                #endregion

                #region Constants

                batchLength.ObservedValue = batchSize;
                numOfPlayers.ObservedValue = batchPriors.Count;

                #endregion

                #region Stats

                killCounts.ObservedValue = batchKillCounts;
                isKillCountMissing.ObservedValue = batchIsKillCountMissing;
                deathCounts.ObservedValue = batchDeathCounts;
                isDeathCountMissing.ObservedValue = batchIsDeathCountMissing;

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

                #region Mapping

                playerMatchMapping.ObservedValue = batchPlayerMatchMapping.ToArray();

                #endregion

                var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);

                // update the priors for the players in this batch
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate()) playerSkill[indexToSteamId[i]] = skillOverTime.Last();

                // update the parameters
                skillClassWidthPriorValue = inferenceEngine.Infer<Gamma>(skillClassWidth);
                skillDynamicsPriorValue = inferenceEngine.Infer<Gamma>(skillDynamics);
                skillSharpnessDecreasePriorValue = inferenceEngine.Infer<Gamma>(skillSharpnessDecrease);

                killWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(killWeightPlayerPerformance);
                killWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(killWeightPlayerOpponentPerformance);
                killCountVariancePriorValue = inferenceEngine.Infer<Gamma>(killCountVariance);

                deathWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deathWeightPlayerPerformance);
                deathWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deathWeightPlayerOpponentPerformance);
                deathCountVariancePriorValue = inferenceEngine.Infer<Gamma>(deathCountVariance);
            }

            using (var file = File.CreateText($"/mnt/win/Andris/Work/WIN/trueskill/tests/ratings_{inferenceEngine.NumberOfIterations}_iteration.json"))
            {
                new JsonSerializer().Serialize(file, playerSkill);
            }

            Console.WriteLine($"skillClassWidthPriorValue: {skillClassWidthPriorValue}");
            Console.WriteLine($"skillDynamicsPriorValue: {skillDynamicsPriorValue}");
            Console.WriteLine($"skillSharpnessDecreasePriorValue: {skillSharpnessDecreasePriorValue}");

            Console.WriteLine($"killWeightPlayerPerformancePriorValue: {killWeightPlayerPerformancePriorValue}");
            Console.WriteLine($"killWeightPlayerOpponentPerformancePriorValue: {killWeightPlayerOpponentPerformancePriorValue}");
            Console.WriteLine($"killCountVariancePriorValue: {killCountVariancePriorValue}");

            Console.WriteLine($"deathWeightPlayerPerformancePriorValue: {deathWeightPlayerPerformancePriorValue}");
            Console.WriteLine($"deathWeightPlayerOpponentPerformancePriorValue: {deathWeightPlayerOpponentPerformancePriorValue}");
            Console.WriteLine($"deathCountVariancePriorValue: {deathCountVariancePriorValue}");
        }

        public static void Test()
        {
            // model
            var nItems = Variable.New<int>().Named("nItems");
            var item = new Range(nItems).Named("item");

            var meanPrior = Variable.New<Gaussian>();

            var variancePrior = Variable.New<Gamma>();
            var variance = Variable<double>.Random(variancePrior);
            variance.AddAttribute(new PointEstimate());
            var time = Variable.Array<double>(item);

            var measurements = Variable.Array<double>(item).Named("measurements");


            using (var itemBlock = Variable.ForEach(item))
            {
                using (Variable.If(itemBlock.Index == 0))
                {
                    measurements[item] = Variable<double>.Random(meanPrior);
                }

                using (Variable.If(itemBlock.Index > 0))
                {
                    measurements[item] = Variable.GaussianFromMeanAndVariance(measurements[itemBlock.Index - 1], variance * time[item]);
                }
            }

            // engine
            var inferenceEngine = new InferenceEngine();

            // observations
            nItems.ObservedValue = 5;

            meanPrior.ObservedValue = Gaussian.FromMeanAndVariance(0, 1);
            variancePrior.ObservedValue = Gamma.FromShapeAndRate(1, 10);
            measurements.ObservedValue = new[] {1, 1.4, 0.6, 0.5, 0.7};
            time.ObservedValue = new[] {0, 1.4, 2.1, 1.15, 1.4};

            var inferred = inferenceEngine.Infer<Gamma>(variance);
            Console.WriteLine(inferred);
        }
    }
}
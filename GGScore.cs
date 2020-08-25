using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GGScore.Classes;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Newtonsoft.Json;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace GGScore
{
    public static class GGScore
    {
        private static Gaussian Decay(Gaussian rating, int timeLapse)
        {
            var newVariance = Math.Pow(Math.Sqrt(rating.GetVariance()) + Math.Max(0, timeLapse - 30), 2);
            return Gaussian.FromMeanAndVariance(rating.GetMean(), newVariance);
        }

        public static void Infer(double skillMean, double skillDeviation, Gamma skillClassWidthPriorValue, Gamma skillDynamicsPriorValue, Gamma skillSharpnessDecreasePriorValue, double skillDamping, int numberOfIterations, string gameName, int[] excludedMatchIds, string inputFileDir, string outputFileDir, int outputLimit, string[] parameterMessages)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            
            #region Parameters

            var skillClassWidthPrior = Variable.Observed(skillClassWidthPriorValue); // β
            var skillDynamicsPrior = Variable.Observed(skillDynamicsPriorValue); // γ
            var skillSharpnessDecreasePrior = Variable.Observed(skillSharpnessDecreasePriorValue); // τ

            var rawMatches = Utils.ReadMatchesFromFile<Match<PlayerStat>>(Path.Join(inputFileDir, $"abios_{gameName}_matches_with_stats.json")).Where(x => !excludedMatchIds.Contains(x.Id)).OrderBy(x => x.Date).ThenBy(x => x.Id).ToArray();

            var players = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText(Path.Join(inputFileDir, $"abios_{gameName}_player_names.json")));
            var playerTiers = JsonConvert.DeserializeObject<Dictionary<int, double>>(File.ReadAllText(Path.Join(inputFileDir, $"abios_{gameName}_player_tiers.json")));
            Console.WriteLine("Done.");

            var batchSize = rawMatches.Length;

            #endregion

            #region Constants

            var batchLength = Variable.New<int>();
            var numOfPlayers = Variable.New<int>();

            #endregion

            #region Ranges

            var nPlayers = new Range(numOfPlayers).Named("nPlayers");
            var nMatches = new Range(batchLength).Named("nMatches");

            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch");

            #endregion

            #region Parameters

            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth"); // β
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamics = Variable<double>.Random(skillDynamicsPrior).Named("skillDynamics"); // γ
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecrease = Variable<double>.Random(skillSharpnessDecreasePrior).Named("skillSharpnessDecrease"); // τ
            skillSharpnessDecrease.AddAttribute(new PointEstimate());

            #endregion

            #region Matches

            // Array to hold the player lookup table. Let's us know which players played in which match
            var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches");

            #endregion

            #region Skill setup

            // This array is used to hold the value of the total number of matches played by the specific players in this current batch.
            // These numbers can be used to create the 2D jagged array holding the skills of the player over time.
            var numberOfMatchesPlayedPerPlayer = Variable.Array<int>(nPlayers).Named("numberOfMatchesPlayedPerPlayer");

            // This range is used to access the 2d jagged array (because players play different amount of matches in the batch) holding the player skills
            var matchCounts = new Range(numberOfMatchesPlayedPerPlayer[nPlayers]).Named("matchCounts");

            // Array to hold the prior of the skill for each of the players
            var skillPriors = Variable.Array<Gaussian>(nPlayers).Named("skillPriors");

            // Jagged array holding the skills for all players through all their matches, the first column is the prior
            var skills = Variable.Array(Variable.Array<double>(matchCounts), nPlayers).Named("skills");

            // Array to hold the time elapsed between matches of each player used for calculating the decay of skills
            var playerTimeLapse = Variable.Array(Variable.Array<double>(matchCounts), nPlayers).Named("playerTimeElapsed");

            // Array used to hold the match length information, used for calculating skill updates
            var matchLengths = Variable.Array<double>(nMatches).Named("matchLengths");

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
            var playerMatchMapping = Variable.Array(Variable.Array<int>(nMatches), nPlayers).Named("playerMatchMapping");

            #endregion

            // Initialize skills variable array
            using (var playerBlock = Variable.ForEach(nPlayers))
            {
                using (var matchBlock = Variable.ForEach(matchCounts))
                {
                    using (Variable.If(matchBlock.Index == 0))
                    {
                        skills[playerBlock.Index][matchBlock.Index] =
                            Variable<double>.Random(skillPriors[playerBlock.Index]);
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[playerBlock.Index][matchBlock.Index] = Variable.GaussianFromMeanAndPrecision(
                            Variable.GaussianFromMeanAndPrecision(skills[playerBlock.Index][matchBlock.Index - 1], skillSharpnessDecrease / playerTimeLapse[playerBlock.Index][matchBlock.Index]), skillDynamics);
                    }
                }
            }

            using (Variable.ForEach(nMatches))
            {
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch).Named("playerPerformance");
                var teamPerformance = Variable.Array<double>(nTeamsPerMatch).Named("teamPerformance");

                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matches[nMatches][nTeamsPerMatch][nPlayersPerTeam].Named("playerIndex");
                        var matchIndex = playerMatchMapping[playerIndex][nMatches];

                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skills[playerIndex][matchIndex], skillDamping);
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = Variable.GaussianFromMeanAndPrecision(dampedSkill, skillClassWidth);
                    }

                    teamPerformance[nTeamsPerMatch] = Variable.Sum(playerPerformance[nTeamsPerMatch]);
                }

                Variable.ConstrainTrue(teamPerformance[0] > teamPerformance[1]);
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                Algorithm = new ExpectationPropagation(),
                NumberOfIterations = numberOfIterations,
                ModelName = "TrueSkill2",
                Compiler =
                {
                    IncludeDebugInformation = true,
                    GenerateInMemory = false,
                    WriteSourceFiles = true,
                    UseParallelForLoops = true,
                    ShowWarnings = false
                },
                OptimiseForVariables = new IVariable[] {skills, skillClassWidth, skillSharpnessDecrease, skillDynamics}
            };

            #region Inference

            // global (w.r.t. the batches) dictionary that keeps track of player skills
            var playerSkill = new Dictionary<int, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<int, DateTime>();


            var abiosIdToBatchIndex = new Dictionary<int, int>();
            var batchIndexToAbiosId = new Dictionary<int, int>();

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

            for (var matchIndex = 0; matchIndex < batchSize; ++matchIndex)
            {
                var match = rawMatches[matchIndex];
                var winnerId = match.Winner;
                var loserId = -1;
                try
                {
                    loserId = match.Rosters.Single(x => x.Key != winnerId).Key;
                }
                catch (InvalidOperationException)
                {
                    Console.WriteLine($"error in match: {match.Id}. WinnerId {winnerId} not in rosters.");
                }


                var teams = new[] {match.Rosters[winnerId].ToArray(), match.Rosters[loserId].ToArray()};
                foreach (var player in teams.SelectMany(x => x))
                {
                    // if this is the first time that we've ever seen this player, initialize his/her global skill with the prior
                    if (!playerSkill.ContainsKey(player))
                    {
                        var playerTier = playerTiers[player];
                        playerSkill[player] = Gaussian.FromMeanAndVariance(skillMean - (playerTier - 1) * skillDeviation, Math.Pow(skillDeviation, 2));
                        globalPlayerLastPlayed[player] = match.Date;
                    }

                    int pIndex;

                    // check if this player has not already appeared in this batch 
                    if (!abiosIdToBatchIndex.ContainsKey(player))
                    {
                        // set the index that the player will receive
                        abiosIdToBatchIndex[player] = batchPriors.Count;
                        batchIndexToAbiosId[batchPriors.Count] = player;
                        pIndex = batchPriors.Count;

                        // init player prior from global skill tracker
                        batchPriors.Add(playerSkill[player]);

                        // init the number of matches played in current batch
                        batchPlayerMatchCount.Add(1);

                        // set the time elapsed since the last time this player has played
                        batchPlayerTimeLapse.Add(new List<double> {(match.Date - globalPlayerLastPlayed[player]).Days});

                        // set up the array that will hold the mapping between the i-th match and the players' matches
                        batchPlayerMatchMapping.Add(new int[batchSize]);
                    }
                    else
                    {
                        // get the index of this player
                        pIndex = abiosIdToBatchIndex[player];

                        // increase the number of matches played in the current batch by the current player
                        batchPlayerMatchCount[pIndex] += 1;

                        // update batchPlayerTimeLapse, so that we can tell how much time has passed (in days) since the last time this player has played
                        var lapse = (match.Date - globalPlayerLastPlayed[player]).Days;
                        if (lapse < 0)
                        {
                            Console.WriteLine("what");
                        }

                        batchPlayerTimeLapse[pIndex].Add((match.Date - globalPlayerLastPlayed[player]).Days);
                    }

                    // set up the mapping between the match index and the player's matches 
                    batchPlayerMatchMapping[pIndex][matchIndex] = batchPlayerMatchCount[pIndex] - 1;

                    // update the date of the last played match for the player
                    globalPlayerLastPlayed[player] = match.Date;
                }

                // set playerIndex
                batchMatches[matchIndex] = teams.Select(t => t.Select(p => abiosIdToBatchIndex[p]).ToArray()).ToArray();

                // set matchLength
                batchMatchLengths[matchIndex] = match.MatchLength;
            }

            Console.WriteLine("Batch is ready to be processed.");

            // process this batch with TS2

            #region Constants

            batchLength.ObservedValue = batchSize;
            numOfPlayers.ObservedValue = batchPriors.Count;

            #endregion

            #region Matches

            matches.ObservedValue = batchMatches;

            #endregion

            #region Skill Setup

            numberOfMatchesPlayedPerPlayer.ObservedValue = batchPlayerMatchCount.ToArray();
            skillPriors.ObservedValue = batchPriors.ToArray();
            playerTimeLapse.ObservedValue = batchPlayerTimeLapse.Select(Enumerable.ToArray).ToArray();
            matchLengths.ObservedValue = batchMatchLengths;

            #endregion

            #region Mapping

            playerMatchMapping.ObservedValue = batchPlayerMatchMapping.ToArray();

            #endregion

            // infer the value of the variables
            var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);
            var posteriors = new Dictionary<string, IDistribution>()
            {
                {"β posterior", inferenceEngine.Infer<Gamma>(skillClassWidth)},
                {"γ posterior", inferenceEngine.Infer<Gamma>(skillDynamics)},
                {"τ posterior", inferenceEngine.Infer<Gamma>(skillSharpnessDecrease)}
            };
            watch.Stop();
            Console.WriteLine($"Inference took {watch.Elapsed.TotalMinutes:F} minutes.");
            var lastMatchDate = globalPlayerLastPlayed.Values.Max();
            foreach (var (i, skillOverTime) in inferredSkills.Enumerate())
            {
                var playerAbiosId = batchIndexToAbiosId[i];
                var playerLapse = lastMatchDate - globalPlayerLastPlayed[playerAbiosId];
                playerSkill[playerAbiosId] = Decay(skillOverTime.Last(), playerLapse.Days);
            }

            #endregion

            var lastGlobalMatchDate = globalPlayerLastPlayed.Values.Max();
            double OrderingFunc(KeyValuePair<int, Gaussian> s) => s.Value.GetMean() - (skillMean / skillDeviation) * Math.Sqrt(s.Value.GetVariance());
            var outputFileName = $"{gameName}_{DateTime.Now:yyyyMMddTHHmmss}.csv";
            var outputFileContent = new StringBuilder();
            foreach (var message in parameterMessages) outputFileContent.AppendLine($";{message}");
            outputFileContent.AppendLine($"\n;Last match date: {lastGlobalMatchDate}");
            outputFileContent.AppendLine($"\n;Inference took {watch.Elapsed.TotalMinutes:F} minutes.\n");
            foreach (var (key, posterior) in posteriors) outputFileContent.AppendLine($";{key}: {posterior}");
            outputFileContent.AppendLine("\n");
            object[] columns = {"Rank", "Name", "AbiosId", "Mean", "Uncertainty", "Last played"};
            Console.WriteLine("{0,5} | {1, -13} ({2}) | {3, 4} | {4} | {5, 13}", columns);
            Console.WriteLine(new string('-', 70));
            outputFileContent.AppendLine(string.Join(';', columns));
            
            var playerRank = 1;
            foreach (var (key, value) in playerSkill.OrderByDescending(OrderingFunc))
            {
                var playerLapse = lastGlobalMatchDate - globalPlayerLastPlayed[key];
                object[] row = {playerRank, players[key], key, value.GetMean(), Math.Sqrt(value.GetVariance()), playerLapse.Days};
                if (playerRank <= outputLimit) Console.WriteLine("{0, 5}   {1, -15} ({2,5})   {3, 4:F0}  {4,10:F}  {5, 7} days ago", row);
                outputFileContent.AppendLine(string.Join(';', row));
                playerRank++;
            }

            var outputFilePath = Path.Combine(outputFileDir, outputFileName);
            File.WriteAllText(outputFilePath, outputFileContent.ToString());
            Console.WriteLine($"Output saved to: {outputFilePath}");
        }
    }
}
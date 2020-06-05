using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public static class Trueskill2
    {
        public static void Run()
        {
            #region Parameters

            const int numberOfStats = 16;

            #endregion

            #region Constants

            var zero = Variable.Observed(0.0).Named("zero");
            var teamSize = Variable.Observed(5.0).Named("teamSize");
            var batchLength = Variable.New<int>();
            var numOfPlayers = Variable.New<int>();

            #endregion

            #region Ranges

            var nPlayers = new Range(numOfPlayers).Named("nPlayers");
            var nMatches = new Range(batchLength).Named("nMatches");
            // nMatches.AddAttribute(new Sequential());

            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch");
            var nStats = new Range(numberOfStats).Named("nStats");
            var nParamsPerStat = new Range(2).Named("nParamsPerStat");
            var nHeroes = new Range(130).Named("nHeroes");

            #endregion

            #region Parameters

            // parameter variables
            var skillPrior = Gaussian.FromMeanAndVariance(1500, 500 * 500); // N ~ (μ, σ)

            var skillClassWidthPrior = Variable.Observed(Gamma.FromMeanAndVariance(250, 100 * 100));
            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth"); // β  
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamicsPrior = Variable.Observed(Gamma.FromMeanAndVariance(10, 10 * 10));
            var skillDynamics = Variable<double>.Random(skillDynamicsPrior).Named("skillDynamics"); // γ
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecreasePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var skillSharpnessDecrease = Variable<double>.Random(skillSharpnessDecreasePrior).Named("skillSharpnessDecrease"); // τ
            skillSharpnessDecrease.AddAttribute(new PointEstimate());


            #region Stats

            var gaussianStatParamsPriors = Variable.Array(Variable.Array(Variable.Array<Gaussian>(nParamsPerStat), nStats), nHeroes);
            var gaussianStatParams = Variable.Array(Variable.Array(Variable.Array<double>(nParamsPerStat), nStats), nHeroes);


            var gammaStatParamsPriors = Variable.Array(Variable.Array<Gamma>(nStats), nHeroes);
            var gammaStatParams = Variable.Array(Variable.Array<double>(nStats), nHeroes);


            using (Variable.ForEach(nHeroes))
            {
                using (var statBlock = Variable.ForEach(nStats))
                {
                    using (var paramBlock = Variable.ForEach(nParamsPerStat))
                    {
                        // If stat positively correlates with performance, e.g.: Kills, Assists, Level
                        using (Variable.If(statBlock.Index != 1))
                        {
                            using (Variable.Case(paramBlock.Index, 0))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(1, 4));
                            }

                            using (Variable.Case(paramBlock.Index, 1))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4));
                            }
                        }

                        // If stat negatively correlates with performance, e.g.: Deaths
                        using (Variable.If(statBlock.Index == 1))
                        {
                            using (Variable.Case(paramBlock.Index, 0))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4));
                            }

                            using (Variable.Case(paramBlock.Index, 1))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(1, 4));
                            }
                        }

                        gaussianStatParams[nHeroes][statBlock.Index][paramBlock.Index] = Variable<double>.Random(gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index]);
                        gaussianStatParams[nHeroes][statBlock.Index][paramBlock.Index].AddAttribute(new PointEstimate());
                    }

                    gammaStatParamsPriors[nHeroes][statBlock.Index] = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
                    gammaStatParams[nHeroes][statBlock.Index] = Variable<double>.Random(gammaStatParamsPriors[nHeroes][statBlock.Index]);
                    gammaStatParams[nHeroes][statBlock.Index].AddAttribute(new PointEstimate());
                }
            }

            #endregion

            #endregion

            #region "Matches"

            // Array to hold the player lookup table. Let's us know which players played in which match
            var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches");
            
            // Array to hold the hero lookup table. Let's us know which hero was played by each player in each match
            var heroesPlayed = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("heroesPlayed");
            
            // Array that let's us know whether hero information is available or not
            var isHeroMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("isHeroMissing");
            
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

            #region Stats

            // Initialize arrays holding player stat(s) (e.g.: kills, deaths, etc.) information and whether they are available
            var stats = Variable.Array(Variable.Array(Variable.Array(Variable.Array<double>(nStats), nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("stats");
            var isStatMissing = Variable.Array(Variable.Array(Variable.Array(Variable.Array<bool>(nStats), nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("isStatMissing");

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
                        skills[playerBlock.Index][matchBlock.Index] = Variable<double>.Random(skillPriors[playerBlock.Index]).Named($"{playerBlock.Index}. player prior");
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[playerBlock.Index][matchBlock.Index] =
                            Variable.GaussianFromMeanAndVariance(
                                Variable.GaussianFromMeanAndVariance(
                                    skills[playerBlock.Index][matchBlock.Index - 1], skillSharpnessDecrease * playerTimeLapse[playerBlock.Index][matchBlock.Index]),
                                skillDynamics);
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
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = Variable.GaussianFromMeanAndVariance(skills[playerIndex][matchIndex], skillClassWidth).Named("playerPerformanceInNthMatchInIthTeam");
                    }

                    teamPerformance[nTeamsPerMatch] = Variable.Sum(playerPerformance[nTeamsPerMatch]);
                }

                Variable.ConstrainTrue(teamPerformance[0] > teamPerformance[1]);

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

                        using (Variable.IfNot(isHeroMissing[nMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            var heroId = heroesPlayed[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                            using (Variable.ForEach(nStats))
                            {
                                using (Variable.IfNot(isStatMissing[nMatches][nTeamsPerMatch][nPlayersPerTeam][nStats]))
                                {
                                    stats[nMatches][nTeamsPerMatch][nPlayersPerTeam][nStats] = Variable.Max(zero,
                                        Variable.GaussianFromMeanAndVariance(
                                            gaussianStatParams[heroId][nStats][0] * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                            gaussianStatParams[heroId][nStats][1] * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[nMatches], gammaStatParams[heroId][nStats] * matchLengths[nMatches]));
                                }
                            }                            
                        }
                    }
                }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 100, ModelName = "TrueSkill2",
                Compiler =
                {
                    IncludeDebugInformation = true,
                    GenerateInMemory = false,
                    WriteSourceFiles = true,
                    UseParallelForLoops = false,
                    ShowWarnings = false,
                    GeneratedSourceFolder = "/mnt/win/Andris/Work/WIN/trueskill/ts.core/generated_source"
                },
            };
            inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));

            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/abios_dota_matches_with_stats.json");
            Console.WriteLine("OK.");
            const int batchSize = 36119; // total matches: 36119

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<int, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<int, DateTime>();

            foreach (var batch in rawMatches.Batch(batchSize))
            {
                var abiosIdToBatchIndex = new Dictionary<int, int>();
                var batchIndexToAbiosId = new Dictionary<int, int>();

                // priors for all the players appearing at least once 
                var batchPriors = new List<Gaussian>();

                // holds the indices of each player in each match w.r.t. the priors array
                var batchMatches = new int[batchSize][][];
                
                // hold the hero ids played by each player in each match
                var batchHeroesPlayed = new int[batchSize][][];

                // hold whether the hero id is missing in a match for a particular player
                var batchIsHeroMissing = new bool[batchSize][][];

                // holds the amount of time that has passed for the m-th player in n-th match since the last time that player has played, index order of this array is [n][m] / [match][player]  
                var batchPlayerTimeLapse = new List<List<double>>();

                // holds the amount of matches the ith player (w.r.t. the priors array) has played in this batch of matches
                var batchPlayerMatchCount = new List<int>();

                // holds the mapping between the batch index of a match and the j-th player's skill in that match
                var batchPlayerMatchMapping = new List<int[]>();

                // holds the length of each match in this batch
                var batchMatchLengths = new double[batchSize];

                // holds the stat counts for the l-th stat for k-th player in the j-th team in the i-th match, index order of this array is [i][j][k][l] / [match][team][player][stat]
                var batchStats = new double[batchSize][][][];

                // Indicates whether the value for the l-th stat for k-th player in the j-th team in the i-th match is missing, index order of this array is [i][j][k][l] / [match][team][player][stat]
                var batchStatMissing = new bool[batchSize][][][];

                foreach (var (matchIndex, match) in batch.Enumerate())
                {
                    var winnerId = match.Winner;
                    var loserId = match.Rosters.Single(x => x.Key != winnerId).Key;
                    var teams = new[] {match.Rosters[winnerId].ToArray(), match.Rosters[loserId].ToArray()};
                    foreach (var player in teams.SelectMany(x => x))
                    {
                        // if this is the first time that we've ever seen this player, initialize his/her global skill with the prior
                        if (!playerSkill.ContainsKey(player))
                        {
                            playerSkill[player] = skillPrior;
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

                    // set stats for each player in the match

                    var statsPerTeam = new double[2][][];
                    var statsMissing = new bool [2][][];
                    
                    var heroesPerTeam = new int[2][];
                    var heroesPerTeamMissing = new bool[2][];
                    
                    for (var teamIndex = 0; teamIndex < 2; ++teamIndex)
                    {
                        var playerStats = new double[5][];
                        var playerStatsMissing = new bool[5][];
                        
                        var heroesPerPlayer = new int[5];
                        var heroesPerPlayerMissing = new bool[5];
                        
                        for (var playerIndex = 0; playerIndex < 5; ++playerIndex)
                        {
                            if (match.PlayerStats?[teams[teamIndex][playerIndex]].HeroId != null)
                            {
                                var statsForPlayer = new[]
                                {
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Kills),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Deaths),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Assists),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Level),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Gpm),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Xpm),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Networth),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].CreepKills),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].CreepDenies),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealt),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].HealingDone),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].ObserversKilled),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].ObserversPlaced),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].SentriesKilled),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].SentriesPlaced),
                                    Validate(match.PlayerStats[teams[teamIndex][playerIndex]].CampsStacked)
                                };

                                var statsForPlayerMissing = new[]
                                {
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Kills),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Deaths),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Assists),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Level),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Gpm),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Xpm),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Networth),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].CreepKills),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].CreepDenies),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealt),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].HealingDone),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].ObserversKilled),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].ObserversPlaced),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].SentriesKilled),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].SentriesPlaced),
                                    IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].CampsStacked)
                                };

                                playerStats[playerIndex] = statsForPlayer;
                                playerStatsMissing[playerIndex] = statsForPlayerMissing;

                                heroesPerPlayer[playerIndex] = match.PlayerStats[teams[teamIndex][playerIndex]].HeroId.Value;
                                heroesPerPlayerMissing[playerIndex] = false;
                            }
                            else
                            {
                                playerStats[playerIndex] = new double[numberOfStats];
                                playerStatsMissing[playerIndex] = Enumerable.Repeat(true, numberOfStats).ToArray();
                                
                                heroesPerPlayer[playerIndex] = 0;
                                heroesPerPlayerMissing[playerIndex] = true;
                            }
                        }

                        statsPerTeam[teamIndex] = playerStats;
                        statsMissing[teamIndex] = playerStatsMissing;
                        
                        heroesPerTeam[teamIndex] = heroesPerPlayer;
                        heroesPerTeamMissing[teamIndex] = heroesPerPlayerMissing;
                    }

                    batchStats[matchIndex] = statsPerTeam;
                    batchStatMissing[matchIndex] = statsMissing;

                    batchHeroesPlayed[matchIndex] = heroesPerTeam;
                    batchIsHeroMissing[matchIndex] = heroesPerTeamMissing;
                }
                
                Console.WriteLine("Batch is ready to be processed.");
                
                // process this batch with TS2

                #region Constants

                batchLength.ObservedValue = batchSize;
                numOfPlayers.ObservedValue = batchPriors.Count;

                #endregion

                #region Stats

                stats.ObservedValue = batchStats;
                isStatMissing.ObservedValue = batchStatMissing;

                heroesPlayed.ObservedValue = batchHeroesPlayed;
                isHeroMissing.ObservedValue = batchIsHeroMissing;
                #endregion

                #region Matches

                matches.ObservedValue = batchMatches;

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
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate()) playerSkill[batchIndexToAbiosId[i]] = skillOverTime.Last();

                // update the parameters
                var skillClassWidthPriorValue = inferenceEngine.Infer<Gamma>(skillClassWidth);
                var skillDynamicsPriorValue = inferenceEngine.Infer<Gamma>(skillDynamics);
                var skillSharpnessDecreasePriorValue = inferenceEngine.Infer<Gamma>(skillSharpnessDecrease);
                var gaussianStatParamPriorValues = inferenceEngine.Infer<Gaussian[][][]>(gaussianStatParams);
                var gammaStatParamPriorValues = inferenceEngine.Infer<Gamma[][]>(gammaStatParams);

                skillClassWidthPrior.ObservedValue = skillClassWidthPriorValue;
                skillDynamicsPrior.ObservedValue = skillDynamicsPriorValue;
                skillSharpnessDecreasePrior.ObservedValue = skillSharpnessDecreasePriorValue;
                gaussianStatParamsPriors.ObservedValue = gaussianStatParamPriorValues;
                gammaStatParamsPriors.ObservedValue = gammaStatParamPriorValues;
            }
        }

        private static IEnumerable<Match> ReadMatchesFromFile(string fileName)
        {
            using (var r = new StreamReader(fileName))
            {
                Console.Write("Reading matches from file...");
                var matches = JsonConvert.DeserializeObject<List<Match>>(r.ReadToEnd());
                return matches.OrderBy(x => x.Date);
            }
        }

        private static double Validate(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return !IsValueMissing(value, min, max) ? value.Value : min;
        }

        private static bool IsValueMissing(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return value == null || value < min || value > max;
        }
    }
}
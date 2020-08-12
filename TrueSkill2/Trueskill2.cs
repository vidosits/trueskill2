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
using ts.core.Classes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

// ReSharper disable PossibleInvalidOperationException

namespace ts.core.TrueSkill2
{
    public static class Trueskill2
    {
        public static void Run()
        {
            #region Parameters

            const int numberOfStats = 13;

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
            var nStats = new Range(numberOfStats).Named("nStats");
            var nParamsPerStat = new Range(2).Named("nParamsPerStat");
            var nHeroes = new Range(148).Named("nHeroes");

            #endregion

            #region Parameters

            const double skillMean = 1500.0; // μ
            const double skillDeviation = 250; // σ

            //Tau was 5% of sigma, not 1% as usual
            // Beta is 50% of sigma
            
            
            var skillClassWidthPrior = Variable.Observed(Gamma.FromShapeAndRate(2, 250 * 250));
            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth"); // β
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamicsPrior = Variable.Observed(Gamma.FromShapeAndRate(2, 25 * 25));
            var skillDynamics = Variable<double>.Random(skillDynamicsPrior).Named("skillDynamics"); // γ
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecreasePrior = Variable.Observed(Gamma.FromShapeAndRate(2, 10 * 10));
            var skillSharpnessDecrease = Variable<double>.Random(skillSharpnessDecreasePrior).Named("skillSharpnessDecrease"); // τ
            skillSharpnessDecrease.AddAttribute(new PointEstimate());

            var skillDecayPrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var skillDecay = Variable<double>.Random(skillDecayPrior).Named("skillDecay");
            skillDecay.AddAttribute(new PointEstimate());

            #region Stats

            var gaussianStatParamsPriors = Variable.Array(Variable.Array(Variable.Array<Gaussian>(nParamsPerStat), nStats), nHeroes);
            var gaussianStatParams = Variable.Array(Variable.Array(Variable.Array<double>(nParamsPerStat), nStats), nHeroes);

            var gammaStatParamsPriors = Variable.Array(Variable.Array<Gamma>(nStats), nHeroes);
            var gammaStatParams = Variable.Array(Variable.Array<double>(nStats), nHeroes);

            // using (Variable.ForEach(nHeroes))
            // {
            //     using (var statBlock = Variable.ForEach(nStats))
            //     {
            //         using (var paramBlock = Variable.ForEach(nParamsPerStat))
            //         {
            //             // If stat positively correlates with performance, e.g.: Kills, Assists, Level
            //             using (Variable.If(statBlock.Index != 1))
            //             {
            //                 using (Variable.Case(paramBlock.Index, 0))
            //                 {
            //                     gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(1, 4));
            //                 }
            //
            //                 using (Variable.Case(paramBlock.Index, 1))
            //                 {
            //                     gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4));
            //                 }
            //             }
            //
            //             // If stat negatively correlates with performance, e.g.: Deaths
            //             using (Variable.If(statBlock.Index == 1))
            //             {
            //                 using (Variable.Case(paramBlock.Index, 0))
            //                 {
            //                     gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4));
            //                 }
            //
            //                 using (Variable.Case(paramBlock.Index, 1))
            //                 {
            //                     gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index] = Variable.Observed(Gaussian.FromMeanAndVariance(1, 4));
            //                 }
            //             }
            //
            //             gaussianStatParams[nHeroes][statBlock.Index][paramBlock.Index] = Variable<double>.Random(gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index]);
            //         }
            //     }
            // }
            //
            // gaussianStatParams.AddAttribute(new PointEstimate());
            //
            // gammaStatParamsPriors[nHeroes][nStats].SetTo(Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10)));
            // gammaStatParams[nHeroes][nStats].SetTo(Variable<double>.Random(gammaStatParamsPriors[nHeroes][nStats]));
            // gammaStatParams.AddAttribute(new PointEstimate());

            #endregion

            #endregion

            #region Matches

            // Array to hold the player lookup table. Let's us know which players played in which match
            var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches");

            // Array to hold the hero lookup table. Let's us know which hero was played by each player in each match
            // var heroesPlayed = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("heroesPlayed");
            //
            // // Array that let's us know whether hero information is available or not
            // var isHeroMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("isHeroMissing");

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
            // var stats = Variable.Array(Variable.Array(Variable.Array(Variable.Array<double>(nStats), nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("stats");
            // var isStatMissing = Variable.Array(Variable.Array(Variable.Array(Variable.Array<bool>(nStats), nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("isStatMissing");

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
                        skills[playerBlock.Index][matchBlock.Index] = Variable<double>.Random(skillPriors[playerBlock.Index]);
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[playerBlock.Index][matchBlock.Index] = Variable.GaussianFromMeanAndPrecision(
                            Variable.GaussianFromMeanAndPrecision(skills[playerBlock.Index][matchBlock.Index - 1] - skillDecay * playerTimeLapse[playerBlock.Index][matchBlock.Index],
                                skillSharpnessDecrease / playerTimeLapse[playerBlock.Index][matchBlock.Index]),
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

                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skills[playerIndex][matchIndex], 0.1);
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = Variable.GaussianFromMeanAndPrecision(dampedSkill, skillClassWidth);
                    }

                    teamPerformance[nTeamsPerMatch] = Variable.Sum(playerPerformance[nTeamsPerMatch]);
                }

                Variable.ConstrainTrue(teamPerformance[0] > teamPerformance[1]);

                // using (var team = Variable.ForEach(nTeamsPerMatch))
                // {
                //     using (Variable.ForEach(nPlayersPerTeam))
                //     {
                //         var opponentTeamIndex = Variable.New<int>();
                //         using (Variable.Case(team.Index, 0))
                //         {
                //             opponentTeamIndex.ObservedValue = 1;
                //         }
                //
                //         using (Variable.Case(team.Index, 1))
                //         {
                //             opponentTeamIndex.ObservedValue = 0;
                //         }
                //
                //         using (Variable.IfNot(isHeroMissing[nMatches][nTeamsPerMatch][nPlayersPerTeam]))
                //         {
                //             var heroId = heroesPlayed[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                //             using (Variable.ForEach(nStats))
                //             {
                //                 using (Variable.IfNot(isStatMissing[nMatches][nTeamsPerMatch][nPlayersPerTeam][nStats]))
                //                 {
                //                     stats[nMatches][nTeamsPerMatch][nPlayersPerTeam][nStats].SetTo(Variable.Max(0,
                //                         Variable.GaussianFromMeanAndPrecision(gaussianStatParams[heroId][nStats][0] * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] + gaussianStatParams[heroId][nStats][1] * (teamPerformance[opponentTeamIndex] / 5) * matchLengths[nMatches],
                //                             gammaStatParams[heroId][nStats] / matchLengths[nMatches])));
                //                 }
                //             }
                //         }
                //     }
                // }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                Algorithm = new ExpectationPropagation(),
                NumberOfIterations = 1000,
                ModelName = "TrueSkill2",
                Compiler =
                {
                    IncludeDebugInformation = true,
                    GenerateInMemory = false,
                    WriteSourceFiles = true,
                    UseParallelForLoops = true,
                    ShowWarnings = false,
                    GeneratedSourceFolder = "/mnt/win/Andris/Work/WIN/trueskill/ts.core/generated_source"
                },
            };
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));

            #region Inference

            // var rawMatches = Utils.ReadMatchesFromFile<Match<LeaguePlayerStat>>("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_lol_matches_with_stats_and_converted_champion_ids.json").OrderBy(x => x.Date).ThenBy(x => x.Id);
            // var rawMatches = Utils.ReadMatchesFromFile<Match<CsgoPlayerStat>>("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_csgo_matches_with_stats.json").OrderBy(x => x.Date).ThenBy(x => x.Id);
            var excluded = new[] {313184};
            var rawMatches = Utils.ReadMatchesFromFile<Match<DotaPlayerStat>>("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_dota2_matches_with_stats.json").Where(x => !excluded.Contains(x.Id)).OrderBy(x => x.Date).ThenBy(x => x.Id);
            
            // var players = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_lol_player_names.json"));
            // var players = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_csgo_player_names.json"));
            var players = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText("/mnt/win/Andris/Work/WIN/trueskill/ts.core/Data/abios_dota2_player_names.json"));
            Console.WriteLine("OK.");
            var batchSize = 38432; // total dota2 matches: 38432, total lol matches: 28636, total csgo matches: 56569

            // global (w.r.t. the batches) dictionary that keeps track of player skills
            var playerSkill = new Dictionary<int, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<int, DateTime>();

            foreach (var batch in rawMatches.Batch(batchSize))
            {
                if (batch.Count() < batchSize) batchSize = batch.Count();
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
                            playerSkill[player] = Gaussian.FromMeanAndVariance(skillMean - 2 * (match.Tier - 1) * skillDeviation, Math.Pow(skillDeviation, 2));
                            // playerSkill[player] = Gaussian.FromMeanAndVariance(skillMean, Math.Pow(skillDeviation, 2));
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

                    // for (var teamIndex = 0; teamIndex < 2; ++teamIndex)
                    // {
                    //     var playerStats = new double[5][];
                    //     var playerStatsMissing = new bool[5][];
                    //
                    //     var heroesPerPlayer = new int[5];
                    //     var heroesPerPlayerMissing = new bool[5];
                    //
                    //     for (var playerIndex = 0; playerIndex < 5; ++playerIndex)
                    //     {
                    //         if (match.PlayerStats != null)
                    //         {
                    //             var statsForPlayer = new[]
                    //             {
                    //                 Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Kills),
                    //                 Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Deaths),
                    //                 Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Assists),
                    //             
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].Level),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].GoldEarned),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].CreepScore),
                    //                 //
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToHeroes),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToObjectives),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToTurrets),
                    //                 //
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].HealingDone),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].CrowdControlTime),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].WardsPlaced),
                    //                 // Validate(match.PlayerStats[teams[teamIndex][playerIndex]].WardsDestroyed)
                    //             };
                    //             
                    //             var statsForPlayerMissing = new[]
                    //             {
                    //                 IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Kills),
                    //                 IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Deaths),
                    //                 IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Assists),
                    //             
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].Level),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].GoldEarned),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].CreepScore),
                    //                 //
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToHeroes),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToObjectives),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].DamageDealtToTurrets),
                    //                 //
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].HealingDone),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].CrowdControlTime),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].WardsPlaced),
                    //                 // IsValueMissing(match.PlayerStats[teams[teamIndex][playerIndex]].WardsDestroyed)
                    //             };
                    //
                    //             playerStats[playerIndex] = statsForPlayer;
                    //             playerStatsMissing[playerIndex] = statsForPlayerMissing;
                    //
                    //             // heroesPerPlayer[playerIndex] = match.PlayerStats[teams[teamIndex][playerIndex]].ChampionId.Value;
                    //             heroesPerPlayerMissing[playerIndex] = false;
                    //         }
                    //         else
                    //         {
                    //             playerStats[playerIndex] = new double[numberOfStats];
                    //             playerStatsMissing[playerIndex] = Enumerable.Repeat(true, numberOfStats).ToArray();
                    //
                    //             heroesPerPlayer[playerIndex] = 0;
                    //             heroesPerPlayerMissing[playerIndex] = true;
                    //         }
                    //     }
                    //
                    //     statsPerTeam[teamIndex] = playerStats;
                    //     statsMissing[teamIndex] = playerStatsMissing;
                    //
                    //     heroesPerTeam[teamIndex] = heroesPerPlayer;
                    //     heroesPerTeamMissing[teamIndex] = heroesPerPlayerMissing;
                    // }
                    //
                    // batchStats[matchIndex] = statsPerTeam;
                    // batchStatMissing[matchIndex] = statsMissing;
                    //
                    // batchHeroesPlayed[matchIndex] = heroesPerTeam;
                    // batchIsHeroMissing[matchIndex] = heroesPerTeamMissing;
                }

                Console.WriteLine("Batch is ready to be processed.");

                // process this batch with TS2

                #region Constants

                batchLength.ObservedValue = batchSize;
                numOfPlayers.ObservedValue = batchPriors.Count;

                #endregion

                #region Stats

                // stats.ObservedValue = batchStats;
                // isStatMissing.ObservedValue = batchStatMissing;
                //
                // heroesPlayed.ObservedValue = batchHeroesPlayed;
                // isHeroMissing.ObservedValue = batchIsHeroMissing;

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

                // update the parameters
                var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate()) playerSkill[batchIndexToAbiosId[i]] = skillOverTime.Last();

                // var skillClassWidthPriorValue = inferenceEngine.Infer<Gamma>(skillClassWidth);
                // var skillDynamicsPriorValue = inferenceEngine.Infer<Gamma>(skillDynamics);
                // var skillSharpnessDecreasePriorValue = inferenceEngine.Infer<Gamma>(skillSharpnessDecrease);
                // var gaussianStatParamPriorValues = inferenceEngine.Infer<Gaussian[][][]>(gaussianStatParams);
                // var gammaStatParamPriorValues = inferenceEngine.Infer<Gamma[][]>(gammaStatParams);

                // skillClassWidthPrior.ObservedValue = skillClassWidthPriorValue;
                // skillDynamicsPrior.ObservedValue = skillDynamicsPriorValue;
                // skillSharpnessDecreasePrior.ObservedValue = skillSharpnessDecreasePriorValue;
                // gaussianStatParamsPriors.ObservedValue = gaussianStatParamPriorValues;
                // gammaStatParamsPriors.ObservedValue = gammaStatParamPriorValues;
            }

            #endregion

            var orderings = new Func<KeyValuePair<int, Gaussian>, double>[]
            {
                s => s.Value.GetMean() - (skillMean / skillDeviation) * Math.Sqrt(s.Value.GetVariance()),
                s => s.Value.GetMean() - 3 * Math.Sqrt(s.Value.GetVariance()),
                s => s.Value.GetMean() - Math.Sqrt(s.Value.GetVariance()),
                s => s.Value.GetMean(),
            };
            
            foreach (var ordering in orderings)
            {
                Console.WriteLine();
                var i = 1;
                foreach (var (key, value) in playerSkill.OrderByDescending(ordering).Take(100))
                {
                    Console.WriteLine($"{i}. {players[key]} ({key}) : {value.GetMean():F0}, {Math.Sqrt(value.GetVariance()):F}");
                    i++;
                }
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
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;

namespace ts.core
{
    public static class Trueskill2
    {
        public const double SkillMean = 25.0; // μ
        public const double SkillDeviation = SkillMean / 3; // σ
        private const double SkillClassWidth = SkillDeviation / 2; // β
        private const double SkillDynamicsFactor = SkillDeviation / 100; // γ
        private const double SkillSharpnessDecreaseFactor = SkillDynamicsFactor / 100; //τ

        private const double KillWeightPlayerTeamPerformance = 1;
        private const double KillWeightPlayerOpponentPerformance = -1;
        private const double KillCountVariance = 1;

        private const double DeathWeightPlayerTeamPerformance = -1;
        private const double DeathWeightPlayerOpponentPerformance = 1;
        private const double DeathCountVariance = 1;

        public static IEnumerable<Gaussian[]> Run(Gaussian[] priors, int[][][] playerMatches, double[][] playerElapsedDays, int[] gameCountsPerPlayer, double[] matchLength, double[][][] killCounts,
            bool[][][] killMissing, double[][][] deathCounts, bool[][][] deathMissing, int[] winnerIndex,
            int[] loserIndex)
        {
            // Defaults
            var zero = Variable.Observed(0.0).Named("zero");

            // Ranges to loop through the data: [games][teams][players]
            var allPlayers = new Range(priors.Length).Named("allPlayers");
            var nGames = new Range(winnerIndex.Length).Named("nGames");
            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerGame = new Range(2).Named("nTeamsPerGame");
            var teamSize = Variable.Observed(5.0).Named("teamSize");

            // Array to hold the index of the winning and losing team in each game
            var winner = Variable.Observed(winnerIndex, nGames).Named("winner");
            var loser = Variable.Observed(loserIndex, nGames).Named("loser");

            // Array to hold the player lookup table. With this array player's details can be found in the skills array (to be defined later)
            var matches = Variable.Observed(playerMatches, nGames, nTeamsPerGame, nPlayersPerTeam).Named("matches");

            // This array is used to hold the value of the total number of games played by this player in this current batch.
            // This number can be used to create the 2D jagged array holding the skills of the player over time.
            var playerGameCounts = Variable.Observed(gameCountsPerPlayer, allPlayers).Named("playerGameCounts");

            // This range is used to access the 2d jagged array (because players play different amount of games in the batch) holding the player skills
            var gameCounts = new Range(playerGameCounts[allPlayers]).Named("gameCounts");

            // Skill matrix for all players through all games, the first column is the prior
            var skills = Variable.Array(Variable.Array<double>(gameCounts), allPlayers).Named("skills");

            // Array to hold the prior of the skill for each of the players
            var skillPriors = Variable.Observed(priors, allPlayers).Named("skillPriors");

            // Array to hold the time elapsed between games of each player
            var playerTimeElapsed = Variable.Observed(playerElapsedDays, allPlayers, gameCounts).Named("playerTimeElapsed");

            // Initialize skills variable array, the outer index is the gameIndex, the innerIndex is the player index
            using (var playerBlock = Variable.ForEach(allPlayers))
            {
                using (var gameBlock = Variable.ForEach(gameCounts))
                {
                    using (Variable.If(gameBlock.Index == 0))
                    {
                        skills[allPlayers][gameCounts] = Variable<double>.Random(skillPriors[allPlayers]).Named($"{playerBlock.Index}. player prior");
                    }

                    using (Variable.If(gameBlock.Index > 0))
                    {
                        skills[allPlayers][gameCounts] =
                            Variable.GaussianFromMeanAndPrecision(skills[allPlayers][gameBlock.Index - 1], Math.Pow(SkillSharpnessDecreaseFactor, 2) * playerTimeElapsed[allPlayers][gameCounts])
                                .Named($"{playerBlock.Index}. player skill in {gameBlock.Index}. game");
                    }
                }
            }
            
            // Initialize arrays holding Kill, Death and Match Length information
            var matchLengths = Variable.Observed(matchLength, nGames).Named("matchLengths");
            var killCount = Variable.Observed(killCounts, nGames, nTeamsPerGame, nPlayersPerTeam).Named("killCount");
            var killCountMissing = Variable.Observed(killMissing, nGames, nTeamsPerGame, nPlayersPerTeam).Named("killCountMissing");
            var deathCount = Variable.Observed(deathCounts, nGames, nTeamsPerGame, nPlayersPerTeam).Named("deathCount");
            var deathCountMissing = Variable.Observed(deathMissing, nGames, nTeamsPerGame, nPlayersPerTeam).Named("deathCountMissing");

            using (Variable.ForEach(nGames))
            {
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerGame).Named("playerPerformance");
                var teamPerformance = Variable.Array<double>(nTeamsPerGame).Named("teamPerformance");

                using (Variable.ForEach(nTeamsPerGame))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        playerPerformance[nTeamsPerGame][nPlayersPerTeam] =
                            Variable.GaussianFromMeanAndVariance(skills[matches[nGames][nTeamsPerGame][nPlayersPerTeam]][nGames], Math.Pow(SkillClassWidth, 2))
                                .Named("playerPerformanceInNthGameInIthTeam");
                    }

                    teamPerformance[nTeamsPerGame] = Variable.Sum(playerPerformance[nTeamsPerGame]);
                }

                Variable.ConstrainTrue(teamPerformance[winner[nGames]] > teamPerformance[loser[nGames]]);

                using (var team = Variable.ForEach(nTeamsPerGame))
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

                        using (Variable.IfNot(killCountMissing[nGames][nTeamsPerGame][nPlayersPerTeam]))
                        {
                            killCount[nGames][nTeamsPerGame][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    KillWeightPlayerTeamPerformance * playerPerformance[nTeamsPerGame][nPlayersPerTeam] +
                                    KillWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[nGames], KillCountVariance * matchLengths[nGames]));
                        }

                        using (Variable.IfNot(deathCountMissing[nGames][nTeamsPerGame][nPlayersPerTeam]))
                        {
                            deathCount[nGames][nTeamsPerGame][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    DeathWeightPlayerTeamPerformance * playerPerformance[nTeamsPerGame][nPlayersPerTeam] +
                                    DeathWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[nGames], DeathCountVariance * matchLengths[nGames]));
                        }
                    }
                }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10, ModelName = "TrueSkill2",
                Compiler = {IncludeDebugInformation = true, GenerateInMemory = false, WriteSourceFiles = true}
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);

            // return updated priors / inferred skills
            return inferredSkills;
        }
    }
}
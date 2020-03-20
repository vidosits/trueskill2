using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
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

        public static IEnumerable<Gaussian[]> RunOnline(Gaussian[][] priors, int[][] playerElapsedDays, double matchLength, double[][] killCounts, double[][] deathCounts, int winnerIndex,
            int loserIndex)
        {
            var team = new Range(2).Named("Team");
            var player = new Range(5).Named("Player");
            var playerSkillPriors = Variable.Array(Variable.Array<double>(player), team).Named("PlayerSkillPrior");

            // initialize player skills with a Gaussian prior

            foreach (var (teamIndex, teamPriors) in priors.Enumerate())
            foreach (var (playerIndex, playerPrior) in teamPriors.Enumerate())
                playerSkillPriors[teamIndex][playerIndex] = Variable.Random(playerPrior).Named("PlayerSkill");


            // player skill may have changed since the last time a player has played
            var timeDecayedPlayerSkill = Variable.Array(Variable.Array<double>(player), team);
            for (var t = 0; t < 2; ++t)
            {
                for (var p = 0; p < 5; ++p)
                {
                    timeDecayedPlayerSkill[t][p] = Variable.GaussianFromMeanAndVariance(playerSkillPriors[t][p], Math.Pow(SkillSharpnessDecreaseFactor, 2) * playerElapsedDays[t][p])
                        .Named("DecayedPlayerSkill");
                }
            }

            // The player performance is a noisy version of their skill
            var playerPerformance = Variable.Array(Variable.Array<double>(player), team);
            playerPerformance[team][player] = Variable.GaussianFromMeanAndVariance(timeDecayedPlayerSkill[team][player], Math.Pow(SkillClassWidth, 2)).Named("PlayerPerformance");

            // The winner performed better in this game
            var teamPerformance = Variable.Array<double>(team);
            teamPerformance[team] = Variable.Sum(playerPerformance[team]);

            Variable.ConstrainTrue(
                teamPerformance[winnerIndex].Named("Winning team performance") > teamPerformance[loserIndex].Named("Losing team performance"));

            // take individual statistics into account
            var playerKills = Variable.Array(Variable.Array<double>(player), team);
            for (var t = 0; t < 2; ++t)
            {
                for (var p = 0; p < 5; ++p)
                {
                    var opponentTeamIndex = t == 0 ? 1 : 0;
                    playerKills[t][p] = Variable.Max(0,
                        Variable.GaussianFromMeanAndVariance(
                            (KillWeightPlayerTeamPerformance * playerPerformance[t][p] + KillWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex]) / 5.0) * matchLength,
                            KillCountVariance * matchLength));
                }
            }

            var playerDeaths = Variable.Array(Variable.Array<double>(player), team);
            for (var t = 0; t < 2; ++t)
            {
                for (var p = 0; p < 5; ++p)
                {
                    var opponentTeamIndex = t == 0 ? 1 : 0;
                    playerDeaths[t][p] = Variable.Max(0,
                        Variable.GaussianFromMeanAndVariance(
                            (DeathWeightPlayerTeamPerformance * playerPerformance[t][p] + DeathWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex]) / 5.0) * matchLength,
                            DeathCountVariance * matchLength));
                }
            }

            // attach observed data
            playerKills.ObservedValue = killCounts;
            playerDeaths.ObservedValue = deathCounts;


            // Run inference
            var inferenceEngine = new InferenceEngine {ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10};

            var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(playerSkillPriors);

            // Add Skill Dynamics to player skill posteriors
            for (var teamIndex = 0; teamIndex < 2; ++teamIndex)
            for (var playerIndex = 0; playerIndex < 5; ++playerIndex)
                inferredSkills[teamIndex][playerIndex] = Gaussian.FromMeanAndVariance(inferredSkills[teamIndex][playerIndex].GetMean(),
                    inferredSkills[teamIndex][playerIndex].GetVariance() + Math.Pow(SkillDynamicsFactor, 2));

            // return updated priors / inferred skills
            return inferredSkills;
        }
    }
}
using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace ts.core
{
    public static class TwoTeamTrueskill
    {
        private const double SkillMean = 25.0; // μ
        private const double SkillDeviation = SkillMean / 3; // σ
        private const double SkillClassWidth = SkillDeviation / 2; // β
        private const double SkillDynamicsFactor = SkillDeviation / 100; // τ or γ

        private const double DrawProbability = 0.1;

        internal static void RunExample(int[][] teams, int[] winnerData, int[] loserData)
        {
            // Calculate draw margin
            var drawMargin = Gaussian.FromMeanAndVariance(0, 1).GetQuantile((DrawProbability + 1.0) / 2.0) * Math.Sqrt(10) * SkillClassWidth;

            // Define the game, team, player ranges and playerSkills 2d jagged array
            var game = new Range(winnerData.Length).Named("Game");
            var team = new Range(2).Named("Team");
            var player = new Range(5).Named("Player");
            var playerSkillPriors = Variable.Array(Variable.Array<double>(player), team).Named("PlayerSkills");

            // initialize player skills with a Gaussian prior
            playerSkillPriors[team][player] = Variable.GaussianFromMeanAndVariance(SkillMean, Math.Pow(SkillDeviation, 2) + Math.Pow(SkillDynamicsFactor, 2)).ForEach(team, player).Named("PlayerSkill");

            var winners = Variable.Array<int>(game).Named("Winners");
            var losers = Variable.Array<int>(game).Named("Losers");

            using (Variable.ForEach(game))
            {
                var playerPerformance = Variable.Array(Variable.Array<double>(player), team);

                // The player performance is a noisy version of their skill
                playerPerformance[team][player] = Variable.GaussianFromMeanAndVariance(playerSkillPriors[team][player], Math.Pow(SkillClassWidth, 2)).Named("PlayerPerformance");

                // The winner performed better in this game
                var teamPerformance = Variable.Array<double>(team);
                teamPerformance[team] = Variable.Sum(playerPerformance[team]);

                Variable.ConstrainTrue(
                    (teamPerformance[winners[game]].Named(("Winning team performance")) - teamPerformance[losers[game]].Named("Losing team performance") > drawMargin).Named("IsWinnerHigher"));
            }

            // Attach the data to the model
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            // Run inference
            var inferenceEngine = new InferenceEngine {ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10};
            var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(playerSkillPriors);

            // The inferred skills are uncertain, which is captured in their variance
            foreach (var inferredTeamSkills in inferredSkills)
            {
                var orderedPlayerSkills = inferredTeamSkills.Select((s, i) => new {Player = i, Skill = s}).OrderByDescending(ps => ps.Skill.GetMean());

                foreach (var playerSkill in orderedPlayerSkills)
                {
                    Console.WriteLine($"Player {playerSkill.Player} skill mean: {playerSkill.Skill.GetMean():F3}, variance: {playerSkill.Skill.GetVariance():F3}");
                }
            }

            var updatedPriors = Variable.Array(Variable.Array<double>(player), team);
            for (var teamIndex = 0; teamIndex < 2; teamIndex++)
            {
                for (var playerIndex = 0; playerIndex < 5; playerIndex++)
                {
                    var inferredPlayerSkill = inferredSkills[teamIndex][playerIndex];
                    updatedPriors[teamIndex][playerIndex] = Variable.GaussianFromMeanAndVariance(inferredPlayerSkill.GetMean(), inferredPlayerSkill.GetVariance() + Math.Pow(SkillDynamicsFactor, 2));
                }
            }

            playerSkillPriors = updatedPriors;
        }
    }
}
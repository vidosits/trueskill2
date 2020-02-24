using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace ts.core
{
    public static class TwoPersonTrueskill
    {
        private const double SkillMean = 25.0; // μ
        private const double SkillDeviation = SkillMean / 3; // σ
        private const double SkillClassWidth = SkillDeviation / 2; // β
        private const double SkillDynamicsFactor = SkillDeviation / 100; // τ or γ

        private const double DrawProbability = 0.1;

        internal static void RunExample(int[] winnerData, int[] loserData)
        {
            // Calculate draw margin
            var drawMargin = Gaussian.FromMeanAndVariance(0, 1).GetQuantile((DrawProbability + 1.0) / 2.0) * Math.Sqrt(2) * SkillClassWidth;
            
            // Define the statistical model as a probabilistic program
            var game = new Range(winnerData.Length).Named("Game");
            var player = new Range(winnerData.Concat(loserData).Max() + 1).Named("Player");
            var playerSkills = Variable.Array<double>(player).Named("PlayerSkills");
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(SkillMean, Math.Pow(SkillDeviation, 2) + Math.Pow(SkillDynamicsFactor, 2)).ForEach(player).Named("PlayerSkill");

            var winners = Variable.Array<int>(game).Named("Winners");
            var losers = Variable.Array<int>(game).Named("Losers");

            using (Variable.ForEach(game))
            {
                // The player performance is a noisy version of their skill
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], Math.Pow(SkillClassWidth, 2)).Named("WinnerPerformance");
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], Math.Pow(SkillClassWidth, 2)).Named("LoserPerformance");

                // The winner performed better in this game
                Variable.ConstrainTrue((winnerPerformance - loserPerformance > drawMargin).Named("IsWinnerHigher"));
            }

            // Attach the data to the model
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            // Run inference
            var inferenceEngine = new InferenceEngine {ShowFactorGraph = false, Algorithm = new ExpectationPropagation()};
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            // The inferred skills are uncertain, which is captured in their variance
            var orderedPlayerSkills = inferredSkills.Select((s, i) => new {Player = i, Skill = s}).OrderByDescending(ps => ps.Skill.GetMean());

            foreach (var playerSkill in orderedPlayerSkills)
            {
                Console.WriteLine($"Player {playerSkill.Player} skill mean: {playerSkill.Skill.GetMean():F5}, variance: {playerSkill.Skill.GetVariance():F5}");
            }
        }
    }
}
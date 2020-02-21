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

        internal static void RunExample(int[] winnerData, int[] loserData)
        {
            // Define the statistical model as a probabilistic program
            var game = new Range(winnerData.Length).Named("Game");
            var player = new Range(winnerData.Concat(loserData).Max() + 1).Named("Player");
            var playerSkills = Variable.Array<double>(player).Named("PlayerSkills");
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(SkillMean, SkillDeviation * SkillDeviation + SkillDynamicsFactor * SkillDynamicsFactor).ForEach(player).Named("PlayerSkill");

            var winners = Variable.Array<int>(game).Named("Winners");
            var losers = Variable.Array<int>(game).Named("Losers");

            using (Variable.ForEach(game))
            {
                // The player performance is a noisy version of their skill
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], SkillClassWidth * SkillClassWidth).Named("WinnerPerformance");
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], SkillClassWidth * SkillClassWidth).Named("LoserPerformance");

                // The winner performed better in this game
                Variable.ConstrainTrue((winnerPerformance > loserPerformance).Named("IsWinnerHigher"));
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
                Console.WriteLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
            }
        }
    }
}
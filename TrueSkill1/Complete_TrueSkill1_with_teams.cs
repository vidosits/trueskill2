using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using ts.core.Classes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace ts.core
{
    public static class TwoTeamTrueskill
    {
        public const double SkillMean = 25.0; // μ
        public const double SkillDeviation = SkillMean / 3; // σ
        private const double SkillClassWidth = SkillDeviation / 2; // β
        public const double SkillDynamicsFactor = SkillDeviation / 100; // τ or γ

        private const double DrawProbability = 0.0;

        public static IDictionary<int, Gaussian> Run(string filepath)
        {
            var drawMargin = Gaussian.FromMeanAndVariance(0, 1).GetQuantile((DrawProbability + 1.0) / 2.0) * Math.Sqrt(2) * SkillClassWidth;
            
            var teams = new Range(2).Named("Team");
            var players = new Range(5).Named("Player");
            var playerSkillPriors = Variable.Array(Variable.Array<Gaussian>(players), teams);
            var playerSkills = Variable.Array(Variable.Array<double>(players), teams);

            playerSkills[teams][players] = Variable<double>.Random(playerSkillPriors[teams][players]);

            var playerPerformance = Variable.Array(Variable.Array<double>(players), teams);

            playerPerformance[teams][players] = Variable.GaussianFromMeanAndVariance(playerSkills[teams][players], Math.Pow(SkillClassWidth, 2));

            Variable.ConstrainTrue((Variable.Sum(playerPerformance[0]) - Variable.Sum(playerPerformance[1]) > drawMargin));

            var inferenceEngine = new InferenceEngine {ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10};

            var matches = Utils.ReadMatchesFromFile<Match<DotaPlayerStat>>(filepath);

            var skills = new Dictionary<int, Gaussian>();
            foreach (var match in matches)
            {
                var winners = match.Rosters[match.Winner];
                var losers = match.Rosters[match.Rosters.Keys.Single(id => id != match.Winner)];

                foreach (var playerId in winners.Union(losers))
                {
                    if (!skills.ContainsKey(playerId))
                    {
                        skills[playerId] = Gaussian.FromMeanAndVariance(SkillMean, Math.Pow(SkillDeviation, 2) + Math.Pow(SkillDynamicsFactor, 2));
                    }
                }

                playerSkillPriors.ObservedValue = new[] {winners.Select(playerId => skills[playerId]).ToArray(), losers.Select(playerId => skills[playerId]).ToArray()};
                var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(playerSkills);
                for (var teamIndex = 0; teamIndex < 2; teamIndex++)
                {
                    var team = teamIndex == 0 ? winners : losers;
                    for (var playerIndex = 0; playerIndex < 5; playerIndex++)
                    {
                        skills[team[playerIndex]] = Gaussian.FromMeanAndVariance(inferredSkills[teamIndex][playerIndex].GetMean(), inferredSkills[teamIndex][playerIndex].GetVariance() + Math.Pow(SkillDynamicsFactor, 2));
                    }
                }
            }

            return skills;
        }
    }
}
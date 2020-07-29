using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using ts.core.Classes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace ts.core
{
    public static class Reimplementation<T>
    {
        public const double SkillMean = 1500.0; // μ
        public const double SkillDeviation = 250; // σ
        private const double SkillClassWidth = 150; // β
        private const double SkillDynamicsFactor = 25; // τ

        private const double DrawProbability = 0.0;

        public static IDictionary<int, Gaussian> Run(string filepath)
        {

            #region Model
            

            var teams = new Range(2);
            var players = new Range(5);
            var nHeroes = new Range(148);
            var nParamsPerStat = new Range(2);
            var nStats = new Range(13);

            var playerSkillPriors = Variable.Array(Variable.Array<Gaussian>(players), teams);
            var playerSkills = Variable.Array(Variable.Array<double>(players), teams);
            
            var gaussianStatParamsPriors = Variable.Array(Variable.Array(Variable.Array<Gaussian>(nParamsPerStat), nStats), nHeroes);
            var gaussianStatParams = Variable.Array(Variable.Array(Variable.Array<double>(nParamsPerStat), nStats), nHeroes);
            // gaussianStatParams.AddAttribute(new PointEstimate());

            var gammaStatParamsPriors = Variable.Array(Variable.Array<Gamma>(nStats), nHeroes);
            var gammaStatParams = Variable.Array(Variable.Array<double>(nStats), nHeroes);
            // gammaStatParams.AddAttribute(new PointEstimate());
            
            
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
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index].SetTo(Variable.Observed(Gaussian.FromMeanAndVariance(1, 4)));
                            }

                            using (Variable.Case(paramBlock.Index, 1))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index].SetTo(Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4)));
                            }
                        }

                        // If stat negatively correlates with performance, e.g.: Deaths
                        using (Variable.If(statBlock.Index == 1))
                        {
                            using (Variable.Case(paramBlock.Index, 0))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index].SetTo(Variable.Observed(Gaussian.FromMeanAndVariance(-1, 4)));
                            }

                            using (Variable.Case(paramBlock.Index, 1))
                            {
                                gaussianStatParamsPriors[nHeroes][statBlock.Index][paramBlock.Index].SetTo(Variable.Observed(Gaussian.FromMeanAndVariance(1, 4)));
                            }
                        }
                    }

                    gammaStatParamsPriors[nHeroes][statBlock.Index].SetTo(Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10)));
                }
            }
            
            gammaStatParams[nHeroes][nStats] = Variable<double>.Random(gammaStatParamsPriors[nHeroes][nStats]);
            playerSkills[teams][players] = Variable<double>.Random(playerSkillPriors[teams][players]);

            var playerPerformance = Variable.Array(Variable.Array<double>(players), teams);

            playerPerformance[teams][players] = Variable.GaussianFromMeanAndVariance(playerSkills[teams][players], Math.Pow(SkillClassWidth, 2));

            Variable.ConstrainTrue(Variable.Sum(playerPerformance[0]) > Variable.Sum(playerPerformance[1]));

            #endregion



            #region Inference

            var inferenceEngine = new InferenceEngine {ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 10};

            
            
            var matches = Utils.ReadMatchesFromFile<Match<T>>(filepath);

            var skills = new Dictionary<int, Gaussian>();
            foreach (var match in matches)
            {
                var winners = match.Rosters[match.Winner];
                var losers = match.Rosters[match.Rosters.Keys.Single(id => id != match.Winner)];

                foreach (var playerId in winners.Union(losers))
                {
                    var meanFormula = SkillMean - 2 * (match.Tier - 1) * SkillDeviation;
                    if (!skills.ContainsKey(playerId) || skills[playerId].GetMean() < meanFormula)
                    {
                        skills[playerId] = Gaussian.FromMeanAndVariance(meanFormula, Math.Pow(SkillDeviation, 2) + Math.Pow(SkillDynamicsFactor, 2));
                    }
                }

                playerSkillPriors.ObservedValue = new[]
                {
                    winners.Select(playerId => skills[playerId]).ToArray(),
                    losers.Select(playerId => skills[playerId]).ToArray()
                };
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

            #endregion
        }
    }
}
using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace ts.core
{
    public class Parameters
    {
        public double DrawMarginMean, DrawMarginPrecision, PerformancePrecision, SkillChangePrecision, DrawMarginChangePrecision, WhiteAdvantage;
        public double[][] Skill, DrawMargin;
    }

    internal static class TrueskillThroughTime
    {
        private static void GenerateData(Parameters parameters, int[] firstYear, out int[][] whiteData, out int[][] blackData, out int[][] outcomeData)
        {
            var nYears = parameters.Skill.Length;
            var nPlayers = parameters.Skill[0].Length;
            const int nGames = 1000;
            var whitePlayer = Util.ArrayInit(nYears, year => new List<int>());
            var blackPlayer = Util.ArrayInit(nYears, year => new List<int>());
            var outcomes = Util.ArrayInit(nYears, year => new List<int>());
            for (var game = 0; game < nGames; game++)
            {
                while (true)
                {
                    var w = Rand.Int(nPlayers);
                    var b = Rand.Int(nPlayers);
                    if (w == b)
                        continue;
                    var minYear = Math.Max(firstYear[w], firstYear[b]);
                    var year = Rand.Int(minYear, nYears);
                    var whiteDelta = parameters.WhiteAdvantage + Gaussian.Sample(parameters.Skill[year][w], parameters.PerformancePrecision) 
                                     - Gaussian.Sample(parameters.Skill[year][b], parameters.PerformancePrecision);
                    var whiteDrawMargin = parameters.DrawMargin[year][w];
                    var blackDrawMargin = parameters.DrawMargin[year][b];
                    int outcome;
                    if (whiteDelta > blackDrawMargin)
                        outcome = 2;  // white wins
                    else if (whiteDelta < -whiteDrawMargin)
                        outcome = 0;  // black wins
                    else
                        outcome = 1;  // draw
                    whitePlayer[year].Add(w);
                    blackPlayer[year].Add(b);
                    outcomes[year].Add(outcome);
                    break;
                }
            }
            whiteData = Util.ArrayInit(nYears, year => whitePlayer[year].ToArray());
            blackData = Util.ArrayInit(nYears, year => blackPlayer[year].ToArray());
            outcomeData = Util.ArrayInit(nYears, year => outcomes[year].ToArray());
        }

        internal static void RunExample()
        {
            var engine = new InferenceEngine();

            const int nPlayers = 10;
            const int nYears = 10;
            Rand.Restart(1);

            var skillPrior = new Gaussian(1200, 800 * 800);
            var drawMarginMeanPrior = new Gaussian(700, 500 * 500);
            var drawMarginPrecisionPrior = Gamma.FromShapeAndRate(2, 500 * 500);
            var performancePrecisionPrior = Gamma.FromShapeAndRate(2, 800 * 800);
            var skillChangePrecisionPrior = Gamma.FromShapeAndRate(2, 26 * 26);
            var drawMarginChangePrecisionPrior = Gamma.FromShapeAndRate(2, 10 * 10);
            var whiteAdvantagePrior = new Gaussian(0, 200 * 200);

            var drawMarginMean = Variable.Random(drawMarginMeanPrior).Named("drawMarginMean");
            var drawMarginPrecision = Variable.Random(drawMarginPrecisionPrior).Named("drawMarginPrecision");
            var performancePrecision = Variable.Random(performancePrecisionPrior).Named("performancePrecision");
            var skillChangePrecision = Variable.Random(skillChangePrecisionPrior).Named("skillChangePrecision");
            var drawMarginChangePrecision = Variable.Random(drawMarginChangePrecisionPrior).Named("drawMarginChangePrecision");
            var whiteAdvantage = Variable.Random(whiteAdvantagePrior).Named("whiteAdvantage");

            var player = new Range(nPlayers).Named("player");
            var year = new Range(nYears).Named("year");
            var firstYear = Variable.Array<int>(player).Named("firstYear");
            var skill = Variable.Array(Variable.Array<double>(player), year).Named("skill");
            var drawMargin = Variable.Array(Variable.Array<double>(player), year).Named("drawMargin");

            using (var yearBlock = Variable.ForEach(year))
            {
                var y = yearBlock.Index;
                using (Variable.If(y == 0))
                {
                    skill[year][player] = Variable.Random(skillPrior).ForEach(player);
                    drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMarginMean, drawMarginPrecision).ForEach(player);
                }
                using (Variable.If(y > 0))
                {
                    using (Variable.ForEach(player))
                    {
                        Variable<bool> isFirstYear = (firstYear[player] >= y).Named("isFirstYear");
                        using (Variable.If(isFirstYear))
                        {
                            skill[year][player] = Variable.Random(skillPrior);
                            drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMarginMean, drawMarginPrecision);
                        }
                        using (Variable.IfNot(isFirstYear))
                        {
                            skill[year][player] = Variable.GaussianFromMeanAndPrecision(skill[y - 1][player], skillChangePrecision);
                            drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMargin[y - 1][player], drawMarginChangePrecision);
                        }
                    }
                }
            }

            // Sample parameter values according to the above model
            firstYear.ObservedValue = Util.ArrayInit(nPlayers, i => Rand.Int(nYears));
            var parameters = new Parameters
            {
                DrawMarginMean = drawMarginMeanPrior.Sample(),
                DrawMarginPrecision = drawMarginPrecisionPrior.Sample(),
                PerformancePrecision = performancePrecisionPrior.Sample(),
                SkillChangePrecision = skillChangePrecisionPrior.Sample(),
                DrawMarginChangePrecision = drawMarginChangePrecisionPrior.Sample(),
                WhiteAdvantage = whiteAdvantagePrior.Sample(),
                Skill = Util.ArrayInit(nYears, y => Util.ArrayInit(nPlayers, i => skillPrior.Sample()))
            };
            parameters.DrawMargin = Util.ArrayInit(nYears, y => Util.ArrayInit(nPlayers, i => Gaussian.Sample(parameters.DrawMarginMean, parameters.DrawMarginPrecision)));
            for (int y = 0; y < nYears; y++)
            {
                for (int i = 0; i < nPlayers; i++)
                {
                    if (y > firstYear.ObservedValue[i])
                    {
                        parameters.Skill[y][i] = Gaussian.Sample(parameters.Skill[y - 1][i], parameters.SkillChangePrecision);
                        parameters.DrawMargin[y][i] = Gaussian.Sample(parameters.DrawMargin[y - 1][i], parameters.DrawMarginChangePrecision);
                    }
                }
            }

            // Sample game outcomes
            int[][] whiteData, blackData, outcomeData;
            GenerateData(parameters, firstYear.ObservedValue, out whiteData, out blackData, out outcomeData);

            const bool inferParameters = false; // make this true to infer additional parameters
            if (!inferParameters)
            {
                // fix the true parameters
                drawMarginMean.ObservedValue = parameters.DrawMarginMean;
                drawMarginPrecision.ObservedValue = parameters.DrawMarginPrecision;
                performancePrecision.ObservedValue = parameters.PerformancePrecision;
                skillChangePrecision.ObservedValue = parameters.SkillChangePrecision;
                drawMarginChangePrecision.ObservedValue = parameters.DrawMarginChangePrecision;
            }

            // Learn the skills from the data
            var nGamesData = Util.ArrayInit(nYears, y => outcomeData[y].Length);
            var nGames = Variable.Observed(nGamesData, year).Named("nGames");
            var game = new Range(nGames[year]).Named("game");
            var whitePlayer = Variable.Observed(whiteData, year, game).Named("whitePlayer");
            var blackPlayer = Variable.Observed(blackData, year, game).Named("blackPlayer");
            var outcome = Variable.Observed(outcomeData, year, game).Named("outcome");
            using (Variable.ForEach(year))
            {
                using (Variable.ForEach(game))
                {
                    var w = whitePlayer[year][game];
                    var b = blackPlayer[year][game];
                    var whitePerformance = Variable.GaussianFromMeanAndPrecision(skill[year][w], performancePrecision);
                    var blackPerformance = Variable.GaussianFromMeanAndPrecision(skill[year][b], performancePrecision);
                    var whiteDrawMargin = Variable.Copy(drawMargin[year][w]);
                    var blackDrawMargin = Variable.Copy(drawMargin[year][b]);
                    var whiteDelta = whitePerformance - blackPerformance + whiteAdvantage;
                    using (Variable.Case(outcome[year][game], 0))
                    { // black wins
                        Variable.ConstrainTrue(whiteDelta + whiteDrawMargin < 0);
                    }
                    using (Variable.Case(outcome[year][game], 1))
                    { // draw
                        Variable.ConstrainBetween(whiteDelta, -whiteDrawMargin, blackDrawMargin);
                    }
                    using (Variable.Case(outcome[year][game], 2))
                    { // white wins
                        Variable.ConstrainTrue(whiteDelta - blackDrawMargin > 0);
                    }
                }
            }
            year.AddAttribute(new Sequential());   // helps inference converge faster

            engine.NumberOfIterations = 10;
            var skillPost = engine.Infer<Gaussian[][]>(skill);
            var drawMarginPost = engine.Infer<Gaussian[][]>(drawMargin);

            // compare estimates to the true values
            // if (inferParameters)
            // {
            //     Console.WriteLine("drawMargin mean = {0} (truth = {1})", engine.Infer<Gaussian>(drawMarginMean), parameters.DrawMarginMean);
            //     Console.WriteLine("drawMargin precision = {0} (truth = {1})", engine.Infer<Gamma>(drawMarginPrecision).GetMean(), parameters.DrawMarginPrecision);
            //     Console.WriteLine("performancePrecision = {0} (truth = {1})", engine.Infer<Gamma>(performancePrecision).GetMean(), parameters.PerformancePrecision);
            //     Console.WriteLine("skillChangePrecision = {0} (truth = {1})", engine.Infer<Gamma>(skillChangePrecision).GetMean(), parameters.SkillChangePrecision);
            //     Console.WriteLine("drawMarginChangePrecision = {0} (truth = {1})", engine.Infer<Gamma>(drawMarginChangePrecision).GetMean(), parameters.DrawMarginChangePrecision);
            // }
            Console.WriteLine("white advantage = {0} (truth = {1})", engine.Infer<Gaussian>(whiteAdvantage), parameters.WhiteAdvantage);
            var countPrinted = 0;
            for (int y = 0; y < nYears; y++)
            {
                for (int p = 0; p < nPlayers; p++)
                {
                    if (y >= firstYear.ObservedValue[p])
                    {
                        if (++countPrinted > 3)
                            break;
                        
                        Console.WriteLine("skill[{0}][{1}] = {2} (truth = {3:g4})", y, p, skillPost[y][p], parameters.Skill[y][p]);
                        Console.WriteLine("drawMargin[{0}][{1}] = {2} (truth = {3:g4})", y, p, drawMarginPost[y][p], parameters.DrawMargin[y][p]);
                    }
                }
            }
        }
    }
}
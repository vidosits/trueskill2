using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CommandLine;
using Microsoft.ML.Probabilistic.Distributions;
using Newtonsoft.Json;
using GGScore.Classes;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace GGScore
{
    internal class Options
    {
        [Option('m', "mu", Default = 1500.0, HelpText = "Default (starting) (tier 1) player rating expected value (mean/mu/μ).")]
        public double Mean { get; set; }

        [Option('s', "std", Default = 250.0, HelpText = "Default player rating standard deviation (std/sigma/σ).")]
        public double Std { get; set; }

        [Option("betaShape", Default = 2.0, HelpText = "Shape value for the Gamma dist. from which β is drawn.")]
        public double BetaShape { get; set; }

        [Option("betaRate", Default = 250, HelpText = "Rate value for the Gamma dist. from which β is drawn.")]
        public double BetaRate { get; set; }

        [Option("gammaShape", Default = 2.0, HelpText = "Shape value for the Gamma dist. from which γ is drawn.")]
        public double GammaShape { get; set; }

        [Option("gammaRate", Default = 25, HelpText = "Rate value for the Gamma dist. from which γ is drawn.")]
        public double GammaRate { get; set; }

        [Option("tauShape", Default = 2.0, HelpText = "Shape value for the Gamma dist. from which τ is drawn.")]
        public double TauShape { get; set; }

        [Option("tauRate", Default = 50.0, HelpText = "Rate value for the Gamma dist. from which τ is drawn.")]
        public double TauRate { get; set; }

        [Option('d', "damping", Default = 0.001, HelpText = "Damping for the Markov-chain part of the model.")]
        public double Damping { get; set; }

        [Option('n', "iterations", Default = 1000, HelpText = "Number of iterations to perform on the model.")]
        public int Iterations { get; set; }

        [Option('g', "game", Required = true, HelpText = "Game selector. Valid options = [lol|dota2|csgo]")]
        public string Game { get; set; }

        [Option('i', "inputDir", Required = true, HelpText = "Input directory for the data files exported from Abios.")]
        public string InputDir { get; set; }

        [Option('o', "outputDir", Default = ".", HelpText = "Output directory for the resulting csv file.")]
        public string OutputDir { get; set; }

        [Option('e', "excludedMatches", HelpText = "List of match Ids to exclude from processing.")]
        public IEnumerable<int> ExcludedMatches { get; set; }

        [Option('l', "limit", Default = 50, HelpText = "Number of players to output from the rankings.")]
        public int Limit { get; set; }
    }

    internal static class Program
    {
        private static void Main(string[] args)
        {
            Parser.Default.ParseArguments<Options>(args).WithParsed(Run);
        }

        private static void Run(Options options)
        {
            var parameterMessages = new[]
            {
                $"Skill prior: Gaussian({options.Mean}, {options.Std}^2)",
                $"β: Gamma({options.BetaShape}, {options.BetaRate}^2)",
                $"γ: Gamma({options.GammaShape}, {options.GammaRate}^2)",
                $"τ: Gamma({options.TauShape}, {options.TauRate}^2)",
                $"Damping: {options.Damping}",
                $"Iterations: {options.Iterations}",
                $"Game: {options.Game}",
                $"Excluded matches: {string.Join(',', options.ExcludedMatches)}",
                $"Input directory: {Path.GetFullPath(options.InputDir)}",
                $"Output directory: {Path.GetFullPath(options.OutputDir)}",
                $"Output limit: {options.Limit}",
            };
            
            Console.WriteLine("Using the following parameters:");
            Console.WriteLine();
            foreach (var message in parameterMessages)
            {
                Console.WriteLine(message);
            }
            Console.WriteLine();
            GGScore.Infer(options.Mean,
                options.Std,
                Gamma.FromShapeAndRate(options.BetaShape, Math.Pow(options.BetaRate, 2)),
                Gamma.FromShapeAndRate(options.GammaShape, Math.Pow(options.GammaRate, 2)),
                Gamma.FromShapeAndRate(options.TauShape, Math.Pow(options.TauRate, 2)),
                options.Damping,
                options.Iterations,
                options.Game,
                options.ExcludedMatches.ToArray(),
                options.InputDir,
                options.OutputDir,
                options.Limit,
                parameterMessages);
        }
    }
}
using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Utilities;

namespace ts.core
{
    public static class TwoCoins
    {
        internal static void RunExample()
        {
            var firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            var secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            var bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            var engine = new InferenceEngine();

            // engine.ShowFactorGraph = true;
            if (engine.Algorithm is Microsoft.ML.Probabilistic.Algorithms.VariationalMessagePassing)
            {
                Console.WriteLine("This example does not run with Variational Message Passing");
                return;
            }

            Console.WriteLine("Probability both coins are heads: " + engine.Infer(bothHeads));
            bothHeads.ObservedValue = false;
            Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin));
        }

        internal static void RunMultiple()
        {
            var observedTosses = new[] {true, false, true, true, true, false, false, false};

            var p = Variable.BetaFromMeanAndVariance(0.5, 1.0 / 12);
            p.AddAttribute(new PointEstimate());
            p.AddAttribute(new ListenToMessages());

            var numTosses = observedTosses.Length;
            var toss = new Range(numTosses);
            var tosses = Variable.Array<bool>(toss).Named("tosses");

            tosses[toss] = Variable.Bernoulli(p).ForEach(toss);

            var inferenceEngine = new InferenceEngine();
            inferenceEngine.MessageUpdated += (algorithm, args) => { Console.WriteLine(args.Message); };

            tosses.ObservedValue = observedTosses;
            var inferredProbability = inferenceEngine.Infer<Beta>(p);

            Console.WriteLine(inferredProbability);
        }
    }
}
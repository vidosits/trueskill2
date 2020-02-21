using System;
using Microsoft.ML.Probabilistic.Models;

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
    }
}
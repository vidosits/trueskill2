using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace ts.core
{
    public class OnlineLearning
    {
        public static void Run()
        {
            Variable<int> nItems = Variable.New<int>().Named("nItems");
            Range item = new Range(nItems).Named("item");

            Variable<Gaussian> meanPrior = Variable.Observed(Gaussian.Uniform()).Named("meanPrior");
            Variable<double> mean = Variable<double>.Random(meanPrior);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean, 1.0).ForEach(item);

            InferenceEngine engine = new InferenceEngine();

            // inference on a single batch  
            double[] data = {2, 3, 4, 5};
            x.ObservedValue = data;
            nItems.ObservedValue = data.Length;
            Gaussian meanExpected = engine.Infer<Gaussian>(mean);

            // online learning in mini-batches  
            int batchSize = 1;
            double[][] dataBatches = new double[data.Length / batchSize][];
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                dataBatches[batch] = data.Skip(batch * batchSize).Take(batchSize).ToArray();
            }

            var meanPriorValue = Gaussian.Uniform();
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                meanPrior.ObservedValue = meanPriorValue;
                nItems.ObservedValue = dataBatches[batch].Length;
                x.ObservedValue = dataBatches[batch];
                var meanMarginal = engine.Infer<Gaussian>(mean);
                Console.WriteLine("mean after batch {0} = {1}", batch, meanMarginal);
                meanPriorValue = meanMarginal;

            }

            // the answers should be identical for this simple model  
            Console.WriteLine("mean = {0} should be {1}", meanPriorValue, meanExpected);
        }
    }
}
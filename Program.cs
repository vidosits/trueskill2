namespace ts.core
{
    internal static class Program
    {
        private static void Main()
        {
            
            // Variable<double> y = Variable.Exp(Variable.GaussianFromMeanAndVariance(0,1));  // y's prior is log-normal
            
            // TwoCoins.RunExample();
            // BasicTrueskill.RunExample(new[] { 0, 0, 0, 1, 3, 4 }, new[] { 1, 3, 4, 2, 1, 2 });
            // TwoPersonTrueskill.RunExample(new[] { 0 }, new[] { 1 });
            // TrueskillThroughTime.RunExample();
            // Test.RunExample();
            
            TwoTeamTrueskill.RunExample(new int[][] {new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9}}, new [] {0}, new[] {1});
        }
    }
}
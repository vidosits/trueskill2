namespace Api.Classes
{
    public class Gaussian
    {
        public double Mean { get; set; }
        public double Variance { get; set; }

        public Gaussian(double mean, double variance)
        {
            Mean = mean;
            Variance = variance;
        }
    }
}
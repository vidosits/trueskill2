using System;
using System.Collections.Generic;

namespace Api.Classes
{
    public class RatingsOptions
    {
        // Dota: 1, Lol: 2, Csgo: 5
        public int GameId { get; set; }
        public double Mu { get; set; }
        public double Sigma { get; set; }
        public Dictionary<string, double[]> PlayerPriors { get; set; }
        public Gamma Beta { get; set; }
        public Gamma Gamma { get; set; }
        public Gamma Tau { get; set; }
        public double SkillDamping { get; set; }
        public double SkillOffset { get; set; }
        public int NumberOfIterations { get; set; }
        public bool UseReversePriors { get; set; }
        public double GracePeriod { get; set; }
    
        public int? Limit { get; set; }
        public DateTime? TillDate { get; set; }
    }
}
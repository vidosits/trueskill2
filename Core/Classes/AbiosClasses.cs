using System;
using System.Collections.Generic;

namespace GGScore.Classes
{
    public class PlayerStat
    {
        public int? HeroId { get; set; }
        public string HeroName { get; set; }
        public double?[] Stats { get; set; }
    }

    public class Match
    {
        public int Id { get; set; }
        public DateTime Date {get; set;}
        public Dictionary<int, IList<int>> Rosters { get; set; }
        
        public int SeriesId { get; set; }
        public int Winner { get; set; }
        public int Tier { get; set; }
        public int MatchLength { get; set; }
        public Dictionary<int, PlayerStat> PlayerStats { get; set; }
    }
}
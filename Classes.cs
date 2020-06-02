using System;
using System.Collections.Generic;

namespace ts.core
{
    public class PlayerStat
    {
        public int? HeroId { get; set; }
        public double? Kills { get; set; }
        public double? Deaths { get; set; }
        public double? Assists { get; set; }
        public double? Level { get; set; }
        public double? Gpm { get; set; }
        public double? Xpm { get; set; }
        public double? Networth { get; set; }
        public double? CreepKills { get; set; }
        public double? CreepDenies { get; set; }
        public double? DamageDealt { get; set; }
        public double? HealingDone { get; set; }
        public double? ObserversKilled { get; set; }
        public double? ObserversPlaced { get; set; }
        public double? SentriesKilled { get; set; }
        public double? SentriesPlaced { get; set; }
        public double? CampsStacked { get; set; }
    }

    public class Match
    {
        public int Id { get; set; }
        public DateTime Date {get; set;}
        public Dictionary<int, IList<int>> Rosters { get; set; }
        // public Dictionary<int, int> Scores { get; set; }
        public int SeriesId { get; set; }
        public int Winner { get; set; }
        public int Tier { get; set; }
        public int MatchLength { get; set; }
        public Dictionary<int, PlayerStat> PlayerStats { get; set; }
    }
}
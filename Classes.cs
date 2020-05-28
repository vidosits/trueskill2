using System;
using System.Collections.Generic;

namespace ts.core
{
    public class PlayerStat
    {
        public int? HeroId { get; set; }
        public int? Kills { get; set; }
        public int? Deaths { get; set; }
        public int? Assists { get; set; }
        public int? Level { get; set; }
        public double? Gpm { get; set; }
        public double? Xpm { get; set; }
        public int? Networth { get; set; }
        public int? CreepKills { get; set; }
        public int? CreepDenies { get; set; }
        public int? DamageDealt { get; set; }
        public int? HealingDone { get; set; }
        public int? ObserversKilled { get; set; }
        public int? ObserversPlaced { get; set; }
        public int? SentriesKilled { get; set; }
        public int? SentriesPlaced { get; set; }
        public int? CampsStacked { get; set; }
    }

    public class Match
    {
        public int Id { get; set; }
        public DateTime? Date {get; set;}
        public Dictionary<int, IList<int>> Rosters { get; set; }
        // public Dictionary<int, int> Scores { get; set; }
        public int SeriesId { get; set; }
        public int Winner { get; set; }
        public int Tier { get; set; }
        public int MatchLength { get; set; }
        public Dictionary<int, PlayerStat> PlayerStats { get; set; }
    }
}
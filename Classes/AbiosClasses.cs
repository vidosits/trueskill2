using System;
using System.Collections.Generic;

namespace GGScore.Classes
{
    public class PlayerStat
    {
        
    }
    public class DotaPlayerStat : PlayerStat
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

    public class CsgoPlayerStat : PlayerStat
    {
        public double? Kills { get; set; }
        public double? Deaths { get; set; }
        public double? Assists { get; set; }
        public double? FlashAssists { get; set; }
        public double? Adr { get; set; }
    }
    
    public class LeaguePlayerStat : PlayerStat
    {
        public int? ChampionId { get; set; }
        public string ChampionName { get; set; }
        
        public double? Kills { get; set; }
        public double? Deaths { get; set; }
        public double? Assists { get; set; }
        
        public double? Level { get; set; }
        public double? GoldEarned { get; set; }
        public double? CreepScore { get; set; }
        
        public double? DamageDealtToHeroes { get; set; }
        public double? DamageDealtToObjectives { get; set; }
        public double? DamageDealtToTurrets { get; set; }
        
        public double? HealingDone { get; set; }
        public double? CrowdControlTime { get; set; }
        public double? WardsPlaced { get; set; }
        public double? WardsDestroyed { get; set; }
        
    }
    
    public class Match<T>
    {
        public int Id { get; set; }
        public DateTime Date {get; set;}
        public Dictionary<int, IList<int>> Rosters { get; set; }
        
        public int SeriesId { get; set; }
        public int Winner { get; set; }
        public int Tier { get; set; }
        public int MatchLength { get; set; }
        public Dictionary<int, T> PlayerStats { get; set; }
    }
}
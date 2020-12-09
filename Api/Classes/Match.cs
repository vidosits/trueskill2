using System.Collections.Generic;
using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;
// ReSharper disable ClassNeverInstantiated.Global

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Match
    {
        [BsonElement("id")]
        [JsonPropertyName("id")]
        public int MatchId { get; set; }

        [BsonElement("rosters")]
        [JsonPropertyName("rosters")]
        public IList<Roster> Rosters { get; set; }

        [BsonElement("series")]
        [JsonPropertyName("series")]
        public Series Series { get; set; }

        [BsonElement("series_id")]
        [JsonPropertyName("series_id")]
        public int SeriesId { get; set; }

        [BsonElement("winner")]
        [JsonPropertyName("winner")]
        public int Winner { get; set; }

        [BsonElement("game")]
        [JsonPropertyName("game")]
        public Game Game { get; set; }
    }
}
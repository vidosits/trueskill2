using System.Collections.Generic;
using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Roster
    {
        [BsonElement("id")]
        [JsonPropertyName("id")]
        public int RosterId { get; set; }

        [BsonElement("players")]
        [JsonPropertyName("players")]
        public IList<Player> Players { get; set; }
    }
}
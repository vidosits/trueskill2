using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class MatchSummary
    {
        [BsonElement("match_length")]
        [JsonPropertyName("match_length")]
        public int MatchLength { get; set; }
    }
}
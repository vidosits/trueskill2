using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Team
    {
        [BsonElement("id")]
        [JsonPropertyName("id")]
        public int TeamId { get; set; }
    }
}
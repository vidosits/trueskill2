using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Player
    {
        [BsonElement("id")]
        [JsonPropertyName("id")]
        public int PlayerId { get; set; }
    }
}
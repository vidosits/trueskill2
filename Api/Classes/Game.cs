using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Game
    {
        [BsonElement("id")]
        [JsonPropertyName("id")]
        public int GameId { get; set; }
    }
}
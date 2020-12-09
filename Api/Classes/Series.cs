using System;
using System.Text.Json.Serialization;
using MongoDB.Bson.Serialization.Attributes;

namespace Api.Classes
{
    [BsonIgnoreExtraElements]
    public class Series
    {
        [BsonElement("start")]
        [JsonPropertyName("start")]
        public DateTime Start { get; set; }

        [BsonElement("tier")]
        [JsonPropertyName("tier")]
        public int Tier { get; set; }
    }
}
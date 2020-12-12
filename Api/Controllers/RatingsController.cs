using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Api.Classes;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using MongoDB.Driver;
using Gaussian = Api.Classes.Gaussian;

namespace Api.Controllers
{
    [ApiController]
    [Route("api/")]
    public class RatingsController : ControllerBase
    {
        private readonly ILogger<RatingsController> _logger;
        private readonly IMongoDatabase _database;

        public RatingsController(ILogger<RatingsController> logger)
        {
            _logger = logger;
            var client = new MongoClient(Environment.GetEnvironmentVariable("MONGO_CONNECTION_STRING"));
            _database = client.GetDatabase(Environment.GetEnvironmentVariable("MONGO_DATABASE"));
        }

        [HttpPost("ratings/")]
        public async Task<Dictionary<string, object>> GetRatings([FromBody] RatingsOptions options)
        {
            var (skills, posteriors) = await CalculateRatings(options);
            return new Dictionary<string, object>
            {
                {"ratings", skills.ToDictionary(rating => rating.Key.ToString(), rating => new Gaussian(rating.Value.GetMean(), rating.Value.GetVariance())) },
                {"posteriors", posteriors.ToDictionary(p => p.Key, p => p.Value.Point)}
            };
        }

        private async Task<(Dictionary<int, Microsoft.ML.Probabilistic.Distributions.Gaussian> skills, Dictionary<string, Microsoft.ML.Probabilistic.Distributions.Gamma> posteriors)> CalculateRatings(RatingsOptions options)
        {
            var collection = _database.GetCollection<Match>(Environment.GetEnvironmentVariable("MONGO_COLLECTION"));
            var matches = await collection.Find(m => m.Series.Start != null && m.Series.Tier > 0 && m.Rosters.Count == 2 && m.Game.GameId == options.GameId).SortBy(m => m.Series.Start)
                .Limit(options.Limit)
                .ToListAsync();

            var convertedMatches = matches.Where(m => m.Rosters.All(r => r.Players.Count == 5) &&
                                                      (m.Rosters[0].RosterId == m.Winner || m.Rosters[1].RosterId == m.Winner)).Select(m => new GGScore.Classes.Match
            {
                Id = m.MatchId,
                Date = m.Series.Start,
                Winner = m.Winner,
                Rosters = new Dictionary<int, IList<int>>
                {
                    {m.Rosters[0].RosterId, m.Rosters[0].Players.Select(p => p.PlayerId).ToList()},
                    {m.Rosters[1].RosterId, m.Rosters[1].Players.Select(p => p.PlayerId).ToList()}
                }
            }).ToArray();


            var (playerSkill, _, posteriors, _, _, _) = GGScore.GGScore.Infer(convertedMatches,
                options.Mu,
                options.Sigma,
                options.PlayerPriors.ToDictionary(element => int.Parse(element.Key), element => element.Value),
                Microsoft.ML.Probabilistic.Distributions.Gamma.FromShapeAndRate(options.Beta.Shape, options.Beta.Rate),
                Microsoft.ML.Probabilistic.Distributions.Gamma.FromShapeAndRate(options.Gamma.Shape, options.Gamma.Rate),
                Microsoft.ML.Probabilistic.Distributions.Gamma.FromShapeAndRate(options.Tau.Shape, options.Tau.Rate),
                options.SkillDamping,
                options.NumberOfIterations,
                options.UseReversePriors,
                options.SkillOffset,
                options.GracePeriod
            );

            return (playerSkill, posteriors);
        }
    }
}
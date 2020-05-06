using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Newtonsoft.Json;

// ReSharper disable PossibleInvalidOperationException

namespace ts.core
{
    public static class Trueskill2
    {
        private static IEnumerable<Match> ReadMatchesFromFile(string fileName)
        {
            using (var r = new StreamReader(fileName))
            {
                Console.Write("Reading matches from file...");
                var matchesByDate = JsonConvert.DeserializeObject<Dictionary<string, List<Match>>>(r.ReadToEnd());
                Console.WriteLine("OK.");

                // set up matches as a chronological list
                Console.Write("Setting up chronological match order...");
                var matches = new List<Match>();

                foreach (var matchesOnDate in matchesByDate)
                foreach (var match in matchesOnDate.Value)
                {
                    match.date = Convert.ToDateTime(matchesOnDate.Key);

                    if (match.radiant.Union(match.dire).All(player => player.steam_id != null)) matches.Add(match);
                }

                Console.WriteLine("OK.");

                return matches;
            }
        }

        public static void Run()
        {
            #region Parameters

            // parameter variables
            var skillPrior = Gaussian.FromMeanAndVariance(1500, 500 * 500); // N ~ (μ, σ)

            var skillClassWidth = Variable.GammaFromMeanAndVariance(250, 100 * 100).Named("skillClassWidth"); // β
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamics = Variable.GammaFromMeanAndVariance(400, 200 * 200).Named("skillDynamics"); // γ
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecrease = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("skillSharpnessDecrease"); // τ
            skillSharpnessDecrease.AddAttribute(new PointEstimate());


            #region Stats

            // kills
            var killWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("killWeightPlayerPerformance");
            killWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var killWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("killWeightPlayerOpponentPerformance");
            killWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var killCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("killCountVariance");
            killCountVariance.AddAttribute(new PointEstimate());

            // deaths
            var deathWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("deathWeightPlayerPerformance");
            deathWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var deathWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("deathWeightPlayerOpponentPerformance");
            deathWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var deathCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("deathCountVariance");
            deathCountVariance.AddAttribute(new PointEstimate());

            // assists
            var assistsWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("assistsWeightPlayerPerformance");
            assistsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var assistsWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("assistsWeightPlayerOpponentPerformance");
            assistsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var assistsCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("assistsCountVariance");
            assistsCountVariance.AddAttribute(new PointEstimate());

            // campsStacked
            var campsStackedWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("campsStackedWeightPlayerPerformance");
            campsStackedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var campsStackedWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("campsStackedWeightPlayerOpponentPerformance");
            campsStackedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var campsStackedCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("campsStackedCountVariance");
            campsStackedCountVariance.AddAttribute(new PointEstimate());

            // denies
            var deniesWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("deniesWeightPlayerPerformance");
            deniesWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var deniesWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("deniesWeightPlayerOpponentPerformance");
            deniesWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var deniesCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("deniesCountVariance");
            deniesCountVariance.AddAttribute(new PointEstimate());

            // goldSpent
            var goldSpentWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("goldSpentWeightPlayerPerformance");
            goldSpentWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var goldSpentWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("goldSpentWeightPlayerOpponentPerformance");
            goldSpentWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var goldSpentCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("goldSpentCountVariance");
            goldSpentCountVariance.AddAttribute(new PointEstimate());

            // heroDamage
            var heroDamageWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("heroDamageWeightPlayerPerformance");
            heroDamageWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var heroDamageWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("heroDamageWeightPlayerOpponentPerformance");
            heroDamageWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var heroDamageCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("heroDamageCountVariance");
            heroDamageCountVariance.AddAttribute(new PointEstimate());

            // heroHealing
            var heroHealingWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("heroHealingWeightPlayerPerformance");
            heroHealingWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var heroHealingWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("heroHealingWeightPlayerOpponentPerformance");
            heroHealingWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var heroHealingCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("heroHealingCountVariance");
            heroHealingCountVariance.AddAttribute(new PointEstimate());

            // lastHits
            var lastHitsWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("lastHitsWeightPlayerPerformance");
            lastHitsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var lastHitsWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("lastHitsWeightPlayerOpponentPerformance");
            lastHitsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var lastHitsCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("lastHitsCountVariance");
            lastHitsCountVariance.AddAttribute(new PointEstimate());

            // observersPlaced
            var observersPlacedWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("observersPlacedWeightPlayerPerformance");
            observersPlacedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var observersPlacedWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("observersPlacedWeightPlayerOpponentPerformance");
            observersPlacedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var observersPlacedCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("observersPlacedCountVariance");
            observersPlacedCountVariance.AddAttribute(new PointEstimate());

            // observerKills
            var observerKillsWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("observerKillsWeightPlayerPerformance");
            observerKillsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var observerKillsWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("observerKillsWeightPlayerOpponentPerformance");
            observerKillsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var observerKillsCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("observerKillsCountVariance");
            observerKillsCountVariance.AddAttribute(new PointEstimate());

            // sentriesPlaced
            var sentriesPlacedWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("sentriesPlacedWeightPlayerPerformance");
            sentriesPlacedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var sentriesPlacedWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("sentriesPlacedWeightPlayerOpponentPerformance");
            sentriesPlacedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var sentriesPlacedCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("sentriesPlacedCountVariance");
            sentriesPlacedCountVariance.AddAttribute(new PointEstimate());

            // sentryKills
            var sentryKillsWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("sentryKillsWeightPlayerPerformance");
            sentryKillsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var sentryKillsWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("sentryKillsWeightPlayerOpponentPerformance");
            sentryKillsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var sentryKillsCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("sentryKillsCountVariance");
            sentryKillsCountVariance.AddAttribute(new PointEstimate());

            // stuns
            var stunsWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("stunsWeightPlayerPerformance");
            stunsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var stunsWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("stunsWeightPlayerOpponentPerformance");
            stunsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var stunsCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("stunsCountVariance");
            stunsCountVariance.AddAttribute(new PointEstimate());

            // teamfightParticipation
            var teamfightParticipationWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("teamfightParticipationWeightPlayerPerformance");
            teamfightParticipationWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var teamfightParticipationWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("teamfightParticipationWeightPlayerOpponentPerformance");
            teamfightParticipationWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var teamfightParticipationCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("teamfightParticipationCountVariance");
            teamfightParticipationCountVariance.AddAttribute(new PointEstimate());

            // totalGold
            var totalGoldWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("totalGoldWeightPlayerPerformance");
            totalGoldWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var totalGoldWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("totalGoldWeightPlayerOpponentPerformance");
            totalGoldWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var totalGoldCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("totalGoldCountVariance");
            totalGoldCountVariance.AddAttribute(new PointEstimate());

            // totalExperience
            var totalExperienceWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("totalExperienceWeightPlayerPerformance");
            totalExperienceWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var totalExperienceWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("totalExperienceWeightPlayerOpponentPerformance");
            totalExperienceWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var totalExperienceCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("totalExperienceCountVariance");
            totalExperienceCountVariance.AddAttribute(new PointEstimate());

            // towerDamage
            var towerDamageWeightPlayerPerformance = Variable.GaussianFromMeanAndVariance(1, 10 * 10).Named("towerDamageWeightPlayerPerformance");
            towerDamageWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var towerDamageWeightPlayerOpponentPerformance = Variable.GaussianFromMeanAndVariance(-1, 10 * 10).Named("towerDamageWeightPlayerOpponentPerformance");
            towerDamageWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var towerDamageCountVariance = Variable.GammaFromMeanAndVariance(1, 10 * 10).Named("towerDamageCountVariance");
            towerDamageCountVariance.AddAttribute(new PointEstimate());

            #endregion

            #endregion

            #region Constants

            var zero = Variable.Observed(0.0).Named("zero");
            var batchLength = Variable.New<int>();
            var numOfPlayers = Variable.New<int>();

            #endregion

            #region Ranges

            // Ranges to loop through the data: [matches][teams][players]
            var allPlayers = new Range(numOfPlayers).Named("allPlayers");
            var allMatches = new Range(batchLength).Named("allMatches");
            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch");
            var teamSize = Variable.Observed(5.0).Named("teamSize");

            #endregion

            #region "Matches & Outcomes"

            // Array to hold the player lookup table. With this array player's details can be found in the skills array (to be defined later)
            var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("matches");

            // Array to hold the index of the winning and losing team in each match
            var winners = Variable.Array<int>(allMatches).Named("winners");
            var losers = Variable.Array<int>(allMatches).Named("losers");

            #endregion

            #region Skill setup

            // This array is used to hold the value of the total number of matches played by the specific players in this current batch.
            // These numbers can be used to create the 2D jagged array holding the skills of the player over time.
            var numberOfMatchesPlayedPerPlayer = Variable.Array<int>(allPlayers).Named("numberOfMatchesPlayedPerPlayer");

            // This range is used to access the 2d jagged array (because players play different amount of matches in the batch) holding the player skills
            var matchCounts = new Range(numberOfMatchesPlayedPerPlayer[allPlayers]).Named("matchCounts");

            // Array to hold the prior of the skill for each of the players
            var skillPriors = Variable.Array<Gaussian>(allPlayers).Named("skillPriors");

            // Jagged array holding the skills for all players through all their matches, the first column is the prior
            var skills = Variable.Array(Variable.Array<double>(matchCounts), allPlayers).Named("skills");

            // Array to hold the time elapsed between matches of each player used for calculating the decay of skills
            var playerTimeLapse = Variable.Array(Variable.Array<double>(matchCounts), allPlayers).Named("playerTimeElapsed");

            // Array used to hold the match length information, used for calculating skill updates
            var matchLengths = Variable.Array<double>(allMatches).Named("matchLengths");

            #endregion

            #region Stats

            // Initialize arrays holding player stat(s) (e.g.: kills, deaths, etc.) information and whether they are available
            var killCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("killCounts");
            var isKillCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isKillCountMissing");

            var deathCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("deathCounts");
            var isDeathCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isDeathCountMissing");

            var assistsCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("assistsCounts");
            var isAssistsCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isAssistsCountMissing");

            var campsStackedCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("campsStackedCounts");
            var isCampsStackedCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isCampsStackedCountMissing");

            var deniesCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("deniesCounts");
            var isDeniesCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isDeniesCountMissing");

            var goldSpentCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("goldSpentCounts");
            var isGoldSpentCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isGoldSpentCountMissing");

            var heroDamageCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("heroDamageCounts");
            var isHeroDamageCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isHeroDamageCountMissing");

            var heroHealingCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("heroHealingCounts");
            var isHeroHealingCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isHeroHealingCountMissing");

            var lastHitsCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("lastHitsCounts");
            var isLastHitsCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isLastHitsCountMissing");

            var observersPlacedCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("observersPlacedCounts");
            var isObserversPlacedCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isObserversPlacedCountMissing");

            var observerKillsCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("observerKillsCounts");
            var isObserverKillsCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isObserverKillsCountMissing");

            var sentriesPlacedCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("sentriesPlacedCounts");
            var isSentriesPlacedCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isSentriesPlacedCountMissing");

            var sentryKillsCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("sentryKillsCounts");
            var isSentryKillsCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isSentryKillsCountMissing");

            var stunsCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("stunsCounts");
            var isStunsCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isStunsCountMissing");

            var teamfightParticipationCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("teamfightParticipationCounts");
            var isTeamfightParticipationCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isTeamfightParticipationCountMissing");

            var totalGoldCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("totalGoldCounts");
            var isTotalGoldCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isTotalGoldCountMissing");

            var totalExperienceCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("totalExperienceCounts");
            var isTotalExperienceCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isTotalExperienceCountMissing");

            var towerDamageCounts = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("towerDamageCounts");
            var isTowerDamageCountMissing = Variable.Array(Variable.Array(Variable.Array<bool>(nPlayersPerTeam), nTeamsPerMatch), allMatches).Named("isTowerDamageCountMissing");

            #endregion

            #region Mapping

            // This array is used to hold the mapping between the index (m) of the m-th match in the batch
            // and the index of the same match for the n-th player in the skills array
            // This is needed because we keep track of player skills in a jagged array like the following:
            //
            // "skills":
            //
            //          m a t c h e s (m)
            //        -------------------
            //      p | 0 1 2 3 4
            //      l | 0 1
            //      a | 0 1 2 3
            //      y | 0 1 2 3 4 5
            //      e | 0 1
            //      r | 0
            //      s | 0 1 2 3
            //     (n)| 0 1 2 3
            // 
            // Because we're iterating through a rectangular table of "player ⨯ matches" we need to translate
            // the indices/elements of the rectangular table to the elements of the jagged "skills" array.
            var playerMatchMapping = Variable.Array(Variable.Array<int>(allMatches), allPlayers).Named("playerMatchMapping");

            #endregion

            // Initialize skills variable array
            using (var playerBlock = Variable.ForEach(allPlayers))
            {
                using (var matchBlock = Variable.ForEach(matchCounts))
                {
                    using (Variable.If(matchBlock.Index == 0))
                    {
                        skills[allPlayers][matchBlock.Index] = Variable<double>.Random(skillPriors[allPlayers]).Named($"{playerBlock.Index}. player prior");
                    }

                    using (Variable.If(matchBlock.Index > 0))
                    {
                        skills[allPlayers][matchBlock.Index] =
                            Variable.GaussianFromMeanAndVariance(
                                Variable.GaussianFromMeanAndVariance(
                                    skills[allPlayers][matchBlock.Index - 1], skillSharpnessDecrease * playerTimeLapse[allPlayers][matchCounts]),
                                skillDynamics);
                    }
                }
            }

            using (Variable.ForEach(allMatches))
            {
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch).Named("playerPerformance");
                var teamPerformance = Variable.Array<double>(nTeamsPerMatch).Named("teamPerformance");

                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matches[allMatches][nTeamsPerMatch][nPlayersPerTeam].Named("playerIndex");
                        var matchIndex = playerMatchMapping[playerIndex][allMatches];
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = Variable.GaussianFromMeanAndVariance(skills[playerIndex][matchIndex], skillClassWidth).Named("playerPerformanceInNthMatchInIthTeam");
                    }

                    teamPerformance[nTeamsPerMatch] = Variable.Sum(playerPerformance[nTeamsPerMatch]);
                }

                Variable.ConstrainTrue(teamPerformance[winners[allMatches]] > teamPerformance[losers[allMatches]]);

                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var opponentTeamIndex = Variable.New<int>();
                        using (Variable.Case(team.Index, 0))
                        {
                            opponentTeamIndex.ObservedValue = 1;
                        }

                        using (Variable.Case(team.Index, 1))
                        {
                            opponentTeamIndex.ObservedValue = 0;
                        }

                        using (Variable.IfNot(isKillCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            killCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    killWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    killWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], killCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isDeathCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            deathCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    deathWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    deathWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], deathCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isAssistsCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            assistsCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    assistsWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    assistsWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], assistsCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isCampsStackedCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            campsStackedCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    campsStackedWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    campsStackedWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], campsStackedCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isDeniesCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            deniesCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    deniesWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    deniesWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], deniesCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isGoldSpentCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            goldSpentCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    goldSpentWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    goldSpentWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], goldSpentCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isHeroDamageCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            heroDamageCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    heroDamageWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    heroDamageWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], heroDamageCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isHeroHealingCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            heroHealingCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    heroHealingWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    heroHealingWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], heroHealingCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isLastHitsCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            lastHitsCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    lastHitsWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    lastHitsWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], lastHitsCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isObserversPlacedCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            observersPlacedCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    observersPlacedWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    observersPlacedWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], observersPlacedCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isObserverKillsCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            observerKillsCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    observerKillsWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    observerKillsWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], observerKillsCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isSentriesPlacedCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            sentriesPlacedCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    sentriesPlacedWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    sentriesPlacedWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], sentriesPlacedCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isSentryKillsCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            sentryKillsCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    sentryKillsWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    sentryKillsWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], sentryKillsCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isStunsCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            stunsCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    stunsWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    stunsWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], stunsCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isTeamfightParticipationCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            teamfightParticipationCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    teamfightParticipationWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    teamfightParticipationWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches],
                                    teamfightParticipationCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isTotalGoldCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            totalGoldCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    totalGoldWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    totalGoldWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], totalGoldCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isTotalExperienceCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            totalExperienceCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    totalExperienceWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    totalExperienceWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], totalExperienceCountVariance * matchLengths[allMatches]));
                        }

                        using (Variable.IfNot(isTowerDamageCountMissing[allMatches][nTeamsPerMatch][nPlayersPerTeam]))
                        {
                            towerDamageCounts[allMatches][nTeamsPerMatch][nPlayersPerTeam] = Variable.Max(zero,
                                Variable.GaussianFromMeanAndVariance(
                                    towerDamageWeightPlayerPerformance * playerPerformance[nTeamsPerMatch][nPlayersPerTeam] +
                                    towerDamageWeightPlayerOpponentPerformance * (teamPerformance[opponentTeamIndex] / teamSize) * matchLengths[allMatches], towerDamageCountVariance * matchLengths[allMatches]));
                        }
                    }
                }
            }

            // Run inference
            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false, Algorithm = new ExpectationPropagation(), NumberOfIterations = 100, ModelName = "TrueSkill2",
                Compiler =
                {
                    IncludeDebugInformation = true,
                    GenerateInMemory = false,
                    WriteSourceFiles = true,
                    ShowWarnings = false,
                    GeneratedSourceFolder = "/mnt/win/Andris/Work/WIN/trueskill/ts.core/generated_source/",
                    UseParallelForLoops = false
                },
                OptimiseForVariables = new List<IVariable>
                {
                    skills,
                    skillClassWidth,
                    skillDynamics,
                    skillSharpnessDecrease,

                    killWeightPlayerPerformance,
                    killWeightPlayerOpponentPerformance,
                    killCountVariance,

                    deathWeightPlayerPerformance,
                    deathWeightPlayerOpponentPerformance,
                    deathCountVariance
                }
            };
            inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));

            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/sorted_dota2_ts2.json");
            // var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/small.json");

            var batchSize = 32873;

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<long, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<long, DateTime>();

            foreach (var batch in rawMatches.Batch(batchSize))
            {
                // batchSize = batch.Count();

                var steamIdToIndex = new Dictionary<long, int>();
                var indexToSteamId = new Dictionary<int, long>();

                // priors for all the players appearing at least once 
                var batchPriors = new List<Gaussian>();

                // holds the indices of each player in each match w.r.t. the priors array
                var batchMatches = new int[batchSize][][];

                // holds the amount of time that has passed for the m-th player in n-th match since the last time that player has played, index order of this array is [n][m] / [match][player]  
                var batchPlayerTimeLapse = new List<List<double>>();

                // holds the amount of matches the ith player (w.r.t. the priors array) has played in this batch of matches
                var batchPlayerMatchCount = new List<int>();

                // holds the mapping between the batch index of a match and the j-th player's skill in that match
                var batchPlayerMatchMapping = new List<int[]>();

                // holds the length of each match in this batch
                var batchMatchLengths = new double[batchSize];

                // holds the kill counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchKillCounts = new double[batchSize][][];
                // holds the value for whether the kill count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsKillCountMissing = new bool[batchSize][][];

                // holds the death counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchDeathCounts = new double[batchSize][][];
                // holds the value for whether the death count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsDeathCountMissing = new bool[batchSize][][];

                // holds the assists counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchAssistsCounts = new double[batchSize][][];
                // holds the value for whether the assists count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsAssistsCountMissing = new bool[batchSize][][];

                // holds the campsStacked counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchCampsStackedCounts = new double[batchSize][][];
                // holds the value for whether the campsStacked count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsCampsStackedCountMissing = new bool[batchSize][][];

                // holds the denies counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchDeniesCounts = new double[batchSize][][];
                // holds the value for whether the denies count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsDeniesCountMissing = new bool[batchSize][][];

                // holds the goldSpent counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchGoldSpentCounts = new double[batchSize][][];
                // holds the value for whether the goldSpent count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsGoldSpentCountMissing = new bool[batchSize][][];

                // holds the heroDamage counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchHeroDamageCounts = new double[batchSize][][];
                // holds the value for whether the heroDamage count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsHeroDamageCountMissing = new bool[batchSize][][];

                // holds the heroHealing counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchHeroHealingCounts = new double[batchSize][][];
                // holds the value for whether the heroHealing count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsHeroHealingCountMissing = new bool[batchSize][][];

                // holds the lastHits counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchLastHitsCounts = new double[batchSize][][];
                // holds the value for whether the lastHits count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsLastHitsCountMissing = new bool[batchSize][][];

                // holds the observersPlaced counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchObserversPlacedCounts = new double[batchSize][][];
                // holds the value for whether the observersPlaced count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsObserversPlacedCountMissing = new bool[batchSize][][];

                // holds the observerKills counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchObserverKillsCounts = new double[batchSize][][];
                // holds the value for whether the observerKills count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsObserverKillsCountMissing = new bool[batchSize][][];

                // holds the sentriesPlaced counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchSentriesPlacedCounts = new double[batchSize][][];
                // holds the value for whether the sentriesPlaced count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsSentriesPlacedCountMissing = new bool[batchSize][][];

                // holds the sentryKills counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchSentryKillsCounts = new double[batchSize][][];
                // holds the value for whether the sentryKills count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsSentryKillsCountMissing = new bool[batchSize][][];

                // holds the stuns counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchStunsCounts = new double[batchSize][][];
                // holds the value for whether the stuns count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsStunsCountMissing = new bool[batchSize][][];

                // holds the teamfightParticipation counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchTeamfightParticipationCounts = new double[batchSize][][];
                // holds the value for whether the teamfightParticipation count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsTeamfightParticipationCountMissing = new bool[batchSize][][];

                // holds the totalGold counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchTotalGoldCounts = new double[batchSize][][];
                // holds the value for whether the totalGold count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsTotalGoldCountMissing = new bool[batchSize][][];

                // holds the totalExperience counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchTotalExperienceCounts = new double[batchSize][][];
                // holds the value for whether the totalExperience count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsTotalExperienceCountMissing = new bool[batchSize][][];

                // holds the towerDamage counts for the k-th player in the j-th team in the i-th match, index order of this array is [i][j][k] / [match][team][player]
                var batchTowerDamageCounts = new double[batchSize][][];
                // holds the value for whether the towerDamage count for k-th player in the j-th team in the i-th match is missing or not
                var batchIsTowerDamageCountMissing = new bool[batchSize][][];


                // holds the index of the winning team for each match in the batch
                var batchWinners = new int[batchSize];

                // holds the index of the losing team for each match in the batch
                var batchLosers = new int[batchSize];

                foreach (var (matchIndex, match) in batch.Enumerate())
                {
                    var teams = new[] {match.radiant, match.dire};
                    foreach (var player in match.radiant.Union(match.dire))
                    {
                        // if this is the first time that we've ever seen this player, initialize his/her global skill with the prior
                        if (!playerSkill.ContainsKey(player.steam_id.Value))
                        {
                            playerSkill[player.steam_id.Value] = skillPrior;
                            globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                        }

                        int pIndex;

                        // check if this player has not already appeared in this batch 
                        if (!steamIdToIndex.ContainsKey(player.steam_id.Value))
                        {
                            // set the index that the player will receive
                            steamIdToIndex[player.steam_id.Value] = batchPriors.Count;
                            indexToSteamId[batchPriors.Count] = player.steam_id.Value;
                            pIndex = batchPriors.Count;

                            // init player prior from global skill tracker
                            batchPriors.Add(playerSkill[player.steam_id.Value]);

                            // init the number of matches played in current batch
                            batchPlayerMatchCount.Add(1);

                            // set the time elapsed since the last time this player has played
                            batchPlayerTimeLapse.Add(new List<double> {(match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days});

                            // set up the array that will hold the mapping between the i-th match and the players' matches
                            batchPlayerMatchMapping.Add(new int[batchSize]);
                        }
                        else
                        {
                            // get the index of this player
                            pIndex = steamIdToIndex[player.steam_id.Value];

                            // increase the number of matches played in the current batch by the current player
                            batchPlayerMatchCount[pIndex] += 1;

                            // update batchPlayerTimeLapse, so that we can tell how much time has passed (in days) since the last time this player has played  
                            batchPlayerTimeLapse[pIndex].Add((match.date - globalPlayerLastPlayed[player.steam_id.Value]).Days);
                        }

                        // set up the mapping between the match index and the player's matches 
                        batchPlayerMatchMapping[pIndex][matchIndex] = batchPlayerMatchCount[pIndex] - 1;

                        // update the date of the last played match for the player
                        globalPlayerLastPlayed[player.steam_id.Value] = match.date;
                    }

                    // set playerIndex
                    batchMatches[matchIndex] = teams.Select(t => t.Select(p => steamIdToIndex[p.steam_id.Value]).ToArray()).ToArray();

                    // set matchLength
                    batchMatchLengths[matchIndex] = match.duration;

                    // set winnerIndex
                    batchWinners[matchIndex] = match.radiant_win ? 0 : 1;

                    // set killCounts and killCountMissing
                    batchKillCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.kills)).ToArray()).ToArray();
                    batchIsKillCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.kills)).ToArray()).ToArray();

                    // set deathCounts and deathCountMissing
                    batchDeathCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.deaths)).ToArray()).ToArray();
                    batchIsDeathCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.deaths)).ToArray()).ToArray();

                    // set assistsCounts and assistsCountMissing
                    batchAssistsCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.assists)).ToArray()).ToArray();
                    batchIsAssistsCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.assists)).ToArray()).ToArray();

                    // set campsStackedCounts and campsStackedCountMissing
                    batchCampsStackedCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.camps_stacked)).ToArray()).ToArray();
                    batchIsCampsStackedCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.camps_stacked)).ToArray()).ToArray();

                    // set deniesCounts and deniesCountMissing
                    batchDeniesCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.dn_t)).ToArray()).ToArray();
                    batchIsDeniesCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.dn_t)).ToArray()).ToArray();

                    // set goldSpentCounts and goldSpentCountMissing
                    batchGoldSpentCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.gold_spent)).ToArray()).ToArray();
                    batchIsGoldSpentCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.gold_spent)).ToArray()).ToArray();

                    // set heroDamageCounts and heroDamageCountMissing
                    batchHeroDamageCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.hero_damage)).ToArray()).ToArray();
                    batchIsHeroDamageCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.hero_damage)).ToArray()).ToArray();

                    // set heroHealingCounts and heroHealingCountMissing
                    batchHeroHealingCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.hero_healing)).ToArray()).ToArray();
                    batchIsHeroHealingCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.hero_healing)).ToArray()).ToArray();

                    // set lastHitsCounts and lastHitsCountMissing
                    batchLastHitsCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.lh_t)).ToArray()).ToArray();
                    batchIsLastHitsCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.lh_t)).ToArray()).ToArray();

                    // set observersPlacedCounts and observersPlacedCountMissing
                    batchObserversPlacedCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.obs_placed)).ToArray()).ToArray();
                    batchIsObserversPlacedCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.obs_placed)).ToArray()).ToArray();

                    // set observerKillsCounts and observerKillsCountMissing
                    batchObserverKillsCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.observer_kills)).ToArray()).ToArray();
                    batchIsObserverKillsCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.observer_kills)).ToArray()).ToArray();

                    // set sentriesPlacedCounts and sentriesPlacedCountMissing
                    batchSentriesPlacedCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.sen_placed)).ToArray()).ToArray();
                    batchIsSentriesPlacedCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.sen_placed)).ToArray()).ToArray();

                    // set sentryKillsCounts and sentryKillsCountMissing
                    batchSentryKillsCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.sentry_kills)).ToArray()).ToArray();
                    batchIsSentryKillsCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.sentry_kills)).ToArray()).ToArray();

                    // set stunsCounts and stunsCountMissing
                    batchStunsCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.stuns)).ToArray()).ToArray();
                    batchIsStunsCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.stuns)).ToArray()).ToArray();

                    // set teamfightParticipationCounts and teamfightParticipationCountMissing
                    batchTeamfightParticipationCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.teamfight_participation, 0, 1)).ToArray()).ToArray();
                    batchIsTeamfightParticipationCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.teamfight_participation, 0, 1)).ToArray()).ToArray();

                    // set totalGoldCounts and totalGoldCountMissing
                    batchTotalGoldCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.total_gold)).ToArray()).ToArray();
                    batchIsTotalGoldCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.total_gold)).ToArray()).ToArray();

                    // set totalExperienceCounts and totalExperienceCountMissing
                    batchTotalExperienceCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.total_xp)).ToArray()).ToArray();
                    batchIsTotalExperienceCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.total_xp)).ToArray()).ToArray();

                    // set towerDamageCounts and towerDamageCountMissing
                    batchTowerDamageCounts[matchIndex] = teams.Select(t => t.Select(p => Validate(p.tower_damage)).ToArray()).ToArray();
                    batchIsTowerDamageCountMissing[matchIndex] = teams.Select(t => t.Select(p => IsValueMissing(p.tower_damage)).ToArray()).ToArray();


                    // set loserIndex
                    batchLosers[matchIndex] = match.radiant_win ? 1 : 0;
                }

                // process this batch with TS2

                #region Constants

                batchLength.ObservedValue = batchSize;
                numOfPlayers.ObservedValue = batchPriors.Count;

                #endregion

                #region Stats

                killCounts.ObservedValue = batchKillCounts;
                isKillCountMissing.ObservedValue = batchIsKillCountMissing;
                deathCounts.ObservedValue = batchDeathCounts;
                isDeathCountMissing.ObservedValue = batchIsDeathCountMissing;
                assistsCounts.ObservedValue = batchAssistsCounts;
                isAssistsCountMissing.ObservedValue = batchIsAssistsCountMissing;
                campsStackedCounts.ObservedValue = batchCampsStackedCounts;
                isCampsStackedCountMissing.ObservedValue = batchIsCampsStackedCountMissing;
                deniesCounts.ObservedValue = batchDeniesCounts;
                isDeniesCountMissing.ObservedValue = batchIsDeniesCountMissing;
                goldSpentCounts.ObservedValue = batchGoldSpentCounts;
                isGoldSpentCountMissing.ObservedValue = batchIsGoldSpentCountMissing;
                heroDamageCounts.ObservedValue = batchHeroDamageCounts;
                isHeroDamageCountMissing.ObservedValue = batchIsHeroDamageCountMissing;
                heroHealingCounts.ObservedValue = batchHeroHealingCounts;
                isHeroHealingCountMissing.ObservedValue = batchIsHeroHealingCountMissing;
                lastHitsCounts.ObservedValue = batchLastHitsCounts;
                isLastHitsCountMissing.ObservedValue = batchIsLastHitsCountMissing;
                observersPlacedCounts.ObservedValue = batchObserversPlacedCounts;
                isObserversPlacedCountMissing.ObservedValue = batchIsObserversPlacedCountMissing;
                observerKillsCounts.ObservedValue = batchObserverKillsCounts;
                isObserverKillsCountMissing.ObservedValue = batchIsObserverKillsCountMissing;
                sentriesPlacedCounts.ObservedValue = batchSentriesPlacedCounts;
                isSentriesPlacedCountMissing.ObservedValue = batchIsSentriesPlacedCountMissing;
                sentryKillsCounts.ObservedValue = batchSentryKillsCounts;
                isSentryKillsCountMissing.ObservedValue = batchIsSentryKillsCountMissing;
                stunsCounts.ObservedValue = batchStunsCounts;
                isStunsCountMissing.ObservedValue = batchIsStunsCountMissing;
                teamfightParticipationCounts.ObservedValue = batchTeamfightParticipationCounts;
                isTeamfightParticipationCountMissing.ObservedValue = batchIsTeamfightParticipationCountMissing;
                totalGoldCounts.ObservedValue = batchTotalGoldCounts;
                isTotalGoldCountMissing.ObservedValue = batchIsTotalGoldCountMissing;
                totalExperienceCounts.ObservedValue = batchTotalExperienceCounts;
                isTotalExperienceCountMissing.ObservedValue = batchIsTotalExperienceCountMissing;
                towerDamageCounts.ObservedValue = batchTowerDamageCounts;
                isTowerDamageCountMissing.ObservedValue = batchIsTowerDamageCountMissing;
                        
                #endregion

                #region Matches & Outcomes

                matches.ObservedValue = batchMatches;
                winners.ObservedValue = batchWinners;
                losers.ObservedValue = batchLosers;

                #endregion

                #region Skill setup

                numberOfMatchesPlayedPerPlayer.ObservedValue = batchPlayerMatchCount.ToArray();
                skillPriors.ObservedValue = batchPriors.ToArray();
                playerTimeLapse.ObservedValue = batchPlayerTimeLapse.Select(Enumerable.ToArray).ToArray();
                matchLengths.ObservedValue = batchMatchLengths;

                #endregion

                #region Mapping

                playerMatchMapping.ObservedValue = batchPlayerMatchMapping.ToArray();

                #endregion

                var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);

                // update the priors for the players in this batch
                foreach (var (i, skillOverTime) in inferredSkills.Enumerate()) playerSkill[indexToSteamId[i]] = skillOverTime.Last();

                // update the parameters
                Console.WriteLine($"killClassWidth: {inferenceEngine.Infer<Gamma>(skillClassWidth)}");
                Console.WriteLine($"killDynamics: {inferenceEngine.Infer<Gamma>(skillDynamics)}");
                Console.WriteLine($"killSharpnessDecrease: {inferenceEngine.Infer<Gamma>(skillSharpnessDecrease)}");

                Console.WriteLine($"killWeightPlayerPerformance: {inferenceEngine.Infer<Gaussian>(killWeightPlayerPerformance)}");
                Console.WriteLine($"killWeightPlayerOpponentPerformance: {inferenceEngine.Infer<Gaussian>(killWeightPlayerOpponentPerformance)}");
                Console.WriteLine($"killCountVariance: {inferenceEngine.Infer<Gamma>(killCountVariance)}");

                Console.WriteLine($"deathWeightPlayerPerformance: {inferenceEngine.Infer<Gaussian>(deathWeightPlayerPerformance)}");
                Console.WriteLine($"deathWeightPlayerOpponentPerformance: {inferenceEngine.Infer<Gaussian>(deathWeightPlayerOpponentPerformance)}");
                Console.WriteLine($"deathCountVariance: {inferenceEngine.Infer<Gamma>(deathCountVariance)}");
            }

            using (var file = File.CreateText($"/mnt/win/Andris/Work/WIN/trueskill/tests/ratings_{inferenceEngine.NumberOfIterations}_iteration.json"))
            {
                new JsonSerializer().Serialize(file, playerSkill);
            }
        }

        private static double Validate(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return !IsValueMissing(value, min, max) ? value.Value : min;
        }

        private static bool IsValueMissing(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return (value == null || value < min || value > max);

        }
    }
}
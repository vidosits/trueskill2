using System;
using System.Collections.Generic;
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
        /// <summary>
        /// This method runs a TrueSkill2 model based on the paper "TrueSkill 2:  An improved Bayesian skill rating system" [Minka et al, 2018].
        /// The data used for training is a sample from a larger Dota2 dataset collected from opendota.com
        /// The main difference(s) between this implementation and the one mentioned in the paper are the inclusion of stats other than kills/deaths.
        /// The extra stats include assists, total gold acquired, number of stuns, etc. Hopefully they improve the ratings further, if not they can always be excluded.
        /// Since each stat is modelled after the implementation of kills/deaths in the paper, each one of them require 3 additional parameters to learn:
        /// A weight for the player performance in that match, a weight for the performance of the opposing _team_ in that match, these two are used for the calculation
        /// of the mean of distribution for that stat) and a variance for the distribution used for modelling the stat.
        /// For each parameters, priors are meant to be broad, but maybe I messed them up somehow?
        /// 
        /// As per the paper + my own Dota2 specific modifications, we model the following things:
        /// - Latent player skill: N ~ (μ, σ) which is their expected contribution to the team.
        /// - Performance: noisy version of skill, specific to a game. Noise is defined by the skillClassWidth parameter β.
        /// - TS1/2 deals with partial play, we don't do that here, because in pro Dota2 matches there's no partial play (all weights are 1).
        /// - The winning team's performance is constrained to be higher than the loser's. There are no draws in Dota2 (epsilon draw margin is 0).
        /// - There's a separate skill variable for each _match_ for a specific player, this follows the Trueskill Through Time model, except there the players only had
        ///   a dedicated skill variable for each year they played.
        /// - After each match we assume that each player (probably) improves in skill, so we add an additive γ term to the variance of the latent skills after each match.
        ///   This also prevents their skill distribution collapsing to a point mass, so there's always a little uncertainty (and dynamicity) about their skill.
        /// - Players' skill decreases as time passes, which is controlled by the `skillSharpnessDecrease` variable τ.
        /// - No quit penalty (no quitting in pro matches)
        /// - No experience or squad offset
        /// - Stats all follow the same model:
        ///     - The player's latent skill is inferrred from their individual statistics such as kill, death, sentries placed, assists, total damage, etc. plus team win/loss.
        ///     - The all follow the following formula of equation (9) on top of page 16. of the TrueSkill2 [Minka et al, 2018] paper. Which is:
        ///             value for stat for the i-th player ∼ max(0, N(player_performance_weight * player_performance + opponent_team_performance_weight * opponent_team_performance, stat_specific_weight * match_length))
        ///
        /// The code implements batching of matches, so both batch mode and online mode (which I think is the equivalent of a batch size of 1) are possible.
        /// Right now the full training set of 32873 Dota2 matches fit memory, but with smaller batches online learning is possible as well.
        ///     - For online learning we infer the distribution of each variable and set them as observed values for the parameter priors after each batch.
        /// </summary>
        public static void Run()
        {
            #region Parameters

            // parameter variables
            var skillPrior = Gaussian.FromMeanAndVariance(1500, 500 * 500); // N ~ (μ, σ)

            var skillClassWidthPrior = Variable.Observed(Gamma.FromMeanAndVariance(250, 100 * 100));
            var skillClassWidth = Variable<double>.Random(skillClassWidthPrior).Named("skillClassWidth"); // β  
            skillClassWidth.AddAttribute(new PointEstimate());

            var skillDynamicsPrior = Variable.Observed(Gamma.FromMeanAndVariance(10, 10 * 10));
            var skillDynamics = Variable<double>.Random(skillDynamicsPrior).Named("skillDynamics"); // γ
            skillDynamics.AddAttribute(new PointEstimate());

            var skillSharpnessDecreasePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var skillSharpnessDecrease = Variable<double>.Random(skillSharpnessDecreasePrior).Named("skillSharpnessDecrease"); // τ
            skillSharpnessDecrease.AddAttribute(new PointEstimate());


            #region Stats

            // kills
            var killWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var killWeightPlayerPerformance = Variable<double>.Random(killWeightPlayerPerformancePrior).Named("killWeightPlayerPerformance");
            killWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var killWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var killWeightPlayerOpponentPerformance = Variable<double>.Random(killWeightPlayerOpponentPerformancePrior).Named("killWeightPlayerOpponentPerformance");
            killWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var killCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var killCountVariance = Variable<double>.Random(killCountVariancePrior).Named("killCountVariance");
            killCountVariance.AddAttribute(new PointEstimate());

            // deaths
            var deathWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var deathWeightPlayerPerformance = Variable<double>.Random(deathWeightPlayerPerformancePrior).Named("deathWeightPlayerPerformance");
            deathWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var deathWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var deathWeightPlayerOpponentPerformance = Variable<double>.Random(deathWeightPlayerOpponentPerformancePrior).Named("deathWeightPlayerOpponentPerformance");
            deathWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var deathCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var deathCountVariance = Variable<double>.Random(deathCountVariancePrior).Named("deathCountVariance");
            deathCountVariance.AddAttribute(new PointEstimate());

            // assists
            var assistsWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var assistsWeightPlayerPerformance = Variable<double>.Random(assistsWeightPlayerPerformancePrior).Named("assistsWeightPlayerPerformance");
            assistsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var assistsWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var assistsWeightPlayerOpponentPerformance = Variable<double>.Random(assistsWeightPlayerOpponentPerformancePrior).Named("assistsWeightPlayerOpponentPerformance");
            assistsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var assistsCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var assistsCountVariance = Variable<double>.Random(assistsCountVariancePrior).Named("assistsCountVariance");
            assistsCountVariance.AddAttribute(new PointEstimate());

            // campsStacked
            var campsStackedWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var campsStackedWeightPlayerPerformance = Variable<double>.Random(campsStackedWeightPlayerPerformancePrior).Named("campsStackedWeightPlayerPerformance");
            campsStackedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var campsStackedWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var campsStackedWeightPlayerOpponentPerformance = Variable<double>.Random(campsStackedWeightPlayerOpponentPerformancePrior).Named("campsStackedWeightPlayerOpponentPerformance");
            campsStackedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var campsStackedCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var campsStackedCountVariance = Variable<double>.Random(campsStackedCountVariancePrior).Named("campsStackedCountVariance");
            campsStackedCountVariance.AddAttribute(new PointEstimate());

            // denies
            var deniesWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var deniesWeightPlayerPerformance = Variable<double>.Random(deniesWeightPlayerPerformancePrior).Named("deniesWeightPlayerPerformance");
            deniesWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var deniesWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var deniesWeightPlayerOpponentPerformance = Variable<double>.Random(deniesWeightPlayerOpponentPerformancePrior).Named("deniesWeightPlayerOpponentPerformance");
            deniesWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var deniesCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var deniesCountVariance = Variable<double>.Random(deniesCountVariancePrior).Named("deniesCountVariance");
            deniesCountVariance.AddAttribute(new PointEstimate());

            // goldSpent
            var goldSpentWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var goldSpentWeightPlayerPerformance = Variable<double>.Random(goldSpentWeightPlayerPerformancePrior).Named("goldSpentWeightPlayerPerformance");
            goldSpentWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var goldSpentWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var goldSpentWeightPlayerOpponentPerformance = Variable<double>.Random(goldSpentWeightPlayerOpponentPerformancePrior).Named("goldSpentWeightPlayerOpponentPerformance");
            goldSpentWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var goldSpentCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var goldSpentCountVariance = Variable<double>.Random(goldSpentCountVariancePrior).Named("goldSpentCountVariance");
            goldSpentCountVariance.AddAttribute(new PointEstimate());

            // heroDamage
            var heroDamageWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var heroDamageWeightPlayerPerformance = Variable<double>.Random(heroDamageWeightPlayerPerformancePrior).Named("heroDamageWeightPlayerPerformance");
            heroDamageWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var heroDamageWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var heroDamageWeightPlayerOpponentPerformance = Variable<double>.Random(heroDamageWeightPlayerOpponentPerformancePrior).Named("heroDamageWeightPlayerOpponentPerformance");
            heroDamageWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var heroDamageCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var heroDamageCountVariance = Variable<double>.Random(heroDamageCountVariancePrior).Named("heroDamageCountVariance");
            heroDamageCountVariance.AddAttribute(new PointEstimate());

            // heroHealing
            var heroHealingWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var heroHealingWeightPlayerPerformance = Variable<double>.Random(heroHealingWeightPlayerPerformancePrior).Named("heroHealingWeightPlayerPerformance");
            heroHealingWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var heroHealingWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var heroHealingWeightPlayerOpponentPerformance = Variable<double>.Random(heroHealingWeightPlayerOpponentPerformancePrior).Named("heroHealingWeightPlayerOpponentPerformance");
            heroHealingWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var heroHealingCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var heroHealingCountVariance = Variable<double>.Random(heroHealingCountVariancePrior).Named("heroHealingCountVariance");
            heroHealingCountVariance.AddAttribute(new PointEstimate());

            // lastHits
            var lastHitsWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var lastHitsWeightPlayerPerformance = Variable<double>.Random(lastHitsWeightPlayerPerformancePrior).Named("lastHitsWeightPlayerPerformance");
            lastHitsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var lastHitsWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var lastHitsWeightPlayerOpponentPerformance = Variable<double>.Random(lastHitsWeightPlayerOpponentPerformancePrior).Named("lastHitsWeightPlayerOpponentPerformance");
            lastHitsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var lastHitsCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var lastHitsCountVariance = Variable<double>.Random(lastHitsCountVariancePrior).Named("lastHitsCountVariance");
            lastHitsCountVariance.AddAttribute(new PointEstimate());

            // observersPlaced
            var observersPlacedWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var observersPlacedWeightPlayerPerformance = Variable<double>.Random(observersPlacedWeightPlayerPerformancePrior).Named("observersPlacedWeightPlayerPerformance");
            observersPlacedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var observersPlacedWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var observersPlacedWeightPlayerOpponentPerformance = Variable<double>.Random(observersPlacedWeightPlayerOpponentPerformancePrior).Named("observersPlacedWeightPlayerOpponentPerformance");
            observersPlacedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var observersPlacedCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var observersPlacedCountVariance = Variable<double>.Random(observersPlacedCountVariancePrior).Named("observersPlacedCountVariance");
            observersPlacedCountVariance.AddAttribute(new PointEstimate());

            // observerKills
            var observerKillsWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var observerKillsWeightPlayerPerformance = Variable<double>.Random(observerKillsWeightPlayerPerformancePrior).Named("observerKillsWeightPlayerPerformance");
            observerKillsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var observerKillsWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var observerKillsWeightPlayerOpponentPerformance = Variable<double>.Random(observerKillsWeightPlayerOpponentPerformancePrior).Named("observerKillsWeightPlayerOpponentPerformance");
            observerKillsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var observerKillsCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var observerKillsCountVariance = Variable<double>.Random(observerKillsCountVariancePrior).Named("observerKillsCountVariance");
            observerKillsCountVariance.AddAttribute(new PointEstimate());

            // sentriesPlaced
            var sentriesPlacedWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var sentriesPlacedWeightPlayerPerformance = Variable<double>.Random(sentriesPlacedWeightPlayerPerformancePrior).Named("sentriesPlacedWeightPlayerPerformance");
            sentriesPlacedWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var sentriesPlacedWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var sentriesPlacedWeightPlayerOpponentPerformance = Variable<double>.Random(sentriesPlacedWeightPlayerOpponentPerformancePrior).Named("sentriesPlacedWeightPlayerOpponentPerformance");
            sentriesPlacedWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var sentriesPlacedCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(10, 10 * 10));
            var sentriesPlacedCountVariance = Variable<double>.Random(sentriesPlacedCountVariancePrior).Named("sentriesPlacedCountVariance");
            sentriesPlacedCountVariance.AddAttribute(new PointEstimate());

            // sentryKills
            var sentryKillsWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var sentryKillsWeightPlayerPerformance = Variable<double>.Random(sentryKillsWeightPlayerPerformancePrior).Named("sentryKillsWeightPlayerPerformance");
            sentryKillsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var sentryKillsWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var sentryKillsWeightPlayerOpponentPerformance = Variable<double>.Random(sentryKillsWeightPlayerOpponentPerformancePrior).Named("sentryKillsWeightPlayerOpponentPerformance");
            sentryKillsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var sentryKillsCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var sentryKillsCountVariance = Variable<double>.Random(sentryKillsCountVariancePrior).Named("sentryKillsCountVariance");
            sentryKillsCountVariance.AddAttribute(new PointEstimate());

            // stuns
            var stunsWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var stunsWeightPlayerPerformance = Variable<double>.Random(stunsWeightPlayerPerformancePrior).Named("stunsWeightPlayerPerformance");
            stunsWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var stunsWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var stunsWeightPlayerOpponentPerformance = Variable<double>.Random(stunsWeightPlayerOpponentPerformancePrior).Named("stunsWeightPlayerOpponentPerformance");
            stunsWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var stunsCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var stunsCountVariance = Variable<double>.Random(stunsCountVariancePrior).Named("stunsCountVariance");
            stunsCountVariance.AddAttribute(new PointEstimate());

            // teamfightParticipation
            var teamfightParticipationWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var teamfightParticipationWeightPlayerPerformance = Variable<double>.Random(teamfightParticipationWeightPlayerPerformancePrior).Named("teamfightParticipationWeightPlayerPerformance");
            teamfightParticipationWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var teamfightParticipationWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var teamfightParticipationWeightPlayerOpponentPerformance = Variable<double>.Random(teamfightParticipationWeightPlayerOpponentPerformancePrior).Named("teamfightParticipationWeightPlayerOpponentPerformance");
            teamfightParticipationWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var teamfightParticipationCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var teamfightParticipationCountVariance = Variable<double>.Random(teamfightParticipationCountVariancePrior).Named("teamfightParticipationCountVariance");
            teamfightParticipationCountVariance.AddAttribute(new PointEstimate());

            // totalGold
            var totalGoldWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var totalGoldWeightPlayerPerformance = Variable<double>.Random(totalGoldWeightPlayerPerformancePrior).Named("totalGoldWeightPlayerPerformance");
            totalGoldWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var totalGoldWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var totalGoldWeightPlayerOpponentPerformance = Variable<double>.Random(totalGoldWeightPlayerOpponentPerformancePrior).Named("totalGoldWeightPlayerOpponentPerformance");
            totalGoldWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var totalGoldCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var totalGoldCountVariance = Variable<double>.Random(totalGoldCountVariancePrior).Named("totalGoldCountVariance");
            totalGoldCountVariance.AddAttribute(new PointEstimate());

            // totalExperience
            var totalExperienceWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var totalExperienceWeightPlayerPerformance = Variable<double>.Random(totalExperienceWeightPlayerPerformancePrior).Named("totalExperienceWeightPlayerPerformance");
            totalExperienceWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var totalExperienceWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var totalExperienceWeightPlayerOpponentPerformance = Variable<double>.Random(totalExperienceWeightPlayerOpponentPerformancePrior).Named("totalExperienceWeightPlayerOpponentPerformance");
            totalExperienceWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var totalExperienceCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var totalExperienceCountVariance = Variable<double>.Random(totalExperienceCountVariancePrior).Named("totalExperienceCountVariance");
            totalExperienceCountVariance.AddAttribute(new PointEstimate());

            // towerDamage
            var towerDamageWeightPlayerPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(1, 10 * 10));
            var towerDamageWeightPlayerPerformance = Variable<double>.Random(towerDamageWeightPlayerPerformancePrior).Named("towerDamageWeightPlayerPerformance");
            towerDamageWeightPlayerPerformance.AddAttribute(new PointEstimate());

            var towerDamageWeightPlayerOpponentPerformancePrior = Variable.Observed(Gaussian.FromMeanAndVariance(-1, 10 * 10));
            var towerDamageWeightPlayerOpponentPerformance = Variable<double>.Random(towerDamageWeightPlayerOpponentPerformancePrior).Named("towerDamageWeightPlayerOpponentPerformance");
            towerDamageWeightPlayerOpponentPerformance.AddAttribute(new PointEstimate());

            var towerDamageCountVariancePrior = Variable.Observed(Gamma.FromMeanAndVariance(1, 10 * 10));
            var towerDamageCountVariance = Variable<double>.Random(towerDamageCountVariancePrior).Named("towerDamageCountVariance");
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
            // allMatches.AddAttribute(new Sequential());
            var nPlayersPerTeam = new Range(5).Named("nPlayersPerTeam");
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch");
            var teamSize = Variable.Observed(5.0).Named("teamSize");

            #endregion

            #region "Matches & Outcomes"

            // Array to hold the player lookup table. Let's us know which players played in which match
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
                    GenerateInMemory = true,
                    WriteSourceFiles = false,
                    UseParallelForLoops = true
                },
            };
            inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianFromMeanAndVarianceOp_PointVariance));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_PointB));
            // inferenceEngine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            
            var rawMatches = ReadMatchesFromFile("/mnt/win/Andris/Work/WIN/trueskill/ts.core/dota2_ts2_matches.json");

            const int batchSize = 32873; // total matches: 32873

            // dictionary keeping track of player skills
            var playerSkill = new Dictionary<long, Gaussian>();

            // dictionary keeping tack of the last time a player has played
            var globalPlayerLastPlayed = new Dictionary<long, DateTime>();
            
            foreach (var batch in rawMatches.Batch(batchSize))
            {
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
                var skillClassWidthPriorValue = inferenceEngine.Infer<Gamma>(skillClassWidth);
                var skillDynamicsPriorValue = inferenceEngine.Infer<Gamma>(skillDynamics);
                var skillSharpnessDecreasePriorValue = inferenceEngine.Infer<Gamma>(skillSharpnessDecrease);
                var killWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(killWeightPlayerPerformance);
                var killWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(killWeightPlayerOpponentPerformance);
                var killCountVariancePriorValue = inferenceEngine.Infer<Gamma>(killCountVariance);
                var deathWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deathWeightPlayerPerformance);
                var deathWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deathWeightPlayerOpponentPerformance);
                var deathCountVariancePriorValue = inferenceEngine.Infer<Gamma>(deathCountVariance);
                var assistsWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(assistsWeightPlayerPerformance);
                var assistsWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(assistsWeightPlayerOpponentPerformance);
                var assistsCountVariancePriorValue = inferenceEngine.Infer<Gamma>(assistsCountVariance);
                var campsStackedWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(campsStackedWeightPlayerPerformance);
                var campsStackedWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(campsStackedWeightPlayerOpponentPerformance);
                var campsStackedCountVariancePriorValue = inferenceEngine.Infer<Gamma>(campsStackedCountVariance);
                var deniesWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deniesWeightPlayerPerformance);
                var deniesWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(deniesWeightPlayerOpponentPerformance);
                var deniesCountVariancePriorValue = inferenceEngine.Infer<Gamma>(deniesCountVariance);
                var goldSpentWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(goldSpentWeightPlayerPerformance);
                var goldSpentWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(goldSpentWeightPlayerOpponentPerformance);
                var goldSpentCountVariancePriorValue = inferenceEngine.Infer<Gamma>(goldSpentCountVariance);
                var heroDamageWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(heroDamageWeightPlayerPerformance);
                var heroDamageWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(heroDamageWeightPlayerOpponentPerformance);
                var heroDamageCountVariancePriorValue = inferenceEngine.Infer<Gamma>(heroDamageCountVariance);
                var heroHealingWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(heroHealingWeightPlayerPerformance);
                var heroHealingWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(heroHealingWeightPlayerOpponentPerformance);
                var heroHealingCountVariancePriorValue = inferenceEngine.Infer<Gamma>(heroHealingCountVariance);
                var lastHitsWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(lastHitsWeightPlayerPerformance);
                var lastHitsWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(lastHitsWeightPlayerOpponentPerformance);
                var lastHitsCountVariancePriorValue = inferenceEngine.Infer<Gamma>(lastHitsCountVariance);
                var observersPlacedWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(observersPlacedWeightPlayerPerformance);
                var observersPlacedWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(observersPlacedWeightPlayerOpponentPerformance);
                var observersPlacedCountVariancePriorValue = inferenceEngine.Infer<Gamma>(observersPlacedCountVariance);
                var observerKillsWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(observerKillsWeightPlayerPerformance);
                var observerKillsWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(observerKillsWeightPlayerOpponentPerformance);
                var observerKillsCountVariancePriorValue = inferenceEngine.Infer<Gamma>(observerKillsCountVariance);
                var sentriesPlacedWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(sentriesPlacedWeightPlayerPerformance);
                var sentriesPlacedWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(sentriesPlacedWeightPlayerOpponentPerformance);
                var sentriesPlacedCountVariancePriorValue = inferenceEngine.Infer<Gamma>(sentriesPlacedCountVariance);
                var sentryKillsWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(sentryKillsWeightPlayerPerformance);
                var sentryKillsWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(sentryKillsWeightPlayerOpponentPerformance);
                var sentryKillsCountVariancePriorValue = inferenceEngine.Infer<Gamma>(sentryKillsCountVariance);
                var stunsWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(stunsWeightPlayerPerformance);
                var stunsWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(stunsWeightPlayerOpponentPerformance);
                var stunsCountVariancePriorValue = inferenceEngine.Infer<Gamma>(stunsCountVariance);
                var teamfightParticipationWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(teamfightParticipationWeightPlayerPerformance);
                var teamfightParticipationWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(teamfightParticipationWeightPlayerOpponentPerformance);
                var teamfightParticipationCountVariancePriorValue = inferenceEngine.Infer<Gamma>(teamfightParticipationCountVariance);
                var totalGoldWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(totalGoldWeightPlayerPerformance);
                var totalGoldWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(totalGoldWeightPlayerOpponentPerformance);
                var totalGoldCountVariancePriorValue = inferenceEngine.Infer<Gamma>(totalGoldCountVariance);
                var totalExperienceWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(totalExperienceWeightPlayerPerformance);
                var totalExperienceWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(totalExperienceWeightPlayerOpponentPerformance);
                var totalExperienceCountVariancePriorValue = inferenceEngine.Infer<Gamma>(totalExperienceCountVariance);
                var towerDamageWeightPlayerPerformancePriorValue = inferenceEngine.Infer<Gaussian>(towerDamageWeightPlayerPerformance);
                var towerDamageWeightPlayerOpponentPerformancePriorValue = inferenceEngine.Infer<Gaussian>(towerDamageWeightPlayerOpponentPerformance);
                var towerDamageCountVariancePriorValue = inferenceEngine.Infer<Gamma>(towerDamageCountVariance);
                
                skillClassWidthPrior.ObservedValue = skillClassWidthPriorValue;
                skillDynamicsPrior.ObservedValue = skillDynamicsPriorValue;
                skillSharpnessDecreasePrior.ObservedValue = skillSharpnessDecreasePriorValue;
                killWeightPlayerPerformancePrior.ObservedValue = killWeightPlayerPerformancePriorValue;
                killWeightPlayerOpponentPerformancePrior.ObservedValue = killWeightPlayerOpponentPerformancePriorValue;
                killCountVariancePrior.ObservedValue = killCountVariancePriorValue;
                deathWeightPlayerPerformancePrior.ObservedValue = deathWeightPlayerPerformancePriorValue;
                deathWeightPlayerOpponentPerformancePrior.ObservedValue = deathWeightPlayerOpponentPerformancePriorValue;
                deathCountVariancePrior.ObservedValue = deathCountVariancePriorValue;
                assistsWeightPlayerPerformancePrior.ObservedValue = assistsWeightPlayerPerformancePriorValue;
                assistsWeightPlayerOpponentPerformancePrior.ObservedValue = assistsWeightPlayerOpponentPerformancePriorValue;
                assistsCountVariancePrior.ObservedValue = assistsCountVariancePriorValue;
                campsStackedWeightPlayerPerformancePrior.ObservedValue = campsStackedWeightPlayerPerformancePriorValue;
                campsStackedWeightPlayerOpponentPerformancePrior.ObservedValue = campsStackedWeightPlayerOpponentPerformancePriorValue;
                campsStackedCountVariancePrior.ObservedValue = campsStackedCountVariancePriorValue;
                deniesWeightPlayerPerformancePrior.ObservedValue = deniesWeightPlayerPerformancePriorValue;
                deniesWeightPlayerOpponentPerformancePrior.ObservedValue = deniesWeightPlayerOpponentPerformancePriorValue;
                deniesCountVariancePrior.ObservedValue = deniesCountVariancePriorValue;
                goldSpentWeightPlayerPerformancePrior.ObservedValue = goldSpentWeightPlayerPerformancePriorValue;
                goldSpentWeightPlayerOpponentPerformancePrior.ObservedValue = goldSpentWeightPlayerOpponentPerformancePriorValue;
                goldSpentCountVariancePrior.ObservedValue = goldSpentCountVariancePriorValue;
                heroDamageWeightPlayerPerformancePrior.ObservedValue = heroDamageWeightPlayerPerformancePriorValue;
                heroDamageWeightPlayerOpponentPerformancePrior.ObservedValue = heroDamageWeightPlayerOpponentPerformancePriorValue;
                heroDamageCountVariancePrior.ObservedValue = heroDamageCountVariancePriorValue;
                heroHealingWeightPlayerPerformancePrior.ObservedValue = heroHealingWeightPlayerPerformancePriorValue;
                heroHealingWeightPlayerOpponentPerformancePrior.ObservedValue = heroHealingWeightPlayerOpponentPerformancePriorValue;
                heroHealingCountVariancePrior.ObservedValue = heroHealingCountVariancePriorValue;
                lastHitsWeightPlayerPerformancePrior.ObservedValue = lastHitsWeightPlayerPerformancePriorValue;
                lastHitsWeightPlayerOpponentPerformancePrior.ObservedValue = lastHitsWeightPlayerOpponentPerformancePriorValue;
                lastHitsCountVariancePrior.ObservedValue = lastHitsCountVariancePriorValue;
                observersPlacedWeightPlayerPerformancePrior.ObservedValue = observersPlacedWeightPlayerPerformancePriorValue;
                observersPlacedWeightPlayerOpponentPerformancePrior.ObservedValue = observersPlacedWeightPlayerOpponentPerformancePriorValue;
                observersPlacedCountVariancePrior.ObservedValue = observersPlacedCountVariancePriorValue;
                observerKillsWeightPlayerPerformancePrior.ObservedValue = observerKillsWeightPlayerPerformancePriorValue;
                observerKillsWeightPlayerOpponentPerformancePrior.ObservedValue = observerKillsWeightPlayerOpponentPerformancePriorValue;
                observerKillsCountVariancePrior.ObservedValue = observerKillsCountVariancePriorValue;
                sentriesPlacedWeightPlayerPerformancePrior.ObservedValue = sentriesPlacedWeightPlayerPerformancePriorValue;
                sentriesPlacedWeightPlayerOpponentPerformancePrior.ObservedValue = sentriesPlacedWeightPlayerOpponentPerformancePriorValue;
                sentriesPlacedCountVariancePrior.ObservedValue = sentriesPlacedCountVariancePriorValue;
                sentryKillsWeightPlayerPerformancePrior.ObservedValue = sentryKillsWeightPlayerPerformancePriorValue;
                sentryKillsWeightPlayerOpponentPerformancePrior.ObservedValue = sentryKillsWeightPlayerOpponentPerformancePriorValue;
                sentryKillsCountVariancePrior.ObservedValue = sentryKillsCountVariancePriorValue;
                stunsWeightPlayerPerformancePrior.ObservedValue = stunsWeightPlayerPerformancePriorValue;
                stunsWeightPlayerOpponentPerformancePrior.ObservedValue = stunsWeightPlayerOpponentPerformancePriorValue;
                stunsCountVariancePrior.ObservedValue = stunsCountVariancePriorValue;
                teamfightParticipationWeightPlayerPerformancePrior.ObservedValue = teamfightParticipationWeightPlayerPerformancePriorValue;
                teamfightParticipationWeightPlayerOpponentPerformancePrior.ObservedValue = teamfightParticipationWeightPlayerOpponentPerformancePriorValue;
                teamfightParticipationCountVariancePrior.ObservedValue = teamfightParticipationCountVariancePriorValue;
                totalGoldWeightPlayerPerformancePrior.ObservedValue = totalGoldWeightPlayerPerformancePriorValue;
                totalGoldWeightPlayerOpponentPerformancePrior.ObservedValue = totalGoldWeightPlayerOpponentPerformancePriorValue;
                totalGoldCountVariancePrior.ObservedValue = totalGoldCountVariancePriorValue;
                totalExperienceWeightPlayerPerformancePrior.ObservedValue = totalExperienceWeightPlayerPerformancePriorValue;
                totalExperienceWeightPlayerOpponentPerformancePrior.ObservedValue = totalExperienceWeightPlayerOpponentPerformancePriorValue;
                totalExperienceCountVariancePrior.ObservedValue = totalExperienceCountVariancePriorValue;
                towerDamageWeightPlayerPerformancePrior.ObservedValue = towerDamageWeightPlayerPerformancePriorValue;
                towerDamageWeightPlayerOpponentPerformancePrior.ObservedValue = towerDamageWeightPlayerOpponentPerformancePriorValue;
                towerDamageCountVariancePrior.ObservedValue = towerDamageCountVariancePriorValue;

            }
        }
        
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

                foreach (var (matchDate, listOfMatchesOnDate) in matchesByDate)
                foreach (var match in listOfMatchesOnDate)
                {
                    match.date = Convert.ToDateTime(matchDate);
                    if (match.radiant.Union(match.dire).All(player => player.steam_id != null)) matches.Add(match);
                }

                Console.WriteLine("OK.");

                return matches.OrderBy(x => x.match_id);
            }
        }

        private static double Validate(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return !IsValueMissing(value, min, max) ? value.Value : min;
        }

        private static bool IsValueMissing(double? value, double min = 0, double max = double.PositiveInfinity)
        {
            return value == null || value < min || value > max;
        }
    }
}
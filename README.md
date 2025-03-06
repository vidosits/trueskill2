## Contents

This repository contains a .NET implementation of the [TrueSkill 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf) rating system combined with [Trueskill Through Time](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf) extension.

The repository also contains a dotnet core 3.1 API server for running inference.

By turning off stats the system is equivalent to simple TrueSkill Through Time. If you only run it on one match then it's just TrueSkill 2. If you turn off stats and only run it on one match, then it's equivalent to simple TrueSkill (1).
## Origins

This work was done for a now defunct esports startup. The idea was that we would show algorithmically generated rankings and ratings of esports players.

While we did that for a while, it turned out that on the pre-match betting market predictions based on our ratings could beat some sportsbooks and hence this algorithm was made into a product.

While working on that our startup went through a couple pivots and eventually went under.
### What is GGScore?
Throughout the repository you'll find references to `GGScore`, which was the name of the feature of our website, which would show player ratings and rankings.
### Features
- Bayesian skill rating system based on TrueSkill 2
- TrueSkill Through Time (TTT) support
- Player decay with grace period support
- Batch inference
- Inference Server with API
- Adjustable player priors
- Adjustable stat priors
- Forward and backward inference
- MongoDB support for input data
### How can I run this?

You probably can't, unless you (re-)produce the input data in the format the API server expects it.

When in use, the rating service was fed with clean (pre-processed) data from Abios, so I can't attach a complete sample, but sample files are included in the `samples` directory for easi_er_ reproducability.

For running it on a set of matches you'll need the following:
- A file containing match ids along with stats (if you enable the usage of stats)
- A file containing player names for a more readable output
- A file containing player priors for seeding players
- A file containing stat priors

You can also check the API server's `/ratings` endpoint for supplying the above through an API call.
### Supplying player priors

If you have better priors than the default for a player's skill you can override it in the `player_priors` input. Make sure to use a list of `[μ, σ^2]`, i.e.: the parameters of the Gaussian prior.

### FAQ

- Q: Where is the stats support?
- A: commit c50caf12da1abcebdffb3714d2404cb5c5e53149

- Q: Do you take PRs/MRs?
- A: No

- Q: What games does it work for?
- A: Any of them, just supply the player priors and the stats priors if you have them enabled

- Q: How does it work?
- A: Long answer is in the TrueSkill 2 and TrueSkill Through Time (TTT) papers. Short answer is that we're using a probabilistic framework ([Infer.NET](https://github.com/dotnet/infer)) for constructing a graphical model and then running Bayesian inference on said model.

- Q: Can I re-implement this in another probabilistic framework?
- A: If it supports expectation progragation (EP) family of inference methods, then sure.

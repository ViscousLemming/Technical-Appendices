<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Search based learners

This page explains the non brute force learners in Layer 3.
They exist for one reason.
Exact enumeration is exponential in the vocabulary size.
So you often want a way to search the induced language without scanning every statement.

The important promise is simple.
Search learners do not change Stack Theory semantics.
They still use the exact correctness equation.
They only change search order.
If you cut them off early, they can fail to find the best policy.
That is a compute budget trade off, not a semantic change.

If you have not read learning.md yet, start there first.

## Shared vocabulary

These terms show up in several learners.

Candidate  
A candidate policy statement \(\pi\) represented by a statement mask.

Node  
A candidate plus any cached information needed to expand it.
For example the current truth set intersection.

Expansion  
Taking a node and generating its children by adding one more vocabulary element.

Budget  
A limit like max_expansions or max_nodes.
Budgets make a learner faster.
They can also make it incomplete.

Heuristic key  
A sortable score computed from PolicyStats.
Bigger keys are better.

Seed  
A fixed number that makes the random parts of a search repeatable.

## Best first search

Function  
`learn_best_first`

Plain English. Best first search keeps a priority queue of candidates.
On each step it expands the candidate with the best heuristic key.
Children are formed by adding one more vocabulary element that keeps the statement in the induced language.

Key parameters

`heuristic`  
The key function used to rank candidates.

`max_expansions`  
How many nodes to pop and expand before stopping.

`assume_monotone_key`  
If True the learner is allowed to stop early.
The stop rule is safe when adding elements cannot increase the key.
This is true for the built in weakness and description length keys.

`rng`  
An optional random number generator.
It is used only to randomise tie breaking among equal keys.

When to use  
When you want a principled search that often finds good policies early.

## Branch and bound

Function  
`learn_branch_and_bound`

Plain English. Branch and bound does a depth first walk over the induced language.
It keeps track of the best correct policy found so far.
When the current partial policy is already worse than the best policy, it prunes that whole branch.

Key parameters

`max_nodes`  
How many nodes to visit before stopping.

`assume_monotone_key`  
If True pruning is allowed.
It is safe under the same monotonicity condition described in heuristics.md.

When to use  
When you want a simple search that can prune aggressively under monotone keys.

## Beam search

Function  
`learn_beam_search`

Plain English. Beam search keeps only the top K candidates at each depth.
K is the beam width.
This makes it much cheaper than best first search, but it is not guaranteed to find the optimal policy.

Key parameters

`beam_width`  
How many candidates to keep per depth.

`max_depth`  
Maximum statement size to explore.
If None, it can explore up to the full vocabulary size.

`heuristic` and `rng`  
As in best first search.

When to use  
When you want a fast heuristic learner and you can tolerate suboptimal results.

## Random search baseline

Function  
`learn_random_search`

Plain English. Random search samples candidate statements uniformly from the induced language masks.
It checks correctness for each sample.
It returns the best correct policy found among the samples.

Key parameters

`n_samples`  
How many random candidates to test.

`seed`  
Controls the random samples.

When to use  
As a sanity check baseline.

## Genetic search

Function  
`learn_genetic`

Plain English. Genetic search keeps a population of candidate masks.
It repeatedly mutates and recombines them.
It uses the heuristic key as a fitness signal and the exact correctness check as the final filter.

Key parameters

`GeneticConfig.population_size`  
How many candidates are in the population.

`GeneticConfig.generations`  
How many evolution steps to run.

`GeneticConfig.seed`  
Controls randomness.

When to use  
When you want a broad global search that can jump around the space.

## Local search

Functions  
`learn_hill_climb` and `learn_simulated_annealing`

Plain English. Local search starts from a candidate and makes small edits.
Hill climbing accepts only improvements.
Simulated annealing sometimes accepts worse moves early, which helps it escape local traps.

Key parameters

`LocalSearchConfig.steps`  
How many moves to attempt per run.

`LocalSearchConfig.restarts`  
How many independent runs to do.

`LocalSearchConfig.seed`  
Controls randomness.

When to use  
When you want a cheap improvement method and you believe good policies live near each other.

## Reinforcement learning scaffolding

Module  
`stacktheory.layer3.rl_env`

Plain English. The suite includes a minimal environment interface where an agent constructs a policy mask one bit at a time.
This is not an RL algorithm.
It is a tiny wrapper that makes it easy to plug Stack Theory policy construction into an RL library if you want.

Key parameters

`max_steps`  
Maximum number of actions per episode.

`invalid_action_penalty`  
Penalty for adding an inconsistent program or repeating an already chosen one.

`terminal_correct_reward` and `terminal_incorrect_penalty`  
Terminal rewards based on whether the constructed policy is correct.

When to use  
When you want to treat policy search as a sequential decision problem.

## Reproducibility note

Search learners can involve randomness.
If you care about comparing results across learners, fix the seed.
Also record the vocabulary and task construction.
For Stack Theory, vocabulary choice is part of the experiment.

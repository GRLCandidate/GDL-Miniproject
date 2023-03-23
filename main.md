---
title: Stochastic Equivariant Subgraph GNNs for Large Heterophilic Datasets
author: "Candidate number: 1045966"
bibliography: references.biblatex
reference-section-title: References
header-includes: ["\\institute{University of Oxford}"]
documentclass: llncs
fontsize: 11pt
numbersections: true
toc: false
abstract: |
    Abstract!
---

Subgraph GNNs have been used successfully on tasks involving small (< 50 nodes) graphs [@bevilacquaEquivariantSubgraphAggregation2022] (TODO?? this might not be the best reference). I speculate that their large expressive power would provide an advantage on large heterophilic datasets. However, current subraph sampling techniques like ego-networks [TODO?? citation needed, also more examples] don't scale well with graph size.

In this paper, I investigate a novel architecture based on low-discrepancy stochastic sampling. I then compare it to ego-networks to ensure performance does not go down too much, and to benchmarks on large heterophilic datasets.

Contributions:

* A novel architecture for applying subgraph GNNs to large graphs
* An experimental regime demonstrating dominance (TODO?? calm down)
* A discussion of stochastic equivariance (I think if I try hard, I can take their [@bevilacquaEquivariantSubgraphAggregation2022] analysis and make it work out in expected value)

# Approach

* I use two benchmarks from the TUDataset [@morrisTUDatasetCollectionBenchmark2020]. MUTAG is smoll [@debnathStructureactivityRelationshipMutagenic1991a], with mean graph size nodes=17.9, edges=39.6, so I use it to show that my thing isn't shit. REDDIT-BINARY is big (mean nodes=429.6, mean edges=995.5) [@yanardagDeepGraphKernels2015], so I use it to show that my thing is fast.

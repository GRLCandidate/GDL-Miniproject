---
title: Investigating Adaptive Step Size in Gradient Flow TODO?? I can do better than that.
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

We have GNNs as gradient flows -- fuck knows what that's all about[@digiovanniGraphNeuralNetworks2022a]. But in there, we derive a GNN update equation _from energy_ (and it's exactly as new-agey as you'd expect). However, this is based on a Euler discretization, which is dumb because he came up with this stuff literal centuries ago and the world has moved on. I'm going to investigate whether I can save those valuable computes everyone's talking about by deriving my update equation from a method which picks a step size adaptively.

# Approach

1. Synthetic dataset with logical formulae, which in principle a GCN should be able to solve, since I can expresse it in graded first order logic [@barceloLogicalExpressivenessGraph2020].
2. Run GCN and GRAFF on experiments of increasing complexity, and wait until it all falls apart. Maybe try again with interleaved GRAFF and equivariant MLPL.
3. Profit!

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

1. For a _suitable_ dataset, do a grid search to find the best combination of T and $\tau$.
2. See if that can be improved upon using an adaptive step size.
3. Profit!

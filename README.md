# Hierarchical Mixtures of Experts

This repo contains two things: an implementation of the Hierarchical Mixtures of Experts (HME) model and a paper.

## 1. Implementation of HME

This is an implementation of the HME model with two levels of gating networks and Gaussian distributions at the experts.

The implementation is an application of the EM algorithm to the HME architecture. The maximization step of the EM algorithm is solved with Iteratively Reweighted Least Squares.

## 2. Paper on HME

The paper presents the probability model of HME and discusses details on the application of the EM algorithm to the HME and how IRLS is used to solve to the maximization step of the EM algorithm.

# Hierarchical Mixtures of Experts

This repo contains two things: an implementation of the Hierarchical Mixtures of Experts (HME) model and a paper.

## 1. Implementation of HME

This is an implementation of the HME model with two levels of gating networks and Gaussian distributions at the experts.

The implementation is an application of the EM algorithm to the HME architecture. The maximization step of the EM algorithm is solved with Iteratively Reweighted Least Squares.

<b>Results comparison on simulated data from a mixed normals distribution </b> (The normal distributions are linear in the input variables; the indicator random variables are generalized linear multinomial models. See details in tests.py.)

```
Results for hme

R^2 0.9336554286020846
fit time 0.038984060287475586
predict time 0.0002620220184326172



Results for sklearn_linear_regression

R^2 0.5893443197365751
fit time 0.0006158351898193359
predict time 4.410743713378906e-05



Results for sklearn_random_forest

R^2 0.91853684721737
fit time 0.033850908279418945
predict time 0.0015201568603515625



Results for sklean_gradient_boosting

R^2 0.9999999382966405
fit time 0.01305699348449707
predict time 0.0001499652862548828
```

## 2. Paper on HME

The paper presents the probability model of HME and discusses details on the application of the EM algorithm to the HME and how IRLS is used to solve to the maximization step of the EM algorithm.

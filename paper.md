---
title: 'EnsembleKalmanProcesses.jl: Derivative-free ensemble-based model calibration'
tags:
  - julia
  - optimization
  - bayesian
  - data assimilation
authors:
  - name: Oliver R. A. Dunbar
    corresponding: true 
    orcid: 0000-0001-7374-0382
    equal-contrib: true
    affiliation: 1 
  - name: Ignacio Lopez-Gomez
    orcid: 0000-0002-7255-5895
    equal-contrib: true 
    affiliation: 1
  - name: Alfredo Garbuno-Iñigo
    orcid: 0000-0003-3279-619X
    affiliation: 2
  - name: Daniel Zhengyu Huang
    orcid: 0000-0001-6072-9352
    affiliation: 1
  - name: Eviatar Bach
    orcid: 0000-0002-9725-0203
    affiliation: 1 
  - name: Jin-long Wu 
    orcid: 0000-0001-7438-4228
    affiliation: 3

affiliations:
 - name: Division of Geological and Planetary Sciences, California Institute of Technology
   index: 1
 - name: Department of Statistics, Mexico Autonomous Institute of Technology
   index: 2
 - name: Department of Mechanical Engineering, University of Wisconsin-Madison
   index: 3
date: 26 September 2022
bibliography: paper.bib
---

# Summary

EnsembleKalmanProcesses.jl is a Julia-based toolbox that can be used for a broad class of black-box gradient-free optimization problems. Specifically, the tools enable the optimization, or calibration, of parameters within a computer model in order to best match user-defined outputs of the model with available observed data [@kennedy_ohagan_2001]. Some of the tools can also approximately quantify parametric uncertainty [@Huang:2022b]. Though the package is written in Julia, a read–write TOML-file interface is provided so that the tools can be applied to computer models implemented in any language. Furthermore, the calibration tools are non-intrusive, relying only on the ability of users to compute an output of their model given a parameter value.

As the package name suggests, the tools are inspired by the well-established class of ensemble Kalman methods. Ensemble Kalman filters are currently one of the only practical ways to assimilate large volumes of observational data into models for operational weather forecasting [@Evensen:1994;@Houtekamer:1998;@Houtekamer:2001]. In the data assimilation setting, a computational weather model is integrated for a short time over a collection, or ensemble, of initial conditions, and the ensemble is updated frequently by a variety of atmospheric observations, allowing the forecasts to keep track of the real system.

The workflow is similar for ensemble Kalman processes. Here, a computer code is run (in parallel) for an ensemble of different values of the parameters that require calibration, producing an ensemble of outputs. This ensemble of outputs is then compared to observed data, and the parameters are updated to a new set of values which reduce the output–data misfit. The computer model is then evaluated for the new ensemble values and the outputs, under certain conditions, are guaranteed to lie closer to the observed data. The process is iterated until a user-defined criterion of convergence is met. Optimal values are selected from statistics of the final ensemble.

# Features

There are different ensemble Kalman algorithms in the literature, which differ in the way that the ensemble update is performed. The following ensemble Kalman processes are implemented tools in our package, and we provide published references for detailed descriptions and evidence of their efficacy:

* Ensemble Kalman Inversion (EKI, @Iglesias:2013),

* Ensemble Kalman Sampler (EKS, @Garbuno-Inigo:2020a;@Garbuno-Inigo:2020b),

* Unscented Kalman Inversion (UKI, @Huang:2022a),

* Sparse Ensemble Kalman Inversion (SEKI, @Schneider:2022).

We also implement some features to improve robustness and flexibility of the ensemble algorithms:

* The \texttt{ParameterDistribution} structure allows us to perform calibrations for parameters with known constraints. It does so by defining transformation maps under-the-hood from the constrained space to an unconstrained space where the optimization problem can be suitably defined. Constrained optimization using this framework has been successfully demonstrated in a variety of settings [@Lopez-Gomez:2022; @Dunbar2022; @Schneider2022covid].

* The \texttt{FailureHandler} structure allows calibrations to continue when several ensemble members fail. Common reasons for failure could be, for instance, simulation blow-up for certain parameter configurations, user termination of slow computations, data corruption, or bad nodes in a high-performance computing facility. This methodology is demonstrated in @Lopez-Gomez:2022.

* The \texttt{Localizer} structure allows us to overcome the restriction of the solution of the calibration to the linear span of the initial ensemble, and to reduce sampling errors due to the finite size of the ensemble. Various such localization and sampling error correction methods are implemented in EnsembleKalmanProcesses.jl [@Lee2021; @Tong2022].

* The TOML-file interface defined in the \texttt{UQParameters} module allows non-intrusive use of EnsembleKalmanProcesses.jl through \text{TOML} files, which are widely used for configuration files and easily read in any programming language. Given the computer model to calibrate and prior distributions on the parameters, EnsembleKalmanProcesses.jl reads these distributions from a file and, after an iteration of the ensemble Kalman algorithm, writes each member of the updated ensemble to a parameter file. Each of these parameter files can be then read individually to initiate the ensemble of the computer model for the next iteration.

# Statement of need

The task of estimating parameters of a computer model or simulator such that its outputs fit with data is ubiquitous in science and engineering, coming under many names such as calibration, inverse problems, and parameter estimation. In statistics and machine learning, when closed-form estimators of parameters of a model are unavailable, similar approaches may need to be employed to fit the model to data. There is a wide variety of algorithms to suit these applications; however, there are many bottlenecks in the practical application of such methods to computer codes:

* Legacy codes: Often code is old, and written in different languages than the packages implementing the calibration algorithms, requiring elaborate interfaces.

* Complex codes: Often large complex codes are difficult to change, so application of intrusive calibration tools to models can be challenging.

* Derivatives: When derivatives of a model output can be taken with respect to parameters, they can often improve the rate of convergence. But in many practical cases, these parameter-to-output maps are not differentiable; they may be chaotic or stochastic. Here one should not – or cannot – apply gradient-based methods.

* Lack of parallelism: There is now widespread access to high-performance computing clusters, cloud computing, and local multi-threading, and such facilities should be exploited where possible.

EnsembleKalmanProcesses.jl aims to provide a flexible and comprehensive solution to address these challenges:

1. It is embarrassingly parallel with respect to the ensemble; therefore, all computer model evaluations within an ensemble can happen simultaneously within an iteration.

2. It is derivative-free, and so is appropriate for computer codes for which derivatives are not available. The optimal updates are robust to noise.

3. It is non-intrusive and so can be applied to black-box computer codes written in any language or style, or to computer models for which the source code is not available to the user.

4. With scalability enhancements, such as the ones provided by the \texttt{Localizer} structure, it can be applied to high-dimensional problems.

# Research projects using the package

* EnsembleKalmanProcesses.jl has been used to train physics-based and machine-learning models of atmospheric turbulence and convection, implemented using \texttt{Flux.jl} and \texttt{TurbulenceConvection.jl} [@Lopez-Gomez:2022]. In this application, the available model outputs are not differentiable with respect to the learnable parameters, so gradient-based optimization was not an option. In addition, the unscented Kalman inversion algorithm was used to approximately quantify parameter uncertainty.

* EnsembleKalmanProcesses.jl features within Calibrate-Emulate-Sample (CES, @Cleary2021), a pipeline used to accelerate parameter uncertainty quantification (by a factor of $10^3$ - $10^4$ with respect to Monte Carlo methods) by using statistical emulators. EnsembleKalmanProcesses.jl is used to choose training points for these emulators. The training points are naturally concentrated by the ensemble Kalman processes into areas of high posterior probability mass. Within CES, the trained emulators are used to sample this probability distribution, and by design are most accurate where they need to be. CES has been successfully used to quantify parameter uncertainty within the moist convection scheme of a simplified climate model [@Dunbar2021;@Howland2022;@Dunbar2022], within a droplet collision-coalescence scheme for cloud microphyiscs [@Bieli:2022], and within boundary layer turbulence schemes for ocean modeling [@Hillier2022].

* EnsembleKalmanProcesses.jl has been used to learn hyperparameters within a machine learning tool known as Random Features within a julia package \texttt{RandomFeatures.jl}. Here, the hyperparameters characterize an infinite family of functions, from which a finite sample is drawn to use as a basis in regression problems. The objective for learning the parameters is noisy and non-differentiable due to the random sampling, so ensemble Kalman processes naturally perform well in this setting.


# Acknowledgements

We acknowledge contributions from several others who played a role in the evolution of this package. These include Jake Bolewski, Navid Constantinou, Gregory L. Wagner, Thomas Jackson, Michael Howland, Melanie Bieli, and Adeline Hillier. The development of this package was supported by the generosity of Eric and Wendy Schmidt by recommendation of the Schmidt Futures program, and by the Defense Advanced Research Projects Agency (Agreement No. HR00112290030).

# References

---
title: 'pyMechT: A Python package for mechanics of soft tissues'
tags:
  - mechanics
  - large deformation
  - hyperelasticity
  - soft tissues
  - biomechanics
  - ex-vivo testing
  - parameter estimation
  - Bayesian inference
authors:
  - name: Ankush Aggarwal
    orcid: 0000-0002-1755-8807
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Ross Williams
    orcid: 
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Claire Rosnel
    orcid: 0009-0000-0038-4321
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Silvia Renon
    orcid: 
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Jude M. Hussain
    orcid: 
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: André F. Schmidt
    orcid: 
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Shiting Huang
    orcid: 
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Sean McGinty
    orcid: 0000-0002-2428-2669
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
  - name: Andrew McBride
    orcid: 0000-0001-7153-3777
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted: "1, 2")
affiliations:
 - name: Glasgow Computational Engineering Centre (GCEC), James Watt School of Engineering, University of Glasgow, UK
   index: 1
date: 17 August 2024
bibliography: paper.bib

---

# Summary
 
`pyMechT` aims to fill an important gap for simulating simplified models in soft tissue mechanics, such as for ex-vivo testing protocols. It is straightforward to perform parameter estimation and Bayesian inference. Unique capabilities include incorporating layered structure of tissues and residual stresses.

# Statement of need

Mechanics of soft tissues plays an important role in several physiological problems, including cardio-vascular and musculoskeletal systems. Common ex-vivo biomechanical testing protocols used to characterize tissues include uniaxial extension for one-dimensional structures, such as tendons and ligaments, biaxial extension for planar tissues, such as heart valves and skin, and inflation-extension for tubular tissue structures, such as blood vessels. These experiments aim to induce a uniform deformation that can be easily related to the generated stresses. 

While several finite element analysis packages are available for performing biomechanical simulation, these are generally intended for more complex scenarios with non-uniform/non-affine deformations. For simulating the ex-vivo experiments, which induce close-to-uniform deformations, in-house codes are commonly developed. However, absence of a common framework can lead to lack of consistency and reproducibility. Moreover, advanced analyses require statistical approaches, such as Monte Carlo simulations and Bayesian inference. To fill this gap, we have developed the open-source Python package `pyMechT`.

# Structure

The package is implemented in Python using an object-oriented structure. The package builds up on widely-used Python libraries: NumPy, SciPy, Pandas, Matplotlib, and PyTorch. `pyMechT` consists of four main modules (see Figure \ref{fig:overview}): 1) MatModel for easily defining (new) material models, 2) SampleExperiment for simulating ex-vivo uniaxial/biaxial/inflation-extension experim-ents, 3) ParamFitter for performing parameter estimation based on experimental data, and 4) MCMC/RandomParameters for performing Bayesian inference using Monte Carlo (MC) or Markov Chain Monte Carlo (MCMC) simulations. 

![Structure of `pyMechT`{caption=Structure of `pyMechT`} \label{fig:overview}](../docs/source/drawing-1.svg)

A particular focus is on parameters, for which a custom dictionary has been implemented named ParamDict. This dictionary facilitates handling large number of parameters via string-based identifiers, and stores lower/upper bounds, fixed/variable flags, in addition to the current parameter values. The dictionary can also be saved/read as csv files. 

# References

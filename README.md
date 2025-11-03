# Functional ultrasound from scratch

A collection of notebooks demonstrating how functional ultrasound imaging (fUSI) works from scratch. Planned are these four notebooks:

1. Ultrasound propagation in homogeneous media: the wave equation, plane waves, focusing, apodization, steering
2. Structural ultrasound reconstruction: how scatterers create an image, the delay-and-sum algorithm
3. Functional ultrasound: measuring the movement of scatterers. Power Doppler, correlation functions, global motion and clutter removal with the SVD.
4. Analyzing real fUS data. Movement artifacts, hemodynamic response functions, General Linear Models

Currently, the first 3 are partially implemented.

## Why this series of notebooks?

There are many comprehensive references on ultrasound, e.g. Szabo et al. (2012). Meanwhile, functional ultrasound of the brain is mostly documented in papers, with information scattered in methods sections that can lack context. Much of the best simulation software is in Matlab, which makes functional ultrasound quite difficult to understand to the Python native (e.g. machine learning researchers). 

I wanted to create Python notebooks where you could transparently see how what the functional ultrasound signal is, from the oscillation of the transducers all the way to activation maps of the brain. I wanted to simulate ultrasound signals from scratch, with legible numpy and scipy code (as few frameworks as possible). That means the code takes some shortcuts, but it makes up for it in legibility and in allowing one to have a full view of the field in one snapshot.

## Related work/inspiration

* [NDK](https://github.com/agencyenterprise/neurotechdevkit): Focused ultrasound toolkit
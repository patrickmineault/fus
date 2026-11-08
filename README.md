# Functional ultrasound from scratch

A collection of notebooks demonstrating how functional ultrasound imaging (fUSI) works from scratch. These six notebooks are implemented:

1. Ultrasound propagation in homogeneous media: the wave equation, plane waves, focusing, apodization, steering
2. Structural ultrasound reconstruction: how scatterers create an image, the delay-and-sum algorithm
3. Functional ultrasound: measuring the movement of scatterers. Power Doppler, correlation functions, global motion and clutter removal with the SVD.
4. Analyzing real fUS data. Movement artifacts, hemodynamic response functions, General Linear Models
5. Decoding saccades from monkey LIP. Direction tuning maps, single-trial decoding with PCA + LDA, and whether fMRI-like resolution or sensitivity limits decoding
6. Doing it in the "real world". Full-wave simulation with k-Wave and what a skull does to the image, GPU beamforming with vbeam and mach, and a map of the ecosystem

## Why this series of notebooks?

There are many comprehensive references on ultrasound, e.g. Szabo et al. (2013). Meanwhile, functional ultrasound of the brain is mostly documented in papers, with information scattered in methods sections that can lack context. Much of the best simulation software is in Matlab, which makes functional ultrasound quite difficult to understand to the Python native (e.g. machine learning researchers). 

I wanted to create Python notebooks where you could transparently see how what the functional ultrasound signal is, from the oscillation of the transducers all the way to activation maps of the brain. I wanted to simulate ultrasound signals from scratch, with legible numpy and scipy code, with as few frameworks as possible. That means the code takes some shortcuts, but it makes up for it in legibility and in allowing one to have a full view of the field in one snapshot.

## Requirements

To maximize legibility, we do as much as possible in base Python, numpy, scipy and matplotlib. The 3rd tutorial uses Numba for acceleration. The 4th tutorial requires the installation of the [CaImAn package](https://github.com/flatironinstitute/CaImAn). The 5th tutorial uses `curl` to download a single session (~300 MB) of the [Griggs et al. (2025) dataset](https://data.caltech.edu/records/p5jan-02r60).

To set up the environment to run tutorials 1-5, in a clean environment:

```
mamba add caiman
pip install -r requirements.txt
```

The 6th tutorial is designed to run on Google Colab with a GPU, where it installs what it needs. Its packages require numpy 2, which conflicts with the numpy pin in `requirements.txt`, so to run it locally, use a separate environment:

```
pip install k-wave-python vbeam "jax[cpu]" numba tqdm jupyter
```

## Related work/inspiration

* [Ultraspy documentation](https://ultraspy.readthedocs.io/en/latest/algorithms/das_algo.html): Excellent tutorials on different reconstruction algorithms
* [NDK](https://github.com/agencyenterprise/neurotechdevkit): Neurotech devkit for focused ultrasound
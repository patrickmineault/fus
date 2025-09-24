# Functional ultrasound imaging sims

Simulates different phantoms in functional ultrasound to demonstrate the basic physical principles underlying functional ultrasound imaging (fUSI) and focused ultrasound (fUS).

## TODO

We'd like this to be a one-stop shop demonstrating signal processing from end-to-end. Currently, we demonstrate a toy structural and functional reconstruction in fUSI in a library-less way (i.e. Numpy and Scipy only). This is to facilitate understanding principles as opposed to practical implementations which may require specialized libraries and GPUs. 

Some of our plans include:

* Demonstrating IQ demodulation with a single scatterer
* Expanding on the delay-and-sum algorithm, demonstrating the sensitivity function of a single element at a single time delay
* Demonstrating acoustic field propagation for a homogeneous medium under dissipative conditions (i.e. replicating S4 here: https://www.medrxiv.org/content/10.1101/2025.08.19.25332261v1.full.pdf)
* Demonstrating simple propagation in focused ultrasound from scratch (i.e. Scenario 0 in NTK) starting from the (dissipative) wave equation


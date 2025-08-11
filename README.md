## Overview

Aquinas (**A** **qu**antum **in**terferometer **as**sembler) is a rudimentary software package for converting linear interferometers into digital quantum circuits. Given a maximum photon number $n$ and an $m \times m$ unitary matrix $U$ (that describes how the creation operators of an $m$ mode linear interferometer are transformed) our `direct_decomposition(U, n)` method returns a quantum circuit with around $m \log_2(n)$ qubits that can simulate the interferometer for any number state containing up to (and including) $n$ photons.

The depth of the interferometer circuits is approximately $n^4 \log n$. **It is therefore recommended** that you build within the range $n\leq 8$

## Methodology

We work in the second quantization picture to encode each of the $m$ interferometer modes with an $n+1$ dimensional quantum register containing between $0$ and $n$ indistinguishable photons. We first decompose $U$ into a grid of $2 \times 2$ unitaries (phase-shifting beamsplitters) using the method of [Clements et. al.](https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743) We then synthesise circuits for each of the beamsplitters and assemble them to make a circuit for the entire interferometer.

For further details, please consult our whitepaper (ArXiv link to come).

## Files

* src
    * `direct_decomposition.py`
        * Home of the `direct_decomposition(U, n)` method plus helper functions
    * `simulation.py`
        * Methods for encoding, decoding, and running sampling experiments. `run_interferom_simulation` is probably what you want.
    * `numeric_truncated_unitaries.py`
        * Methods for truncating ladder operators. (i.e. converting unbounded creation or annihilation operators into finite Hermitian matrices)
    * `boson_sampling_probabilities.py`
        * Borrowed code for calculating boson sampling probabilities using the Aaronson and Arkipov method. (Used to verify that our circuits are working)

* paper
    * `results_and_plots.ipynb`
        * The results and plots used in our whitepaper
    * `scaling_sizes.ipynb`
        * Looking at how circuit depth scales with $n$ for an arbitrary beamsplitter
    * `simulation_precision.ipynb`
        * Investigating the accuracy of our circuits for the two experiments

## Installation

**(Optional)**: It is recommended that you build this package in a *virtual environment* (conda or venv for example). If using conda, create a new environment with ``conda create --name <my-env>`` and activate it with ``conda activate <my-env>``

When ready to install, simply run:

`pip install Aquinas`

## Thanks!

Thank you for taking an interest in this project. Please don't hesitate to contact if there are any issues or if there's anything that I can improve on.
